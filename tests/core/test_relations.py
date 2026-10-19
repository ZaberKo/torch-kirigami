"""core / relations contracts."""

import itertools

import pytest
import torch
from torch import nn

from tests.support.graph_helpers import removed
from torch_kirigami import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    DependencyGraph,
    IndexSet,
    PermuteRelation,
    Region,
    ReshapeRelation,
    Selection,
    SliceRelation,
    TensorRef,
)


def test_broadcast_partial_regions_match_dense_reference():
    from torch_kirigami import BroadcastRelation

    small = TensorRef("small", (1, 3, 4))
    big = TensorRef("big", (2, 3, 4))
    chosen = Selection(small, (Region((IndexSet.of([0]), IndexSet.of([1]), IndexSet.of([2]))),))
    relation = BroadcastRelation(small, big)
    forward = relation.propagate(chosen)[0]
    assert forward.count == 2
    assert relation.propagate(forward)[0] == chosen
    # Removing only one broadcast copy cannot remove the shared source position.
    partial = Selection(big, (Region((IndexSet.of([0]), IndexSet.of([1]), IndexSet.of([2]))),))
    assert not relation.propagate(partial)[0]


def test_reshape_keeps_large_token_prefix_symbolic():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 24)

        def forward(self, x):
            return self.proj(x).reshape(x.shape[0], x.shape[1], 3, 4, 2)

    graph = DependencyGraph.build(Model(), args=(torch.randn(1, 5000, 8),))
    call = next(c for c in graph.operations() if c.name == "reshape")
    impact = graph.propagate(remove=[call.outputs[0].axis(3).select([1])])
    assert impact.status == "resolved"
    assert set(impact.selection(graph.parameter("proj.weight")).fully_selected_indices(0)) == {
        2,
        3,
        10,
        11,
        18,
        19,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"source_start": -1},
        {"target_start": True},
        {"count": 1.0},
        {"source_block": 0},
        {"target_block": 0},
        {"require_full_source": 1},
    ],
)
def test_block_map_constructor_rejects_invalid_fields(changes):
    with pytest.raises((ValueError, TypeError)):
        BlockMap(**{"source_start": 0, "target_start": 0, "count": 2, **changes})


def test_relation_constructors_and_endpoint_checks():
    a, b = TensorRef("a", (2, 3)), TensorRef("b", (3, 2))
    with pytest.raises(ValueError):
        ReshapeRelation(a, TensorRef("wrong", (2, 4)))
    with pytest.raises(ValueError):
        PermuteRelation(a, b, (0, 0))
    with pytest.raises(ValueError):
        PermuteRelation(a, b, (0, 1))
    with pytest.raises(ValueError):
        AxisRelation(AxisPort(a.axis(0)), AxisPort(b.axis(1)), (BlockMap(0, 0, 3),))
    with pytest.raises(IndexError):
        AxisPort(a.axis(0), Region((IndexSet.span(0, 3), IndexSet.span(0, 3))))
    relations = (
        AxisRelation.equal(a.axis(0), b.axis(1)),
        ReshapeRelation(a, b),
        PermuteRelation(a, b, (-1, 0)),
        BroadcastRelation(TensorRef("small", (1, 3)), a),
        SliceRelation(a, TensorRef("slice", (2, 2)), (slice(None), slice(1, 3))),
    )
    for relation in relations:
        with pytest.raises(ValueError, match="endpoint"):
            relation.propagate(Selection(TensorRef("foreign", (2, 3))))
    with pytest.raises(ValueError):
        AxisPort(a.axis(0)).fully_selected_indices(Selection(TensorRef("foreign", a.shape)))


@pytest.mark.parametrize(
    "index, shape",
    [
        ((slice(None),), (2,)),
        ((2, slice(None)), (3,)),
        ((slice(None), slice(None, None, -1)), (2, 3)),
        ((slice(None), slice(1, 3)), (2, 3)),
        ((True, slice(None)), (3,)),
    ],
)
def test_slice_constructor_validates_index_and_output(index, shape):
    with pytest.raises((ValueError, IndexError, TypeError)):
        SliceRelation(TensorRef("big", (2, 3)), TensorRef("small", shape), index)


def test_broadcast_does_not_delete_scalar_operand():
    class Model(nn.Module):
        def forward(self, x, bias):
            return x + bias

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 4, 3), torch.randn(1, 4, 1)))
    call = graph.operations()[0]
    result = graph.propagate(remove=[call.outputs[0].axis(0).select([0])])
    assert not result.selection(call.inputs[1])
    result = graph.propagate(remove=[call.outputs[0].axis(1).select([1])])
    assert removed(result, call.inputs[1], 1) == {1}


def test_axis_port_full_and_partitioned_scopes_match_coordinate_fibers():
    """Use explicit coordinate sets independently of interval subtraction."""
    ref = TensorRef("port-fibers", (2, 3))
    positions = tuple(itertools.product(range(2), range(3)))
    scopes = (
        None,
        Region((IndexSet.span(0, 2), IndexSet.span(0, 3))),
        Region((IndexSet.span(0, 2), IndexSet(((0, 1), (2, 3))))),
        Region((IndexSet.span(1, 2), IndexSet.span(1, 3))),
    )
    for mask in range(64):
        selected = {p for i, p in enumerate(positions) if mask & (1 << i)}
        selection = Selection(
            ref, tuple(Region(tuple(IndexSet.span(i, i + 1) for i in point)) for point in selected)
        )
        for dim, scope in itertools.product(range(2), scopes):
            port = AxisPort(ref.axis(dim), scope)
            fibers = set(itertools.product(*(tuple(axis) for axis in port.region.axes)))
            expected = {
                i for i in port.region.axes[dim] if {p for p in fibers if p[dim] == i} <= selected
            }
            assert set(port.fully_selected_indices(selection)) == expected


@pytest.mark.parametrize("extension", ["selection", "region", "indices", "extent"])
def test_port_fast_path_retains_explicit_scope_for_extensions(extension, monkeypatch):
    class ExtendedSelection(Selection):
        pass

    class ExtendedRegion(Region):
        pass

    class ExtendedIndices(IndexSet):
        pass

    class ExtendedExtent(int):
        pass

    ref = TensorRef("port-extension", (ExtendedExtent(2) if extension == "extent" else 2, 3))
    indices = ExtendedIndices.span(0, 1) if extension == "indices" else IndexSet.span(0, 1)
    region_type = ExtendedRegion if extension == "region" else Region
    selection_type = ExtendedSelection if extension == "selection" else Selection
    selection = selection_type(ref, (region_type((indices, IndexSet.span(0, 3))),))
    port = AxisPort(ref.axis(0))
    original = Selection.fully_selected_indices
    scopes = []

    def observed(self, dim, scope=None):
        scopes.append(scope)
        return original(self, dim, scope)

    monkeypatch.setattr(Selection, "fully_selected_indices", observed)
    assert port.fully_selected_indices(selection) == IndexSet.span(0, 1)
    assert scopes == [port.region]
