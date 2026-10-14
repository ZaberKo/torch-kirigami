"""Shared identity proofs agree with propagation and reject unproved mappings."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from torch_kirigami import AxisRelation, DependencyGraph, IndexSet, Region, TensorRef
from torch_kirigami.dep_graph.relations import (
    AxisPort,
    BlockMap,
    BroadcastRelation,
    ReshapeRelation,
)
from torch_kirigami.dep_graph.selection import full_region
from torch_kirigami.pruning import IdentityAxisIndex


def test_identity_component_matches_every_single_position(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    root = graph.parameter("0.weight").axis(0)
    index = IdentityAxisIndex(graph.relations)
    axes = index.component(root)
    assert axes is not None
    for position in range(6):
        actual = graph.propagate(remove=(root.select([position]),))
        expected = {axis.tensor.id: axis.select([position]) for axis in axes}
        assert expected == dict(actual.selections)
    assert all(index.component(axis) == axes for axis in axes)


def test_full_scope_has_the_same_identity_proof_as_an_implicit_scope(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    root = graph.parameter("0.weight").axis(0)
    explicit = tuple(
        replace(
            relation,
            left=replace(relation.left, scope=full_region(relation.left.tensor.shape)),
            right=replace(relation.right, scope=full_region(relation.right.tensor.shape)),
        )
        if type(relation) is AxisRelation
        else relation
        for relation in graph.relations
    )
    ordinary, scoped = IdentityAxisIndex(graph.relations), IdentityAxisIndex(explicit)
    assert scoped.component(root) == ordinary.component(root)
    assert scoped.equivalent_axes(root) == ordinary.equivalent_axes(root)


def test_identity_component_rejects_partial_scopes_extensions_and_full_width(execution_device):
    class ExtendedRelation(AxisRelation):
        """A subclass is not covered by the exact built-in proof."""

    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    root = graph.parameter("0.weight").axis(0)
    relations = list(graph.relations)
    offset = next(
        i
        for i, relation in enumerate(relations)
        if type(relation) is AxisRelation and root.tensor in relation.refs
    )
    relation = relations[offset]
    partial_axes = list(full_region(relation.left.tensor.shape).axes)
    partial_axes[relation.left.axis.dim] = IndexSet.span(
        0, relation.left.tensor.shape[relation.left.axis.dim] - 1
    )
    relations[offset] = replace(
        relation, left=replace(relation.left, scope=Region(tuple(partial_axes)))
    )
    assert IdentityAxisIndex(relations).component(root) is None
    relations[offset] = ExtendedRelation(
        relation.left, relation.right, relation.maps, relation.reason
    )
    assert IdentityAxisIndex(relations).component(root) is None
    singleton = nn.Linear(3, 1)
    small = DependencyGraph.build(singleton, args=(torch.randn(2, 3),))
    assert IdentityAxisIndex(small.relations).component(small.parameter("weight").axis(0)) is None


@pytest.mark.parametrize("nested", ["mapping_count", "scope_indices", "tensor_extent"])
def test_identity_proofs_reject_nested_extension_records(nested):
    """Value equality cannot prove the behavior of an embedded extension."""

    class ExtendedInteger(int):
        """An accepted scalar subclass remains outside native arithmetic proofs."""

    class ExtendedIndices(IndexSet):
        """A full-looking scope may implement different selection behavior."""

    left = TensorRef("left", (ExtendedInteger(4) if nested == "tensor_extent" else 4,))
    right = TensorRef("right", (4,))
    root = left.axis(0)
    scope = Region((ExtendedIndices(((0, 4),)),)) if nested == "scope_indices" else None
    count = ExtendedInteger(4) if nested == "mapping_count" else 4
    relation = AxisRelation(
        AxisPort(root, scope), AxisPort(right.axis(0)), (BlockMap(0, 0, count),)
    )
    index = IdentityAxisIndex((relation,))
    assert index.component(root) is None
    assert index.equivalent_axes(root) == (root,)


@pytest.mark.parametrize("relation_type", [BroadcastRelation, ReshapeRelation])
def test_non_axis_relations_do_not_hide_nested_tensor_extensions(relation_type):
    class ExtendedInteger(int):
        """Equal dimensions do not prove ordinary arithmetic semantics."""

    left = TensorRef("left", (4,))
    right = TensorRef("right", (ExtendedInteger(4),))
    index = IdentityAxisIndex((relation_type(left, right),))
    assert index.component(left.axis(0)) is None
    assert index.equivalent_axes(left.axis(0)) == (left.axis(0),)


def test_equivalent_seeds_keep_full_closure_with_nonidentity_relations(execution_device):
    model = nn.Sequential(
        nn.Conv2d(4, 6, 1, groups=2), nn.BatchNorm2d(6), nn.ReLU(), nn.Conv2d(6, 2, 1)
    ).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 4, 2, 2),))
    root = graph.parameter("0.weight").axis(0)
    index = IdentityAxisIndex(graph.relations)
    assert index.component(root) is None
    axes = index.equivalent_axes(root)
    assert len(axes) > 1
    for positions in ([0], [3], [0, 3], list(range(6))):
        reference = graph.propagate(remove=(root.select(positions),))
        for axis in axes:
            actual = graph.propagate(remove=(axis.select(positions),))
            assert dict(actual.selections) == dict(reference.selections)
            assert actual.diagnostics == reference.diagnostics
    assert all(index.equivalent_axes(axis) == axes for axis in axes)
