"""Operation counts guard batching; numerical checks remain independent."""

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, IndexSet, Region, Selection
from torch_kirigami.pruning import CandidateSpace, ParameterGroup
from torch_kirigami.pruning.groups import unique_groups
from torch_kirigami.sparsity import GroupLasso, scale_groups_, set_group_norms_
from torch_kirigami.sparsity import values as sparse_values


def test_disjoint_groups_do_not_require_pairwise_equivalence(monkeypatch):
    model = nn.Linear(2, 128, bias=False)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2),))
    groups = tuple(
        ParameterGroup(graph, (graph.parameter("weight").axis(0).select([i]),)) for i in range(128)
    )
    original, comparisons = ParameterGroup.equivalent, []

    def count(left, right):
        comparisons.append((left, right))
        return original(left, right)

    monkeypatch.setattr(ParameterGroup, "equivalent", count)
    regularizer = GroupLasso(groups + groups)
    assert len(regularizer.groups) == 128 and len(comparisons) <= 128
    torch.testing.assert_close(regularizer(), model.weight.norm(dim=1).sum())


def test_equivalence_buckets_preserve_geometric_equality_and_resolve_collisions():
    model = nn.Linear(3, 3, bias=False)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3),))
    ref = graph.parameter("weight")
    a = ref.axis(0).select([0]).union(ref.axis(1).select([0]))
    b = Selection(
        ref,
        (
            Region((IndexSet.span(0, 3), IndexSet.of([0]))),
            Region((IndexSet.of([0]), IndexSet.span(1, 3))),
        ),
    )
    # Diagonals have equal element counts and identical axis projections,
    # but must remain different groups.
    diagonals = [
        Selection(
            ref, tuple(Region((IndexSet.of([i]), IndexSet.of([j]))) for i, j in enumerate(columns))
        )
        for columns in ((0, 1, 2), (2, 1, 0))
    ]
    groups = tuple(ParameterGroup(graph, (s,)) for s in (a, b, *diagonals))
    assert len(unique_groups(groups)) == 3
    with pytest.raises(ValueError, match="conflicting coefficients"):
        GroupLasso(groups, coefficients=(1, 2, 1, 1))


def test_scaling_gathers_only_the_combined_union(monkeypatch, execution_device):
    model = nn.Linear(3, 3, bias=False)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3),))
    ref = graph.parameter("weight")
    groups = tuple(ParameterGroup(graph, (ref.axis(0).select([i]),)) for i in range(3))
    original, calls = sparse_values.gather_region, []

    def gather(tensor, region):
        calls.append(region)
        return original(tensor, region)

    monkeypatch.setattr(sparse_values, "gather_region", gather)
    before = model.weight.detach().clone()
    scale_groups_(groups * 2, 0.5)
    torch.testing.assert_close(model.weight, before * 0.5)
    assert len(calls) == 1


@pytest.mark.parametrize("activation", [nn.Identity, nn.ReLU])
def test_protected_pointwise_domains_do_not_enumerate_channels(activation, monkeypatch):
    model = nn.Sequential(nn.Linear(4, 128), activation())
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4),))
    original, calls = graph.propagate, []

    def propagate(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "propagate", propagate)
    space = CandidateSpace(graph)
    assert not space.axes and not space.candidates
    assert not calls


def test_group_bucketing_does_not_exceed_valid_region_coordinate_budgets():
    model = nn.Linear(16384, 64, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 16384),))
    ref = graph.parameter("weight")
    selection = Selection(
        ref,
        tuple(
            Region((IndexSet.of([row]), IndexSet.of(range(256 * row, 256 * (row + 1), 2))))
            for row in range(64)
        ),
    )
    group = ParameterGroup(graph, (selection,))
    assert selection.count == 8192 and len(selection.regions) == 64
    assert unique_groups((group, group)) == (group,)
    loss = GroupLasso((group, group))()
    torch.testing.assert_close(loss, loss.new_tensor(8192**0.5))
    loss.backward()
    expected = torch.zeros_like(model.weight)
    for row in range(64):
        expected[row, 256 * row : 256 * (row + 1) : 2] = 1 / 8192**0.5
    torch.testing.assert_close(model.weight.grad, expected)


def test_bucket_collisions_do_not_make_equality_depend_on_difference_fragmentation():
    model = nn.Linear(40960, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 40960),))
    ref = graph.parameter("weight")
    left = IndexSet(tuple((10 * i, 10 * i + 4) for i in range(4096)))
    right = IndexSet(
        ((0, 10240), *((10 * i + 1, 10 * i + 3) for i in range(1025, 4095)), (40950, 40954))
    )
    selections = tuple(
        Selection(ref, (Region((IndexSet.of([0]), indices)),)) for indices in (left, right)
    )
    assert selections[0].count == selections[1].count == 16384
    assert set(left) != set(right)
    assert selections[0] != selections[1] and selections[1] != selections[0]
    groups = tuple(ParameterGroup(graph, (s,)) for s in selections)
    expected = torch.zeros_like(model.weight)
    for indices in (left, right):
        expected[0, list(indices)] += 1 / 16384**0.5
    for ordered in (groups, groups[::-1]):
        assert len(unique_groups(ordered)) == 2
        loss = GroupLasso(ordered)()
        torch.testing.assert_close(loss, loss.new_tensor(256))
        (gradient,) = torch.autograd.grad(loss, model.weight)
        torch.testing.assert_close(gradient, expected)


def test_norm_projection_preserves_small_coordinates_beside_large_values(execution_device):
    model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[1e300, 1e-300]]))
    original = model.weight.detach().clone()
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=torch.float64),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    set_group_norms_((group,), (1e300,))
    torch.testing.assert_close(model.weight, original, rtol=1e-12, atol=0)
