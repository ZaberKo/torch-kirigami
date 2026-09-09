"""graph / propagation contracts."""

import torch
from torch import nn

from tests.support.graph_helpers import indices
from torch_kirigami import (
    Balanced,
    DependencyGraph,
    Fixed,
    IndexSet,
)


def test_linear_chain_and_joint_requests(execution_device):
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    a = graph.parameter("0.weight").axis(0).select([1])
    b = graph.parameter("2.weight").axis(1).select([4])
    impact = graph.propagate(remove=[a, b])
    assert impact.status == "resolved"
    assert indices(impact, graph.parameter("0.bias"), 0) == {1, 4}
    assert indices(impact, graph.parameter("2.weight"), 1) == {1, 4}
    assert impact.selections == graph.propagate(remove=[b, a, a]).selections
    assert impact.selections == graph.propagate(remove=list(impact.selections.values())).selections
    assert "via" in graph.explain(impact)
    protected = graph.propagate(remove=[a], constraints=[Fixed(graph.parameter("0.bias").axis(0))])
    assert protected.status == "conflict"


def test_unequal_group_choices_are_not_silently_completed():
    graph = DependencyGraph.build(nn.Conv2d(6, 4, 1, groups=2), args=(torch.randn(2, 6, 3, 3),))
    request = graph.calls("")[0].input().axis(1).select([0])
    impact = graph.propagate(remove=[request])
    assert impact.status == "unresolved"
    assert indices(impact, request.tensor, 1) == {0}
    assert any(d.code == "unbalanced_groups" for d in impact.diagnostics)


def test_multiple_partition_constraints():
    graph = DependencyGraph.build(nn.Linear(12, 12), args=(torch.randn(2, 12),))
    axis = graph.parameter("weight").axis(0)
    constraints = [
        Balanced(axis, tuple(IndexSet.span(i, i + size) for i in range(0, 12, size)))
        for size in (6, 4)
    ]
    impact = graph.propagate(remove=[axis.select([0, 4, 8])], constraints=constraints)
    assert impact.status == "unresolved"
    assert any(d.code == "unbalanced_groups" for d in impact.diagnostics)
