"""integration / shared constraints contracts."""

import torch
from torch import nn

from tests.support.graph_helpers import indices
from tests.support.pruning import StaticMetric, build
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
)


def test_shared_convolution_and_linear_require_both_layouts():
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 1))

        def forward(self, image, vector):
            return F.conv1d(image, self.weight, groups=2), F.linear(vector, self.weight.squeeze(-1))

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 6, 5), torch.randn(2, 3)))
    input_ = next(v for v in graph.values() if v.kind == "input" and len(v.shape) == 3)
    result = graph.propagate(remove=[input_.axis(1).select([0, 4])])
    assert result.status == "unresolved"
    assert any(d.code == "unsupported_layout" for d in result.diagnostics)


def test_shared_weight_different_groupings_are_intersected():
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 1))

        def forward(self, a, b):
            return F.conv1d(a, self.weight, groups=2), F.conv1d(b, self.weight, groups=4)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 6, 5), torch.randn(2, 12, 5)))
    input_ = next(v for v in graph.values() if v.kind == "input" and v.shape[1] == 12)
    invalid = graph.propagate(remove=[input_.axis(1).select([0, 4, 6, 10])])
    assert invalid.status == "unresolved"
    valid = graph.propagate(remove=[input_.axis(1).select([0, 3, 7, 10])])
    assert valid.status == "resolved"


def test_normalization_cannot_bypass_shared_weight_layout():
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 1))

        def forward(self, image, normalized):
            return F.conv1d(image, self.weight, groups=2), F.layer_norm(
                normalized, (4, 3, 1), weight=self.weight
            )

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 6, 5), torch.randn(2, 4, 3, 1)))
    input_ = next(v for v in graph.values() if v.kind == "input" and len(v.shape) == 3)
    impact = graph.propagate(remove=[input_.axis(1).select([0, 4])])
    assert impact.status == "unresolved"
    assert any(
        d.code == "unsupported_layout" and d.node == "layer_norm" for d in impact.diagnostics
    )


def test_shared_module_paths_and_calls(execution_device):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 4)
            self.right = self.left

        def forward(self, x):
            return self.left(x) + self.right(x)

    graph = DependencyGraph.build(Shared(), args=(torch.randn(2, 4),))
    assert graph.parameter("left.weight") is graph.parameter("right.weight")
    assert len(graph.calls("left")) == len(graph.calls("right")) == 2
    result = graph.propagate(remove=[graph.parameter("left.weight").axis(0).select([2])])
    assert result.status == "resolved"
    assert all(indices(result, c.output(), 1) == {2} for c in graph.calls("left"))


def test_residual_and_shared_weight_cycle_reaches_fixed_point(execution_device):
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4, bias=False)
            self.b = nn.Linear(4, 4, bias=False)
            self.b.weight = self.a.weight

        def forward(self, x):
            return self.a(x) + self.b(x)

    graph = DependencyGraph.build(Residual(), args=(torch.randn(2, 4),))
    impact = graph.propagate(remove=[graph.calls("a")[0].output().axis(1).select([1, 3])])
    assert impact.status == "resolved"
    assert indices(impact, graph.calls("b")[0].output(), 1) == {1, 3}


def test_shared_module_parameters_and_aliases(execution_device):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)
            self.alias = self.a
            self.b = nn.Linear(4, 6)
            self.b.weight = self.a.weight
            self.out = nn.Linear(6, 2)

        def forward(self, x):
            return self.out(self.a(x) + self.alias(x) + self.b(x))

    model = Shared()
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    old = model.a.weight
    plan = pruner.plan_remove([graph.parameter("a.weight").axis(0).select([1, 3])])
    _returned, result = pruner.apply(plan)
    assert model.a.weight is model.alias.weight is model.b.weight
    assert len([p for p in result.parameter_map if p is old]) == 1
    assert model(x).shape == (2, 2)


def test_joint_rows_columns_greedy_grouped_chain(execution_device):
    model = nn.Sequential(nn.Conv1d(4, 6, 1), nn.Conv1d(6, 6, 1, groups=2), nn.Conv1d(6, 2, 1))
    x = torch.randn(2, 4, 5)
    _graph, pruner = build(model, x)

    @StaticMetric
    def metric(ctx, batch):
        return [
            (0 if c.key.startswith("0.") else 10)
            + (0 if int(c.key.rsplit(":", 1)[1]) in (0, 4) else 2)
            for c in batch
        ]

    plan = pruner.plan(
        pruner.discover_candidates(), budget=ChannelRatio(0.34), strategy=Greedy(metric)
    )
    assert plan.selection_report.removed == (2, 2)
    pruner.apply(plan)
    model(x).sum().backward()
    assert model[1].weight.shape == (4, 2, 1)
