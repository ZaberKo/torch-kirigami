"""operators / shapes contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)


def test_squeeze_requires_rank_preserving_rewrite_when_axis_becomes_one():
    class Model(nn.Module):
        def forward(self, x):
            return x.squeeze()

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 3),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([0, 1])])
    assert impact.status == "resolved"
    assert any(r.kind == "dimension_transform" for r in impact.requirements)


def test_unflatten_has_explicit_attribute_binding():
    graph = DependencyGraph.build(nn.Unflatten(1, (4, 2)), args=(torch.randn(2, 8),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([2, 3])])
    assert impact.status == "resolved"
    req = next(r for r in impact.requirements if r.kind == "attribute")
    assert req.target == "unflattened_size"
    assert tuple(a.dim for a in dict(req.data)["axes"]) == (1, 2)


@pytest.mark.parametrize(
    "mode", ["repeat", "interleave", "tile", "stack", "chunk", "narrow", "unfold"]
)
def test_axis_family_compaction(mode, execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 1)

        def forward(self, x):
            y = self.conv(x)
            if mode == "repeat":
                return y.repeat(1, 2, 1, 1)
            if mode == "interleave":
                return y.repeat_interleave(2, dim=1)
            if mode == "tile":
                return torch.tile(y, (1, 2, 1, 1))
            if mode == "stack":
                return torch.stack([y, y], dim=2)
            if mode == "chunk":
                return y.chunk(2, dim=1)
            if mode == "narrow":
                return y.narrow(2, 0, 2)
            return F.unfold(y, 2)

    model = Net()
    x = torch.randn(2, 3, 4, 4)
    old = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("conv.weight").axis(0).select([1, 4])], preserve_io=False
    )
    if mode == "chunk":
        a, b = old(x)
        expected = (a[:, [0, 2]], b[:, [0, 2]])
    elif mode in ("repeat", "tile"):
        expected = old(x)[:, [0, 2, 3, 5, 6, 8, 9, 11]]
    elif mode == "interleave":
        expected = old(x)[:, [0, 1, 4, 5, 6, 7, 10, 11]]
    elif mode == "unfold":
        expected = F.unfold(old.conv(x)[:, [0, 2, 3, 5]], 2)
    else:
        expected = old(x)[:, [0, 2, 3, 5]]
    torch.testing.assert_close(model(x), expected)


def test_same_shape_different_reshape_provenance():
    class A(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], 8)

    class B(nn.Module):
        def forward(self, x):
            return x.reshape(2, -1)

    results = []
    for model in (A(), B()):
        graph = DependencyGraph.build(model, args=(torch.randn(2, 8),))
        input_ = next(v for v in graph.values() if v.kind == "input")
        impact = graph.propagate(remove=[input_.axis(1).select([1, 3])])
        assert impact.status == "resolved"
        results.append(next(r for r in impact.requirements if r.kind == "shape_arguments"))
    assert dict(results[0].data)["expression"] != dict(results[1].data)["expression"]
    assert all(r.kind != "attribute" for r in results)


def test_shape_arithmetic_and_squeeze_unsqueeze():
    class Model(nn.Module):
        def forward(self, x):
            return x.reshape(x.size(0), x.size(1) // 2, 2).unsqueeze(1).squeeze(1)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 8),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([2, 3])])
    assert impact.status == "resolved"
    assert any(e.kind == "floordiv" for e in graph.shape_expressions.values())


def test_nonrectangular_reshape_is_unresolved():
    class Model(nn.Module):
        def forward(self, x):
            return x.reshape(2, 4, 2)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 8),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    result = graph.propagate(remove=[input_.axis(1).select([0])])
    assert result.status == "unresolved"
    assert any(d.code == "unsupported_layout" for d in result.diagnostics)


def test_channels_last_conv_view_keeps_layout(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 1)

        def forward(self, x):
            y = self.conv(x)
            return y.permute(0, 2, 3, 1).view(-1, y.size(1))

    model = Model()
    x = torch.randn(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("conv.weight").axis(0).select([0, 1])], preserve_io=False
    )
    torch.testing.assert_close(model(x), original(x)[:, 2:])


@pytest.mark.parametrize("dynamic", [True, False])
def test_reshape_original_size_provenance(dynamic, execution_device):
    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return y.reshape(y.size(0), y.size(1) if dynamic else 6)

    model = Reshape()
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    request = [graph.parameter("fc.weight").axis(0).select([1, 3])]
    if not dynamic:
        with pytest.raises(PlanningError, match="original forward"):
            pruner.plan(remove=request, preserve_io=False)
    else:
        plan = pruner.plan(remove=request, preserve_io=False)
        pruner.apply(plan)
        assert model(x).shape == (2, 4)


def test_view_noncontiguous_parameter_is_rejected():
    class View(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(16.0).reshape(4, 4).t())

        def forward(self, x):
            return self.weight.t().view(-1) + x

    model = View()
    graph, pruner = build(model, torch.zeros(()))
    with pytest.raises(PlanningError, match=r"view.*original forward"):
        pruner.plan(remove=[graph.parameter("weight").axis(0).select([0])], preserve_io=False)


def test_squeeze_new_singleton_rejected():
    class Squeeze(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x).squeeze()

    model = Squeeze()
    graph, pruner = build(model, torch.randn(2, 4))
    with pytest.raises(PlanningError, match="compact shape"):
        pruner.plan(remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False)


def test_unflatten_bound_tuple_and_unbind_ports():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.unflatten = nn.Unflatten(1, (2, 3))

        def forward(self, x):
            return self.unflatten(self.fc(x)).unbind(1)

    model = Net()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = pruner.plan(
        remove=[graph.parameter("fc.weight").axis(0).select([1, 4])], preserve_io=False
    )
    pruner.apply(plan)
    assert model.unflatten.unflattened_size == (2, 2)
    for actual, reference in zip(model(x), old(x), strict=True):
        torch.testing.assert_close(actual, reference[:, [0, 2]])


def test_channels_last_weight_layout_is_validated_before_execution(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 3).to(memory_format=torch.channels_last)

        def forward(self, x):
            y = self.conv(x)
            return y.permute(0, 2, 3, 1).view(-1, y.size(1))

    model = Model()
    x = torch.randn(2, 3, 6, 6)
    expected = model(x)[:, [0, 2, 3, 5]]
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("conv.weight").axis(0).select([1, 4])], preserve_io=False
    )
    assert (
        next(r for r in plan.recipes if r.tensor.paths == ("conv.weight",)).memory_format
        == "channels_last"
    )
    Pruner(model).apply(plan)
    torch.testing.assert_close(model(x), expected)
