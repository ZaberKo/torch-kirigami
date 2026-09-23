"""Optimized paths preserve independent objectives, ownership and bounded work."""

import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import StaticMetric
from torch_kirigami import DependencyGraph, IndexSet, Region, Selection
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
    Magnitude,
    ParameterGroup,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import GroupLasso, GroupSquaredL2


@pytest.mark.parametrize("method", ["sin", "cos", "abs", "square", "neg", "clone"])
@pytest.mark.parametrize("functional", [False, True])
def test_allocating_registered_families_support_inplace_consumers(
    method, functional, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            y = self.first(x)
            y = getattr(torch, method)(y) if functional else getattr(y, method)()
            return self.last(y.relu_())

    model = Model()
    x = torch.randn(2, 4)
    keep = [0, 2, 3]
    y = F.linear(x, model.first.weight[keep], model.first.bias[keep])
    expected = F.linear(
        getattr(torch, method)(y).relu(), model.last.weight[:, keep], model.last.bias
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("first.weight").axis(0).select([1])]
        )
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize("width", [8, 1024])
def test_zero_budget_protected_head_does_not_scan_channels(width, monkeypatch, execution_device):
    model = nn.Linear(4, width)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4),))
    original = graph.propagate
    requests = []

    def counted(*args, **kwargs):
        requests.append(tuple(kwargs["remove"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "propagate", counted)
    plan = Pruner(model, graph=graph).plan(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0),
        strategy=Greedy(Magnitude(), max_trials=0),
    )
    assert len(requests) <= 3 and not any(requests)
    assert not plan.recipes and plan.selection_report.widths == ()
    before = model.weight
    Pruner(model).apply(plan)
    assert model.weight is before


@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
@pytest.mark.parametrize("zeros", [False, True])
def test_batched_regularizer_has_independent_loss_and_gradient_reference(
    kind, zeros, execution_device
):
    model = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 2)).double()
    if zeros:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 16, dtype=torch.float64),))
    groups = Pruner(graph.model, graph=graph).parameter_groups(
        Pruner(graph.model, graph=graph).discover_candidates().candidates
    )
    penalty = kind(groups)
    actual = penalty()
    rows = torch.cat((model[0].weight, model[0].bias[:, None], model[1].weight.T), dim=1)
    expected = rows.norm(dim=1).sum() if kind is GroupLasso else rows.square().sum() * 0.5
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters())
    left = torch.autograd.grad(actual, parameters, allow_unused=True)
    right = torch.autograd.grad(expected, parameters, allow_unused=True)
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b)
    # The count is structural, not a noisy wall-clock threshold: each selected
    # weight is reduced once; no full-parameter gather backward per candidate.
    seen, pending = set(), [actual.grad_fn]
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        pending.extend(n for n, _ in node.next_functions)
    assert sum("IndexSelectBackward" in type(n).__name__ for n in seen) <= 3


def test_batched_and_irregular_overlap_groups_preserve_selected_domain(execution_device):
    model = nn.Linear(4, 4, bias=False).double()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4, dtype=torch.float64),))
    ref = graph.parameter("weight")
    rectangle = Selection(ref, (Region((IndexSet.of([1, 3]), IndexSet.of([0, 2]))),))
    groups = (
        ParameterGroup(graph, (ref.axis(0).select([1]),)),
        ParameterGroup(graph, (rectangle,)),
    )
    with torch.no_grad():
        model.weight[0].fill_(float("nan"))
    actual = GroupLasso(groups, coefficients=(2, 3))()
    expected = 2 * model.weight[1].norm() + 3 * model.weight[[1, 3]][:, [0, 2]].norm()
    torch.testing.assert_close(actual, expected)
    (a,) = torch.autograd.grad(actual, model.weight)
    (b,) = torch.autograd.grad(expected, model.weight)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all()


def test_tied_checkpoint_uses_storage_identity_but_checks_independent_payloads(
    monkeypatch, execution_device
):
    model = nn.Linear(4, 4)
    model.alias = model.weight
    calls = []
    original = torch.isnan

    def counted(value):
        calls.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch, "isnan", counted)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    assert not calls
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    payload["state_dict"]["alias"] = payload["state_dict"]["alias"].clone()
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    load_checkpoint(model, stream)
    assert calls
    assert model.alias is model.weight


def test_zero_budget_uses_explicit_space_without_rediscovery_or_scoring(
    monkeypatch, execution_device
):
    model = nn.Sequential(nn.Linear(4, 1024), nn.Linear(1024, 2))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4),))

    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()

    def unexpected(*args, **kwargs):
        pytest.fail("Zero-budget planning rediscovered candidates or requested scores")

    monkeypatch.setattr(pruner, "discover_candidates", unexpected)
    plan = pruner.plan(
        space,
        budget=ChannelRatio(0),
        strategy=Greedy(StaticMetric(unexpected)),
    )
    assert plan.selection_report.widths == (1024,) and plan.selection_report.removed == (0,)
    before = tuple(model.parameters())
    Pruner(model).apply(plan)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_batched_lasso_extreme_values_match_float64_reference(scale, execution_device):
    model = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, 13).reshape(4, 3) * scale)
    graph = DependencyGraph.build(model, args=(torch.zeros(2, 3),))
    ref = graph.parameter("weight")
    groups = tuple(ParameterGroup(graph, (ref.axis(0).select([i]),)) for i in range(4))
    actual = GroupLasso(groups)()
    expected = model.weight.double().norm(dim=1).sum()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=0)
    (a,) = torch.autograd.grad(actual, model.weight)
    (b,) = torch.autograd.grad(expected, model.weight)
    torch.testing.assert_close(a, b)


def test_batched_and_irregular_groups_keep_common_accumulation_precision(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.low = nn.Parameter(torch.ones(3, 3, dtype=torch.float32))
            self.high = nn.Parameter(torch.full((3,), 1e30, dtype=torch.float64))

        def forward(self, x):
            return x @ self.low + self.high

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.ones(3),))
    low, high = graph.parameter("low"), graph.parameter("high")
    rectangle = Selection(low, (Region((IndexSet.of([0, 2]), IndexSet.of([1, 2]))),))
    groups = (
        ParameterGroup(graph, (rectangle,)),
        ParameterGroup(graph, (high.axis(0).select([1]),)),
    )
    with torch.no_grad():
        model.low.fill_(1e30)
    actual = GroupSquaredL2(groups)()
    expected = (model.low[[0, 2]][:, [1, 2]].double().square().sum() + model.high[1].square()) * 0.5
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
    left = torch.autograd.grad(actual, tuple(model.parameters()))
    right = torch.autograd.grad(expected, tuple(model.parameters()))
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b)
