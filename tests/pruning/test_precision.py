"""pruning / precision contracts."""

import math

import pytest
import torch
from torch import nn

from tests.support.pruning import KeyStrategy, build
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningContext,
    Pruner,
    WeightTaylor,
)


@pytest.mark.parametrize(
    "scale,dtype",
    [
        (1e-30, torch.float32),
        (1e20, torch.float32),
        (1e-300, torch.float64),
        (1e300, torch.float64),
    ],
)
def test_l2_extreme_finite_values_preserve_scores_and_order(scale, dtype, execution_device):
    model = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2 * scale, 2 * scale], [scale, scale]], dtype=dtype))
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=dtype),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(2))
    context = PlanningContext(graph, graph.operations(), candidates, ChannelRatio(0.5), (axis,), ())
    expected = [math.hypot(*row) for row in model.weight.detach().cpu().tolist()]
    assert context.score(Magnitude(), candidates) == pytest.approx(expected, rel=1e-6, abs=0)
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        Pruner(model, graph=graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(Magnitude()),
    )
    assert tuple(plan.analysis.selection(graph.parameter("weight")).fully_selected_indices(0)) == (
        1,
    )


def test_stable_l2_complex_components_and_zero_region(execution_device):
    model = nn.Linear(2, 2, bias=False, dtype=torch.complex64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3e38 + 3e38j, 0], [0, 0]], dtype=torch.complex64))
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=torch.complex64),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(2))
    context = PlanningContext(graph, graph.operations(), candidates, ChannelRatio(0.5), (axis,), ())
    value = model.weight.detach().cpu()[0, 0].item()
    assert context.score(Magnitude(), candidates) == pytest.approx(
        (math.hypot(value.real, value.imag), 0)
    )


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
@pytest.mark.parametrize("scale", [1e-30, 1e20])
def test_taylor_promotes_before_multiplying(scale, mode, execution_device):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2 * scale, 2 * scale], [scale, scale]]))
    model.weight.grad = torch.full_like(model.weight, scale)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),)) for i in range(2))
    metric = WeightTaylor(mode)
    context = PlanningContext(graph, graph.operations(), candidates, ChannelRatio(0.5), (axis,), ())
    rows = model.weight.detach().cpu().tolist()
    gradients = model.weight.grad.cpu().tolist()
    expected = [
        math.fsum(w * g for w, g in zip(row, grad, strict=True))
        for row, grad in zip(rows, gradients, strict=True)
    ]
    assert context.score(metric, candidates) == pytest.approx(expected, rel=1e-12, abs=0)
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        Pruner(model, graph=graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(metric),
    )
    assert tuple(plan.analysis.selection(axis.tensor).fully_selected_indices(0)) == (1,)


def test_complex_l1_promotes_before_absolute_value(execution_device):
    model = nn.Linear(2, 2, bias=False, dtype=torch.complex64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3e38 + 3e38j, 0], [1, 0]], dtype=torch.complex64))
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=torch.complex64),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),)) for i in range(2))
    context = PlanningContext(graph, graph.operations(), candidates, ChannelRatio(0.5), (axis,), ())
    expected = [
        math.fsum(math.hypot(v.real, v.imag) for v in row)
        for row in model.weight.detach().cpu().tolist()
    ]
    assert context.score(Magnitude(1), candidates) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_magnitude_low_precision_accumulation_and_filter(dtype):
    model = nn.Linear(4, 6).to(dtype)
    with torch.no_grad():
        model.weight.fill_(200)
        model.bias.fill_(10)
    graph, pruner = build(model, torch.randn(2, 4, dtype=dtype))
    axis = graph.parameter("weight").axis(0)
    candidates = [Candidate("one", (axis.select([0]),))]

    @KeyStrategy
    def strategy(ctx):
        expected = (4 * 200**2 + 10**2) ** 0.5
        assert ctx.score(Magnitude(), ctx.candidates)[0] == pytest.approx(expected)
        filtered = Magnitude(parameter_filter=lambda ref, param: ref.paths[0] == "weight")
        assert ctx.score(filtered, ctx.candidates) == (400.0,)
        return ["one"]

    Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        CandidateSpace(candidates=candidates, channel_axes=(axis,)),
        budget=ChannelRatio(0.2),
        strategy=strategy,
    )
