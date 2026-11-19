"""pruning / metrics contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import KeyStrategy, build
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    Greedy,
    Magnitude,
    PlanningContext,
    Pruner,
    WeightTaylor,
)


@pytest.mark.parametrize(
    "metric",
    [Magnitude(1), Magnitude(2), WeightTaylor("elementwise_abs"), WeightTaylor("joint_abs")],
)
def test_metric_union_formula_and_precision(metric, execution_device):
    model = nn.Linear(4, 5, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(-10, 10, dtype=torch.float64).reshape(5, 4))
    model.weight.grad = torch.linspace(-2, 2, 20, dtype=torch.float64).reshape(5, 4)
    graph, pruner = build(model, torch.randn(2, 4, dtype=torch.float64))
    ref = graph.parameter("weight")
    candidate = Candidate("joint", (ref.axis(0).select([1]), ref.axis(1).select([2])))

    @KeyStrategy
    def strategy(ctx):
        score = ctx.score(metric, [candidate])[0]
        mask = torch.zeros_like(model.weight, dtype=torch.bool)
        mask[1] = True
        mask[:, 2] = True
        if isinstance(metric, Magnitude):
            expected = (
                model.weight[mask].abs().sum()
                if metric.p == 1
                else model.weight[mask].square().sum().sqrt()
            )
        else:
            terms = (model.weight * model.weight.grad)[mask]
            expected = terms.abs().sum() if metric.mode == "elementwise_abs" else terms.sum().abs()
        assert score == pytest.approx(expected.item(), rel=1e-12)
        return ["joint"]

    Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        CandidateSpace(candidates=[candidate], channel_axes=(ref.axis(0),)),
        budget=ChannelCount((4,), (ref.axis(0),)),
        strategy=strategy,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires parameters on CPU and CUDA")
@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (Magnitude(1), (7.0, 5.0, 11.0)),
        (Magnitude(2), (5.0, 17**0.5, 73**0.5)),
        (WeightTaylor("elementwise_abs"), (10.0, 8.0, 25.0)),
        (WeightTaylor("joint_abs"), (2.0, 0.0, 7.0)),
    ],
)
def test_metric_scores_regions_on_cpu_and_cuda(metric, expected):
    """Combine the CPU row and CUDA column of each removal before ranking it."""

    class SplitDevices(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 3, bias=False, device="cpu")
            self.last = nn.Linear(3, 1, bias=False, device="cuda")

        def forward(self, x):
            return self.last(self.first(x).to("cuda"))

    model = SplitDevices().eval()
    with torch.no_grad():
        model.first.weight.copy_(torch.tensor([[3.0, 0.0], [4.0, 0.0], [8.0, 0.0]]))
        model.last.weight.copy_(torch.tensor([[4.0, 1.0, 3.0]], device="cuda"))
    model.first.weight.grad = torch.tensor([[2.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    model.last.weight.grad = torch.tensor([[-1.0, -4.0, -3.0]], device="cuda")
    x = torch.tensor([[1.0, 2.0]])
    graph, pruner = build(model, x)
    space = pruner.discover_candidates(targets=("first",))
    budget = ChannelCount((2,), space.channel_axes)
    context = PlanningContext(
        graph, space.candidates, budget, space.channel_axes, pruner.constraints
    )
    assert context.score(metric, space.candidates) == pytest.approx(expected, rel=1e-12)
    plan = pruner.plan(space, budget=budget, strategy=Greedy(metric))
    axis = graph.parameter("first.weight").axis(0)
    assert tuple(plan.analysis.selection(axis.tensor).fully_selected_indices(axis.dim)) == (1,)
    reference = F.linear(
        F.linear(x, model.first.weight[[0, 2]]).to("cuda"), model.last.weight[:, [0, 2]]
    )
    pruner.apply(plan)
    torch.testing.assert_close(model(x), reference)
