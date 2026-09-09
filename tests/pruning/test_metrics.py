"""pruning / metrics contracts."""

import pytest
import torch
from torch import nn

from tests.support.pruning import build
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    Magnitude,
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

    def strategy(ctx):
        score = ctx.score([candidate])[0]
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

    pruner.plan(
        candidates=[candidate],
        metric=metric,
        strategy=strategy,
        budget=ChannelRatio(0.5, axes=(ref.axis(0),)),
        preserve_io=False,
    )
