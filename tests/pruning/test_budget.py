"""pruning / budget contracts."""

import pytest
import torch
from torch import nn

from tests.support.pruning import build
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    Magnitude,
)


def test_depthwise_candidate_denominator_and_custom_blocks():
    model = nn.Conv1d(4, 8, 1, groups=4)
    graph, pruner = build(model, torch.randn(2, 4, 5))
    plan = pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.25), preserve_io=False)
    assert plan.budget.widths == (8,) and plan.budget.removed == (2,)
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("pair", (axis.select([0, 1]),))
    with pytest.raises(ValueError, match="explicit"):
        pruner.plan(
            candidates=[candidate], metric=Magnitude(), budget=ChannelRatio(0.25), preserve_io=False
        )
    manual = pruner.plan(
        candidates=[candidate],
        metric=Magnitude(),
        budget=ChannelRatio(0.25, axes=(axis,)),
        preserve_io=False,
    )
    assert manual.budget.widths == plan.budget.widths


def test_global_budget_no_hidden_local_cap():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return self.a(x), self.b(x)

    model = Branches()
    _graph, pruner = build(model, torch.randn(2, 4))

    def metric(ctx, batch):
        return [0 if c.key.startswith("a.") else 100 for c in batch]

    plan = pruner.plan(metric=metric, budget=ChannelRatio(0.25, scope="global"), preserve_io=False)
    assert plan.budget.widths == (6, 6)
    assert plan.budget.removed == (3, 0)


def test_explicit_protected_axis_keeps_budget_baseline_and_seed_alias_dedup():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    plan = pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.5, axes=(axis, axis)))
    assert plan.budget.widths == (6,) and plan.budget.shortfall == 3
    assert not plan.recipes
    pruner.apply(plan)
    graph.validate(model)


def test_unknown_branch_underfill_keeps_denominator():
    class Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return torch.special.gammaln(self.a(x)), self.b(x)

    model = Unknown()
    _graph, pruner = build(model, torch.randn(2, 4))
    plan = pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.5), preserve_io=False)
    assert plan.budget.widths == (6, 6)
    assert plan.budget.removed == (0, 3)
    assert plan.budget.shortfall == 3
    assert any("Incomplete" in reason for _, reason in plan.budget.exclusions)
