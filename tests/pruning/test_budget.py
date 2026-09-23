"""pruning / budget contracts."""

import pytest
import torch
from torch import nn

from tests.support.pruning import StaticMetric, build
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Greedy,
    Magnitude,
    Pruner,
)


def test_depthwise_candidate_denominator_and_custom_blocks():
    model = nn.Conv1d(4, 8, 1, groups=4)
    graph, pruner = build(model, torch.randn(2, 4, 5))
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.25),
        strategy=Greedy(Magnitude()),
    )
    assert plan.selection_report.widths == (8,) and plan.selection_report.removed == (2,)
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("pair", (axis.select([0, 1]),))
    with pytest.raises(TypeError):
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
            CandidateSpace(candidates=[candidate], channel_axes=None),
            budget=ChannelRatio(0.25),
            strategy=Greedy(Magnitude()),
        )
    manual = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        CandidateSpace(candidates=[candidate], channel_axes=(axis,)),
        budget=ChannelRatio(0.25),
        strategy=Greedy(Magnitude()),
    )
    assert manual.selection_report.widths == plan.selection_report.widths


def test_global_budget_no_hidden_local_cap():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return self.a(x), self.b(x)

    model = Branches()
    _graph, pruner = build(model, torch.randn(2, 4))

    @StaticMetric
    def metric(ctx, batch):
        return [0 if c.key.startswith("a.") else 100 for c in batch]

    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.25, scope="global"),
        strategy=Greedy(metric),
    )
    assert plan.selection_report.widths == (6, 6)
    assert plan.selection_report.removed == (3, 0)


def test_explicit_protected_axis_keeps_budget_baseline_and_seed_alias_dedup():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    plan = pruner.plan(
        CandidateSpace(
            candidates=pruner.discover_candidates().candidates, channel_axes=(axis, axis)
        ),
        budget=ChannelRatio(0.5),
        strategy=Greedy(Magnitude()),
    )
    assert plan.selection_report.widths == (6,) and plan.selection_report.shortfall == 3
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
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(Magnitude()),
    )
    assert plan.selection_report.widths == (6, 6)
    assert plan.selection_report.removed == (0, 3)
    assert plan.selection_report.shortfall == 3
    assert any("Incomplete" in reason for _, reason in plan.selection_report.exclusions)
