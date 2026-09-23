"""Actionable diagnostics survive joint search, portable plans, and retries."""

import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import StaticMetric
from torch_kirigami import AxisRef, DependencyGraph, Diagnostic, Divisible
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    PruningPlan,
)


class RewriteExample(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.bad = nn.Linear(3, 4)
        self.good = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        self.register_buffer("scale", torch.ones(2, 4))
        self.mode = mode

    def forward(self, x):
        y = self.bad(x)
        if self.mode in ("whole_shape", "dimension"):
            rows = self.scale.shape[0] if self.mode == "whole_shape" else self.scale.size(0)
            y = y * self.scale if rows == 2 else -y * self.scale
        elif self.mode == "column_size":
            y = y * self.scale if self.scale.size(1) == 4 else -y * self.scale
        elif self.mode == "hardcoded":
            y = y.reshape(y.size(0), 4)
        elif self.mode == "dynamic":
            y = y.reshape(y.size(0), y.size(1))
        elif self.mode == "newaxis":
            y = y[:, None]
        elif self.mode == "unsqueeze":
            y = y.unsqueeze(1)
        elif self.mode in ("inplace", "outplace"):
            y = y.transpose(0, 1)
            y = y.relu_() if self.mode == "inplace" else y.relu()
        elif self.mode in ("view", "reshape"):
            y = F.max_pool1d(y.unsqueeze(-1), 1)
            y = y.view(-1) if self.mode == "view" else y.reshape(-1)
        elif self.mode == "advanced":
            y = y[:, [0, 2]]
        return y, self.good(x)


@StaticMetric
def ranked(context, batch):
    return [0 if c.key in ("bad", "early", "pair") else 1 for c in batch]


@pytest.mark.parametrize(
    ("mode", "location", "hint"),
    [
        ("view", "view", "If a copy is acceptable"),
        ("whole_shape", "metadata scale.shape", "If only one dimension is needed"),
        ("hardcoded", "reshape", "If a fixed number was intended"),
        ("newaxis", "getitem", "unsqueeze(dim)"),
        ("inplace", "relu_", "If no consumer relies"),
    ],
)
def test_manual_and_automatic_rewrite_diagnostics_survive_plan_roundtrip(
    mode, location, hint, execution_device
):
    model = RewriteExample(mode)
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    bad, good = (graph.parameter(p).axis(0) for p in ("bad.weight", "good.0.weight"))
    request = bad.select([1])
    before = tuple(model.parameters()), model.scale
    with pytest.raises(PlanningError) as error:
        Pruner(model, graph=graph, preserve_io=False).plan_remove([request])
    assert location in str(error.value) and hint in str(error.value)
    impact = graph.propagate(remove=[request])
    if impact.status != "resolved":
        assert location in graph.explain(impact) and hint in graph.explain(impact)
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        CandidateSpace(
            candidates=[Candidate("bad", (request,)), Candidate("good", (good.select([1]),))],
            channel_axes=(bad, good),
        ),
        budget=ChannelRatio(0.25),
        strategy=Greedy(ranked),
    )
    assert plan.selected == ("good",)
    assert len(plan.selection_report.exclusions) == 1
    reason = dict(plan.selection_report.exclusions)["bad"]
    assert location in reason and hint in reason
    assert "Earlier attempt" not in reason  # Retried after the independent cut was accepted.
    assert all(a is b for a, b in zip(before[0], model.parameters(), strict=True))
    assert model.scale is before[1]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name])
    restored = PruningPlan.from_dict(plan.to_dict())
    assert restored.explain() == plan.explain()
    target = copy.deepcopy(model)
    Pruner(target).apply(restored)
    expected_bad = original(x)[0]
    keep = [0, 2, 3]
    first, last = original.good
    expected_good = F.linear(
        F.linear(x, first.weight[keep], first.bias[keep]), last.weight[:, keep], last.bias
    )
    torch.testing.assert_close(target(x), (expected_bad, expected_good))
    sum(value.sum() for value in target(x)).backward()
    assert target.good[0].weight.grad is not None


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("view", "reshape"),
        ("whole_shape", "dimension"),
        ("hardcoded", "dynamic"),
        ("newaxis", "unsqueeze"),
        ("inplace", "outplace"),
    ],
)
def test_suggested_rewrites_preserve_original_values_and_allow_compaction(
    before, after, execution_device
):
    model = RewriteExample(before)
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    model.mode = after
    torch.testing.assert_close(model(x), original(x))
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph, preserve_io=False).plan_remove(
        [graph.parameter("bad.weight").axis(0).select([1])]
    )
    Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3]
    expected = F.linear(x, original.bad.weight[keep], original.bad.bias[keep])
    if after == "dimension":
        expected *= original.scale[:, keep]
    elif after == "unsqueeze":
        expected = expected.unsqueeze(1)
    elif after == "outplace":
        expected = expected.t().clamp_min(0)
    elif after == "reshape":
        expected = expected.flatten()
    torch.testing.assert_close(model(x), (expected, original.good(x)))
    sum(value.sum() for value in model(x)).backward()


@pytest.mark.parametrize(
    ("mode", "absent"), [("column_size", "If only one dimension"), ("advanced", "unsqueeze")]
)
def test_unrelated_failures_do_not_receive_rewrite_advice(mode, absent):
    model = RewriteExample(mode)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    with pytest.raises(PlanningError) as error:
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("bad.weight").axis(0).select([1])]
        )
    assert absent not in str(error.value)


@dataclass(frozen=True)
class NeedsPartner:
    axis: AxisRef

    @property
    def refs(self):
        return (self.axis.tensor,)

    def check(self, selections):
        selection = selections.get(self.axis.tensor.id)
        if selection is not None:
            removed = selection.fully_selected_indices(self.axis.dim)
            if 0 in removed and 1 not in removed:
                return Diagnostic(
                    "needs_partner", "Coordinate 0 requires coordinate 1", node="pair_rule"
                )
        return None


@pytest.mark.parametrize("limit", [2, 100])
def test_retry_replaces_old_failure_and_distinguishes_unfinished_search(limit, execution_device):
    model = nn.Linear(3, 4)
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("weight").axis(0)
    plan = Pruner(model, graph=graph, preserve_io=False, constraints=[NeedsPartner(axis)]).plan(
        CandidateSpace(
            candidates=[
                Candidate("early", (axis.select([0]),)),
                Candidate("later", (axis.select([1]),)),
            ],
            channel_axes=(axis,),
        ),
        budget=ChannelRatio(0.5),
        strategy=Greedy(ranked, max_trials=limit),
    )
    if limit == 2:
        assert plan.selected == ("later",) and plan.selection_report.limit_reached
        reason = dict(plan.selection_report.exclusions)["early"]
        assert "Earlier attempt" in reason and "needs_partner at pair_rule" in reason
        assert "Not retried after the accepted selection changed" in reason
        keep = [0, 2, 3]
    else:
        assert set(plan.selected) == {"early", "later"}
        assert not plan.selection_report.exclusions and not plan.selection_report.limit_reached
        keep = [2, 3]
    restored = PruningPlan.from_dict(plan.to_dict())
    assert restored.explain() == plan.explain()
    Pruner(model).apply(restored)
    torch.testing.assert_close(model(x), F.linear(x, original.weight[keep], original.bias[keep]))
    model(x).sum().backward()


def test_covered_candidate_has_no_stale_exclusion():
    model = nn.Linear(3, 4)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("weight").axis(0)
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        CandidateSpace(
            candidates=[
                Candidate("pair", (axis.select([0, 1]),)),
                Candidate("covered", (axis.select([0]),)),
            ],
            channel_axes=(axis,),
        ),
        budget=ChannelRatio(0.5),
        strategy=Greedy(ranked),
    )
    assert plan.selected == ("pair",) and not plan.selection_report.exclusions


@pytest.mark.parametrize("scope", ["local", "global"])
def test_budget_blocked_completion_retains_constraint_and_actual_cap(scope):
    model = nn.Linear(3, 4)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("weight").axis(0)
    plan = Pruner(model, graph=graph, preserve_io=False, constraints=[Divisible(axis, 2)]).plan(
        Pruner(
            model, graph=graph, preserve_io=False, constraints=[Divisible(axis, 2)]
        ).discover_candidates(),
        budget=ChannelRatio(0.25, scope=scope),
        strategy=Greedy(Magnitude()),
    )
    assert not plan.recipes
    assert len(plan.selection_report.exclusions) == 4
    for _, reason in plan.selection_report.exclusions:
        assert "divisible" in reason
        assert f"{scope} channel budget" in reason and "2 removals > 1 allowed" in reason
    assert (
        PruningPlan.from_dict(plan.to_dict()).selection_report.exclusions
        == plan.selection_report.exclusions
    )


def test_trial_limit_reports_untested_candidates_without_inventing_constraints():
    model = nn.Linear(3, 4)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        Pruner(model, graph=graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(Magnitude(), max_trials=1),
    )
    assert len(plan.selected) == 1 and len(plan.selection_report.exclusions) == 3
    assert all(
        "before this candidate could be tested" in reason
        for _, reason in plan.selection_report.exclusions
    )


def test_invalid_empty_request_preserves_reason_and_failure_state():
    model = nn.Linear(3, 10)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    original = model.weight
    with pytest.raises(PlanningError) as error:
        Pruner(
            model,
            graph=graph,
            preserve_io=False,
            constraints=[Divisible(graph.parameter("weight").axis(0), 4)],
        ).plan(
            Pruner(
                model,
                graph=graph,
                preserve_io=False,
                constraints=[Divisible(graph.parameter("weight").axis(0), 4)],
            ).discover_candidates(),
            budget=ChannelRatio(0.1),
            strategy=Greedy(Magnitude()),
        )
    assert "Empty request:" in str(error.value) and "divisible" in str(error.value)
    assert "Candidate attempt" in str(error.value) and "budget" in str(error.value)
    assert model.weight is original


@pytest.mark.parametrize("limit", [0, 1])
def test_count_feasible_batch_uses_one_trial_and_zero_limit_still_skips_search(limit):
    model = nn.Linear(3, 4)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    plan = Pruner(
        model,
        graph=graph,
        preserve_io=False,
        constraints=[Divisible(graph.parameter("weight").axis(0), 2)],
    ).plan(
        Pruner(
            model,
            graph=graph,
            preserve_io=False,
            constraints=[Divisible(graph.parameter("weight").axis(0), 2)],
        ).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(Magnitude(), max_trials=limit),
    )
    if limit:
        assert plan.selection_report.removed == (2,)
        assert plan.selection_report.trials == 1 and not plan.selection_report.limit_reached
        assert plan.analysis.status == "resolved"
    else:
        assert not plan.recipes and plan.selection_report.limit_reached
        assert all("limit is zero" in reason for _, reason in plan.selection_report.exclusions)
        assert "no claim of infeasibility" in plan.explain()
