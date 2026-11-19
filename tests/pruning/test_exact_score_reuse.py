"""Cached built-in scores match ordinary scoring in value, completeness and memory."""

import pytest
import torch
from torch import nn

from torch_kirigami import Barrier, DependencyGraph, config
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    Magnitude,
    ParameterBudget,
    PlanningContext,
    Pruner,
)
from torch_kirigami.pruning.candidate_checks import CandidateFilter
from torch_kirigami.pruning.score_reuse import IndependentRanking


class OrdinaryMagnitude(Magnitude):
    """Force ordinary per-candidate joint scoring for the union norm."""


class OrdinaryGroupMagnitude(GroupMagnitude):
    """Force ordinary per-candidate joint scoring for the normalized energy."""


ORDINARY = {Magnitude: OrdinaryMagnitude, GroupMagnitude: OrdinaryGroupMagnitude}


class TwoBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(10, 4), nn.Linear(4, 2))
        self.b = nn.Sequential(nn.Linear(10, 4), nn.Linear(4, 2))

    def forward(self, x):
        return self.a(x), self.b(x)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("metric_type", [Magnitude, GroupMagnitude])
def test_incomplete_candidates_never_receive_cached_scores(strategy_type, metric_type):
    config.max_parts = 4
    model = TwoBranches().eval()
    with torch.no_grad():
        for parameter in model.a.parameters():
            parameter.zero_()
        for parameter in model.b.parameters():
            parameter.fill_(1.0)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 10),))
    rows = graph.parameter("a.0.weight").axis(0)
    # Any removal from branch `a` is incomplete, so its zero-valued channels
    # must not be ranked first.
    barrier = Barrier((rows.tensor,), "Unproved branch")
    pruner = Pruner(model, graph=graph, constraints=(barrier,))
    space = pruner.discover_candidates(targets=("a.0", "b.0"))
    assert not graph.propagate(remove=(rows.select([0]),), constraints=pruner.constraints).complete
    budget = ParameterBudget(sum(p.numel() for p in model.parameters()) - 13)
    plans = [
        pruner.plan(space, budget=budget, strategy=strategy_type(metric, max_trials=1))
        for metric in (metric_type(1), ORDINARY[metric_type](1))
    ]
    assert plans[0].selected == plans[1].selected == ("b.0.weight:0:000000000000",)
    assert plans[0].selection_report == plans[1].selection_report
    pruner.apply(plans[0])
    assert model.a[0].out_features == 4 and model.b[0].out_features == 3
    assert [tuple(y.shape) for y in model(torch.ones(1, 10))] == [(1, 2), (1, 2)]


class Broadcasts(nn.Module):
    """Two broadcast additions, so accepted slices on three axes meet one parameter."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, 3, 3))
        self.b = nn.Parameter(torch.zeros(3, 3, 3))
        self.tail = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.a[0, 1:, :2] = 0.01
            self.a[1, 1:, :2] = 0.1
            self.tail.weight.fill_(1.0)

    def forward(self, x, y):
        return (x + self.a).relu() + self.b, self.tail(y)


@pytest.mark.parametrize("impact_cache_size", [0, None])
def test_exact_closure_survives_an_intermediate_limit(impact_cache_size):
    """Verified identity slices recover a complete closure after a traversal limit."""
    config.max_parts = 3
    if impact_cache_size is not None:
        config.impact_cache_size = impact_cache_size
    model = Broadcasts()
    args = (torch.ones(3, 3, 3), torch.ones(1, 3))
    graph = DependencyGraph.build(model, args=args)
    values = {ref.id.split(":")[-2]: ref for ref in graph.values() if ":value:" in ref.id}
    axis = graph.parameter("a").axis(0)
    tail = graph.parameter("tail.weight").axis(0)
    first = Candidate(
        "initial",
        (
            values["add_1"].axis(0).select([2]),
            values["add"].axis(1).select([0]),
            values["x"].axis(2).select([2]),
        ),
    )
    blocked = Candidate("blocked", (axis.select([0]),), axis)
    safe = Candidate("safe", (tail.select([0]),), tail)
    space = CandidateSpace((first, blocked, safe), (axis, tail))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    budget = ParameterBudget(22)
    context = PlanningContext(
        graph, space.candidates, budget, space.channel_axes, pruner.constraints
    )
    accepted = context.impact(first.remove)
    joint = context.impact((*first.remove, *blocked.remove))
    # The final joint selection fits in three regions; an intermediate union does not.
    assert accepted.complete and joint.complete
    assert all(len(selection.regions) <= 3 for selection in joint.selections.values())
    ranking = IndependentRanking.build(context, Magnitude(1), space.channel_axes)
    report = None if ranking is None else ranking.score(accepted)
    assert report is not None and report.scores["blocked"] == pytest.approx(0.04)
    plans = [
        pruner.plan(space, budget=budget, strategy=DynamicGreedy(metric, max_trials=2))
        for metric in (Magnitude(1), OrdinaryMagnitude(1))
    ]
    assert plans[0].selected == plans[1].selected == ("initial", "blocked")
    assert plans[0].selection_report == plans[1].selection_report
    assert plans[0].selection_report.trials == 2
    expected, tail_output = model(*args)
    pruner.apply(plans[0])
    out, tail_after = model(args[0][1:2, 1:, :2], args[1])
    torch.testing.assert_close(out, expected.detach()[1:2, 1:, :2])
    torch.testing.assert_close(tail_after, tail_output.detach())


def test_cached_completeness_checks_only_candidate_positions(monkeypatch):
    """`Magnitude` proves its candidates' positions, not every position of the axis."""
    model = nn.Sequential(nn.Linear(16, 4096), nn.ReLU(), nn.Linear(4096, 16)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 16),))
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("0.weight").axis(0)
    space = CandidateSpace((Candidate("0", (axis.select([7]),), axis),), (axis,))
    checks = []
    original = CandidateFilter._members_complete

    def counted(self, *args, **kwargs):
        checks.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CandidateFilter, "_members_complete", counted)
    budget = ChannelCount((4095,), (axis,))
    plan = pruner.plan(space, budget=budget, strategy=Greedy(Magnitude(1)))
    assert plan.selected == ("0",) and 0 < len(checks) <= 2
    ordinary = pruner.plan(space, budget=budget, strategy=Greedy(OrdinaryMagnitude(1)))
    assert ordinary.selection_report == plan.selection_report


def _permuted_columns(width: int, exponent: int) -> nn.Module:
    """Give every column the same values in a different row order."""
    model = nn.Sequential(nn.Linear(1, width, bias=False), nn.Linear(width, 16, bias=False))
    model = model.double().eval()
    with torch.no_grad():
        model[0].weight.zero_()
        weight = torch.ones(16, width, dtype=torch.float64)
        for column in range(width):
            weight[(column * 13 + 3) % 16, column] = 2.0**exponent
        model[1].weight.copy_(weight)
    return model


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("metric_type", [Magnitude, GroupMagnitude])
def test_cached_scores_are_bitwise_ordinary_scores(metric_type, p, execution_device):
    # Near 2**53, summation order changes the rounded float64 sum.
    model = _permuted_columns(16, 53)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 1, dtype=torch.float64),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.5), space.channel_axes, pruner.constraints
    )
    ranking = IndependentRanking.build(context, metric_type(p), space.channel_axes)
    assert ranking is not None
    rows = graph.parameter("0.weight").axis(0)
    for accepted in (context.impact(()), context.impact((rows.select([5]),))):
        cached = ranking.score(accepted).scores
        remaining = tuple(c for c in space.candidates if c.key in cached)
        assert len(remaining) == len(space.candidates) - (0 if not accepted.requested else 1)
        ordinary = context.score(ORDINARY[metric_type](p), remaining, accepted_impact=accepted)
        assert [cached[c.key] for c in remaining] == list(ordinary)
    for strategy_type in (Greedy, DynamicGreedy):
        plans = [
            pruner.plan(space, budget=ChannelRatio(2 / 16), strategy=strategy_type(metric))
            for metric in (metric_type(p), ORDINARY[metric_type](p))
        ]
        assert plans[0].selected == plans[1].selected
        assert plans[0].selection_report == plans[1].selection_report


@pytest.mark.parametrize("metric_type", [Magnitude, GroupMagnitude])
def test_cached_statistics_reduce_one_slice_at_a_time(metric_type, execution_device):
    if execution_device != "cuda":
        pytest.skip("Peak allocation is measured with the CUDA allocator")
    model = nn.Sequential(nn.Linear(3, 1024, bias=False), nn.Linear(1024, 4096, bias=False))
    model = model.eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    axis = graph.parameter("0.weight").axis(0)
    space = type(space)((space.candidates[2],), space.channel_axes)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    plan = pruner.plan(
        space, budget=ChannelCount((1023,), (axis,)), strategy=Greedy(metric_type(p=1))
    )
    torch.cuda.synchronize()
    # The consumer weight has 16 MiB; one float64 column slice has 32 KiB.
    assert torch.cuda.max_memory_allocated() - before < 1024**2
    assert plan.selected == (space.candidates[0].key,)


@pytest.mark.parametrize("p", [1, 2])
def test_row_and_column_scores_reduce_one_region_at_a_time(p, execution_device):
    """Overlapping rows and columns count once, without gathering every selected element."""
    width, block = 2048, 256
    torch.manual_seed(3)
    model = nn.Linear(width, width, bias=False).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, width),))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    weight = graph.parameter("weight")
    candidate = Candidate(
        "block", (weight.axis(0).select(range(block)), weight.axis(1).select(range(block)))
    )
    context = PlanningContext(
        graph, (candidate,), ParameterBudget(1), (weight.axis(0), weight.axis(1)), ()
    )
    mask = torch.zeros(width, width, dtype=torch.bool)
    mask[:block] = True
    mask[:, :block] = True
    expected = torch.linalg.vector_norm(model.weight.detach().double()[mask], ord=p)
    (score,) = context.score(Magnitude(p), (candidate,))
    # Counting the 256 x 256 intersection twice would raise the score by several percent.
    assert score == pytest.approx(expected.item(), rel=1e-6)
    if execution_device != "cuda":
        return
    del mask, expected
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    plan = pruner.plan(
        CandidateSpace((candidate,), (weight.axis(0), weight.axis(1))),
        budget=ParameterBudget((width - block) ** 2),
        strategy=Greedy(Magnitude(p)),
    )
    torch.cuda.synchronize()
    # The largest region has 256 x 2048 elements, 4 MiB in float64. One gathered
    # vector of all selected elements and its sorted indices needed over 50 MiB.
    assert torch.cuda.max_memory_allocated() - before <= 12 * 1024**2
    assert plan.selected == ("block",)
