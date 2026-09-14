"""pruning / strategy contracts."""

import pytest
import torch
from torch import nn

from tests.support.models import Chain
from tests.support.pruning import build
from torch_kirigami import (
    Balanced,
    DependencyGraph,
    Divisible,
    IndexSet,
)
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    WeightTaylor,
)


@pytest.mark.parametrize("width", [32, 33, 64])
def test_builtin_scoring_does_not_thrash_small_impact_cache(width, monkeypatch):
    model = nn.Sequential(nn.Linear(4, width), nn.Linear(width, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    count = 0
    propagate = graph.propagate

    def counted(**kwargs):
        nonlocal count
        count += 1
        return propagate(**kwargs)

    monkeypatch.setattr(graph, "propagate", counted)
    Pruner(model, graph=graph, preserve_io=False).plan(
        Pruner(model, graph=graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.2),
        strategy=Greedy(Magnitude(), max_trials=1),
    )
    assert count <= width + 12


def test_custom_metric_keeps_whole_batch_and_shared_expressions_are_readonly():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 64)

        def forward(self, x):
            y = self.fc(x)
            for _ in range(8):
                y = y.reshape(y.size(0), y.size(1))
            return y

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    operations = graph.operations()
    assert len({id(op.expressions) for op in operations}) == 1
    with pytest.raises(TypeError):
        operations[0].expressions[operations[0].node] = None
    batch_sizes = []

    def metric(context, batch):
        assert isinstance(batch, tuple)
        batch_sizes.append(len(batch))
        return [len(batch)] * len(batch)

    Pruner(model, graph=graph, preserve_io=False).plan(
        Pruner(model, graph=graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.2),
        strategy=Greedy(metric, max_trials=1),
    )
    assert batch_sizes == [64]


def test_greedy_initial_invalid_divisibility_and_limit():
    model = nn.Linear(4, 10)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    aligned = Pruner(model, graph=graph, preserve_io=False, constraints=[Divisible(axis, 4)])
    space = aligned.discover_candidates()
    plan = aligned.plan(space, budget=ChannelRatio(0.2), strategy=Greedy(Magnitude()))
    assert plan.selection_report.removed == (2,)
    with pytest.raises(PlanningError, match="empty request"):
        aligned.plan(space, budget=ChannelRatio(0.1), strategy=Greedy(Magnitude()))
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.4),
        strategy=Greedy(Magnitude(), max_trials=1),
    )
    assert plan.selection_report.limit_reached
    assert plan.analysis.status == "resolved"
    assert plan.selection_report.removed == (1,)


def test_balanced_completion_multiple_constraints_and_stable_ties():
    model = nn.Linear(4, 12, bias=False)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    constraints = [
        Balanced(axis, tuple(IndexSet.span(i, i + n) for i in range(0, 12, n))) for n in (6, 4)
    ]

    def metric(ctx, batch):
        return [0.0] * len(batch)

    pruner = Pruner(model, graph=graph, constraints=constraints, preserve_io=False)
    space = pruner.discover_candidates()
    a = pruner.plan(space, budget=ChannelRatio(0.5), strategy=Greedy(metric))
    b = pruner.plan(space, budget=ChannelRatio(0.5), strategy=Greedy(metric))
    assert a.selected == b.selected
    assert a.analysis.status == "resolved"
    assert 0 <= sum(a.selection_report.removed) <= 6
    manual = Pruner(
        pruner.model, graph=pruner.graph, preserve_io=False, constraints=constraints
    ).plan_remove([axis.select([0, 1, 4, 6, 8, 9])])
    assert manual.analysis.status == "resolved"
    assert a.selection_report.trials <= 10_000


def test_metric_errors_and_nonadditive_custom_scoring():
    model = nn.Linear(4, 6)
    _graph, pruner = build(model, torch.randn(2, 4))
    pruner = Pruner(model, graph=_graph, preserve_io=False)
    space = pruner.discover_candidates()
    common = {"budget": ChannelRatio(0.4)}
    with pytest.raises(PlanningError, match="gradients"):
        pruner.plan(space, strategy=Greedy(WeightTaylor()), **common)
    for metric in (lambda c, b: [float("nan")] * len(b), lambda c, b: [1]):
        with pytest.raises(PlanningError, match=r"nonfinite|length"):
            pruner.plan(space, strategy=Greedy(metric), **common)
    seen = []

    def metric(ctx, batch):
        seen.extend(len(c.remove) for c in batch)
        return [float(len(c.remove) ** 2) for c in batch]

    def strategy(ctx):
        a, b = ctx.candidates[:2]
        joint = Candidate("temporary", (*a.remove, *b.remove))
        assert ctx.score(metric, [joint]) == (4.0,)
        return [a.key, b.key]

    plan = pruner.plan(space, strategy=strategy, **common)
    assert seen == [2] and len(plan.selected) == 2
    with pytest.raises(PlanningError, match="unregistered"):
        pruner.plan(space, strategy=lambda c: ["bad"], **common)


def test_default_zero_budget_avoids_candidate_scoring(monkeypatch):
    model = nn.Sequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    calls = []
    original = DependencyGraph.propagate

    def counted(self, **kwargs):
        calls.append(1)
        return original(self, **kwargs)

    def forbidden_metric(context, batch):
        raise AssertionError("Zero budget must not score candidates")

    monkeypatch.setattr(DependencyGraph, "propagate", counted)
    plan = Pruner(model, graph=graph).plan(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0),
        strategy=Greedy(forbidden_metric),
    )
    assert not plan.recipes and len(calls) <= 5


def test_zero_trial_strategy_skips_scoring_but_validates_empty(monkeypatch):
    model = Chain()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    calls = []

    def metric(context, batch):
        calls.append(batch)
        raise AssertionError("No score should be needed")

    plan = Pruner(model, graph=graph).plan(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(metric, max_trials=0),
    )
    assert not calls and not plan.recipes and plan.selection_report.limit_reached
