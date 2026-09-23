"""Static and conditional score policies share verified physical execution."""

import copy
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRef,
    DependencyGraph,
    Diagnostic,
    Divisible,
    Impact,
    Selection,
    TensorRef,
)
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    DynamicGreedy,
    Granularity,
    Greedy,
    Magnitude,
    MetricContext,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
    StrategyResult,
)


def test_conditional_magnitude_changes_choice_and_matches_manual_model(
    execution_device: str,
) -> None:
    """A previously removed row/column intersection cannot influence later scores."""
    source = nn.Sequential(
        nn.Linear(1, 3, bias=False), nn.Linear(3, 3, bias=False), nn.Linear(3, 1, bias=False)
    )
    with torch.no_grad():
        source[0].weight.copy_(torch.tensor([[1.0], [2.0], [3.0]]))
        source[1].weight.copy_(torch.tensor([[4.0, 2.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]]))
        source[2].weight.copy_(torch.tensor([[1.0, 1.0, 20.0]]))
    x = torch.randn(4, 1)
    for strategy_type, expected_keys, second_kept in (
        (Greedy, ("a", "c"), [0, 2]),
        (DynamicGreedy, ("a", "b"), [1, 2]),
    ):
        model = copy.deepcopy(source)
        graph = DependencyGraph.build(model, args=(x,))
        first, second = graph.parameter("0.weight").axis(0), graph.parameter("1.weight").axis(0)
        space = CandidateSpace(
            (
                Candidate("a", (first.select([0]),), first),
                Candidate("b", (second.select([0]),), second),
                Candidate("c", (second.select([1]),), second),
            ),
            (first, second),
        )
        before = tuple(model.parameters())
        pruner = Pruner(model, graph=graph)
        plan = pruner.plan(
            space,
            budget=ChannelRatio(1 / 3, scope="global"),
            strategy=strategy_type(Magnitude(p=1)),
        )
        assert plan.selected == expected_keys
        assert plan.selection_report.trials == 2
        assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
        torch.testing.assert_close(model(x), source(x))
        pruner.apply(PruningPlan.from_dict(plan.to_dict()))
        first_kept = [1, 2]
        hidden = F.linear(x, source[0].weight[first_kept])
        middle = F.linear(hidden, source[1].weight[second_kept][:, first_kept])
        expected = F.linear(middle, source[2].weight[:, second_kept])
        torch.testing.assert_close(model(x), expected)
        model(x).sum().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


class TrackingMetric:
    """Record accepted removal counts while returning deterministic local scores."""

    def __init__(self, *, fail_after_commit: bool = False) -> None:
        self.selected_counts: list[int] = []
        self.fail_after_commit = fail_after_commit

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        selected: Impact,
    ) -> list[float]:
        axis = candidates[0].axis
        assert axis is not None
        count = len(selected.selection(axis.tensor).fully_selected_indices(axis.dim))
        self.selected_counts.append(count)
        if self.fail_after_commit and count:
            return [float("nan")] * len(candidates)
        return [
            float(sum(candidate.remove[0].fully_selected_indices(0))) for candidate in candidates
        ]


def test_dynamic_reranks_after_whole_constraint_completion(execution_device: str) -> None:
    """Intermediate one-channel proposals must not be presented as commitments."""
    model = nn.Linear(2, 8, bias=False)
    x = torch.randn(3, 2)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, preserve_io=False, granularity=Granularity(default=4))
    metric = TrackingMetric()
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=DynamicGreedy(metric),
    )
    assert metric.selected_counts == [0, 4]
    assert plan.selection_report.removed == (4,)
    assert plan.selection_report.trials == 1
    original = copy.deepcopy(model)
    pruner.apply(plan)
    torch.testing.assert_close(model(x), F.linear(x, original.weight[4:]))


def test_dynamic_accepts_initially_unresolved_divisibility(execution_device: str) -> None:
    """Scoring needs complete influence, not an executable empty request."""
    model = nn.Linear(2, 10, bias=False)
    x = torch.randn(3, 2)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("weight").axis(0)
    pruner = Pruner(model, graph=graph, preserve_io=False, constraints=(Divisible(axis, 4),))
    metric = TrackingMetric()
    original = copy.deepcopy(model)
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.2),
        strategy=DynamicGreedy(metric),
    )
    assert metric.selected_counts == [0, 2]
    assert plan.selection_report.removed == (2,)
    pruner.apply(plan)
    torch.testing.assert_close(model(x), F.linear(x, original.weight[2:]))


def test_dynamic_score_failure_keeps_model_and_graph_usable(execution_device: str) -> None:
    model = nn.Linear(2, 4, bias=False)
    x = torch.randn(3, 2)
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    parameters = tuple(model.parameters())
    with pytest.raises(PlanningError, match="nonfinite"):
        pruner.plan(
            pruner.discover_candidates(),
            budget=ChannelRatio(0.5),
            strategy=DynamicGreedy(TrackingMetric(fail_after_commit=True)),
        )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    torch.testing.assert_close(model(x), original(x))
    graph.validate()
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=DynamicGreedy(TrackingMetric()),
    )
    assert plan.selection_report.removed == (2,)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_empty_target_and_zero_trials_do_not_score(strategy_type: type[Greedy]) -> None:
    model = nn.Linear(2, 4, bias=False)
    graph = DependencyGraph.build(model, args=(torch.randn(1, 2),))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    space = pruner.discover_candidates()
    metric = TrackingMetric()
    plan = pruner.plan(space, budget=ParameterBudget(8), strategy=strategy_type(metric))
    assert not metric.selected_counts and not plan.selected
    plan = pruner.plan(
        space, budget=ChannelRatio(0.5), strategy=strategy_type(metric, max_trials=0)
    )
    assert not metric.selected_counts and plan.selection_report.limit_reached
    with pytest.raises(PlanningError, match="Parameter target not reached"):
        pruner.plan(space, budget=ParameterBudget(4), strategy=strategy_type(metric, max_trials=0))


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_greedy_options_and_result_are_immutable(strategy_type: type[Greedy]) -> None:
    strategy = strategy_type(Magnitude(), max_trials=2)
    with pytest.raises(FrozenInstanceError):
        strategy.max_trials = 3
    result = StrategyResult(("one",), "exhausted", (("two", "fixed"),))
    with pytest.raises(FrozenInstanceError):
        result.keys = ()
    with pytest.raises(ValueError, match="unique"):
        StrategyResult(("one", "one"))
    with pytest.raises(TypeError, match="score"):
        strategy_type(lambda context, batch: [])
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            strategy_type(Magnitude(), max_trials=value)


@pytest.mark.parametrize(
    ("keys", "exclusions", "error", "message"),
    [
        ("abc", (), TypeError, "bare string"),
        (b"abc", (), TypeError, "bare string"),
        ((), "ab", TypeError, "bare string"),
        ((), ("ab",), TypeError, "not a string"),
        ((), (("a", "first"), ("a", "second")), ValueError, "unique"),
        (("a",), (("a", "rejected"),), ValueError, "both select and exclude"),
        ((), (("", "rejected"),), ValueError, "nonempty"),
        ((), (("a", ""),), ValueError, "nonempty"),
    ],
)
def test_strategy_result_rejects_ambiguous_records(
    keys: object, exclusions: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        StrategyResult(keys, exclusions=exclusions)


@dataclass(frozen=True)
class NeedsPartner:
    """Permit the first coordinate only after the second is also selected."""

    axis: AxisRef

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        selection = selections.get(self.axis.tensor.id)
        if selection is not None:
            indices = selection.fully_selected_indices(self.axis.dim)
            if 0 in indices and 1 not in indices:
                return Diagnostic("needs_partner", "Coordinate 0 requires coordinate 1")
        return None


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_parameter_target_preserves_earlier_rejection_diagnostics(
    strategy_type: type[Greedy], execution_device: str
) -> None:
    """Stopping at a resource target must not discard previous failed trials."""
    model = nn.Linear(2, 4, bias=False)
    x = torch.randn(3, 2)
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("weight").axis(0)
    pruner = Pruner(model, graph=graph, preserve_io=False, constraints=(NeedsPartner(axis),))
    space = CandidateSpace(
        (
            Candidate("early", (axis.select([0]),), axis),
            Candidate("later", (axis.select([1]),), axis),
        ),
        (axis,),
    )
    plan = pruner.plan(space, budget=ParameterBudget(6), strategy=strategy_type(TrackingMetric()))
    assert plan.selected == ("later",)
    assert plan.selection_report.trials == 2
    exclusions = dict(plan.selection_report.exclusions)
    assert set(exclusions) == {"early"}
    assert "requires coordinate 1" in exclusions["early"]
    assert "Not retried after the accepted selection changed" in exclusions["early"]
    restored = PruningPlan.from_dict(plan.to_dict())
    assert restored.selection_report.exclusions == plan.selection_report.exclusions
    pruner.apply(restored)
    torch.testing.assert_close(model(x), F.linear(x, original.weight[[0, 2, 3]]))
