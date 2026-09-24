"""Proved score reuse agrees with joint analysis and preserves execution checks."""

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.contracts import AxisBarrier, Diagnostic, Divisible
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    DynamicGreedy,
    Granularity,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
)
from torch_kirigami.pruning.ranking import IndependentRanking
from torch_kirigami.selection import AxisRef, Selection, TensorRef


class ReferenceMagnitude(GroupMagnitude):
    """Force ordinary per-candidate joint scoring with the same metric formula."""


def context_for(
    graph: DependencyGraph, candidates: tuple[Candidate, ...], constraints: tuple = ()
) -> PlanningContext:
    """Build a scoring context with explicit logical domains."""
    axes = tuple(dict.fromkeys(c.axis for c in candidates if c.axis is not None))
    return PlanningContext(
        graph, graph.operations(), candidates, ChannelRatio(0.5), axes, constraints
    )


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("convolution", [False, True])
def test_public_plan_restore_apply_matches_generic_joint_scoring(
    p: int, strategy_type: type[Greedy], convolution: bool, execution_device: str
) -> None:
    """Both policies preserve complete plans and compact-model outputs on CPU/CUDA."""
    torch.manual_seed(17)
    layer = (lambda a, b: nn.Conv2d(a, b, 1)) if convolution else nn.Linear
    model = nn.Sequential(
        layer(4, 12), nn.ReLU(), layer(12, 4), layer(4, 12), nn.ReLU(), layer(12, 4)
    ).eval()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, 3, 3) if convolution else torch.randn(2, 3, 4)
    plans, compact = [], []
    for current, metric in ((model, GroupMagnitude(p)), (reference, ReferenceMagnitude(p))):
        graph = DependencyGraph.build(current, args=(x,))
        pruner = Pruner(current, graph=graph, granularity=Granularity(default=2))
        space = pruner.discover_candidates(targets=("0", "3"))
        plan = pruner.plan(
            space, budget=ParameterBudget.from_ratio(current, 0.25), strategy=strategy_type(metric)
        )
        plans.append(plan)
        restored = PruningPlan.from_dict(plan.to_dict())
        compact.append(pruner.apply(restored)[0])
    assert plans[0].selected == plans[1].selected
    assert plans[0].selection_report == plans[1].selection_report
    for key, value in compact[0].state_dict().items():
        torch.testing.assert_close(value, compact[1].state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(compact[0](x), compact[1](x), rtol=0, atol=0)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("scale", [0.0, 1e-200, 1.0, 1e200])
def test_batched_statistics_and_conditional_order_have_independent_reference(
    p: int, scale: float, execution_device: str
) -> None:
    """Bias broadcasting, BN weights, aliases and noncontiguous columns count once."""
    model = nn.Sequential(nn.Linear(3, 6), nn.BatchNorm1d(6), nn.Linear(6, 3)).double().eval()
    model.register_parameter("alias", model[0].weight)
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(1, 19, dtype=torch.float64).reshape(6, 3) * scale)
        model[1].weight.copy_(torch.arange(1, 7, dtype=torch.float64) * scale)
        model[2].weight = nn.Parameter(
            (torch.arange(1, 19, dtype=torch.float64).reshape(6, 3) * scale).T
        )
        model[0].bias.fill_(1e100)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)
    # A candidate subset must still normalize against all surviving positions.
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in (0, 2, 4))
    context = context_for(graph, candidates)
    ranking = IndependentRanking.build(context, GroupMagnitude(p))
    assert ranking is not None
    rows = model[0].weight.detach().cpu().tolist()
    bn = model[1].weight.detach().cpu().tolist()
    columns = model[2].weight.detach().T.cpu().tolist()
    expected = [
        math.fsum(abs(v) for v in (*a, b, *c)) if p == 1 else math.hypot(*a, b, *c)
        for a, b, c in zip(rows, bn, columns, strict=True)
    ]
    assert ranking.norms[0] == pytest.approx(expected, rel=1e-12, abs=0)
    selected = context.impact((axis.select([0]),))
    actual, removals = ranking.rank(selected, (axis,))
    remaining = candidates[1:]
    scores = context.score(ReferenceMagnitude(p), remaining, selected=selected)
    assert actual == [
        c for _, c in sorted(zip(scores, remaining, strict=True), key=lambda x: (x[0], x[1].key))
    ]
    assert all(removals[c.key] == {axis: c.remove[0].fully_selected_indices(0)} for c in actual)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_batched_norms_preserve_low_precision_and_complex_values(
    dtype: torch.dtype, execution_device: str
) -> None:
    model = nn.Linear(3, 4, bias=False, dtype=dtype)
    with torch.no_grad():
        values = torch.tensor([[1, 2, 3], [0, 0, 0], [4, 2, 1], [1, 1, 1]], dtype=dtype)
        if dtype.is_complex:
            values *= 1 + 2j
        model.weight.copy_(values)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3, dtype=dtype),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(4))
    context = context_for(graph, candidates)
    ranking = IndependentRanking.build(context, GroupMagnitude())
    assert ranking is not None
    expected = [math.hypot(*(abs(v) for v in row)) for row in values.cpu().tolist()]
    assert ranking.norms[0] == pytest.approx(expected, rel=1e-6, abs=0)


def test_dynamic_propagations_follow_joint_trials_instead_of_candidate_count(monkeypatch) -> None:
    model = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 3)).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 4}))
    space = pruner.discover_candidates(targets=("0",))
    calls = []
    original = DependencyGraph.propagate

    def counted(self: DependencyGraph, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DependencyGraph, "propagate", counted)
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=DynamicGreedy(GroupMagnitude()))
    assert len(plan.selected) == 16
    assert plan.selection_report.trials == 4
    assert len(calls) <= plan.selection_report.trials + 3


@dataclass(frozen=True)
class IncompletePosition:
    """A custom constraint must never lose its scoring-completeness diagnostic."""

    axis: AxisRef

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        selection = selections.get(self.axis.tensor.id)
        if selection and 1 in selection.fully_selected_indices(self.axis.dim):
            return Diagnostic("opaque", "Unproved custom position", complete=False)
        return None


@pytest.mark.parametrize("case", ["shared", "overlap", "reshape", "blocks", "custom", "barrier"])
def test_unproved_candidates_use_generic_scoring(case: str) -> None:
    linear = nn.Linear(4, 4)
    if case == "shared":
        model = nn.Sequential(linear, linear)
    elif case == "reshape":
        model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.Flatten(), nn.Linear(8, 4))
    else:
        model = nn.Sequential(linear, nn.ReLU(), nn.Linear(4, 4))
    x = torch.ones(1, 1, 2, 2) if case == "reshape" else torch.ones(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    positions = [0, 1] if case == "blocks" else [0]
    candidates = (Candidate("first", (axis.select(positions),), axis),)
    if case == "overlap":
        other = graph.parameter("2.weight").axis(0)
        candidates += (Candidate("other", (other.select([1]),), other),)
    constraints = ()
    if case == "custom":
        constraints = (IncompletePosition(axis),)
    elif case == "barrier":
        constraints = (AxisBarrier(axis, "Opaque axis"),)
    context = context_for(graph, candidates, constraints)
    assert IndependentRanking.build(context, GroupMagnitude()) is None
    if case == "custom":
        with pytest.raises(PlanningError, match="Unproved custom position"):
            context.score(GroupMagnitude(), candidates)


@pytest.mark.parametrize("kind", ["subclass", "filter", "instance_override", "inference"])
def test_extensions_and_inference_tensors_do_not_reuse_statistics(kind: str) -> None:
    with torch.inference_mode(kind == "inference"):
        model = nn.Linear(3, 4)
        graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
        axis = graph.parameter("weight").axis(0)
        candidates = (Candidate("first", (axis.select([0]),), axis),)
        context = context_for(graph, candidates)
        metric = ReferenceMagnitude() if kind == "subclass" else GroupMagnitude()
        if kind == "filter":
            metric.parameter_filter = lambda ref, tensor: True
        elif kind == "instance_override":
            metric.score = lambda *args, **kwargs: [7.0]
        assert IndependentRanking.build(context, metric) is None


def test_last_position_and_weight_updates_disable_reuse() -> None:
    model = nn.Linear(3, 4)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(4))
    context = context_for(graph, candidates)
    ranking = IndependentRanking.build(context, GroupMagnitude())
    assert ranking is not None
    assert ranking.rank(context.impact((axis.select([0, 1, 2]),)), (axis,)) is None
    empty = context.impact(())
    with torch.no_grad():
        model.weight.add_(1)
    assert ranking.rank(empty, (axis,)) is None


def test_initial_divisibility_violation_is_still_repaired() -> None:
    model = nn.Sequential(nn.Linear(3, 10), nn.Linear(10, 3))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, constraints=(Divisible(axis, 4),))
    plan = pruner.plan(
        pruner.discover_candidates(targets=("0",)),
        budget=ChannelRatio(0.2),
        strategy=DynamicGreedy(GroupMagnitude()),
    )
    assert len(plan.selected) == 2
    pruner.apply(plan)
    assert model[0].out_features == model[1].in_features == 8


@pytest.mark.parametrize("value", [0.0, 1.0, float("nan"), float("inf")])
def test_ties_and_invalid_statistics_match_generic_public_planning(value: float) -> None:
    model = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 3)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)
    x = torch.ones(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    plans = []
    for metric in (GroupMagnitude(), ReferenceMagnitude()):
        if not math.isfinite(value):
            with pytest.raises(PlanningError, match="nonfinite"):
                pruner.plan(space, budget=ChannelRatio(0.5), strategy=DynamicGreedy(metric))
        else:
            plans.append(
                pruner.plan(space, budget=ChannelRatio(0.5), strategy=DynamicGreedy(metric))
            )
    if plans:
        assert plans[0].to_dict() == plans[1].to_dict()


@pytest.mark.parametrize("scope", ["local", "global"])
def test_equivalent_axes_keep_joint_budget_and_divisibility_behavior(scope: str) -> None:
    """Two entry axes in one component must deduplicate their actual impact."""
    model = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 3)).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    first, second = graph.parameter("0.weight").axis(0), graph.parameter("1.weight").axis(1)
    candidates = tuple(
        Candidate(f"{label}:{i}", (axis.select([i]),), axis)
        for label, axis in (("a", first), ("b", second))
        for i in range(6)
    )
    pruner = Pruner(model, graph=graph, constraints=(Divisible(first, 2),))
    space = CandidateSpace(candidates, (first, second))
    plans = [
        pruner.plan(space, budget=ChannelRatio(0.5, scope=scope), strategy=DynamicGreedy(metric))
        for metric in (GroupMagnitude(), ReferenceMagnitude())
    ]
    assert plans[0].to_dict() == plans[1].to_dict()


class RepeatedProducer(nn.Module):
    """Exercise repeated calls and consumers without counting a weight twice."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 6)
        self.left, self.right = nn.Linear(6, 3), nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = self.stem(x), self.stem(2 * x)
        return self.left(first) + self.right(first + second)


def test_repeated_calls_and_multiple_consumers_preserve_shared_statistics(
    execution_device: str,
) -> None:
    model = RepeatedProducer().double().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3, dtype=torch.float64),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("stem",))
    context = context_for(graph, space.candidates)
    ranking = IndependentRanking.build(context, GroupMagnitude())
    assert ranking is not None
    energy = (
        model.stem.weight.square().sum(1)
        + model.left.weight.square().sum(0)
        + model.right.weight.square().sum(0)
    )
    assert ranking.norms[0] == pytest.approx(energy.sqrt().tolist())
    plans = [
        pruner.plan(space, budget=ChannelRatio(0.5), strategy=DynamicGreedy(metric))
        for metric in (GroupMagnitude(), ReferenceMagnitude())
    ]
    assert plans[0].to_dict() == plans[1].to_dict()
