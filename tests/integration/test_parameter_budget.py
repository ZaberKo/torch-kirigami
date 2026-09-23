"""Whole-model resource targets through public planning, persistence and execution."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.support.pruning import KeyStrategy, StaticMetric
from torch_kirigami import DependencyGraph, Divisible
from torch_kirigami.measurement import count_parameters
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    ExecutionError,
    Granularity,
    Greedy,
    Magnitude,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
)


@StaticMetric
def ordered(context, batch):
    order = {candidate.key: index for index, candidate in enumerate(context.candidates)}
    return [order[candidate.key] for candidate in batch]


@pytest.mark.parametrize("cap", [58, 51, 50, 24, 100])
def test_parameter_target_counts_whole_model_and_stops(cap, execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ParameterBudget(cap), strategy=Greedy(ordered)
    )
    kept = min(8, (cap - 2) // 7)
    assert plan.selection_report.before_params == 58
    assert plan.selection_report.after_params == 7 * kept + 2
    assert plan.selection_report.target_met
    assert sum(p.numel() for p in model.parameters()) == 58
    # Portable data replays without the graph, metric, or original Parameter objects.
    loaded = PruningPlan.from_dict(plan.to_dict())
    replay = copy.deepcopy(original)
    compact, _ = Pruner(replay).apply(loaded)
    assert compact is replay
    assert sum(p.numel() for p in compact.parameters()) == 7 * kept + 2
    with torch.no_grad():
        reference = nn.functional.linear(
            nn.functional.linear(x, original[0].weight[-kept:], original[0].bias[-kept:]),
            original[1].weight[:, -kept:],
            original[1].bias,
        )
    torch.testing.assert_close(compact(x), reference)
    assert model[0].out_features == 8


def test_frozen_uncaptured_and_aliased_parameters_count_once(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 8), nn.Linear(8, 2)
            self.alias = self.a.weight
            self.unused = nn.Parameter(torch.ones(100), requires_grad=False)
            self.register_buffer("state", torch.zeros(1000))

        def forward(self, x):
            return self.b(self.a(x))

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph)
    compact, result = pruner.prune(
        pruner.discover_candidates(), budget=ParameterBudget(150), strategy=Greedy(ordered)
    )
    assert result.plan.selection_report.before_params == 158
    assert result.plan.selection_report.after_params == 144
    assert sum(p.numel() for p in compact.parameters()) == 144
    assert compact.alias is compact.a.weight
    assert compact.unused.numel() == 100 and compact.state.numel() == 1000


def test_joint_row_column_overlap_and_wrapped_candidates(execution_device):
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False)
    )
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    a, b = graph.parameter("0.weight").axis(0), graph.parameter("1.weight").axis(0)
    combined = Candidate("both", (a.select([0]), b.select([0])))
    # Channel accounting metadata does not determine the whole-model count.
    space = CandidateSpace((combined,), ())
    pruner = Pruner(model, graph=graph)
    compact, result = pruner.prune(space, budget=ParameterBudget(27), strategy=Greedy(Magnitude()))
    assert result.plan.selection_report.after_params == 12 + 9 + 6
    assert sum(p.numel() for p in compact.parameters()) == 27


@pytest.mark.parametrize("cap,limit", [(1, 100), (40, 0), (40, 1)])
def test_unmet_parameter_target_never_applies(cap, limit, execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    originals = tuple(model.parameters())
    values = tuple(p.detach().clone() for p in originals)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="Parameter target not reached"):
        pruner.prune(
            pruner.discover_candidates(),
            budget=ParameterBudget(cap),
            strategy=Greedy(ordered, max_trials=limit),
        )
    assert all(a is b for a, b in zip(originals, model.parameters(), strict=True))
    for p, value in zip(model.parameters(), values, strict=True):
        torch.testing.assert_close(p, value)
    graph.validate()


def test_custom_strategy_must_reach_target_and_budget_types_coexist():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    with pytest.raises(PlanningError, match="58 remain"):
        pruner.plan(space, budget=ParameterBudget(40), strategy=KeyStrategy(lambda ctx: ()))
    channel = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(ordered))
    parameter = pruner.plan(space, budget=ParameterBudget(44), strategy=Greedy(ordered))
    assert channel.recipes == parameter.recipes
    assert parameter.selection_report.after_params == 44


def test_parameter_target_completes_alignment_and_invalid_empty_request():
    model = nn.Linear(4, 10)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    axis = graph.parameter("weight").axis(0)
    pruner = Pruner(model, graph=graph, preserve_io=False, constraints=(Divisible(axis, 4),))
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ParameterBudget(100), strategy=Greedy(ordered)
    )
    assert (
        plan.selection_report.after_params == 40
    )  # Empty is below the cap but structurally invalid.
    aligned = Pruner(model, graph=graph, preserve_io=False, granularity=Granularity(default=4))
    plan = aligned.plan(
        aligned.discover_candidates(), budget=ParameterBudget(39), strategy=Greedy(ordered)
    )
    assert plan.selection_report.after_params == 20


def test_already_below_target_does_not_score_or_require_candidates():
    model = nn.Linear(4, 2)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph)

    @StaticMetric
    def unexpected_score(context, batch):
        pytest.fail("A satisfied target must not invoke the metric")

    plan = pruner.plan(
        CandidateSpace((), ()),
        budget=ParameterBudget(10),
        strategy=Greedy(unexpected_score, max_trials=0),
    )
    assert not plan.recipes and plan.selection_report.after_params == 10
    assert plan.selection_report.trials == 0
    with pytest.raises(PlanningError, match="10 remain"):
        pruner.plan(
            CandidateSpace((), ()), budget=ParameterBudget(9), strategy=Greedy(unexpected_score)
        )


def test_unsupported_branch_stays_fixed_and_counts_toward_parameter_limit(execution_device):
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.out = nn.Linear(4, 6), nn.Linear(4, 6), nn.Linear(6, 2)

        def forward(self, x):
            return torch.special.gammaln(self.a(x).exp()), self.out(self.b(x))

    model = Branches().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()

    @StaticMetric
    def metric(context, batch):
        assert all(c.axis.tensor != graph.parameter("a.weight") for c in batch)
        return ordered.score(context, batch, selected=context.impact(()))

    plan = pruner.plan(space, budget=ParameterBudget(46), strategy=Greedy(metric))
    assert plan.selection_report.before_params == 74 and plan.selection_report.after_params == 46
    assert any("Incomplete" in reason for _, reason in plan.selection_report.exclusions)
    with pytest.raises(PlanningError, match="39 remain"):
        pruner.plan(space, budget=ParameterBudget(38), strategy=Greedy(metric))
    graph.validate()
    compact, _ = pruner.apply(plan)
    actual_fixed, actual_pruned = compact(x)
    torch.testing.assert_close(actual_fixed, original(x)[0], rtol=0, atol=0)
    reference = nn.functional.linear(
        nn.functional.linear(x, original.b.weight[-2:], original.b.bias[-2:]),
        original.out.weight[:, -2:],
        original.out.bias,
    )
    torch.testing.assert_close(actual_pruned, reference)


@pytest.mark.parametrize(
    "field,value", [("before_params", 59), ("after_params", 43), ("max_params", 1)]
)
def test_parameter_report_is_checked_against_portable_structure(field, value):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ParameterBudget(44), strategy=Greedy(ordered)
    )
    altered = replace(plan, selection_report=replace(plan.selection_report, **{field: value}))
    with pytest.raises(ValueError, match="Parameter report"):
        PruningPlan.from_dict(altered.to_dict())
    with pytest.raises(ExecutionError, match="Parameter report"):
        pruner.apply(altered)
    graph.validate()


@pytest.mark.parametrize("value", [-1, True, 1.5, float("inf"), "100"])
def test_parameter_budget_requires_an_integer_count(value):
    with pytest.raises(ValueError, match="max_params"):
        ParameterBudget(value)


@pytest.mark.parametrize("ratio,expected", [(0, 100), (0.1, 90), (0.9, 10), (0.58, 42), (0.999, 0)])
def test_ratio_conversion_is_exact_at_integer_boundaries(ratio, expected, execution_device):
    model = nn.Linear(9, 10)
    assert count_parameters(model) == 100
    budget = ParameterBudget.from_ratio(model, ratio)
    assert budget.max_params == expected
    # The budget retains neither the model nor a live denominator.
    model.weight = nn.Parameter(torch.ones(1, 9))
    assert budget.max_params == expected


@pytest.mark.parametrize(
    "ratio", [-0.1, 1, 2, 10**400, float("nan"), float("inf"), True, "0.1", None]
)
def test_ratio_conversion_rejects_invalid_values(ratio):
    with pytest.raises(ValueError, match="pruning_ratio"):
        ParameterBudget.from_ratio(nn.Linear(1, 1), ratio)


def test_counting_is_readonly_and_distinguishes_parameter_identity(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(10), requires_grad=False)
            self.alias = self.weight
            self.other = nn.Parameter(self.weight.detach())  # Same storage, distinct entity.
            self.register_buffer("buffer", torch.ones(100))

        def forward(self, x):
            pytest.fail("Parameter counting must not execute forward")

    model = Model()
    assert count_parameters(model) == 20
    assert ParameterBudget.from_ratio(model, 0.25).max_params == 15
    assert model.training and model.alias is model.weight and not model.weight.requires_grad
    assert count_parameters(nn.Identity()) == 0


def test_ratio_budget_uses_the_same_plan_and_portable_execution(execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    ratio_budget = ParameterBudget.from_ratio(model, 0.25)  # floor(58 * 0.75) = 43
    ratio_plan = pruner.plan(space, budget=ratio_budget, strategy=Greedy(ordered))
    absolute_plan = pruner.plan(space, budget=ParameterBudget(43), strategy=Greedy(ordered))
    assert ratio_plan.to_dict() == absolute_plan.to_dict()
    compact, _ = pruner.apply(PruningPlan.from_dict(ratio_plan.to_dict()))
    assert count_parameters(compact) == 37  # Three complete feature removals, not two.
    compact(x).sum().backward()
