"""pruning / context contracts."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.support.models import _context
from tests.support.pruning import StaticMetric
from torch_kirigami import (
    CandidateAxis,
    DependencyGraph,
    Fixed,
    OperatorRegistry,
    OperatorRule,
    Selection,
)
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    Magnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
    StrategyResult,
    WeightTaylor,
    strategies,
)


class FalseyFilter:
    def __bool__(self):
        return False

    def __call__(self, ref, parameter):
        return False


class FalseyStrategy:
    def __bool__(self):
        return False

    def select(self, context):
        return StrategyResult(())


@pytest.mark.parametrize(
    "metric", [Magnitude(), GroupMagnitude(), WeightTaylor(), WeightTaylor(mode="joint_abs")]
)
def test_metric_keyword_contract_and_accepted_impact_match_independent_reference(
    metric, execution_device
):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    axis = graph.parameter("0.weight").axis(0)
    candidate = Candidate("next", (axis.select([1]),), axis)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.5), space.channel_axes, pruner.constraints
    )
    accepted_impact = context.impact((axis.select([0]),))
    values = torch.cat((original[0].weight[1], original[0].bias[1:2], original[1].weight[:, 1]))
    if isinstance(metric, GroupMagnitude):
        energies = original[0].weight[1:].square().sum(1) + original[1].weight[:, 1:].square().sum(
            0
        )
        expected = (energies[0] / energies.mean()).item()
    elif isinstance(metric, WeightTaylor):
        expected = (
            values.abs().sum() if metric.mode == "elementwise_abs" else values.sum().abs()
        ).item()
    else:
        expected = values.norm(p=2).item()
    direct = metric.score(context=context, candidates=(candidate,), accepted_impact=accepted_impact)
    scores = context.score(metric=metric, candidates=(candidate,), accepted_impact=accepted_impact)
    assert direct == pytest.approx([expected])
    assert scores == pytest.approx((expected,))
    assert isinstance(scores, tuple)
    # Scoring is read-only; the baseline's requests have not yet been applied.
    torch.testing.assert_close(model(x), original(x))
    plan = pruner.plan_remove((*accepted_impact.requested, *candidate.remove))
    compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    hidden = nn.functional.linear(x, original[0].weight[2:], original[0].bias[2:])
    reference = nn.functional.linear(hidden, original[1].weight[:, 2:], original[1].bias)
    torch.testing.assert_close(compact(x), reference)
    compact(x).sum().backward()


@pytest.mark.parametrize("empty_constraints", [(), [], iter(())])
def test_graph_free_apply_accepts_empty_constraint_iterables(empty_constraints, execution_device):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    plan = Pruner(model, graph=graph).plan_remove((axis.select([0]),))
    restored = copy.deepcopy(model)
    compact, _ = Pruner(restored, constraints=empty_constraints).apply(
        PruningPlan.from_dict(plan.to_dict())
    )
    reference = nn.functional.linear(
        nn.functional.linear(x, original[0].weight[1:], original[0].bias[1:]),
        original[1].weight[:, 1:],
        original[1].bias,
    )
    torch.testing.assert_close(compact(x), reference)
    # Nonempty iterators still require graph-bound planning configuration.
    before = tuple(model.parameters())
    with pytest.raises(ValueError, match="configuration requires"):
        Pruner(model, constraints=iter((Fixed(axis),)))
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    graph.validate()


def test_cache_hit_validates_full_reference_and_impact_owner():
    context, candidate = _context()
    context.impact(candidate.remove)
    altered = replace(candidate.remove[0].tensor, paths=("wrong",))
    invalid = Candidate("altered", (Selection(altered, candidate.remove[0].regions),))
    with pytest.raises(ValueError, match="altered"):
        context.impact(invalid.remove)
    with pytest.raises(ValueError, match="altered"):
        context.score(Magnitude(), (invalid,))
    context.compile(context.impact(()))
    other, _ = _context()
    with pytest.raises(ValueError, match="another graph"):
        context.compile(other.impact(()))


def test_context_premises_are_readonly():
    context, _ = _context()
    for name in (
        "graph",
        "operations",
        "candidates",
        "budget",
        "channel_axes",
        "constraints",
        "widths",
        "targets",
        "trials",
    ):
        with pytest.raises(AttributeError):
            setattr(context, name, None)
    context.attempt(())
    result = StrategyResult((), exclusions=(("one", "test"),))
    report = context.report(context.impact(()), result)
    assert report.trials == 1 and report.exclusions == (("one", "test"),)


@pytest.mark.parametrize("conflict", [None, "key", "block"])
def test_candidate_domains_use_axes_not_keys(conflict):
    registry = OperatorRegistry.default()
    base = registry.modules[nn.Linear]

    class DuplicateDomains(OperatorRule):
        def analyze(self, ctx):
            spec = base.analyze(ctx)
            domain = spec.candidates[0]
            if ctx.module_path == "a":
                twin = CandidateAxis(
                    domain.key if conflict == "key" else "duplicate",
                    ctx.binding("weight").axis(1) if conflict == "key" else domain.axis,
                    2 if conflict == "block" else domain.block_size,
                )
                return replace(spec, candidates=(domain, twin))
            return spec

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)
            self.b = nn.Linear(4, 6)

        def forward(self, x):
            return self.a(x), self.b(x)

    registry.modules[nn.Linear] = DuplicateDomains()
    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),), operators=registry)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    options = {
        "strategy": Greedy(
            StaticMetric(
                lambda ctx, batch: [0 if c.axis.tensor.paths[0] == "b.weight" else 1 for c in batch]
            )
        ),
        "budget": ChannelRatio(0.25, scope="global"),
    }
    if conflict:
        with pytest.raises(PlanningError, match="Conflicting"):
            pruner.discover_candidates()
    else:
        plan = pruner.plan(pruner.discover_candidates(), **options)
        assert plan.selection_report.widths == (6, 6)
        assert plan.selection_report.targets == (9,) and plan.selection_report.removed == (0, 3)


def test_falsey_strategy_and_parameter_filters_are_called():
    model = nn.Linear(4, 6)
    context, candidate = _context(model)
    plan = Pruner(model, graph=context.graph, preserve_io=False).plan(
        Pruner(model, graph=context.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0),
        strategy=FalseyStrategy(),
    )
    assert plan.selected == ()
    # No gradient is needed because the supplied filter excludes all parameters.
    for metric in (
        Magnitude(parameter_filter=FalseyFilter()),
        WeightTaylor(parameter_filter=FalseyFilter()),
    ):
        assert context.score(metric, (candidate,)) == (0.0,)


def test_conditional_scoring_rejects_another_graphs_accepted_impact():
    context, candidate = _context()
    other, _ = _context()
    with pytest.raises(ValueError, match="another graph"):
        context.score(Magnitude(), (candidate,), accepted_impact=other.impact(()))


@pytest.mark.parametrize(
    "reply",
    [
        (),
        StrategyResult(("not_registered",)),
        StrategyResult((), exclusions=(("not_registered", "unsupported"),)),
    ],
)
def test_custom_strategy_result_is_checked_and_failure_preserves_model(reply):
    context, _ = _context()
    model = context.graph.model
    pruner = Pruner(model, graph=context.graph, preserve_io=False)
    before = tuple(model.parameters())

    class InvalidStrategy:
        def select(self, context):
            return reply

    error = TypeError if not isinstance(reply, StrategyResult) else PlanningError
    with pytest.raises(error, match=r"StrategyResult|unregistered"):
        pruner.plan(
            pruner.discover_candidates(), budget=ChannelRatio(0.5), strategy=InvalidStrategy()
        )
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    context.graph.validate()


def test_strategy_stop_reason_cannot_override_measured_budget():
    context, _ = _context()
    pruner = Pruner(context.graph.model, graph=context.graph, preserve_io=False)

    class ClaimedSuccess:
        def select(self, context):
            return StrategyResult((), stop_reason="target_reached")

    with pytest.raises(PlanningError, match="Parameter target not reached"):
        pruner.plan(
            pruner.discover_candidates(), budget=ParameterBudget(10), strategy=ClaimedSuccess()
        )
    context.graph.validate()


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_normalized_plans_are_independent_of_internal_score_batching(strategy_type, monkeypatch):
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 6), nn.Linear(6, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    plans = []
    for size in (1, 3, 32):
        monkeypatch.setattr(strategies, "_SCORE_BATCH_SIZE", size)
        plan = pruner.plan(
            space, budget=ChannelRatio(0.25), strategy=strategy_type(GroupMagnitude())
        )
        plans.append(plan.to_dict())
    assert plans[0] == plans[1] == plans[2]
    graph.validate()
