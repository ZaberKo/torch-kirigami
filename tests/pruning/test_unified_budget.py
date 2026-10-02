"""Public final-size budgets, independent references and failure-state contracts."""

import copy
import io
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import KeyStrategy, StaticMetric
from torch_kirigami import DependencyGraph, Divisible
from torch_kirigami.pruning import (
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    DynamicGreedy,
    ExecutionError,
    Granularity,
    Greedy,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
    SelectionReport,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import CumulativeChannelBudget


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("resource", ["channels", "parameters"])
def test_final_size_targets_share_completion_and_portable_execution(
    strategy_type, resource, execution_device
):
    model = nn.Sequential(nn.Linear(4, 64), nn.Linear(64, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 8}))
    space = pruner.discover_candidates()
    # 52 hidden units would have 52*(4+1+2)+2 parameters. Alignment requires 48.
    budget = (
        ChannelCount((52,), space.channel_axes) if resource == "channels" else ParameterBudget(366)
    )
    metric = StaticMetric(lambda context, candidates: [0.0] * len(candidates))
    plan = pruner.plan(space, budget=budget, strategy=strategy_type(metric))
    assert len(plan.selected) == 16
    assert plan.selection_report.target_met
    torch.testing.assert_close(model(x), original(x))
    compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    reference = F.linear(
        F.linear(x, original[0].weight[16:], original[0].bias[16:]),
        original[1].weight[:, 16:],
        original[1].bias,
    )
    assert compact[0].out_features == 48
    torch.testing.assert_close(compact(x), reference)
    compact(x).sum().backward()


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_local_targets_do_not_keep_pruning_a_satisfied_axis(strategy_type, execution_device):
    model = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 8), nn.Linear(8, 1)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(3, 2)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    metric = StaticMetric(
        lambda context, candidates: [
            0 if c.axis == space.channel_axes[0] else 1 for c in candidates
        ]
    )
    plan = pruner.plan(
        space, budget=ChannelCount((6, 4), space.channel_axes), strategy=strategy_type(metric)
    )
    assert plan.selection_report.remaining == (6, 4)
    compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    y = F.linear(x, original[0].weight[2:], original[0].bias[2:])
    y = F.linear(y, original[1].weight[4:, 2:], original[1].bias[4:])
    torch.testing.assert_close(compact(x), F.linear(y, original[2].weight[:, 4:], original[2].bias))


def test_local_joint_effects_may_overshoot_one_axis_and_reports_cannot_cancel(execution_device):
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.out = nn.Linear(2, 8), nn.Linear(2, 8), nn.Linear(8, 1)

        def forward(self, x):
            return self.out(self.a(x) + self.b(x))

    model = Residual().eval()
    x = torch.randn(2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    plan = pruner.plan(
        space,
        budget=ChannelCount((6, 4), space.channel_axes),
        strategy=Greedy(StaticMetric(lambda ctx, cs: [0] * len(cs))),
    )
    assert plan.selection_report.remaining == (4, 4)
    # A surplus on the first axis cannot compensate for missing reduction on the second.
    report = SelectionReport(space.channel_axes, (8, 8), (4, 2), (6, 4), "local")
    assert report.shortfall == 2 and not report.target_met
    with pytest.raises(ValueError, match="targets"):
        PruningPlan.from_dict(
            replace(plan, selection_report=replace(plan.selection_report, targets=(3, 4))).to_dict()
        )
    with pytest.raises(ValueError, match="analysis"):
        PruningPlan.from_dict(
            replace(plan, selection_report=replace(plan.selection_report, removed=(5, 4))).to_dict()
        )
    missing = replace(
        plan.selection_report.channel_axes[0].tensor,
        id="missing_parameter",
        paths=("missing.weight",),
    ).axis(0)
    malformed = replace(
        plan,
        selection_report=replace(
            plan.selection_report,
            channel_axes=(missing, plan.selection_report.channel_axes[1]),
        ),
    )
    before = tuple(model.parameters())
    with pytest.raises(ValueError, match="absent"):
        PruningPlan.from_dict(malformed.to_dict())
    with pytest.raises(ExecutionError, match="absent"):
        pruner.apply(malformed)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    graph.validate()
    pruner.apply(plan)


@pytest.mark.parametrize("failure", ["underfill", "limit", "protected"])
def test_missed_channel_target_never_mutates_and_has_valid_alternative(failure, execution_device):
    model = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    before = tuple(model.parameters())
    if failure == "protected":
        space = CandidateSpace((), (graph.parameter("1.weight").axis(0),))
        budget, strategy = (
            ChannelCount((0,), space.channel_axes),
            Greedy(StaticMetric(lambda c, cs: [0] * len(cs))),
        )
    else:
        budget = ChannelRatio(0.5)
        strategy = (
            KeyStrategy(lambda ctx: (ctx.candidates[0].key,))
            if failure == "underfill"
            else Greedy(StaticMetric(lambda c, cs: [0] * len(cs)), max_trials=1)
        )
    with pytest.raises(PlanningError, match="Channel target not reached"):
        pruner.plan(space, budget=budget, strategy=strategy)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    torch.testing.assert_close(model(x), original(x))
    graph.validate()
    # A normal target remains attainable after failed planning on the same model.
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(StaticMetric(lambda c, cs: [0] * len(cs))),
    )
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    expected = F.linear(
        F.linear(x, original[0].weight[4:], original[0].bias[4:]),
        original[1].weight[:, 4:],
        original[1].bias,
    )
    torch.testing.assert_close(model(x), expected)


def test_already_met_budget_still_repairs_invalid_empty_structure(execution_device):
    model = nn.Sequential(nn.Linear(2, 10), nn.Linear(10, 1)).eval()
    x = torch.randn(2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, constraints=(Divisible(axis, 4),))
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0),
        strategy=Greedy(StaticMetric(lambda c, cs: [0] * len(cs))),
    )
    assert plan.selection_report.remaining == (8,)
    pruner.apply(plan)
    model(x).sum().backward()


def test_cumulative_targets_keep_original_baseline_after_overshoot_and_restore(execution_device):
    model = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 1)).eval()
    x = torch.randn(2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 8}))
    space = pruner.discover_candidates()
    cumulative = CumulativeChannelBudget(graph, space)
    _, result = pruner.prune(
        space,
        budget=cumulative.budget(graph, space, 0.1),
        strategy=Greedy(StaticMetric(lambda c, cs: [0] * len(cs))),
    )
    assert model[0].out_features == 56
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 8}))
    space = pruner.discover_candidates()
    cumulative.update(result, graph, space)
    assert cumulative.budget(graph, space, 0.1).max_channels == (57,)
    plan = pruner.plan(
        space,
        budget=cumulative.budget(graph, space, 0.1),
        strategy=Greedy(StaticMetric(lambda c, cs: [0] * len(cs))),
    )
    assert not plan.recipes
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 1)).eval(), stream)
    fresh = DependencyGraph.build(restored, args=(x,))
    restored_pruner = Pruner(restored, graph=fresh, granularity=Granularity(by_path={"0": 8}))
    restored_space = restored_pruner.discover_candidates()
    resumed = CumulativeChannelBudget(fresh, restored_space)
    resumed.load_state_dict(cumulative.state_dict(), fresh, restored_space)
    _, result = restored_pruner.prune(
        restored_space,
        budget=resumed.budget(fresh, restored_space, 0.2),
        strategy=Greedy(StaticMetric(lambda c, cs: [0] * len(cs))),
    )
    assert restored[0].out_features == 48 and result.plan.selection_report.target_met
    restored(x).sum().backward()
