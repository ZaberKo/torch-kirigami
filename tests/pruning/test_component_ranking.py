"""Local proof invalidation must preserve generic candidates and dynamic decisions."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
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
from torch_kirigami.pruning.strategies import _rank_candidates


class GenericMagnitude(GroupMagnitude):
    """Use the complete conditional-scoring implementation as a reference."""


class MixedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 6), nn.Linear(6, 2))
        self.b = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.a(x) + self.b(x)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("invalidate", ["cross_axis", "last_position"])
def test_component_fallback_preserves_other_scores_and_fixed_summaries(
    p, invalidate, execution_device
):
    torch.manual_seed(93)
    model = MixedBranches().double().eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4, dtype=torch.float64),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("a.0", "a.2", "b.0"))
    context = PlanningContext(graph, space.candidates, ChannelRatio(0.5), space.channel_axes, ())
    ranking = IndependentRanking.build(context, GroupMagnitude(p), space.channel_axes)
    assert ranking is not None
    original = ranking.score(context.impact(()))
    first = graph.parameter("a.0.weight").axis(0)
    second = graph.parameter("a.2.weight").axis(0)
    independent = graph.parameter("b.0.weight").axis(0)
    requested = (
        second.select([0, 1]) if invalidate == "cross_axis" else first.select(range(7)),
        independent.select([0]),
    )
    accepted = context.impact(requested)
    result = ranking.score(accepted)
    assert result is not None
    live = tuple(
        c for c in space.candidates if c.axis == independent and c.key not in result.covered
    )
    covered = {
        c.key
        for c in space.candidates
        if c.axis == independent and c.remove == (independent.select([0]),)
    }
    assert result.covered == covered
    assert set(result.scores) == {c.key for c in live}
    assert result.removals is original.removals is ranking.removals
    for candidate in live:
        assert result.removals[candidate.key] is original.removals[candidate.key]
    expected = context.score(GenericMagnitude(p), live, accepted_impact=accepted)
    assert tuple(result.scores[c.key] for c in live) == pytest.approx(expected, rel=1e-12)
    ranked, _ = _rank_candidates(
        context, ranking.metric, space.candidates, accepted, space.channel_axes, {}, ranking
    )
    reference, _ = _rank_candidates(
        context, GenericMagnitude(p), space.candidates, accepted, space.channel_axes, {}
    )
    assert ranked == reference  # Invalid components must remain in the generic ranking.
    with pytest.raises(TypeError):
        result.removals[live[0].key][independent] = None


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_mixed_component_public_plan_apply_matches_generic(
    p, strategy_type, execution_device, monkeypatch
):
    torch.manual_seed(97)
    source = MixedBranches().double().eval()
    with torch.no_grad():
        source.a[2].weight[:2].mul_(0.0001)
        source.a[3].weight[:, :2].mul_(0.0001)
    x = torch.randn(2, 4, dtype=torch.float64)
    original_score = IndependentRanking.score
    partial = []

    def observe(ranking, accepted):
        result = original_score(ranking, accepted)
        if result is not None:
            cached = {candidate.key for candidate, _, _ in ranking.candidates}
            if cached - result.scores.keys() - result.covered:
                partial.append(result)
        return result

    monkeypatch.setattr(IndependentRanking, "score", observe)
    plans = []
    for metric in (GroupMagnitude(p), GenericMagnitude(p)):
        model = copy.deepcopy(source)
        graph = DependencyGraph.build(model, args=(x,))
        pruner = Pruner(model, graph=graph, granularity=Granularity(default=2))
        space = pruner.discover_candidates(targets=("a.0", "a.2", "b.0"))
        old_parameters = tuple(model.parameters())
        with pytest.raises(PlanningError, match="target not reached"):
            pruner.plan(
                space, budget=ParameterBudget(1), strategy=strategy_type(metric, max_trials=1)
            )
        assert all(a is b for a, b in zip(old_parameters, model.parameters(), strict=True))
        graph.validate()
        plan = pruner.plan(
            space, budget=ParameterBudget.from_ratio(model, 0.35), strategy=strategy_type(metric)
        )
        plans.append(plan.to_dict())
        retained = {}
        for path in ("a.0", "a.2", "b.0"):
            ref = graph.parameter(f"{path}.weight")
            removed = set(plan.analysis.selection(ref).fully_selected_indices(0))
            retained[path] = [i for i in range(ref.shape[0]) if i not in removed]
        a, b, c = (retained[path] for path in ("a.0", "a.2", "b.0"))
        hidden = F.relu(F.linear(x, source.a[0].weight[a], source.a[0].bias[a]))
        hidden = F.linear(hidden, source.a[2].weight[b][:, a], source.a[2].bias[b])
        expected = F.linear(hidden, source.a[3].weight[:, b], source.a[3].bias)
        hidden = F.relu(F.linear(x, source.b[0].weight[c], source.b[0].bias[c]))
        expected = expected + F.linear(hidden, source.b[2].weight[:, c], source.b[2].bias)
        compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
        torch.testing.assert_close(compact(x), expected, rtol=1e-12, atol=1e-12)
        compact(x).square().sum().backward()
        assert all(parameter.grad is not None for parameter in compact.parameters())
    assert plans[0] == plans[1]
    if strategy_type is DynamicGreedy:
        assert partial
