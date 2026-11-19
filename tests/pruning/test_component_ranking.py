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
from torch_kirigami.pruning.score_reuse import IndependentRanking
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
    # Row removals on one weight remove parts of the column slices of the other
    # component. Its scores then reduce the remaining parts of those slices.
    # Keeping one position of `a.0` leaves only that component to generic scoring.
    valid = (first, second, independent) if invalidate == "cross_axis" else (second, independent)
    live = tuple(c for c in space.candidates if c.axis in valid and c.key not in result.covered)
    covered = {
        c.key
        for c in space.candidates
        if c.axis in valid and all(not s.subtract(accepted.selection(s.tensor)) for s in c.remove)
    }
    assert result.covered == covered
    assert set(result.scores) == {c.key for c in live}
    assert result.removals is original.removals is ranking.removals
    for candidate in live:
        assert result.removals[candidate.key] is original.removals[candidate.key]
    expected = context.score(GenericMagnitude(p), live, accepted_impact=accepted)
    assert [result.scores[c.key] for c in live] == list(expected)
    ranked, _ = _rank_candidates(
        context, ranking.metric, space.candidates, accepted, space.channel_axes, {}, ranking
    )
    reference, _ = _rank_candidates(
        context, GenericMagnitude(p), space.candidates, accepted, space.channel_axes, {}
    )
    assert ranked == reference  # Invalid components must remain in the generic ranking.
    with pytest.raises(TypeError):
        result.removals[live[0].key][independent] = None


class Bottleneck(nn.Module):
    """Convolutions whose middle weight joins two components on different axes."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        hidden = F.relu(self.bn1(self.conv1(x)))
        return x + self.conv3(F.relu(self.bn2(self.conv2(hidden))))


@pytest.mark.parametrize("p", [1, 2])
def test_bottleneck_dynamic_plan_scores_shared_weights_conditionally(p, monkeypatch):
    """Both components of `conv2.weight` keep exact scores after either one shrinks."""
    torch.manual_seed(11)
    source = Bottleneck().double().eval()
    with torch.no_grad():
        for module in (source.bn1, source.bn2):
            module.running_mean.uniform_(-1, 1)
            module.running_var.uniform_(0.5, 2)
            module.weight.uniform_(0.5, 2)
    x = torch.randn(2, 4, 5, 5, dtype=torch.float64)
    original_score = IndependentRanking.score
    states = []

    def observe(ranking, accepted):
        result = original_score(ranking, accepted)
        if result is not None:
            states.append(len(ranking.conditional))
        return result

    monkeypatch.setattr(IndependentRanking, "score", observe)
    plans = []
    for metric in (GroupMagnitude(p), GenericMagnitude(p)):
        model = copy.deepcopy(source)
        graph = DependencyGraph.build(model, args=(x,))
        pruner = Pruner(model, graph=graph)
        space = pruner.discover_candidates(targets=("conv1", "conv2"))
        plan = pruner.plan(
            space, budget=ParameterBudget.from_ratio(model, 0.4), strategy=DynamicGreedy(metric)
        )
        plans.append(plan.to_dict())
        compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
        removed = {
            name: set(
                plan.analysis.selection(graph.parameter(f"{name}.weight")).fully_selected_indices(0)
            )
            for name in ("conv1", "conv2")
        }
        first = [i for i in range(8) if i not in removed["conv1"]]
        second = [i for i in range(8) if i not in removed["conv2"]]
        assert first and second and len(first) + len(second) < 16
        hidden = F.relu(
            F.batch_norm(
                F.conv2d(x, source.conv1.weight[first], source.conv1.bias[first]),
                source.bn1.running_mean[first],
                source.bn1.running_var[first],
                source.bn1.weight[first],
                source.bn1.bias[first],
                eps=source.bn1.eps,
            )
        )
        hidden = F.relu(
            F.batch_norm(
                F.conv2d(
                    hidden,
                    source.conv2.weight[second][:, first],
                    source.conv2.bias[second],
                    padding=1,
                ),
                source.bn2.running_mean[second],
                source.bn2.running_var[second],
                source.bn2.weight[second],
                source.bn2.bias[second],
                eps=source.bn2.eps,
            )
        )
        expected = x + F.conv2d(hidden, source.conv3.weight[:, second], source.conv3.bias)
        torch.testing.assert_close(compact(x), expected, rtol=1e-12, atol=1e-12)
    assert plans[0] == plans[1]
    # After the first acceptance, the other component of `conv2.weight` is scored
    # conditionally instead of by ordinary propagation.
    assert any(states)


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
    conditional = []

    def observe(ranking, accepted):
        result = original_score(ranking, accepted)
        if result is not None and ranking.conditional:
            conditional.append(result)
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
        assert conditional
