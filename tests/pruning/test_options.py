"""Resource settings preserve public planning, restoration, and failure checks."""

import copy
from dataclasses import fields

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph, StaleGraphError, config
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
    WeightTaylor,
)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "32", None])
def test_config_validates_every_field_atomically(invalid):
    original = config.as_dict()
    for field in fields(config):
        with pytest.raises(ValueError, match=field.name):
            config.update(**{field.name: invalid})
        assert config.as_dict() == original
    with pytest.raises(TypeError, match="Unknown"):
        config.update(score_batch_size=1, unknown=2)
    assert config.as_dict() == original


def test_assignment_reset_and_detached_dictionary():
    original = config.as_dict()
    config.score_batch_size = 1
    assert config.score_batch_size == 1
    with pytest.raises(ValueError, match="score_batch_size"):
        config.score_batch_size = 0
    detached = config.as_dict()
    detached["max_parts"] = 1
    assert config.max_parts == original["max_parts"]
    config.reset()
    assert config.as_dict() == original


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_context_cache_bounds_and_stale_failures(capacity, execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).to(execution_device)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3, device=execution_device),))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(3))
    config.update(
        impact_cache_size=capacity, recipe_cache_size=capacity, attribute_cache_size=capacity
    )
    context = PlanningContext(graph, candidates, ChannelRatio(0.5), (axis,), ())
    other = PlanningContext(graph, candidates, ChannelRatio(0.5), (axis,), ())
    for candidate in candidates:
        impact = context.impact(candidate.remove)
        assert context.compile(impact) == other.compile(other.impact(candidate.remove))
        assert len(context._cache) <= capacity
        assert len(context._compiled) <= capacity
        assert len(context._attribute_checks) <= capacity
    assert len(other._cache) <= capacity
    model.eval()
    with pytest.raises(StaleGraphError):
        context.impact(candidates[-1].remove)
    with pytest.raises(StaleGraphError):
        context.compile(impact)


@pytest.mark.parametrize("capacity", [0, 1])
@pytest.mark.parametrize("metric_type", [GroupMagnitude, WeightTaylor])
@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_configured_public_plans_restore_and_apply(
    capacity, metric_type, strategy_type, execution_device, tmp_path
):
    torch.manual_seed(43)
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).to(
        device=execution_device, dtype=torch.float64
    )
    x = torch.randn(2, 3, device=execution_device, dtype=torch.float64)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    reference = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    ordinary = Pruner(model, graph=graph)
    space = ordinary.discover_candidates()
    baseline = ordinary.plan(space, budget=ChannelRatio(0.5), strategy=strategy_type(metric_type()))
    config.update(
        score_batch_size=1,
        impact_cache_size=capacity,
        recipe_cache_size=capacity,
        attribute_cache_size=capacity,
        normalization_cache_size=capacity,
        taylor_cache_size=capacity,
        taylor_cache_positions=8,
        taylor_chunk_elements=7,
    )
    configured = Pruner(model, graph=graph)
    metric = metric_type()
    before = tuple(model.parameters())
    with pytest.raises(PlanningError, match="target not reached"):
        configured.plan(
            space, budget=ParameterBudget(1), strategy=strategy_type(metric, max_trials=1)
        )
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    graph.validate()
    plan = configured.plan(space, budget=ChannelRatio(0.5), strategy=strategy_type(metric))
    assert plan.to_dict() == baseline.to_dict()
    if metric_type is GroupMagnitude:
        assert len(metric._normalizers) <= capacity
    else:
        assert len(metric._axis_statistics) <= capacity
        assert metric._statistic_positions <= 8
    destination = tmp_path / "plan.pt"
    torch.save(plan.to_dict(), destination)
    restored = PruningPlan.from_dict(torch.load(destination, weights_only=True))
    model, _result = configured.apply(restored)
    # Independently reconstruct surviving hidden positions from unchanged rows.
    kept = [
        next(i for i, old in enumerate(reference[0].weight) if torch.equal(row, old))
        for row in model[0].weight
    ]
    expected = F.linear(
        F.relu(F.linear(x, reference[0].weight[kept], reference[0].bias[kept])),
        reference[2].weight[:, kept],
        reference[2].bias,
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(StaleGraphError):
        graph.validate()


@pytest.mark.parametrize("capacity", [0, 1, 2])
def test_metric_caches_follow_global_limits(capacity, execution_device):
    model = nn.Linear(3, 4).to(execution_device)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3, device=execution_device),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(3))
    context = PlanningContext(graph, candidates, ChannelRatio(0.5), (axis,), ())
    config.update(normalization_cache_size=capacity, taylor_cache_size=capacity)
    grouped, taylor = GroupMagnitude(), WeightTaylor()
    for candidate in candidates:
        accepted = context.impact(candidate.remove)
        remaining = tuple(c for c in candidates if c is not candidate)
        scores = context.score(grouped, remaining, accepted_impact=accepted)
        reference = GroupMagnitude()
        assert scores == pytest.approx(
            context.score(reference, remaining, accepted_impact=accepted)
        )
        assert len(grouped._normalizers) <= capacity
        with torch.no_grad():
            model.weight.grad.add_(0.5)
        actual = context.score(taylor, (candidate,))[0]
        i = int(candidate.key)
        expected = (model.weight[i] * model.weight.grad[i]).abs().sum()
        expected += (model.bias[i] * model.bias.grad[i]).abs()
        assert actual == pytest.approx(expected.item(), rel=1e-6)
        assert len(taylor._axis_statistics) <= capacity


@pytest.mark.parametrize("batch_size", [1, 3, 64])
def test_generic_metric_receives_configured_batches(batch_size):
    class ObservedMagnitude(GroupMagnitude):
        def score(self, context, candidates, *, accepted_impact):
            visits.append(len(candidates))
            return super().score(context, candidates, accepted_impact=accepted_impact)

    visits = []
    model = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3),))
    config.update(score_batch_size=batch_size)
    pruner = Pruner(model, graph=graph)
    pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(ObservedMagnitude()),
    )
    assert visits and max(visits) == min(batch_size, 6)


@pytest.mark.parametrize("capacity", [0, 1])
def test_failed_attribute_checks_remain_bounded_and_invalidate(capacity, monkeypatch):
    model = nn.Linear(3, 5)
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3),))
    axis = graph.parameter("weight").axis(0)
    config.update(attribute_cache_size=capacity)
    context = PlanningContext(
        graph,
        (),
        ChannelRatio(0.5),
        (axis,),
        (),
    )
    validate = graph.validate_attribute_changes

    def reject(_changes):
        raise CaptureError("test attribute rejection")

    monkeypatch.setattr(graph, "validate_attribute_changes", reject)
    for count in (1, 2, 3):
        impact = context.impact((axis.select(range(count)),))
        with pytest.raises(PlanningError, match="test attribute rejection"):
            context.compile(impact)
        assert len(context._attribute_checks) <= capacity
        assert not context._compiled
        assert model.out_features == 5
    monkeypatch.setattr(graph, "validate_attribute_changes", validate)
    with torch.no_grad():
        model.weight.add_(1)  # Invalidates cached failures, without structural changes.
    assert context.compile(impact)


def test_existing_owners_follow_configuration_changes(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 2)).to(execution_device)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3, device=execution_device),))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(3))
    context = PlanningContext(graph, candidates, ChannelRatio(0.5), (axis,), ())
    grouped, taylor = GroupMagnitude(), WeightTaylor()
    expected_group = context.score(grouped, candidates)
    expected_taylor = context.score(taylor, candidates)
    impact = context.impact(candidates[0].remove)
    expected_recipes = context.compile(impact)
    assert context._cache and context._compiled
    assert grouped._normalizers and taylor._axis_statistics

    config.update(
        impact_cache_size=0,
        recipe_cache_size=0,
        attribute_cache_size=0,
        normalization_cache_size=0,
        taylor_cache_size=0,
    )
    assert context.score(grouped, candidates) == pytest.approx(expected_group)
    assert context.score(taylor, candidates) == pytest.approx(expected_taylor)
    assert context.compile(context.impact(candidates[0].remove)) == expected_recipes
    assert not context._cache and not context._compiled and not context._attribute_checks
    assert not grouped._normalizers and not taylor._axis_statistics
    assert taylor._statistic_positions == 0
    graph.validate()  # Cache tuning does not change structural assumptions.

    config.update(taylor_cache_size=1, taylor_cache_positions=6, taylor_chunk_elements=6)
    assert context.score(taylor, candidates) == pytest.approx(expected_taylor)
    assert len(taylor._axis_statistics) == 1
    assert taylor._statistic_positions <= 6
    config.update(taylor_cache_size=0)
    assert context.score(taylor, ()) == ()
    assert not taylor._axis_statistics
