"""Conditional scoring uses original coordinates and complete joint effects."""

import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.contracts import Diagnostic
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    PlanningContext,
    PlanningError,
    Pruner,
)
from torch_kirigami.pruning.metrics import GroupMagnitude, Magnitude, WeightTaylor
from torch_kirigami.pruning.types import StrategyResult


def context_for(graph, candidates):
    return PlanningContext(
        graph,
        graph.operations(),
        tuple(candidates),
        ChannelRatio(0.5),
        tuple(dict.fromkeys(c.axis for c in candidates if c.axis is not None)),
        (),
    )


@pytest.mark.parametrize(
    "metric", [Magnitude(1), Magnitude(2), WeightTaylor(), WeightTaylor("joint_abs")]
)
def test_conditional_metric_counts_new_union_once(metric, execution_device):
    model = nn.Linear(4, 5, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(-10, 10, dtype=torch.float64).reshape(5, 4))
    model.weight.grad = torch.linspace(-2, 2, 20, dtype=torch.float64).reshape(5, 4)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4, dtype=torch.float64),))
    ref = graph.parameter("weight")
    candidate = Candidate("joint", (ref.axis(0).select([1]), ref.axis(1).select([2])))
    context = context_for(graph, (candidate,))
    selected = context.impact((ref.axis(0).select([0]), ref.axis(1).select([3])))

    score = metric.score(context, (candidate,), selected=selected)[0]
    mask = torch.zeros_like(model.weight, dtype=torch.bool)
    mask[1] = True
    mask[:, 2] = True
    mask[0] = False
    mask[:, 3] = False
    if isinstance(metric, Magnitude):
        expected = torch.linalg.vector_norm(model.weight[mask], ord=metric.p)
    else:
        terms = (model.weight * model.weight.grad)[mask]
        expected = terms.abs().sum() if metric.mode == "elementwise_abs" else terms.sum().abs()
    assert score == pytest.approx(expected.item(), rel=1e-12)


@pytest.mark.parametrize("metric", [Magnitude(1), Magnitude(2), WeightTaylor()])
def test_conditional_metric_includes_joint_only_upstream_effect(metric, execution_device):
    model = nn.Sequential(
        nn.Conv2d(1, 2, 1, bias=False), nn.Flatten(), nn.Linear(4, 2, bias=False)
    ).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([7.0, 11.0]).reshape(2, 1, 1, 1))
        model[2].weight.copy_(torch.arange(1, 9, dtype=torch.float64).reshape(2, 4))
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 1, 1, 2, dtype=torch.float64),))
    axis = graph.parameter("2.weight").axis(1)
    candidate = Candidate("second_spatial_position", (axis.select([1]),), axis)
    context = context_for(graph, (candidate,))
    selected = context.impact((axis.select([0]),))
    assert selected.complete
    assert not selected.selection(graph.parameter("0.weight"))
    assert not context.impact(candidate.remove).selection(graph.parameter("0.weight"))

    values = torch.cat((model[0].weight[0].flatten(), model[2].weight[:, 1]))
    expected = (
        torch.linalg.vector_norm(values, ord=metric.p)
        if isinstance(metric, Magnitude)
        else (2 * values).abs().sum()
    )
    assert metric.score(context, (candidate,), selected=selected)[0] == pytest.approx(
        expected.item()
    )


@pytest.mark.parametrize("p", [1, 2])
def test_group_magnitude_formula_includes_dependencies_and_affine_weights(p, execution_device):
    model = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).double().eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(1, 9, dtype=torch.float64).reshape(4, 2))
        model[1].weight.copy_(torch.tensor([1.0, 2.0, 4.0, 8.0]))
        model[2].weight.copy_(torch.arange(2, 10, dtype=torch.float64).reshape(2, 4))
        for layer in model:
            layer.bias.fill_(1000)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(4))
    context = context_for(graph, candidates)
    scores = GroupMagnitude(p).score(context, candidates, selected=context.impact(()))
    energy = (
        model[0].weight.abs().pow(p).sum(1)
        + model[1].weight.abs().pow(p)
        + model[2].weight.abs().pow(p).sum(0)
    )
    assert scores == pytest.approx((energy / energy.mean()).tolist())


def test_group_normalization_independent_of_batch_subset_and_block_wrapping(execution_device):
    model = nn.Linear(2, 4, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, 9, dtype=torch.float64).reshape(4, 2))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    axis = graph.parameter("weight").axis(0)
    first = Candidate("first", (axis.select([0]),), axis)
    second = Candidate("second", (axis.select([1]),), axis)
    block = Candidate("block", (axis.select([0, 1]),), axis)
    context = context_for(graph, (first, second, block))
    selected = context.impact(())
    metric = GroupMagnitude()
    joint = metric.score(context, (first, second, block), selected=selected)
    assert metric.score(context, (block, first), selected=selected) == pytest.approx(
        [joint[2], joint[0]]
    )
    assert metric.score(context, (second,), selected=selected) == pytest.approx([joint[1]])
    restricted = context_for(graph, (block,))
    assert metric.score(restricted, (block,), selected=restricted.impact(())) == pytest.approx(
        [joint[2]]
    )
    energy = model.weight.square().sum(1)
    assert joint == pytest.approx(
        [
            (energy[0] / energy.mean()).item(),
            (energy[1] / energy.mean()).item(),
            ((energy[0] + energy[1]) / energy.mean()).item(),
        ]
    )


def test_group_dynamic_normalization_uses_surviving_rows_and_columns(execution_device):
    model = nn.Sequential(nn.Linear(2, 4, bias=False), nn.Linear(4, 3, bias=False)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(1, 9, dtype=torch.float64).reshape(4, 2))
        model[1].weight.copy_(torch.arange(2, 14, dtype=torch.float64).reshape(3, 4))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)
    candidate = Candidate("channel2", (axis.select([2]),), axis)
    context = context_for(graph, (candidate,))
    selected = context.impact((axis.select([1]), graph.parameter("1.weight").axis(0).select([0])))
    energy = model[0].weight.square().sum(1) + model[1].weight[1:].square().sum(0)
    expected = energy[2] / energy[[0, 2, 3]].mean()
    score = GroupMagnitude().score(context, (candidate,), selected=selected)[0]
    assert score == pytest.approx(expected.item())


class AliasedBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
        self.weight_alias = self.linear.bias

    def forward(self, x):
        return self.linear(x)


def test_group_excludes_bias_with_misleading_alias(execution_device):
    model = AliasedBias().double()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
        model.linear.bias.copy_(torch.tensor([1e6, 1e3, 1.0]))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    axis = graph.parameter("linear.weight").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    context = context_for(graph, (candidate,))
    energy = model.linear.weight.square().sum(1)
    assert GroupMagnitude().score(
        context, (candidate,), selected=context.impact(())
    ) == pytest.approx([(energy[0] / energy.mean()).item()])


class FunctionalLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.arange(1.0, 9.0).reshape(4, 2))
        self.offset = nn.Parameter(torch.full((4,), 1e6))

    def forward(self, x):
        return F.linear(x, weight=self.matrix, bias=self.offset)


def test_group_functional_roles_do_not_require_weight_attribute_names(execution_device):
    model = FunctionalLinear().double()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    axis = graph.parameter("matrix").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    context = context_for(graph, (candidate,))
    energy = model.matrix.square().sum(1)
    assert GroupMagnitude().score(
        context, (candidate,), selected=context.impact(())
    ) == pytest.approx([(energy[0] / energy.mean()).item()])


@pytest.mark.parametrize("invalid", ["missing_axis", "two_axes", "partial_slice"])
def test_group_ambiguous_domain_is_explicit_error(invalid):
    model = nn.Linear(3, 4, bias=False)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    ref = graph.parameter("weight")
    axis = ref.axis(0)
    if invalid == "missing_axis":
        candidate = Candidate("bad", (axis.select([0]),))
    elif invalid == "two_axes":
        candidate = Candidate("bad", (axis.select([0]), ref.axis(1).select([0])), axis)
    else:
        partial = axis.select([0]).subtract(ref.axis(1).select([1]))
        candidate = Candidate("bad", (partial,), axis)
    context = context_for(graph, (candidate,))
    with pytest.raises(PlanningError, match="declared axis"):
        GroupMagnitude().score(context, (candidate,), selected=context.impact(()))


@pytest.mark.parametrize("scale", [0.0, 1e-200, 1e200])
def test_group_normalization_is_finite_at_extreme_scales(scale, execution_device):
    model = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64) * scale)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    context = context_for(graph, (candidate,))
    score = GroupMagnitude().score(context, (candidate,), selected=context.impact(()))[0]
    assert math.isfinite(score)
    assert score == pytest.approx(0.0 if scale == 0 else 1 / 3)


def test_group_normalization_cache_observes_weight_updates():
    model = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(1)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    context = context_for(graph, (candidate,))
    selected = context.impact(())
    metric = GroupMagnitude()
    assert metric.score(context, (candidate,), selected=selected) == pytest.approx([1])
    with torch.no_grad():
        model.weight[0].mul_(2)
    assert metric.score(context, (candidate,), selected=selected) == pytest.approx([1.6])


def test_group_tied_weight_rows_columns_and_repeated_calls_count_once(execution_device):
    linear = nn.Linear(3, 3, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.arange(1, 10, dtype=torch.float64).reshape(3, 3))
    model = nn.Sequential(linear, linear)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)
    assert axis.tensor == graph.parameter("1.weight")
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(3))
    context = context_for(graph, candidates)
    energy = linear.weight.square()
    union_energy = energy.sum(0) + energy.sum(1) - energy.diag()
    assert GroupMagnitude().score(
        context, candidates, selected=context.impact(())
    ) == pytest.approx((union_energy / union_energy.mean()).tolist())


def test_group_custom_strategy_public_plan_apply_keeps_model_unchanged_during_scores(
    execution_device,
):
    model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(1, 9, dtype=torch.float64).reshape(4, 2))
        model[1].weight.copy_(torch.arange(2, 10, dtype=torch.float64).reshape(2, 4))
    x = torch.randn(3, 2, dtype=torch.float64)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in (1, 2))
    energy = model[0].weight.square().sum(1) + model[1].weight.square().sum(0)

    class ConditionalStrategy:
        def select(self, context):
            selected = context.impact(candidates[0].remove)
            scores = context.score(GroupMagnitude(), (candidates[1],), selected=selected)
            assert scores == pytest.approx([(energy[2] / energy[[0, 2, 3]].mean()).item()])
            return StrategyResult(("1", "2"))

    pruner = Pruner(model, graph=graph)
    plan = pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=ChannelRatio(0.5),
        strategy=ConditionalStrategy(),
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    expected = F.linear(
        F.linear(x, original["0.weight"][[0, 3]], original["0.bias"][[0, 3]]),
        original["1.weight"][:, [0, 3]],
        original["1.bias"],
    )
    compact, _ = pruner.apply(plan)
    assert compact is model
    torch.testing.assert_close(compact(x), expected)


def test_group_inference_parameters_do_not_use_stale_normalization_cache():
    with torch.inference_mode():
        model = nn.Linear(2, 2, bias=False).double()
        model.weight.fill_(1)
        graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
        axis = graph.parameter("weight").axis(0)
        candidate = Candidate("first", (axis.select([0]),), axis)
        context = context_for(graph, (candidate,))
        selected = context.impact(())
        metric = GroupMagnitude()
        assert metric.score(context, (candidate,), selected=selected) == pytest.approx([1])
        model.weight[0].mul_(2)
        assert metric.score(context, (candidate,), selected=selected) == pytest.approx([1.6])


@pytest.mark.parametrize("kind", ["magnitude", "taylor"])
def test_parameter_filter_applies_to_every_candidate_and_parameter(kind):
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2)).double()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(3))
    seen = []

    def only_downstream(ref, _parameter):
        seen.append(ref)
        return ref == graph.parameter("1.weight")

    metric = (
        Magnitude(1, parameter_filter=only_downstream)
        if kind == "magnitude"
        else WeightTaylor(parameter_filter=only_downstream)
    )
    context = context_for(graph, candidates)
    scores = metric.score(context, candidates, selected=context.impact(()))
    assert scores == pytest.approx(model[1].weight.abs().sum(0).tolist())
    assert seen.count(graph.parameter("1.weight")) == 3
    assert seen.count(graph.parameter("0.weight")) == 3
    assert seen.count(graph.parameter("0.bias")) == 3


def test_group_cache_cannot_hide_incomplete_domain_in_another_context():
    model = nn.Linear(2, 2, bias=False)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2),))
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    context = context_for(graph, (candidate,))
    metric = GroupMagnitude()
    metric.score(context, (candidate,), selected=context.impact(()))

    class IncompleteSecondPosition:
        refs = (axis.tensor,)

        def check(self, selections):
            selection = selections.get(axis.tensor.id)
            if selection and 1 in selection.fully_selected_indices(axis.dim):
                return Diagnostic("incomplete_test", "Unknown second position", complete=False)
            return None

    constrained = PlanningContext(
        graph,
        graph.operations(),
        (candidate,),
        ChannelRatio(0.5),
        (axis,),
        (IncompleteSecondPosition(),),
    )
    with pytest.raises(PlanningError, match="Unknown second position"):
        metric.score(constrained, (candidate,), selected=constrained.impact(()))


def test_group_singleton_cache_avoids_duplicate_propagation(monkeypatch):
    model = nn.Linear(2, 64, bias=False)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(64))
    context = context_for(graph, candidates)
    selected = context.impact(())
    original = DependencyGraph.propagate
    calls = []

    def counted_propagation(self, *args, **kwargs):
        calls.append(kwargs.get("remove"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DependencyGraph, "propagate", counted_propagation)
    metric = GroupMagnitude()
    first = metric.score(context, candidates[:4], selected=selected)
    assert len(calls) == 64  # Full original domain once, not 64 + 4 requested candidates.
    again = metric.score(context, (candidates[20], candidates[0]), selected=selected)
    assert len(calls) == 64
    assert again[1] == first[0]

    committed = context.impact(candidates[0].remove)
    scores = metric.score(context, candidates[:2], selected=committed)
    energy = model.weight.square().sum(1)
    assert scores == pytest.approx([0.0, (energy[1] / energy[1:].mean()).item()])


def test_group_cached_singletons_do_not_replace_joint_block_score(execution_device):
    linear = nn.Linear(3, 3, bias=False).double()
    with torch.no_grad():
        linear.weight.copy_(torch.arange(1, 10, dtype=torch.float64).reshape(3, 3))
    model = nn.Sequential(linear, linear)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)
    singles = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in (0, 1))
    block = Candidate("block", (axis.select([0, 1]),), axis)
    context = context_for(graph, (*singles, block))
    selected = context.impact(())
    metric = GroupMagnitude()
    scores = metric.score(context, singles, selected=selected)
    joint_score = metric.score(context, (block,), selected=selected)[0]
    energy = linear.weight.square()
    singleton_energies = energy.sum(0) + energy.sum(1) - energy.diag()
    joint_energy = energy.sum() - energy[2, 2]
    assert joint_score == pytest.approx((joint_energy / singleton_energies.mean()).item())
    assert joint_score != pytest.approx(sum(scores))
