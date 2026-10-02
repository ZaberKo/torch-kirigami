"""Exact axis statistics preserve Taylor formulas and public pruning behavior."""

import copy
import json
import math

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, IndexSet, Region, Selection, StaleGraphError
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
    WeightTaylor,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.pruning import metrics as metric_module


class GenericTaylor(WeightTaylor):
    """Use the original region reductions, without native statistics reuse."""


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def linear_context(device, dtype=torch.float64):
    model = nn.Linear(4, 5).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        model.weight.copy_(torch.arange(-10, 10, device=device, dtype=dtype).reshape(5, 4))
        model.bias.copy_(torch.arange(-2, 3, device=device, dtype=dtype))
    model.weight.grad = torch.linspace(-2, 2, 20, device=device, dtype=dtype).reshape(5, 4)
    model.bias.grad = torch.tensor([2, -3, 4, -5, 6], device=device, dtype=dtype)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4, device=device, dtype=dtype),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(5))
    context = PlanningContext(graph, candidates, ChannelRatio(0.4), (axis,), ())
    return model, axis, candidates, context


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_axis_scores_match_independent_products_and_batch_order(mode, dtype, execution_device):
    model, _axis, candidates, context = linear_context(execution_device, dtype)
    weights = model.weight.detach().double().cpu().tolist()
    gradients = model.weight.grad.double().cpu().tolist()
    biases = model.bias.detach().double().cpu().tolist()
    bias_gradients = model.bias.grad.double().cpu().tolist()
    expected = []
    for row, gradient, bias, bias_gradient in zip(
        weights, gradients, biases, bias_gradients, strict=True
    ):
        terms = [w * g for w, g in zip(row, gradient, strict=True)] + [bias * bias_gradient]
        expected.append(
            math.fsum(map(abs, terms)) if mode == "elementwise_abs" else abs(math.fsum(terms))
        )
    metric = WeightTaylor(mode)
    assert context.score(metric, candidates) == pytest.approx(expected, rel=1e-12)
    assert context.score(metric, candidates[::-1]) == pytest.approx(expected[::-1], rel=1e-12)
    assert tuple(context.score(metric, (c,))[0] for c in candidates) == pytest.approx(
        expected, rel=1e-12
    )
    assert bool(metric._axis_statistics) == (mode == "elementwise_abs")
    assert all(
        isinstance(value, float)
        for _w, _g, values in metric._axis_statistics.values()
        for value in values
    )


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_conditional_rows_columns_use_union_and_preserve_signed_cancellation(
    mode, execution_device
):
    model, axis, _candidates, context = linear_context(execution_device)
    rows, columns = {1, 3}, {0, 2}
    candidate = Candidate("union", (axis.select(rows), axis.tensor.axis(1).select(columns)))
    accepted = context.impact((axis.select([0]), axis.tensor.axis(1).select([3])))
    terms = []
    weights, gradients = model.weight.detach().cpu().tolist(), model.weight.grad.cpu().tolist()
    for row in range(5):
        for column in range(4):
            if (row in rows or column in columns) and row != 0 and column != 3:
                terms.append(weights[row][column] * gradients[row][column])
    for row in rows:
        terms.append(float(model.bias.detach()[row]) * float(model.bias.grad[row]))
    expected = math.fsum(map(abs, terms)) if mode == "elementwise_abs" else abs(math.fsum(terms))
    metric = WeightTaylor(mode)
    assert context.score(metric, (candidate,), accepted_impact=accepted) == pytest.approx(
        (expected,)
    )
    assert context.score(
        GenericTaylor(mode), (candidate,), accepted_impact=accepted
    ) == pytest.approx((expected,))
    # Conditional regions have two partial axes; no full-axis statistics apply.
    assert all(
        weight() is not model.weight
        for weight, _gradient, _values in metric._axis_statistics.values()
    )


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_statistics_refresh_and_unselected_nonfinite_values(mode, execution_device):
    model, _axis, candidates, context = linear_context(execution_device)
    metric = WeightTaylor(mode)

    def compare():
        assert context.score(metric, (candidates[0],)) == pytest.approx(
            context.score(GenericTaylor(metric.mode), (candidates[0],))
        )

    compare()
    model.weight.grad.mul_(2)
    compare()
    model.weight.grad = -model.weight.grad.clone()
    compare()
    with torch.no_grad():
        model.weight.add_(0.125)
    compare()
    metric.mode = "joint_abs" if mode == "elementwise_abs" else "elementwise_abs"
    compare()
    model.weight.grad[4].fill_(float("nan"))
    compare()  # Finite selected row; another cached position is NaN.
    with pytest.raises(PlanningError, match="nonfinite"):
        context.score(metric, (candidates[4],))


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_column_deletion_invalidates_full_row_statistics_for_conditional_score(
    mode, execution_device
):
    model, axis, candidates, context = linear_context(execution_device)
    metric = WeightTaylor(mode)
    context.score(metric, candidates[:1])  # Populate full-row sums before deleting a column.
    selected = context.impact((axis.tensor.axis(1).select([1]),))
    assert context.score(metric, candidates[:1], accepted_impact=selected) == pytest.approx(
        context.score(GenericTaylor(mode), candidates[:1], accepted_impact=selected)
    )
    assert model.weight.shape == (5, 4)  # Conditional analysis never mutates weights.


def test_signed_regions_preserve_backend_cancellation_and_valid_apply(execution_device):
    model = nn.Linear(3, 3, bias=False).double().to(execution_device).eval()
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[1e308, -1e308, 0]] * 3, dtype=torch.float64, device=execution_device)
        )
    model.weight.grad = torch.ones_like(model.weight)
    graph = DependencyGraph.build(
        model, args=(torch.zeros(1, 3, dtype=torch.float64, device=execution_device),)
    )
    axis = graph.parameter("weight").axis(1)
    candidate = Candidate("cancel", (axis.select([0, 1]),), axis)
    context = PlanningContext(graph, (candidate,), ChannelRatio(2 / 3), (axis,), ())
    generic = GenericTaylor("joint_abs")
    reference = generic.score(context, (candidate,), accepted_impact=context.impact(()))[0]
    if not math.isfinite(reference):
        # CUDA's parallel flattened reduction can itself overflow here. The
        # shortcut must preserve that backend's original failure too.
        for metric in (WeightTaylor("joint_abs"), generic):
            with pytest.raises(PlanningError, match="nonfinite"):
                context.score(metric, (candidate,))
        before = model.weight.detach().clone()
        pruner = Pruner(model, graph=graph, preserve_io=False)
        with pytest.raises(PlanningError, match="nonfinite"):
            pruner.plan(
                CandidateSpace((candidate,), (axis,)),
                budget=ChannelRatio(2 / 3),
                strategy=Greedy(WeightTaylor("joint_abs")),
            )
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        with torch.no_grad():
            model.weight.mul_(0.1)  # Valid alternative that does not overflow on CUDA.
    assert context.score(WeightTaylor("joint_abs"), (candidate,)) == (0.0,)
    assert context.score(generic, (candidate,)) == (0.0,)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    plan = pruner.plan(
        CandidateSpace((candidate,), (axis,)),
        budget=ChannelRatio(2 / 3),
        strategy=Greedy(WeightTaylor("joint_abs")),
    )
    compact, _result = pruner.apply(plan)
    assert compact.in_features == 1
    torch.testing.assert_close(
        compact(torch.ones(2, 1, dtype=torch.float64, device=execution_device)),
        torch.zeros(2, 3, dtype=torch.float64, device=execution_device),
    )


def test_joint_regions_keep_parameter_sum_order(execution_device):
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 3, bias=False).double()
            self.b = nn.Linear(3, 3, bias=False).double()
            self.c = nn.Linear(3, 3, bias=False).double()

        def forward(self, inputs):
            return self.a(inputs), self.b(inputs), self.c(inputs)

    model = Mixed().to(execution_device).eval()
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), (1e308, -1e308, 1e308), strict=True):
            parameter.zero_()
            parameter[0, 0] = value
            parameter.grad = torch.ones_like(parameter)
    graph = DependencyGraph.build(
        model, args=(torch.zeros(1, 3, dtype=torch.float64, device=execution_device),)
    )
    first = graph.parameter("a.weight").axis(0)
    middle = graph.parameter("b.weight")
    last = graph.parameter("c.weight").axis(0)
    candidate = Candidate(
        "mixed",
        (
            first.select([0]),
            Selection(middle, (Region((IndexSet.of([0]), IndexSet.of([0]))),)),
            last.select([0]),
        ),
    )
    context = PlanningContext(graph, (candidate,), ChannelRatio(0), (), ())
    assert context.score(WeightTaylor("joint_abs"), (candidate,)) == (1e308,)
    assert context.score(GenericTaylor("joint_abs"), (candidate,)) == (1e308,)


@pytest.mark.parametrize("single_position", [False, True])
def test_joint_requests_keep_generic_overflow_failure_and_valid_alternative(
    single_position, execution_device
):
    rows = (
        [[-1e308, 0, 0]] * 3 + [[1e308, 0, 0]] * 3
        if single_position
        else [[-1e308, -1e308, 0], [1e308, 1e308, 0]]
    )
    model = nn.Linear(3, len(rows), bias=False).double().to(execution_device).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor(rows, dtype=torch.float64, device=execution_device))
    model.weight.grad = torch.ones_like(model.weight)
    graph = DependencyGraph.build(
        model, args=(torch.zeros(1, 3, dtype=torch.float64, device=execution_device),)
    )
    axis = graph.parameter("weight").axis(1)
    candidate = Candidate("columns", (axis.select([0] if single_position else [0, 1]),), axis)
    budget = ChannelRatio(1 / 3 if single_position else 2 / 3)
    context = PlanningContext(graph, (candidate,), budget, (axis,), ())
    metric = WeightTaylor("joint_abs")
    reference = GenericTaylor("joint_abs").score(
        context, (candidate,), accepted_impact=context.impact(())
    )[0]
    score = metric.score(context, (candidate,), accepted_impact=context.impact(()))[0]
    assert score == reference or (math.isnan(score) and math.isnan(reference))
    assert not metric._axis_statistics  # Signed reductions never use axis statistics.
    pruner = Pruner(model, graph=graph, preserve_io=False)
    parameter, gradient, before = model.weight, model.weight.grad, model.weight.detach().clone()
    if not math.isfinite(reference):
        with pytest.raises(PlanningError, match="nonfinite"):
            pruner.plan(
                CandidateSpace((candidate,), (axis,)),
                budget=budget,
                strategy=Greedy(metric),
            )
        assert model.weight is parameter and model.weight.grad is gradient
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    with torch.no_grad():
        model.weight.mul_(0.1)
    plan = pruner.plan(
        CandidateSpace((candidate,), (axis,)), budget=budget, strategy=Greedy(metric)
    )
    compact, _result = pruner.apply(plan)
    width = 2 if single_position else 1
    assert compact.in_features == width and compact.out_features == len(rows)
    torch.testing.assert_close(
        compact(torch.ones(2, width, dtype=torch.float64, device=execution_device)),
        torch.zeros(2, len(rows), dtype=torch.float64, device=execution_device),
    )


def test_absolute_taylor_near_float64_overflow_retains_region_failure_semantics(execution_device):
    maximum = torch.finfo(torch.float64).max
    ulp = maximum - math.nextafter(maximum, 0)
    model = nn.Linear(3, 6, bias=False).double().to(execution_device).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[:, 0].copy_(
            torch.tensor([maximum] + [ulp / 4] * 5, dtype=torch.float64, device=execution_device)
        )
    model.weight.grad = torch.ones_like(model.weight)
    graph = DependencyGraph.build(
        model, args=(torch.zeros(1, 3, dtype=torch.float64, device=execution_device),)
    )
    axis = graph.parameter("weight").axis(1)
    candidate = Candidate("column", (axis.select([0]),), axis)
    budget = ChannelRatio(1 / 3)
    context = PlanningContext(graph, (candidate,), budget, (axis,), ())
    metric = WeightTaylor()
    generic = GenericTaylor()
    reference = generic.score(context, (candidate,), accepted_impact=context.impact(()))[0]
    score = metric.score(context, (candidate,), accepted_impact=context.impact(()))[0]
    assert score == reference
    assert not metric._axis_statistics
    before = model.weight.detach().clone()
    pruner = Pruner(model, graph=graph, preserve_io=False)
    if not math.isfinite(reference):
        with pytest.raises(PlanningError, match="nonfinite"):
            pruner.plan(
                CandidateSpace((candidate,), (axis,)), budget=budget, strategy=Greedy(metric)
            )
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    with torch.no_grad():
        model.weight.mul_(0.1)
    assert context.score(metric, (candidate,)) == pytest.approx(
        context.score(generic, (candidate,))
    )
    plan = pruner.plan(
        CandidateSpace((candidate,), (axis,)), budget=budget, strategy=Greedy(metric)
    )
    compact, _result = pruner.apply(plan)
    assert compact.in_features == 2
    torch.testing.assert_close(
        compact(torch.ones(2, 2, dtype=torch.float64, device=execution_device)),
        torch.zeros(2, 6, dtype=torch.float64, device=execution_device),
    )


def test_filter_callbacks_are_validated_and_statistics_bound(execution_device):
    model, _axis, candidates, context = linear_context(execution_device)
    visits = []
    metric = WeightTaylor(parameter_filter=lambda ref, _parameter: visits.append(ref) or True)
    context.score(metric, candidates[:2])
    assert len(visits) == 4  # Weight and bias for each candidate, even with cached statistics.
    for _ in range(130):
        model.weight.grad = torch.ones_like(model.weight)
        metric._axis_sums(model.weight, 0)
    assert len(metric._axis_statistics) <= 128
    assert metric._statistic_positions <= 262_144
    assert metric._statistic_positions == sum(
        len(values) for _w, _g, values in metric._axis_statistics.values()
    )


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_axis_statistics_chunk_promotions_and_fall_back_for_oversized_slices(
    mode, execution_device, monkeypatch
):
    model, axis, candidates, context = linear_context(execution_device)
    monkeypatch.setattr(metric_module, "_TAYLOR_STATISTIC_ELEMENTS", 7)
    promoted = []
    original_to = torch.Tensor.to

    def record_to(tensor, *args, **kwargs):
        if args and args[0] is torch.float64:
            promoted.append(tensor.numel())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    metric = WeightTaylor(mode)
    assert context.score(metric, candidates) == pytest.approx(
        context.score(GenericTaylor(mode), candidates)
    )
    assert promoted and max(promoted) <= 7
    column = Candidate("column", (axis.tensor.axis(1).select([2]),), axis.tensor.axis(1))
    assert context.score(metric, (column,)) == pytest.approx(
        context.score(GenericTaylor(mode), (column,))
    )
    metric = WeightTaylor(mode)
    monkeypatch.setattr(metric_module, "_TAYLOR_STATISTIC_ELEMENTS", 2)
    assert context.score(metric, candidates[:1]) == pytest.approx(
        context.score(GenericTaylor(mode), candidates[:1])
    )
    assert all(
        weight() is not model.weight
        for weight, _gradient, _values in metric._axis_statistics.values()
    )


def test_position_capacity_evicts_vectors_and_oversized_axes_use_generic_path(
    execution_device, monkeypatch
):
    model, _axis, candidates, context = linear_context(execution_device)
    monkeypatch.setattr(metric_module, "_TAYLOR_CACHE_POSITIONS", 8)
    metric = WeightTaylor()
    for _ in range(3):
        model.weight.grad = torch.ones_like(model.weight)
        assert metric._axis_sums(model.weight, 0) is not None
        assert metric._statistic_positions == 5
    monkeypatch.setattr(metric_module, "_TAYLOR_CACHE_POSITIONS", 4)
    metric = WeightTaylor()
    assert context.score(metric, candidates) == pytest.approx(
        context.score(GenericTaylor(), candidates)
    )
    assert not metric._axis_statistics


@pytest.mark.parametrize("failure", ["missing", "sparse"])
def test_cached_statistics_do_not_hide_invalid_gradients(failure, execution_device):
    model, _axis, candidates, context = linear_context(execution_device)
    metric = WeightTaylor()
    context.score(metric, candidates[:1])
    model.weight.grad = None if failure == "missing" else model.weight.grad.to_sparse()
    with pytest.raises(PlanningError, match="dense current gradients"):
        context.score(metric, candidates[:1])


@pytest.mark.parametrize("accept", [True, False])
def test_mutating_filter_is_validated_even_when_it_excludes_parameter(accept, execution_device):
    model, _axis, candidates, context = linear_context(execution_device)

    def mutate(_ref, _parameter):
        model.train()
        return accept

    metric = WeightTaylor(parameter_filter=mutate)
    with pytest.raises(StaleGraphError):
        context.score(metric, candidates[:1])
    assert not metric._axis_statistics


@pytest.mark.parametrize("override", ["subclass", "instance", "class"])
def test_score_overrides_do_not_use_native_statistics(override, execution_device, monkeypatch):
    _model, _axis, candidates, context = linear_context(execution_device)
    metric = GenericTaylor() if override == "subclass" else WeightTaylor()
    original = WeightTaylor.score

    def custom_score(self, context, candidates, *, accepted_impact):
        return original(self, context, candidates, accepted_impact=accepted_impact)

    if override == "class":
        monkeypatch.setattr(WeightTaylor, "score", custom_score)
    elif override == "instance":
        monkeypatch.setattr(
            metric,
            "score",
            lambda context, candidates, *, accepted_impact: custom_score(
                metric, context, candidates, accepted_impact=accepted_impact
            ),
        )
    context.score(metric, candidates)
    assert not metric._axis_statistics


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_weight_aliases_are_counted_once(mode, execution_device):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 5, bias=False).double()
            self.b = nn.Linear(3, 5, bias=False).double()
            self.b.weight = self.a.weight
            self.output = nn.Linear(5, 2, bias=False).double()

        def forward(self, inputs):
            return self.output(self.a(inputs) + self.b(inputs))

    model = Shared().to(execution_device).eval()
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    graph = DependencyGraph.build(
        model, args=(torch.randn(2, 3, dtype=torch.float64, device=execution_device),)
    )
    axis = graph.parameter("a.weight").axis(0)
    candidate = Candidate("alias", (axis.select([1]),), axis)
    context = PlanningContext(graph, (candidate,), ChannelRatio(0.2), (axis,), ())
    products = torch.cat(
        (
            (model.a.weight[1] * model.a.weight.grad[1]).detach(),
            (model.output.weight[:, 1] * model.output.weight.grad[:, 1]).detach(),
        )
    )
    expected = float(products.abs().sum() if mode == "elementwise_abs" else products.sum().abs())
    assert context.score(WeightTaylor(mode), (candidate,)) == pytest.approx((expected,))


@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_inference_statistics_are_not_retained(mode, execution_device):
    with torch.inference_mode():
        _model, _axis, candidates, context = linear_context(execution_device)
        metric = WeightTaylor(mode)
        assert context.score(metric, candidates) == pytest.approx(
            context.score(GenericTaylor(mode), candidates)
        )
        assert not metric._axis_statistics


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("mode", ["elementwise_abs", "joint_abs"])
def test_public_plans_compact_models_and_restore_match_generic_statistics(
    strategy_type, mode, execution_device, tmp_path
):
    torch.manual_seed(71)
    dense = (
        nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
        .double()
        .to(execution_device)
        .eval()
    )
    for parameter in dense.parameters():
        parameter.grad = torch.randn_like(parameter)
    inputs = torch.randn(3, 4, dtype=torch.float64, device=execution_device)
    states, outputs, keys = [], [], []
    for metric in (WeightTaylor(mode), GenericTaylor(mode)):
        model = copy.deepcopy(dense)
        for copied, original in zip(model.parameters(), dense.parameters(), strict=True):
            copied.grad = original.grad.clone()
        graph = DependencyGraph.build(model, args=(inputs,))
        pruner = Pruner(model, graph=graph)
        plan = pruner.plan(
            pruner.discover_candidates(targets=("0", "2")),
            budget=ParameterBudget.from_ratio(model, 0.2),
            strategy=strategy_type(metric),
        )
        for parameter, original in zip(model.parameters(), dense.parameters(), strict=True):
            torch.testing.assert_close(parameter, original)
        plan = PruningPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        keys.append(plan.selected)
        compact, _result = pruner.apply(plan)
        states.append({name: value.clone() for name, value in compact.state_dict().items()})
        outputs.append(compact(inputs).detach())
        compact(inputs).sum().backward()
        assert all(parameter.grad is not None for parameter in compact.parameters())
        checkpoint = tmp_path / f"{type(metric).__name__}.pt"
        save_checkpoint(compact, checkpoint)
        restored = load_checkpoint(
            copy.deepcopy(dense), checkpoint, map_location=execution_device
        ).eval()
        torch.testing.assert_close(restored(inputs), outputs[-1])
    assert keys[0] == keys[1]
    assert states[0].keys() == states[1].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
