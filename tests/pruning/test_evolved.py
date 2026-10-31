"""Independent fixed-formula, conditional scoring and public lifecycle checks."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    EvolvedMetric,
    Greedy,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


def setup_linear():
    model = nn.Linear(4, 5).double().eval()
    with torch.no_grad():
        model.weight.copy_(torch.arange(-10, 10, dtype=torch.float64).reshape(5, 4))
    model.weight.grad = torch.linspace(-2, 3, 20, dtype=torch.float64).reshape(5, 4)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4, dtype=torch.float64),))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(5))
    context = PlanningContext(graph, candidates, ChannelRatio(0.2), (axis,), ())
    return model, graph, candidates, context


def reference(weight):
    g = weight.grad.abs()
    return weight.detach().square() * ((g - g.min()) / (g.max() - g.min()))


def test_formula_uses_full_tensor_minmax_and_is_batch_independent(execution_device):
    model, _graph, candidates, context = setup_linear()
    metric = EvolvedMetric()
    empty = context.impact(())
    expected = reference(model.weight).sum(1).tolist()
    assert metric.score(context, candidates, accepted_impact=empty) == pytest.approx(expected)
    for candidate, value in zip(candidates, expected, strict=True):
        assert metric.score(context, (candidate,), accepted_impact=empty) == pytest.approx([value])
    assert model.bias.grad is None  # Bias excluded deliberately, not treated as missing data.


def test_conditional_intersections_deduplicate_aliases_and_preserve_normalization():
    model, graph, _candidates, _context = setup_linear()
    model.register_parameter("alias", model.weight)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4, dtype=torch.float64),))
    ref = graph.parameter("weight")
    candidate = Candidate("joint", (ref.axis(0).select([1]), ref.axis(1).select([2])))
    context = PlanningContext(graph, (candidate,), ChannelRatio(0.2), (ref.axis(0),), ())
    accepted = context.impact((ref.axis(0).select([0]), ref.axis(1).select([3])))
    mask = torch.zeros_like(model.weight, dtype=torch.bool)
    mask[1] = True
    mask[:, 2] = True
    mask[0] = False
    mask[:, 3] = False
    metric = EvolvedMetric()
    score = metric.score(context, (candidate,), accepted_impact=accepted)[0]
    assert score == pytest.approx(float(reference(model.weight)[mask].sum()))
    covered = context.impact((*accepted.requested, *candidate.remove))
    assert metric.score(context, (candidate,), accepted_impact=covered) == [0.0]


def test_cache_invalidates_on_gradient_replacement_and_weight_mutation():
    model, _graph, candidates, context = setup_linear()
    metric = EvolvedMetric()
    empty = context.impact(())
    metric.score(context, candidates, accepted_impact=empty)
    model.weight.grad = model.weight.grad.flip(0)
    assert metric.score(context, candidates, accepted_impact=empty) == pytest.approx(
        reference(model.weight).sum(1).tolist()
    )
    with torch.no_grad():
        model.weight.mul_(2)
    assert metric.score(context, candidates, accepted_impact=empty) == pytest.approx(
        reference(model.weight).sum(1).tolist()
    )
    model.weight.grad.fill_(3)
    assert metric.score(context, candidates, accepted_impact=empty) == [0.0] * 5


@pytest.mark.parametrize("bad", ["missing", "sparse", "nan"])
def test_invalid_gradients_fail_without_mutation(bad):
    model, _graph, candidates, context = setup_linear()
    original = model.weight.clone()
    if bad == "missing":
        model.weight.grad = None
    elif bad == "sparse":
        model.weight.grad = model.weight.grad.to_sparse()
    else:
        model.weight.grad[0, 0] = float("nan")
    with pytest.raises(PlanningError):
        EvolvedMetric().score(context, candidates, accepted_impact=context.impact(()))
    torch.testing.assert_close(original, model.weight)


def test_plan_apply_recovery_and_checkpoint_lifecycle(tmp_path, execution_device):
    torch.manual_seed(8)
    model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).double().eval()
    dense = copy.deepcopy(model)
    x = torch.randn(5, 3, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 1, 0])
    nn.functional.cross_entropy(model(x), labels).backward()
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan(
        pruner.discover_candidates(targets=("0",)),
        budget=ParameterBudget.from_ratio(model, 0.15),
        strategy=Greedy(EvolvedMetric()),
    )
    removed = tuple(plan.analysis.selection(graph.parameter("0.weight")).fully_selected_indices(0))
    kept = [i for i in range(6) if i not in removed]
    expected = nn.functional.linear(
        nn.functional.gelu(dense[0](x))[:, kept], dense[2].weight[:, kept], dense[2].bias
    )
    pruner.apply(plan)
    torch.testing.assert_close(model(x), expected)
    assert model[0].out_features < 6
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model[2].weight.clone()
    optimizer.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(x), labels).backward()
    optimizer.step()
    assert not torch.equal(model[2].weight, before)
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(dense, tmp_path / "model.pt", map_location=execution_device).eval()
    torch.testing.assert_close(restored(x), model(x))
