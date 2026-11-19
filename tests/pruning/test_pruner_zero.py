"""Pruner-Zero scoring and lifecycle through the public pruning API."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    DynamicGreedy,
    Greedy,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PrunerZero,
)


def test_pruner_zero_scores_new_coordinates_with_full_tensor_scaling(execution_device):
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 3), nn.Linear(3, 2)).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(2)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    first, middle, last = (graph.parameter(f"{index}.weight") for index in range(3))
    row0 = first.axis(0).select([0])
    row1 = middle.axis(0).select([1])
    remove_first = Candidate("first", (row0,))
    remove_second = Candidate("second", (row1,))
    joint = Candidate("joint", (row0, row1))
    candidates = (remove_first, remove_second, joint)
    context = PlanningContext(graph, candidates, ParameterBudget(0), (), ())
    energy = {
        first: torch.arange(6, dtype=torch.float64).reshape(3, 2),
        middle: torch.arange(9, dtype=torch.float64).reshape(3, 3),
        last: torch.ones((2, 3), dtype=torch.float64),
    }
    for value in energy.values():
        value.requires_grad_()

    def unexpected_saved_tensor(tensor):
        pytest.fail(f"Pruner-Zero saved a tensor for backward: {tensor.shape}")

    with (
        torch.enable_grad(),
        torch.autograd.graph.saved_tensors_hooks(unexpected_saved_tensor, lambda tensor: tensor),
    ):
        metric = PrunerZero(graph, energy)
        scores = context.score(metric, candidates)
        assert torch.is_grad_enabled()
    # First removal contributes first-weight row 0 and middle-weight column 0.
    # The second contributes middle-weight row 1. Its intersection with column
    # 0 was already counted, and the last weight has constant statistics.
    assert scores == pytest.approx((4 / 5 + 4.5, 6.0, 4 / 5 + 9))
    accepted = context.impact(remove_first.remove)
    assert context.score(metric, candidates, accepted_impact=accepted) == pytest.approx(
        (0.0, 4.5, 4.5)
    )
    with pytest.raises(PlanningError, match="no statistics"):
        context.score(PrunerZero(graph, {first: energy[first]}), (remove_first,))


@pytest.mark.parametrize("strategy", [Greedy, DynamicGreedy])
def test_pruner_zero_uses_supplied_statistics_through_plan_and_apply(strategy, execution_device):
    model = nn.Sequential(nn.Linear(2, 3, bias=False), nn.Linear(3, 2, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(2)
        model[1].weight.fill_(3)
    x = torch.randn(2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    left, right = (graph.parameter(f"{index}.weight") for index in range(2))
    energy = {
        left: torch.tensor([[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]),
        right: torch.tensor([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
    }
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    with pytest.raises(PlanningError, match="no statistics"):
        pruner.plan(space, budget=ParameterBudget(8), strategy=strategy(PrunerZero(graph, {})))
    assert model[0].out_features == model[1].in_features == 3
    assert all(parameter.grad is None for parameter in model.parameters())
    expected = F.linear(F.linear(x, model[0].weight[[0, 2]]), model[1].weight[:, [0, 2]])
    plan = pruner.plan(
        space, budget=ParameterBudget(8), strategy=strategy(PrunerZero(graph, energy))
    )
    assert tuple(plan.analysis.selection(left).fully_selected_indices(0)) == (1,)
    pruner.apply(plan)
    assert model[0].out_features == model[1].in_features == 2
    torch.testing.assert_close(model(x), expected)


def test_pruner_zero_keeps_representable_importance_at_extreme_scales(execution_device):
    model = nn.Linear(2, 3, bias=False).eval()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1e20, 1e20], [1.0, 1.0], [1e20, 1e20]]))
    x = torch.full((2, 2), 1e-10)
    graph = DependencyGraph.build(model, args=(x,))
    ref = graph.parameter("weight")
    pruner = Pruner(model, graph=graph, preserve_io=False)
    space = pruner.discover_candidates()
    budget = ParameterBudget(4)
    context = PlanningContext(graph, space.candidates, budget, space.channel_axes, ())

    constant = PrunerZero(graph, {ref: torch.ones_like(model.weight)})
    assert context.score(constant, space.candidates) == (0.0, 0.0, 0.0)
    energy = torch.tensor([[0.0, 0.0], [1e20, 1e20], [1e-30, 1e-30]])
    metric = PrunerZero(graph, {ref: energy})
    reference = (energy.double() / energy.double().max()) * model.weight.detach().double().square()
    assert context.score(metric, space.candidates) == pytest.approx(
        reference.sum(1).tolist(), rel=1e-6, abs=0
    )
    expected = F.linear(x, model.weight[1:])
    plan = pruner.plan(space, budget=budget, strategy=Greedy(metric))
    assert tuple(plan.analysis.selection(ref).fully_selected_indices(0)) == (0,)
    pruner.apply(plan)
    torch.testing.assert_close(model(x), expected)
