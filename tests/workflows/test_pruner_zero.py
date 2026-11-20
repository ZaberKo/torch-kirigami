"""Numerical coverage for the Pruner-Zero vision adaptation."""

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Candidate, ParameterBudget, PlanningContext

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

from prune_finetune import calibrate_pruner_zero


def test_pruner_zero_squares_each_batch_gradient_before_accumulating(execution_device):
    model = nn.Linear(2, 2, bias=False).double().train()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    model.weight.grad = torch.ones_like(model.weight)
    original_gradient = model.weight.grad
    x = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(2))
    context = PlanningContext(graph, candidates, ParameterBudget(2), (axis,), ())

    metric = calibrate_pruner_zero(
        model, graph, [(x, torch.tensor([0])), (x, torch.tensor([1]))], execution_device, 0
    )
    # Equal logits give opposite +/-0.5 gradients on column zero and zero on
    # column one. Summing before squaring would give zero on both columns.
    assert context.score(metric, candidates) == pytest.approx((1.0, 1.0))
    assert model.training and model.weight.grad is original_gradient
    torch.testing.assert_close(model.weight, torch.tensor([[1.0, 2.0], [1.0, 2.0]]).double())
    torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))
    with pytest.raises(ValueError, match="empty loader"):
        calibrate_pruner_zero(model, graph, [], execution_device, 1)
    assert model.training and model.weight.grad is original_gradient


def test_pruner_zero_minmax_scales_gradient_norms_and_ranks_columns(execution_device):
    model = nn.Linear(4, 2, bias=False).to(device=execution_device, dtype=torch.float64).eval()
    rows = torch.tensor([[1.0, 2.0, 1.0, 1.0]] * 2, device=execution_device, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(rows)
    samples = [[0.0, 1.0, 3.0, 3.0], [0.0, 0.0, 0.0, 4.0]]
    inputs = torch.tensor(samples, dtype=torch.float64, device=execution_device)
    graph = DependencyGraph.build(model, args=(inputs[:1],))
    axis = graph.parameter("weight").axis(1)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in (1, 2))
    context = PlanningContext(graph, candidates, ParameterBudget(8), (axis,), ())
    loader = [(inputs[i : i + 1], torch.tensor([i])) for i in range(2)]

    # The logits are equal, so each batch gradient is half the input with
    # opposite signs on the two rows. Dividing by the maximum norm of 2.5
    # gives column statistics 0.2 and 0.6. Scaling squared norms reverses their rank.
    metric = calibrate_pruner_zero(model, graph, loader, execution_device, 2)
    assert context.score(metric, candidates) == pytest.approx([2 * 4 * 0.2, 2 * 0.6])
