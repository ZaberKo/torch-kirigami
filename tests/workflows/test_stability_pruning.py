"""Step-based stability training matches an independent sparse-loss reference."""

import copy
import json
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_stability_step_training_matches_loss_reference(schedule, execution_device, capsys):
    module = importlib.import_module("stability_pruning")
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64, device=execution_device)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[0.2, -0.1], [0.3, 0.4]]))
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    batches = [
        (
            torch.tensor([[1.0, -0.5]], dtype=torch.float64).repeat(n, 1),
            torch.zeros(n, dtype=torch.long),
        )
        for n in (3, 1, 2, 1)
    ]
    factors = (
        [0.25, 0.5, 0.75, 1.0]
        if schedule == "linear"
        else [(2 - math.sqrt(2)) / 4, 0.5, (2 + math.sqrt(2)) / 4, 1.0]
    )
    expected_weighted_losses = []
    for (images, labels), factor in zip(batches, factors, strict=True):
        reference_optimizer.zero_grad(set_to_none=True)
        task_loss = F.cross_entropy(
            reference(images.to(execution_device)), labels.to(execution_device)
        )
        penalty = reference.weight.square().sum() / 2
        expected_weighted_losses.append(0.01 * factor * penalty.item())
        (task_loss + 0.01 * factor * penalty).backward()
        reference_optimizer.step()

    iterator = iter(batches)
    for offset in (0, 2):
        module.train_steps(
            model,
            iterator,
            optimizer,
            execution_device,
            2,
            regularizer=lambda: model.weight.square().sum() / 2,
            sparse_loss_weight=0.01,
            sparsity_schedule=schedule,
            completed_steps=offset,
            total_steps=4,
        )
    torch.testing.assert_close(model.weight, reference.weight)
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["sparse_loss_weight"] for row in logs] == pytest.approx([0.005, 0.01])
    assert [row["training_steps"] for row in logs] == [2, 4]
    assert [row["weighted_sparse_loss"] for row in logs] == pytest.approx(
        [
            (3 * expected_weighted_losses[0] + expected_weighted_losses[1]) / 4,
            (2 * expected_weighted_losses[2] + expected_weighted_losses[3]) / 3,
        ]
    )
