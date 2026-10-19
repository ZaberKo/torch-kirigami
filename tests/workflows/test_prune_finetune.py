"""Task-gradient calibration and basic pruning preserve model interfaces."""

import copy
import json
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_kirigami.pruning import (
    PlanningError,
    load_checkpoint,
)

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib

import prune_finetune
from torchvision.models.resnet import Bottleneck


def test_taylor_calibration_preserves_bn_state_and_uses_task_only_gradients():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 3)).train()
    reference = copy.deepcopy(model).eval()
    x, labels = torch.randn(3, 4), torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(x, labels), batch_size=3)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    prune_finetune.collect_task_gradients(model, loader, "cpu")
    F.cross_entropy(reference(x), labels).backward()
    assert model.training and model[0].training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("recipe", ["prune_finetune", "iterative_pruning"])
@pytest.mark.parametrize("pruning_ratio", [0.4, 0.6])
def test_basic_workflow_prunes_both_bottleneck_widths_and_preserves_interfaces(
    monkeypatch, tmp_path, recipe, pruning_ratio, execution_device
):
    """Every bottleneck contributes both internal widths, even across rebuilds."""
    module = importlib.import_module(recipe)
    weights = module.MODELS["resnet50"][1]
    models, snapshots = [], []

    def factory(*, weights):
        model = nn.Sequential(
            nn.Conv2d(3, 24, 1),
            Bottleneck(24, 6),
            Bottleneck(24, 6),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, 10),
        )
        models.append(model)
        snapshots.append(
            {name: value.detach().clone() for name, value in model.state_dict().items()}
        )
        return model

    monkeypatch.setitem(module.MODELS, "resnet50", (factory, weights))
    monkeypatch.setattr(module, "measure_model", lambda *args: {})
    monkeypatch.setattr(
        module,
        "load_images",
        lambda *args, **kwargs: (
            None,
            TensorDataset(torch.randn(2, 3, 16, 16), torch.tensor([0, 1])),
            {},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe,
            "--model",
            "resnet50",
            "--device",
            execution_device,
            "--val_workers",
            "0",
            "--pruning_ratio",
            str(pruning_ratio),
            "--granularity",
            "3",
            "--output",
            str(tmp_path),
            *(
                ["--rounds", "2" if pruning_ratio == 0.4 else "1"]
                if recipe == "iterative_pruning"
                else []
            ),
        ],
    )
    with torch.device("cpu"):
        if pruning_ratio == 0.6:
            # Internal widths can remove less than half this model's weights.
            # A larger target must fail, not include stem or residual widths.
            with pytest.raises(PlanningError, match="Parameter target not reached"):
                module.main()
        else:
            module.main()
    if pruning_ratio == 0.6:
        assert len(models) == 1
        for name, expected in snapshots[0].items():
            torch.testing.assert_close(models[0].state_dict()[name].cpu(), expected)
        assert not (tmp_path / "model.pt").exists()
        return
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert {item["parameter"] for item in saved["config"]["candidate_axes"]} == {
        "1.conv1.weight",
        "1.conv2.weight",
        "2.conv1.weight",
        "2.conv2.weight",
    }
    restored = load_checkpoint(
        factory(weights=None), tmp_path / "model.pt", map_location=execution_device
    ).eval()
    # The protected classifier's width is deliberately not divisible by three.
    assert restored[-1].out_features == 10
    assert restored[0].out_channels == restored[-1].in_features == 24
    for block in (restored[1], restored[2]):
        assert block.conv1.in_channels == block.conv3.out_channels == 24
        assert block.conv1.out_channels == block.conv2.in_channels == 3
        assert block.conv2.out_channels == block.conv3.in_channels == 3
    assert sum(p.numel() for p in restored.parameters()) <= saved["config"]["max_params"]
    restored(torch.randn(1, 3, 16, 16, device=execution_device)).square().sum().backward()


@pytest.mark.parametrize("max_batches", [0, 1, 2, 16])
def test_taylor_multibatch_matches_sample_mean_with_unequal_batches(max_batches, execution_device):
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 3)).double().train()
    reference = copy.deepcopy(model).eval()
    x = torch.randn(5, 4, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 0, 2])
    loader = DataLoader(TensorDataset(x, labels), batch_size=3)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    prune_finetune.collect_task_gradients(model, loader, execution_device, max_batches)
    count = 3 if max_batches == 1 else 5
    F.cross_entropy(reference(x[:count]), labels[:count]).backward()
    assert model.training and model[0].training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_taylor_empty_calibration_restores_modes():
    model = nn.Sequential(nn.Linear(4, 3)).train()
    model[0].eval()
    with pytest.raises(ValueError, match="empty loader"):
        prune_finetune.collect_task_gradients(model, [], "cpu", 0)
    assert model.training and not model[0].training
    with pytest.raises(ValueError, match="nonnegative"):
        prune_finetune.collect_task_gradients(model, [], "cpu", -1)
