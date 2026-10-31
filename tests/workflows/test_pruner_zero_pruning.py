"""Fixed formula calibration and executable CE-only recovery without downloads."""

import copy
import json
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import pruner_zero_pruning as workflow

from tests.support.torchvision_models import tiny_vit, workflow_resnet
from torch_kirigami.pruning import load_checkpoint


def test_calibration_matches_dataset_mean_with_unequal_batches_and_restores_state():
    model = nn.Sequential(
        nn.Linear(3, 5), nn.BatchNorm1d(5), nn.Dropout(0.3), nn.Linear(5, 2)
    ).double()
    model[0].eval()
    modes = tuple(m.training for m in model.modules())
    dense = copy.deepcopy(model).eval()
    x, labels = torch.randn(5, 3, dtype=torch.float64), torch.tensor([0, 1, 0, 1, 1])
    F.cross_entropy(dense(x), labels).backward()
    state = copy.deepcopy(model.state_dict())
    report = workflow.collect_task_gradients(
        model, [(x[:2], labels[:2]), (x[2:], labels[2:])], "cpu", 2
    )
    assert report == {"samples": 5, "batches": 2}
    for parameter, expected in zip(model.parameters(), dense.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected.grad)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key])
    assert tuple(m.training for m in model.modules()) == modes
    with pytest.raises(ValueError, match="batches"):
        workflow.collect_task_gradients(model, [(x, labels)], "cpu", 2)
    assert all(p.grad is None for p in model.parameters())
    assert tuple(m.training for m in model.modules()) == modes


@pytest.mark.parametrize("name", ["resnet18", "vit_b_16"])
@pytest.mark.parametrize("epochs", [0, 1])
def test_entry_prunes_recovers_with_ce_and_roundtrips(
    name, epochs, monkeypatch, tmp_path, execution_device
):
    def builder(*, weights):
        return workflow_resnet() if name == "resnet18" else tiny_vit(224)

    selected_weights = workflow.MODELS[name][1]
    monkeypatch.setitem(workflow.MODELS, name, (builder, selected_weights))
    data = TensorDataset(
        torch.randn(3, 3, 224, 224, device="cpu"), torch.tensor([0, 1, 2], device="cpu")
    )
    monkeypatch.setattr(workflow, "load_images", lambda *a, **kw: (data, data, {"fixture": True}))
    stages = []

    def measurement(model, example, options):
        stages.append(sum(p.numel() for p in model.parameters()))
        return {"#Params": stages[-1]}

    monkeypatch.setattr(workflow, "measure_model", measurement)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pruner_zero_pruning.py",
            "--model",
            name,
            "--device",
            execution_device,
            "--train_batch_size",
            "3",
            "--val_batch_size",
            "3",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--calibration_batches",
            "1",
            "--granularity",
            "1",
            "--pruning_ratio",
            "0.01",
            "--finetune_epochs",
            str(epochs),
            "--output",
            str(tmp_path),
        ],
    )
    workflow.main()
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert [r["stage"] for r in result["stages"]] == ["pretrained", "pruned"] + (
        ["finetuned"] if epochs else []
    )
    assert result["stages"][1]["target_met"]
    assert stages[1] < stages[0]
    compact = load_checkpoint(
        workflow.make_model(name, pretrained=False),
        tmp_path / "model.pt",
        map_location=execution_device,
    ).eval()
    pruned = load_checkpoint(
        workflow.make_model(name, pretrained=False),
        tmp_path / "pruned.pt",
        map_location=execution_device,
    ).eval()
    if epochs:
        assert any(
            not torch.equal(p, q)
            for p, q in zip(compact.parameters(), pruned.parameters(), strict=True)
        )
    assert compact(data.tensors[0].to(execution_device)).shape[0] == 3


@pytest.mark.parametrize(
    "args",
    [
        ["--calibration_batches", "-1"],
        ["--pruning_ratio", "nan"],
        ["--lr", "0"],
        ["--finetune_epochs", "-1"],
        ["--tree_path", "custom.json"],
    ],
)
def test_cli_rejects_invalid_and_out_of_scope_options(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pruner_zero_pruning.py", "--device", "cpu", *args])
    with pytest.raises(SystemExit):
        workflow.parse_args()
