"""Real MLP discovery, numerical reconstruction and CE-only executable workflow."""

import copy
import json
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import dvbp_pruning as workflow
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

from tests.support.torchvision_models import tiny_vit
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
    apply_ob2c_updates,
    collect_covariances,
    denoise_covariance,
    load_checkpoint,
    prepare_ob2c_updates,
    save_checkpoint,
)


def miniature(name):
    if name == "vit_b_16":
        return tiny_vit(224)
    return ConvNeXt(
        [
            CNBlockConfig(4, 8, 1),
            CNBlockConfig(8, 12, 1),
            CNBlockConfig(12, 16, 1),
            CNBlockConfig(16, None, 1),
        ],
        num_classes=5,
    )


@pytest.mark.parametrize("name", ["vit_b_16", "convnext_tiny"])
def test_real_pairs_reconstruct_apply_backward_and_restore(
    name, tmp_path, monkeypatch, execution_device
):
    weights = workflow.MLP_MODELS[name][1]
    monkeypatch.setitem(workflow.MLP_MODELS, name, (lambda *, weights: miniature(name), weights))
    model = workflow.make_mlp_model(name, pretrained=False).to(execution_device).eval()
    dense = copy.deepcopy(model)
    x = torch.randn(3, 3, 224, 224)
    graph = DependencyGraph.build(model, args=(x[:1],), operators=workflow.vbp_operators())
    pairs, _ = workflow.discover_mlps(graph)
    assert len(pairs) == (2 if name.startswith("vit") else 4)
    moments = collect_covariances(
        model,
        tuple(pairs.values()),
        [(x, torch.zeros(3))],
        execution_device,
        1,
        dtype=torch.float64,
    )
    statistics = {p: denoise_covariance(s, 0.5) for p, s in moments.items()}
    first, second = next(iter(pairs.items()))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter(f"{first}.weight").axis(0).select([0, 1])])
    updates = prepare_ob2c_updates(graph, plan, pairs, statistics)
    assert updates[f"{second}.weight"].shape[1] == dense.get_submodule(second).in_features - 2
    pruner.apply(plan)
    apply_ob2c_updates(model, updates)
    assert tuple((p, type(m)) for p, m in model.named_modules()) == tuple(
        (p, type(m)) for p, m in dense.named_modules()
    )
    model(x).sum().backward()
    assert model.get_submodule(first).weight.grad is not None
    save_checkpoint(model, tmp_path / "compact.pt")
    restored = load_checkpoint(dense, tmp_path / "compact.pt", map_location=execution_device).eval()
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("name", ["vit_b_16", "convnext_tiny"])
@pytest.mark.parametrize("epochs", [0, 1])
def test_entry_calibrates_compensates_and_recovers(
    name, epochs, monkeypatch, tmp_path, execution_device
):
    weights = workflow.MLP_MODELS[name][1]
    monkeypatch.setitem(workflow.MLP_MODELS, name, (lambda *, weights: miniature(name), weights))
    data = TensorDataset(
        torch.randn(3, 3, 224, 224, device="cpu"), torch.tensor([0, 1, 2], device="cpu")
    )
    monkeypatch.setattr(workflow, "load_images", lambda *a, **kw: (data, data, {"fixture": True}))

    def measurement(model, example, options):
        assert next(model.parameters()).device.type == execution_device
        assert example.device.type == execution_device
        return {"#Params": sum(p.numel() for p in model.parameters())}

    monkeypatch.setattr(workflow, "measure_model", measurement)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dvbp_pruning.py",
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
    # Match normal CLI execution: loaders and their RNG stay on CPU, while
    # --device still moves the model and input batches to the requested GPU.
    with torch.device("cpu"):
        workflow.main()
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert [r["stage"] for r in result["stages"]] == ["pretrained", "pruned"] + (
        ["finetuned"] if epochs else []
    )
    assert all(s["images"] == 3 for s in result["config"]["calibration"].values())
    assert result["stages"][1]["target_met"]
    assert result["stages"][1]["#Params"] < result["stages"][0]["#Params"]
    compact = load_checkpoint(
        workflow.make_mlp_model(name, pretrained=False),
        tmp_path / "model.pt",
        map_location=execution_device,
    ).eval()
    pruned = load_checkpoint(
        workflow.make_mlp_model(name, pretrained=False),
        tmp_path / "pruned.pt",
        map_location=execution_device,
    ).eval()
    if epochs:
        assert any(
            not torch.equal(p, q)
            for p, q in zip(compact.parameters(), pruned.parameters(), strict=True)
        )


@pytest.mark.parametrize(
    "args",
    [
        ["--damping", "nan"],
        ["--score_ratio", "1"],
        ["--ridge_floor", "-1"],
        ["--calibration_batches", "-1"],
        ["--model", "resnet18"],
    ],
)
def test_cli_rejects_invalid_and_unsupported_inputs(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dvbp_pruning.py", "--device", "cpu", *args])
    with pytest.raises(SystemExit):
        workflow.parse_args()
