import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_kirigami.pruning import Pruner, load_checkpoint, save_checkpoint
from torch_kirigami.sparsity import ChannelGate

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib

import imagenet_data as imagenet
import imagenet_models
import workflow_utils
from datasets import ClassLabel, Dataset, Features, Image
from huggingface_hub import HfApi, constants
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image as PILImage
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.vision_transformer import VisionTransformer


def test_imagenet_label_order_and_synonyms():
    categories = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
    names = [f"{category}, synonym {i}" for i, category in enumerate(categories)]
    assert imagenet.label_mapping(names, categories) == tuple(range(1000))
    names[134], names[517], names[639] = "crane", "crane2", "maillot, tank suit"
    assert imagenet.label_mapping(names, categories) == tuple(range(1000))
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="class order"):
        imagenet.label_mapping(names, categories)
    with pytest.raises(ValueError, match="1000"):
        imagenet.label_mapping(names[:10], categories)


@pytest.mark.parametrize("need_train", [False, True])
@pytest.mark.parametrize("use_cache", [False, True])
def test_data_loading_requests_only_needed_splits(tmp_path, monkeypatch, need_train, use_cache):
    weights = ResNet18_Weights.IMAGENET1K_V1
    features = Features({"image": Image(), "label": ClassLabel(names=weights.meta["categories"])})
    folder = tmp_path / "data"
    folder.mkdir()
    for split in ("validation", "train") if need_train else ("validation",):
        labels = [0, 1, 2, 3] if split == "validation" else [4, 5, 6, 7]
        rows = Dataset.from_dict(
            {"image": [PILImage.new("L", (8, 8))] * 4, "label": labels}, features=features
        )
        rows.to_parquet(str(folder / f"{split}-00000.parquet"))
    data_dir = tmp_path
    if use_cache:
        cache = tmp_path / "hub"
        repo = cache / "datasets--ILSVRC--imagenet-1k"
        revision = "a" * 40
        snapshot = repo / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "data").symlink_to(folder, target_is_directory=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(revision)
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
        data_dir = None

    def no_network(*args, **kwargs):
        pytest.fail("Pre-downloaded ImageNet must be resolved without Hub requests")

    monkeypatch.setattr(HfApi, "repo_info", no_network)
    train, validation, metadata = imagenet.load_images(
        weights, data_dir, need_train=need_train, train_samples=2, val_samples=3
    )
    assert len(validation) == 3
    assert validation[0][0].shape == (3, 224, 224)
    assert metadata["validation"]["samples"] == 3
    if need_train:
        samples = list(train)
        assert len(samples) == len(train) == 2
        assert {label for _, label in samples}.isdisjoint({validation[i][1] for i in range(3)})
        assert set(metadata["files"]) == {"validation", "train"}
    else:
        assert train is None and set(metadata["files"]) == {"validation"}
        with pytest.raises(FileNotFoundError, match="train"):
            imagenet.load_images(weights, data_dir, need_train=True)


def test_missing_hf_cache_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))

    def no_network(*args, **kwargs):
        pytest.fail("Missing cached data must not trigger a download")

    monkeypatch.setattr(HfApi, "repo_info", no_network)
    with pytest.raises(LocalEntryNotFoundError):
        imagenet.load_images(ResNet18_Weights.IMAGENET1K_V1)


def test_accuracy_uses_all_logits_weights_partial_batches_and_restores_modes():
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])
    model = nn.Sequential(nn.Identity(), nn.Dropout(0.9)).train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    result = imagenet.evaluate(
        model, DataLoader(TensorDataset(logits, labels), batch_size=2), "cpu"
    )
    assert result["samples"] == 3
    assert result["top1"] == pytest.approx(100 / 3)
    assert result["top5"] == pytest.approx(200 / 3)
    assert result["loss"] == pytest.approx(F.cross_entropy(logits, labels).item())
    assert [m.training for m in model.modules()] == modes
    with pytest.raises(ValueError, match="empty"):
        imagenet.evaluate(model, [], "cpu")
    assert [m.training for m in model.modules()] == modes


def test_torchvision_resnet_compaction_matches_masked_reference_and_checkpoint(
    tmp_path, execution_device
):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    # Compare structural mathematics in float64: changing convolution widths can
    # change cuDNN's TF32 algorithm, which is not an exact FP32 numerical reference.
    model = resnet18(weights=None).to(device=execution_device, dtype=torch.float64).eval()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 64, 64, device=execution_device, dtype=torch.float64)
    removed = [0, 2, 7]
    # BN output is exactly zero for these channels, including its offset. ReLU
    # preserves zero, so removing corresponding conv2 inputs has this reference.
    with torch.no_grad():
        reference.layer1[0].bn1.weight[removed] = 0
        reference.layer1[0].bn1.bias[removed] = 0
        expected = reference(x)
    space = imagenet_models.build_space(model, x, "resnet18", ("layer1.0",))
    pruner = Pruner(model, graph=space.graph)
    pruner.apply(pruner.plan(remove=(space.axes[0].select(removed),)))
    torch.testing.assert_close(model(x), expected, atol=1e-9, rtol=1e-9)
    assert model.layer1[0].conv1.out_channels == 61
    assert model.layer1[0].conv2.in_channels == 61
    assert model.fc.out_features == 1000
    model(x).square().mean().backward()
    assert model.layer1[0].conv1.weight.grad is not None
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(
        resnet18(weights=None), tmp_path / "model.pt", map_location=execution_device
    ).eval()
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("model_name", ["resnet18", "vit_b_16"])
@pytest.mark.parametrize(
    "recipe,extra",
    [
        ("prune_finetune", []),
        ("prune_finetune", ["--metric", "taylor"]),
        ("iterative_pruning", ["--rounds", "2"]),
        ("bn_sparsity", []),
        ("group_sparsity", []),
        ("group_sparsity", ["--penalty", "squared"]),
        ("soft_pruning", ["--operation", "zero"]),
        ("soft_pruning", ["--operation", "decay"]),
        ("gate_pruning", []),
        ("stability_pruning", []),
    ],
)
def test_pretrained_workflow(monkeypatch, tmp_path, model_name, recipe, extra, execution_device):
    # Only test fixtures use random weights/data. Every executable entry must
    # request an official pretrained weight enum; there is no random-model CLI.
    requested = []
    weights = imagenet_models.MODELS[model_name][1]

    def factory(*, weights):
        requested.append(weights)
        if model_name == "resnet18":
            return resnet18(weights=None)
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=1,
            num_heads=2,
            hidden_dim=16,
            mlp_dim=32,
            num_classes=1000,
        )
        with torch.no_grad():
            model.heads.head.weight.normal_(std=0.02)
        return model

    def data(selected_weights, data_dir, *, need_train, **kwargs):
        assert selected_weights is weights
        x = torch.randn(2, 3, 224, 224)
        return (
            TensorDataset(x, torch.tensor([0, 1])) if need_train else None,
            TensorDataset(x + 0.1, torch.tensor([2, 3])),
            {"test_fixture": True},
        )

    monkeypatch.setitem(imagenet_models.MODELS, model_name, (factory, weights))
    monkeypatch.setattr(workflow_utils, "load_images", data)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe + ".py",
            "--model",
            model_name,
            "--device",
            execution_device,
            "--ratio",
            "0.5",
            "--batch-size",
            "2",
            "--threads",
            "1",
            "--finetune-epochs",
            "1",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--output",
            str(tmp_path),
            *extra,
        ],
    )
    module = importlib.import_module(recipe)
    if recipe == "bn_sparsity" and model_name == "vit_b_16":
        with pytest.raises(ValueError, match="LayerNorm"):
            module.main()
        assert not requested
        return
    # Match the CLI: CPU data loading, followed by explicit model/input transfer.
    with torch.device("cpu"):
        module.main()
    results = json.loads((tmp_path / "metrics.json").read_text())["stages"]
    assert requested == [weights, None]
    baseline = results[0]
    assert baseline["stage"] == "pretrained"
    assert all(r["samples"] == 2 for r in results)
    assert all(r["top1_delta_pp"] == pytest.approx(r["top1"] - baseline["top1"]) for r in results)
    pruned = [r for r in results if r["stage"].endswith("pruned")]
    assert pruned[-1]["actual_ratio"] == 0.5
    assert pruned[-1]["#Params"] < baseline["#Params"]
    assert pruned[-1]["#MACs"] < baseline["#MACs"]
    if recipe == "iterative_pruning":
        assert [row["actual_ratio"] for row in pruned] == [0.25, 0.5]
    assert all(row["latency_ms"] > 0 for row in results)
    assert results[-1]["stage"].endswith("finetuned")


def test_default_pretrained_path_never_trains(monkeypatch, tmp_path):
    weights = imagenet_models.MODELS["resnet18"][1]
    monkeypatch.setitem(
        imagenet_models.MODELS, "resnet18", (lambda *, weights: resnet18(weights=None), weights)
    )

    def data(weights, data_dir, *, need_train, **kwargs):
        assert not need_train
        return None, TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1])), {}

    def unexpected_training(*args, **kwargs):
        pytest.fail("Evaluation-only pruning must not train")

    monkeypatch.setattr(workflow_utils, "load_images", data)
    monkeypatch.setattr(workflow_utils, "train_epoch", unexpected_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prune_finetune.py",
            "--threads",
            "1",
            "--warmup",
            "0",
            "--repetitions",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    importlib.import_module("prune_finetune").main()
    stages = json.loads((tmp_path / "metrics.json").read_text())["stages"]
    assert [row["stage"] for row in stages] == ["pretrained", "pruned"]


@pytest.mark.parametrize("gated", [False, True])
def test_vit_adapter_gate_identity_and_ffn_compaction(gated, tmp_path, execution_device):
    torch.set_num_threads(1)
    # torchvision architecture with smaller dimensions keeps numerical regression
    # independent and cheap; full pretrained ViT is also checked manually.
    official = (
        VisionTransformer(
            image_size=32,
            patch_size=16,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            mlp_dim=32,
            num_classes=1000,
        )
        .to(device=execution_device, dtype=torch.float64)
        .eval()
    )
    with torch.no_grad():
        official.heads.head.weight.normal_(std=0.02)
    model = imagenet_models.TraceableViT(copy.deepcopy(official)).eval()
    x = torch.randn(2, 3, 32, 32, device=execution_device, dtype=torch.float64)
    layers = ("encoder.layers.encoder_layer_0",)
    if gated:
        model.encoder.layers[0].mlp[1] = nn.Sequential(
            model.encoder.layers[0].mlp[1],
            ChannelGate(32, -1).to(device=execution_device, dtype=torch.float64),
        )
    skeleton = copy.deepcopy(model)
    with torch.no_grad():
        torch.testing.assert_close(model(x), official(x), rtol=1e-10, atol=1e-10)
        official.encoder.layers[0].mlp[0].weight[[0, 2]] = 0
        official.encoder.layers[0].mlp[0].bias[[0, 2]] = 0
        expected = official(x)
    space = imagenet_models.build_space(model, x, "vit_b_16", layers, gated=gated)
    pruner = Pruner(model, graph=space.graph)
    pruner.apply(pruner.plan(remove=(space.axes[0].select([0, 2]),)))
    torch.testing.assert_close(model(x), expected, rtol=1e-9, atol=1e-9)
    assert model.encoder.layers[0].mlp[0].out_features == 30
    model(x).square().mean().backward()
    assert model.encoder.layers[0].mlp[0].weight.grad is not None
    if gated:
        assert model.encoder.layers[0].mlp[1][1].size == 30
    save_checkpoint(model, tmp_path / "vit.pt")
    restored = load_checkpoint(skeleton, tmp_path / "vit.pt", map_location=execution_device)
    torch.testing.assert_close(restored(x), model(x))


def test_taylor_calibration_preserves_bn_state_and_uses_task_only_gradients():
    run = workflow_utils.Experiment.__new__(workflow_utils.Experiment)
    run.model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 3)).train()
    reference = copy.deepcopy(run.model).eval()
    x, labels = torch.randn(3, 4), torch.tensor([0, 1, 2])
    run.train_loader = DataLoader(TensorDataset(x, labels), batch_size=3)
    run.options = SimpleNamespace(device="cpu")
    before = {name: value.clone() for name, value in run.model.state_dict().items()}
    run.task_gradients()
    F.cross_entropy(reference(x), labels).backward()
    assert run.model.training and run.model[0].training
    for name, value in run.model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for actual, expected in zip(run.model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
