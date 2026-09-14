import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import Pruner, load_checkpoint, save_checkpoint
from torch_kirigami.sparsity import ChannelGate, register_gate_operators

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib

import imagenet_data as imagenet
import imagenet_models
import prune_finetune
from datasets import ClassLabel, Dataset, Features, Image
from huggingface_hub import HfApi, constants
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image as PILImage
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.vision_transformer import VisionTransformer

WORKFLOWS = (
    "prune_finetune",
    "iterative_pruning",
    "bn_sparsity",
    "group_sparsity",
    "soft_pruning",
    "gate_pruning",
    "stability_pruning",
)


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
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("layer1.0.conv1",))
    pruner.apply(pruner.plan_remove((space.channel_axes[0].select(removed),)))
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
        ("soft_pruning", ["--operation", "decay", "--cycles", "1", "--projection-epochs", "2"]),
        ("gate_pruning", []),
        ("stability_pruning", []),
        ("stability_pruning", ["--search-steps", "1", "--window", "2", "--threshold", "1"]),
    ],
)
def test_pretrained_workflow(monkeypatch, tmp_path, model_name, recipe, extra, execution_device):
    # Only test fixtures use random weights/data. Every executable entry must
    # request an official pretrained weight enum; there is no random-model CLI.
    requested = []
    module = importlib.import_module(recipe)
    weights = prune_finetune.MODELS[model_name][1]

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

    monkeypatch.setitem(module.MODELS, model_name, (factory, weights))
    monkeypatch.setattr(module, "load_images", data)
    model_args = (
        [] if recipe == "bn_sparsity" and model_name == "resnet18" else ["--model", model_name]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe + ".py",
            *model_args,
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
    if recipe == "bn_sparsity" and model_name == "vit_b_16":
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2
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
    if recipe == "stability_pruning" and "--search-steps" in extra:
        search = next(row for row in results if row["stage"] == "search_completed")
        assert search["selection_checks"] == 1
        assert search["training_epochs"] == 0
        assert search["stable"] is False
    if recipe == "soft_pruning" and "--cycles" in extra:
        assert [row["stage"] for row in results if row["stage"].endswith("projected")] == [
            "cycle_1_projected"
        ]
    assert all(row["latency_ms"] > 0 for row in results)
    assert results[-1]["stage"].endswith("finetuned")


@pytest.mark.parametrize(
    "recipe,extra",
    [
        ("prune_finetune", []),
        ("iterative_pruning", ["--rounds", "1"]),
        ("bn_sparsity", ["--sparse-epochs", "0"]),
        ("group_sparsity", ["--sparse-epochs", "0"]),
        ("gate_pruning", ["--sparse-epochs", "0"]),
        ("stability_pruning", ["--search-steps", "1"]),
    ],
)
def test_evaluation_only_workflows_do_not_request_training(
    monkeypatch, tmp_path, execution_device, recipe, extra
):
    module = importlib.import_module(recipe)
    weights = module.MODELS["resnet18"][1]
    monkeypatch.setitem(
        module.MODELS, "resnet18", (lambda *, weights: resnet18(weights=None), weights)
    )

    def data(weights, data_dir, *, need_train, **kwargs):
        assert not need_train
        return None, TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1])), {}

    def unexpected_training(*args, **kwargs):
        pytest.fail("Evaluation-only pruning must not train")

    monkeypatch.setattr(module, "load_images", data)
    monkeypatch.setattr(torch.optim.SGD, "step", unexpected_training)
    device_args = ["--device", "cpu"] if execution_device == "cpu" else []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe + ".py",
            *device_args,
            "--threads",
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
    module.main()
    stages = json.loads((tmp_path / "metrics.json").read_text())["stages"]
    expected = ["pretrained", "pruned"]
    if recipe == "iterative_pruning":
        expected = ["pretrained", "round_1_pruned"]
    elif recipe == "stability_pruning":
        expected = ["pretrained", "search_completed", "pruned"]
    assert [row["stage"] for row in stages] == expected
    assert all(row["device"].split(":")[0] == execution_device for row in stages)


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
    operators = (
        register_gate_operators(OperatorRegistry.default()) if gated else OperatorRegistry.default()
    )
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    pruner = Pruner(model, graph=graph)
    pruner.apply(
        pruner.plan_remove((graph.parameter(f"{layers[0]}.mlp.0.weight").axis(0).select([0, 2]),))
    )
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


@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("explicit_cpu", [False, True])
@pytest.mark.parametrize("recipe", WORKFLOWS)
def test_workflow_device_default_requires_cuda(
    monkeypatch, capsys, cuda_available, explicit_cpu, recipe
):
    module = importlib.import_module(recipe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(sys, "argv", ["workflow", *(["--device", "cpu"] if explicit_cpu else [])])
    if not cuda_available and not explicit_cpu:
        with pytest.raises(SystemExit) as error:
            module.parse_args()
        assert error.value.code == 2
        assert "CUDA is unavailable" in capsys.readouterr().err
    else:
        options = module.parse_args()
        assert options.device == ("cpu" if explicit_cpu else "cuda")


@pytest.mark.parametrize("recipe", WORKFLOWS)
def test_example_launches_with_only_shared_support_files(tmp_path, recipe):
    source = Path(__file__).resolve().parents[2] / "examples" / "workflows" / f"{recipe}.py"
    target = tmp_path / source.name
    shutil.copyfile(source, target)
    for name in ("imagenet_data.py", "imagenet_models.py", "model_metrics.py"):
        shutil.copyfile(source.parent / name, tmp_path / name)
    # Only the selected entry and its three support modules are available.
    # Ignore PYTHONPATH while allowing Python to import from the script directory.
    completed = subprocess.run(
        [sys.executable, "-E", str(target), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--ratio" in completed.stdout


@pytest.mark.parametrize(
    "recipe,flag",
    [
        ("prune_finetune", "--rounds"),
        ("prune_finetune", "--strength"),
        ("iterative_pruning", "--sparse-epochs"),
        ("iterative_pruning", "--metric"),
        ("bn_sparsity", "--model"),
        ("bn_sparsity", "--rounds"),
        ("group_sparsity", "--metric"),
        ("gate_pruning", "--penalty"),
        ("soft_pruning", "--strength"),
        ("soft_pruning", "--sparse-epochs"),
        ("stability_pruning", "--sparse-epochs"),
        ("stability_pruning", "--operation"),
    ],
)
def test_workflow_rejects_unrelated_task_options(monkeypatch, capsys, recipe, flag):
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", flag, "1"])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(recipe).parse_args()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
