"""Executable workflow stages, training, failure handling, and saved outputs."""

import decimal
import json
import math
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib

import imagenet_data as imagenet
import prune_finetune
from datasets import Dataset
from PIL import Image as PILImage
from torchvision.models.vision_transformer import VisionTransformer

from tests.support.torchvision_models import workflow_resnet


@pytest.mark.parametrize("model_name", ["resnet18", "vit_b_16"])
@pytest.mark.parametrize(
    "recipe,extra",
    [
        ("prune_finetune", []),
        pytest.param("prune_finetune", ["--train_augmentation"], id="augmented-finetune"),
        ("prune_finetune", ["--train_workers", "2", "--val_workers", "2"]),
        ("prune_finetune", ["--metric", "taylor"]),
        pytest.param("prune_finetune", ["--selection", "dynamic"], id="dynamic-magnitude"),
        pytest.param(
            "prune_finetune",
            ["--metric", "taylor", "--selection", "dynamic"],
            id="dynamic-taylor",
        ),
        pytest.param("prune_finetune", ["--metric", "geometric_median"], id="geometric-median"),
        ("iterative_pruning", ["--rounds", "2"]),
        ("bn_guided_pruning", []),
        ("group_sparsity", []),
        ("group_sparsity", ["--penalty", "squared"]),
        ("group_sparsity", ["--penalty", "squared", "--train_samples", "2"]),
        (
            "group_sparsity",
            [
                "--penalty",
                "squared",
                "--sparse_epochs",
                "3",
                "--selection_interval_steps",
                "2",
                "--sparsity_schedule",
                "linear",
            ],
        ),
        ("soft_pruning", ["--operation", "zero"]),
        ("soft_pruning", ["--operation", "decay"]),
        ("soft_pruning", ["--operation", "decay", "--cycles", "1", "--projection_epochs", "2"]),
        ("gate_pruning", []),
        ("stability_pruning", []),
        ("stability_pruning", ["--max_selection_checks", "1", "--window", "2", "--threshold", "1"]),
        ("stability_pruning", ["--max_selection_checks", "2", "--selection_interval_steps", "1"]),
        (
            "stability_pruning",
            ["--max_selection_checks", "5", "--threshold", "0", "--sparsity_schedule", "linear"],
        ),
    ],
)
def test_pretrained_workflow(
    monkeypatch, tmp_path, model_name, recipe, extra, execution_device, capsys
):
    # Only test fixtures use random weights/data. Every executable entry must
    # request an official pretrained weight enum; there is no random-model CLI.
    requested = []
    module = importlib.import_module(recipe)
    weights = prune_finetune.MODELS[model_name][1]
    graph_builds = 0
    if recipe == "iterative_pruning":
        original_build = module.DependencyGraph.build

        def observed_build(*args, **kwargs):
            nonlocal graph_builds
            graph_builds += 1
            return original_build(*args, **kwargs)

        monkeypatch.setattr(module.DependencyGraph, "build", observed_build)
    # Observe the coefficient reaching autograd, not just a scheduler helper or
    # printed value. Keep real group penalties and public plan/apply/save/load.
    penalty_weights, selection_steps = [], []
    training_samples_seen = []
    optimizer_steps = 0
    if recipe in ("group_sparsity", "stability_pruning"):
        original_penalty, original_plan = module.GroupSquaredL2, module.make_plan
        original_step = torch.optim.SGD.step
        original_groups = Pruner.parameter_groups

        def repeated_groups(self, *args, **kwargs):
            # Equivalent candidate entries must not double the workflow's loss.
            groups = original_groups(self, *args, **kwargs)
            return groups + groups

        monkeypatch.setattr(Pruner, "parameter_groups", repeated_groups)
        if recipe == "group_sparsity":
            original_lasso = module.GroupLasso

            def observed_lasso(groups):
                assert len({id(g) for g in groups}) == len(groups)
                return original_lasso(groups)

            monkeypatch.setattr(module, "GroupLasso", observed_lasso)

        def observed_penalty(*args, **kwargs):
            assert len({id(g) for g in args[0]}) == len(args[0])
            regularizer = original_penalty(*args, **kwargs)

            def evaluate_penalty():
                value = regularizer()
                value.register_hook(lambda gradient: penalty_weights.append(gradient.item()))
                return value

            return evaluate_penalty

        def observed_plan(*args, **kwargs):
            selection_steps.append(optimizer_steps)
            return original_plan(*args, **kwargs)

        def observed_step(optimizer, *args, **kwargs):
            nonlocal optimizer_steps
            result = original_step(optimizer, *args, **kwargs)
            optimizer_steps += 1
            return result

        monkeypatch.setattr(module, "GroupSquaredL2", observed_penalty)
        monkeypatch.setattr(module, "make_plan", observed_plan)
        monkeypatch.setattr(torch.optim.SGD, "step", observed_step)

    def factory(*, weights):
        requested.append(weights)
        if model_name == "resnet18":
            return workflow_resnet()
        model = VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=2,
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
        training = TensorDataset(x, torch.tensor([0, 1])) if need_train else None
        if "--train_augmentation" in extra:
            rows = Dataset.from_dict(
                {
                    "image": [PILImage.new("RGB", (280, 260), (i * 50, 60, 120)) for i in range(2)],
                    "label": [0, 1],
                }
            )
            training = imagenet.Images(rows, weights.transforms(), (0, 1))
        if need_train and (
            recipe == "stability_pruning"
            or (recipe == "group_sparsity" and "squared" in extra and kwargs["train_samples"] != 2)
        ):

            class ObservedTrainingData(TensorDataset):
                def __getitem__(self, index):
                    training_samples_seen.append(index)
                    return super().__getitem__(index)

            training = ObservedTrainingData(x.repeat(3, 1, 1, 1), torch.arange(6))
        return (
            training,
            TensorDataset(x + 0.1, torch.tensor([2, 3])),
            {"test_fixture": True},
        )

    monkeypatch.setitem(module.MODELS, model_name, (factory, weights))
    monkeypatch.setattr(module, "load_images", data)
    model_args = (
        []
        if recipe == "bn_guided_pruning" and model_name == "resnet18"
        else ["--model", model_name]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe + ".py",
            *model_args,
            "--device",
            execution_device,
            "--pruning_ratio",
            "0.15" if model_name == "resnet18" else "0.02",
            "--train_batch_size",
            "2",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--val_batch_size",
            "1",
            "--finetune_epochs",
            "1",
            "--latency_warmup",
            "0",
            "--latency_repetitions",
            "1",
            "--output",
            str(tmp_path),
            *(["--selection_interval_steps", "2"] if recipe == "stability_pruning" else []),
            *extra,
        ],
    )
    if recipe == "bn_guided_pruning" and model_name == "vit_b_16":
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2
        assert not requested
        return
    # Match the CLI: CPU data loading, followed by explicit model/input transfer.
    with torch.device("cpu"):
        module.main()
    progress_output = capsys.readouterr().err.lower()
    assert "evaluation" in progress_output and "fine-tuning" in progress_output
    assert "100%" in progress_output and "2/2" in progress_output
    if recipe == "prune_finetune" and "taylor" in extra:
        assert "taylor calibration" in progress_output
    saved = json.loads((tmp_path / "metrics.json").read_text())
    results = saved["stages"]
    expected_layers = (
        ["layer1.0", "layer1.1", "layer2.0", "layer3.0", "layer4.0"]
        if model_name == "resnet18"
        else ["encoder.layers.encoder_layer_0", "encoder.layers.encoder_layer_1"]
    )
    if recipe in ("prune_finetune", "iterative_pruning"):
        axes = {item["parameter"] for item in saved["config"]["candidate_axes"]}
        if model_name == "resnet18":
            assert axes == {f"{path}.conv1.weight" for path in expected_layers}
        else:
            assert axes == {f"{path}.mlp.0.weight" for path in expected_layers}
        assert saved["config"]["pruning_scope"] == "block_internal"
    else:
        assert saved["config"]["layers"] == expected_layers
    assert saved["config"]["train_batch_size"] == 2
    assert saved["config"]["val_batch_size"] == 1
    assert requested == [weights, None]
    baseline = results[0]
    # Independent decimal arithmetic verifies the CLI conversion, including
    # learned gates and the fixed initial baseline used by iterative pruning.
    requested_ratio = decimal.Decimal(str(saved["config"]["pruning_ratio"]))
    assert saved["config"]["max_params"] == int(baseline["#Params"] * (1 - requested_ratio))
    assert baseline["stage"] == "pretrained"
    assert all(r["samples"] == 2 for r in results)
    assert all(r["top1_delta_pp"] == pytest.approx(r["top1"] - baseline["top1"]) for r in results)
    pruned = [r for r in results if r["stage"].endswith("pruned")]
    assert pruned[-1]["max_params"] == saved["config"]["max_params"]
    assert all(row["target_met"] and row["#Params"] <= row["max_params"] for row in pruned)
    assert all(row["#Params"] == row["after_params"] for row in pruned)
    assert pruned[-1]["#Params"] < baseline["#Params"]
    assert pruned[-1]["#MACs"] < baseline["#MACs"]
    if recipe == "iterative_pruning":
        assert graph_builds == saved["config"]["rounds"]
        initial = baseline["#Params"]
        cap = saved["config"]["max_params"]
        assert [row["max_params"] for row in pruned] == [initial - (initial - cap) // 2, cap]
        assert pruned[1]["before_params"] == pruned[0]["after_params"]
    if recipe == "stability_pruning":
        search = next(row for row in results if row["stage"] == "search_completed")
        interval = saved["config"]["selection_interval_steps"]
        checks = saved["config"]["max_selection_checks"]
        total_steps = (checks - 1) * interval
        actual_steps = search["training_steps"]
        assert actual_steps == (search["selection_checks"] - 1) * interval
        assert selection_steps == list(range(0, actual_steps + 1, interval))
        if checks == 1:
            assert actual_steps == 0 and search["stable"] is False
        if saved["config"]["threshold"] == 0:
            assert search["stable"] and search["selection_checks"] == 3
            assert actual_steps < total_steps
        # Three batches per traversal, checks every two: a check must neither
        # rewind to the first batch nor discard the remaining third batch.
        sparse_samples = training_samples_seen[: actual_steps * 2]
        for start in range(0, len(sparse_samples), 6):
            traversal = sparse_samples[start : start + 6]
            assert len(set(traversal)) == len(traversal)
            assert set(traversal) <= set(range(6))
        assert sorted(training_samples_seen[actual_steps * 2 :]) == list(range(6))
    elif recipe == "group_sparsity" and "squared" in extra:
        batches_per_epoch = 1 if saved["config"]["train_samples"] == 2 else 3
        total_steps = actual_steps = saved["config"]["sparse_epochs"] * batches_per_epoch
        interval = saved["config"]["selection_interval_steps"]
        assert selection_steps == [*range(0, actual_steps, interval), actual_steps]
    if recipe == "stability_pruning" or (recipe == "group_sparsity" and "squared" in extra):
        fractions = [step / total_steps for step in range(1, actual_steps + 1)]
        if saved["config"]["sparsity_schedule"] == "cosine":
            # sin²(x/2) is an independent expression for the half-cosine ramp.
            fractions = [math.sin(math.pi * p / 2) ** 2 for p in fractions]
        expected = [saved["config"]["sparse_loss_weight"] * p for p in fractions]
        assert penalty_weights == pytest.approx(expected)
        finetune_steps = 3 if recipe == "stability_pruning" else batches_per_epoch
        assert optimizer_steps == actual_steps + finetune_steps
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
        ("bn_guided_pruning", ["--sparse_epochs", "0"]),
        ("group_sparsity", ["--sparse_epochs", "0"]),
        ("gate_pruning", ["--sparse_epochs", "0"]),
        ("stability_pruning", ["--max_selection_checks", "1"]),
    ],
)
def test_evaluation_only_workflows_do_not_request_training(
    monkeypatch, tmp_path, execution_device, recipe, extra
):
    module = importlib.import_module(recipe)
    weights = module.MODELS["resnet18"][1]
    monkeypatch.setitem(module.MODELS, "resnet18", (lambda *, weights: workflow_resnet(), weights))

    def data(weights, data_dir, *, need_train, **kwargs):
        assert not need_train
        return None, TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1])), {}

    def unexpected_training(*args, **kwargs):
        pytest.fail("Evaluation-only pruning must not train")

    monkeypatch.setattr(module, "load_images", data)
    monkeypatch.setattr(torch.optim.SGD, "step", unexpected_training)
    device_args = ["--device", "cpu"] if execution_device == "cpu" else []
    # The narrow fixture contains about 130k parameters; require real compaction.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe + ".py",
            *device_args,
            "--pruning_ratio",
            "0.05",
            "--val_batch_size",
            "1",
            "--val_workers",
            "0",
            "--latency_warmup",
            "0",
            "--latency_repetitions",
            "1",
            "--output",
            str(tmp_path),
            *extra,
        ],
    )
    # Real Arrow datasets produce CPU tensors; the CUDA fixture's default
    # device must not turn the mocked dataset into already-device-resident data.
    with torch.device("cpu"):
        module.main()
    stages = json.loads((tmp_path / "metrics.json").read_text())["stages"]
    expected = ["pretrained", "pruned"]
    if recipe == "iterative_pruning":
        expected = ["pretrained", "round_1_pruned"]
    elif recipe == "stability_pruning":
        expected = ["pretrained", "search_completed", "pruned"]
    assert [row["stage"] for row in stages] == expected
    assert all(row["device"].split(":")[0] == execution_device for row in stages)
    assert stages[-1]["#Params"] < stages[0]["#Params"]


def test_workflow_unmet_parameter_target_stops_before_apply(monkeypatch, tmp_path):
    module = prune_finetune
    strategy = module.Greedy
    weights = module.MODELS["resnet18"][1]
    monkeypatch.setitem(module.MODELS, "resnet18", (lambda *, weights: workflow_resnet(), weights))
    monkeypatch.setattr(module, "Greedy", lambda metric: strategy(metric, max_trials=0))
    monkeypatch.setattr(module, "measure_model", lambda *args: {})
    monkeypatch.setattr(
        module,
        "load_images",
        lambda *args, **kwargs: (
            None,
            TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])),
            {},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow",
            "--device",
            "cpu",
            "--val_workers",
            "0",
            "--pruning_ratio",
            "0.05",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(PlanningError, match=r"Parameter target not reached.*strategy trial limit"):
        module.main()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert "layer2.0.conv1.weight" in {
        item["parameter"] for item in saved["config"]["candidate_axes"]
    }
    assert [row["stage"] for row in saved["stages"]] == ["pretrained"]
    assert not (tmp_path / "model.pt").exists()
