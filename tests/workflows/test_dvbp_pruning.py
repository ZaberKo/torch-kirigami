"""Independent DVBP numerical references and complete workflow tests."""

import copy
import json
import math
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import dvbp_pruning as workflow
from imagenet_models import MLP_MODELS
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

from tests.support.torchvision_models import tiny_vit
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Pruner, load_checkpoint, save_checkpoint


@pytest.mark.parametrize(
    "aspect, expected", [(0.25, 0.91600407), (1.0, 0.65277594), (4.0, 3.66401628)]
)
def test_mp_median_matches_continuous_bulk_quantiles(aspect, expected):
    assert workflow.mp_median(aspect) == pytest.approx(expected, abs=1e-7)


def test_streaming_covariance_matches_concatenated_rows(execution_device):
    values = torch.randn(7, 611, 4, dtype=torch.float64)
    values[3:] += 20
    stats = workflow.ActivationCovariance()
    for batch in values.split((2, 5)):
        stats.update(batch)
    rows = values.flatten(0, 1)
    assert stats.images == len(values)
    assert stats.rows == len(rows)
    torch.testing.assert_close(stats.mean, rows.mean(0))
    torch.testing.assert_close(stats.scatter / stats.rows, torch.cov(rows.T, correction=0))


def test_hybrid_scores_use_image_count_for_noise_estimation(execution_device):
    # Six features, six images, and many rows give an MP aspect of one.
    # With unit bulk eigenvalues the cutoff is 4 / 0.65277594.
    axes = torch.eye(6, dtype=torch.float64)
    radii = torch.tensor([1.0, 1.0, 1.0, 1.0, math.sqrt(4), math.sqrt(8)])
    positive = math.sqrt(6) * radii[:, None] * axes
    rows = torch.cat((positive, -positive)).tile(4, 1)
    stats = workflow.ActivationCovariance()
    stats.update(rows.reshape(6, -1, 6))
    denoised = workflow.denoise(stats, 0.25, 0.1)
    expected = torch.diag(torch.tensor([0, 0, 0, 0, 0, 8], dtype=torch.float64, device="cpu"))
    torch.testing.assert_close(denoised.signal @ denoised.signal.T, expected)
    hybrid = 0.75 * torch.cov(rows.T, correction=0).diag().cpu() + 0.25 * expected.diag()
    torch.testing.assert_close(denoised.scores, hybrid)
    assert denoised.ridge == pytest.approx(0.1 * 8 / 6)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_calibration_observes_keyword_consumer_input_and_restores_modes(
    nonfinite, execution_device
):
    class Model(nn.Sequential):
        def forward(self, x):
            return self[2](input=self[1](self[0](x)))

    model = Model(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2)).double().train()
    model[0].eval()
    x = torch.randn(4, 3, dtype=torch.float64)
    if nonfinite:
        x[0, 0] = float("nan")
    stats = workflow.collect_covariances(model, ("2",), [(x, torch.zeros(4))], execution_device, 1)
    assert model.training and not model[0].training and model[2].training
    assert not model[2]._forward_pre_hooks
    assert all(p.grad is None for p in model.parameters())
    if nonfinite:
        with pytest.raises(ValueError, match="nonfinite"):
            workflow.denoise(stats["2"], 0.2, 0.1)
    else:
        rows = F.gelu(F.linear(x, model[0].weight, model[0].bias))
        torch.testing.assert_close(stats["2"].mean, rows.mean(0))
        torch.testing.assert_close(stats["2"].scatter / len(x), torch.cov(rows.T, correction=0))
    with pytest.raises(ValueError, match="at least two"):
        workflow.denoise(workflow.ActivationCovariance(), 0.2, 0.1)


def test_constant_features_use_mean_compensation(execution_device):
    stats = workflow.ActivationCovariance()
    center = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
    stats.update(center.expand(8, -1))
    fit = workflow.denoise(stats, 0.5, 0.1)
    weight, bias = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, dtype=torch.float64)
    compact_weight, compact_bias = workflow.compensate(weight, bias, fit, (1,))
    torch.testing.assert_close(compact_weight, weight[:, [0, 2]].cpu())
    torch.testing.assert_close(
        F.linear(center[[0, 2]].cpu(), compact_weight, compact_bias),
        F.linear(center, weight, bias).cpu(),
    )


def test_global_selection_ob2c_compact_forward_and_checkpoint(execution_device, tmp_path):
    torch.manual_seed(26)
    model = (
        nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 4), nn.GELU(), nn.Linear(4, 2))
        .double()
        .eval()
    )
    original = copy.deepcopy(model)
    sample = torch.randn(32, 3, dtype=torch.float64) + 3
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph)
    pairs = {"0": "2", "2": "4"}
    stats = workflow.collect_covariances(
        model, pairs.values(), [(sample, torch.zeros(len(sample)))], execution_device, 1
    )
    fitted = {path: workflow.denoise(value, 0.3, 0.1) for path, value in stats.items()}
    # Force a known cross-layer ranking, including adjacent pairs whose shared
    # boundary weight loses rows and columns in the same plan.
    scores = {
        "0": torch.tensor([5, 0, 4, 3, 1, 6]),
        "2": torch.tensor([8, 9, 2, 7]),
    }
    selected = workflow.select_hidden_neurons(scores, 3, 1)
    assert selected == {"0": (1, 4), "2": (2,)}
    plan = pruner.plan_remove(
        graph.parameter(f"{path}.weight").axis(0).select(indices)
        for path, indices in selected.items()
    )
    updates = workflow.prepare_compensation(graph, plan, pairs, fitted)

    def inverse_reference(path, remove, keep_rows):
        layer = original.get_submodule(path)
        fit = fitted[path]
        signal = fit.signal.to(execution_device)
        mean = fit.mean.to(execution_device)
        system = signal @ signal.T + torch.eye(layer.in_features, dtype=signal.dtype) * fit.ridge
        inverse = torch.linalg.inv(system)
        delta = (
            -layer.weight[:, remove]
            @ torch.linalg.inv(inverse[remove][:, remove])
            @ inverse[remove]
        )
        retained = [i for i in range(layer.in_features) if i not in remove]
        return (
            (layer.weight + delta)[keep_rows][:, retained],
            (layer.bias - delta @ mean)[keep_rows],
        )

    bridge = inverse_reference("2", [1, 4], [0, 1, 3])
    output = inverse_reference("4", [2], [0, 1])
    for path, (weight, bias) in zip(("2", "4"), (bridge, output), strict=True):
        torch.testing.assert_close(updates[f"{path}.weight"], weight.cpu())
        torch.testing.assert_close(updates[f"{path}.bias"], bias.cpu())
    compact, _ = pruner.apply(plan)
    with torch.no_grad():
        for path, value in updates.items():
            compact.get_parameter(path).copy_(value)
        hidden = original[1](original[0](sample))[:, [0, 2, 3, 5]]
        expected = F.linear(original[3](F.linear(hidden, *bridge)), *output)
        torch.testing.assert_close(compact(sample), expected)
    compact(sample).square().sum().backward()
    assert all(p.grad is not None for p in compact.parameters())
    checkpoint = tmp_path / "compact.pt"
    save_checkpoint(compact, checkpoint)
    restored = load_checkpoint(original, checkpoint, map_location=execution_device)
    torch.testing.assert_close(restored(sample), expected)


def test_alignment_reserves_survivors_and_reports_unattainable_targets():
    scores = {
        "a": torch.tensor([5, 0, 2, 3, 1]),
        "b": torch.tensor([4, 4, 7, 6, 8, 9]),
    }
    # The first width has a mandatory remainder. Remove the cheapest aligned
    # blocks until the global target is reached.
    assert workflow.select_hidden_neurons(scores, 4, 2) == {"a": (1, 4, 2), "b": (0, 1)}
    with pytest.raises(ValueError, match="at most 7"):
        workflow.select_hidden_neurons(scores, 8, 2)
    with pytest.raises(ValueError, match="retain 6 hidden neurons"):
        workflow.select_hidden_neurons(scores, 1, 6)


def test_alignment_ranks_blocks_by_total_score():
    scores = {
        "a": torch.tensor([0, 100, 101, 102]),
        "b": torch.tensor([3, 2, 1, 4]),
    }
    assert workflow.select_hidden_neurons(scores, 2, 2) == {"a": (), "b": (2, 1)}
    assert workflow.select_hidden_neurons(scores, 2, 1) == {"a": (0,), "b": (2,)}


def test_prepare_rejects_unrelated_recipe_without_changing_model(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(4, 3),))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    plan = pruner.plan_remove([graph.parameter("2.weight").axis(0).select([0])])
    before = tuple(model.parameters())
    with pytest.raises(ValueError, match="matching producer rows and consumer columns"):
        workflow.prepare_compensation(graph, plan, {"0": "2"}, {})
    assert all(current is old for current, old in zip(model.parameters(), before, strict=True))
    graph.validate()


@pytest.mark.parametrize("name, epochs", [("vit_b_16", 0), ("convnext_tiny", 1)])
def test_dvbp_entry_runs_on_torchvision_models(
    name, epochs, monkeypatch, tmp_path, execution_device, capsys
):
    weights = MLP_MODELS[name][1]
    requested = []
    images = torch.randn(2, 3, 224, 224, device="cpu")
    labels = torch.tensor([0, 1], device="cpu")
    dataset = TensorDataset(images, labels)

    def skeleton():
        if name == "vit_b_16":
            return tiny_vit(image_size=224)
        return ConvNeXt(
            [CNBlockConfig(4, 8, 1), CNBlockConfig(8, 12, 1), CNBlockConfig(12, None, 1)],
            stochastic_depth_prob=0.2,
            num_classes=5,
        )

    def builder(*, weights):
        requested.append(weights)
        return skeleton()

    def data(selected_weights, data_dir, **options):
        assert selected_weights is weights and options["need_train"]
        return dataset, dataset, {"test_fixture": True}

    monkeypatch.setitem(MLP_MODELS, name, (builder, weights))
    monkeypatch.setattr(workflow, "load_images", data)
    monkeypatch.setattr(workflow, "measure_model", lambda *args: {})
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
            "2",
            "--val_batch_size",
            "2",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--calibration_batches",
            "1",
            "--granularity",
            "2",
            "--hidden_pruning_ratio",
            "0.2",
            "--finetune_epochs",
            str(epochs),
            "--output",
            str(tmp_path),
        ],
    )
    with torch.device("cpu"):
        workflow.main()
    assert requested == [weights, None]
    result = json.loads((tmp_path / "metrics.json").read_text())
    stages = ["pretrained", "pruned", "finetuned"] if epochs else ["pretrained", "pruned"]
    assert [row["stage"] for row in result["stages"]] == stages
    output = capsys.readouterr().out
    logged = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert [row for row in logged if "stage" in row] == result["stages"]
    assert "Checkpoint verified; results:" in output
    assert bool([row for row in logged if "task_loss" in row]) == bool(epochs)
    report = result["stages"][1]
    assert report["target_removed_hidden"] == report["original_hidden"] // 5
    assert (
        report["target_removed_hidden"]
        <= report["removed_hidden"]
        <= report["target_removed_hidden"] + 1
    )
    assert all(value["images"] == 2 for value in result["config"]["calibration"].values())
    assert (tmp_path / "model.pt").is_file()
    pruned_checkpoint = tmp_path / "pruned.pt"
    assert pruned_checkpoint.is_file() == bool(epochs)
    assert not (tmp_path / "training.pt").exists()
    if epochs:
        restored = load_checkpoint(skeleton(), pruned_checkpoint, map_location=execution_device)
        with torch.no_grad():
            logits = restored.eval()(images.to(execution_device))
            assert F.cross_entropy(logits, labels.to(execution_device)).item() == pytest.approx(
                report["loss"]
            )


@pytest.mark.parametrize(
    "args",
    [
        ["--damping", "0"],
        ["--calibration_batches", "-1"],
        ["--granularity", "0"],
        ["--model", "resnet18"],
        ["--repair_covariance", "raw"],
        ["--ridge_floor", "1e-8"],
    ],
)
def test_invalid_cli_settings_are_reported(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dvbp_pruning.py", "--device", "cpu", *args])
    with pytest.raises(SystemExit, match="2"):
        workflow.parse_args()
