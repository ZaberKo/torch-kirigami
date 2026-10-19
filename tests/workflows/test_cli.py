"""Workflow command-line contracts, defaults, and documented commands."""

import re
import shlex
import shutil
import subprocess
import sys
from inspect import signature
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib

import model_metrics

WORKFLOWS = (
    "prune_finetune",
    "iterative_pruning",
    "bn_guided_pruning",
    "group_sparsity",
    "soft_pruning",
    "gate_pruning",
    "stability_pruning",
    "variance_pruning",
    "isomorphic_pruning",
    "osscar_pruning",
    "vit_head_pruning",
)


@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("explicit_cpu", [False, True])
@pytest.mark.parametrize("recipe", WORKFLOWS)
def test_workflow_device_default_requires_cuda(
    monkeypatch, capsys, cuda_available, explicit_cpu, recipe
):
    module = importlib.import_module(recipe)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        sys,
        "argv",
        ["workflow", *(["--device", "cpu"] if explicit_cpu else [])],
    )
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
    ratio_flag = "--family_pruning_ratio" if recipe == "isomorphic_pruning" else "--pruning_ratio"
    assert ratio_flag in completed.stdout
    assert "--max_params" not in completed.stdout
    assert "--channel_pruning_ratio" not in completed.stdout
    assert "--max_macs" not in completed.stdout


@pytest.mark.parametrize(
    "recipe,flag",
    [
        ("prune_finetune", "--rounds"),
        ("prune_finetune", "--sparse_loss_weight"),
        ("iterative_pruning", "--sparse_epochs"),
        ("iterative_pruning", "--metric"),
        ("bn_guided_pruning", "--metric"),
        ("bn_guided_pruning", "--rounds"),
        ("group_sparsity", "--metric"),
        ("gate_pruning", "--penalty"),
        ("soft_pruning", "--sparse_loss_weight"),
        ("soft_pruning", "--sparse_epochs"),
        ("stability_pruning", "--sparse_epochs"),
        ("stability_pruning", "--operation"),
    ],
)
def test_workflow_rejects_unrelated_task_options(monkeypatch, capsys, recipe, flag):
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", flag, "1"])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(recipe).parse_args()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("recipe", WORKFLOWS)
def test_workflow_full_data_defaults_and_explicit_cli_scope(monkeypatch, recipe):
    module = importlib.import_module(recipe)
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu"])
    options = module.parse_args()
    assert options.train_samples == options.val_samples == 0
    assert options.train_batch_size == options.val_batch_size == 256
    assert options.train_workers == options.val_workers == 8
    latency_defaults = signature(model_metrics.measure_module_latency).parameters
    assert options.latency_warmup == latency_defaults["warmup"].default
    assert options.latency_repetitions == latency_defaults["repetitions"].default
    assert not options.compile_latency
    ratio_name = "family_pruning_ratio" if recipe == "isomorphic_pruning" else "pruning_ratio"
    assert getattr(options, ratio_name) == 0.05
    if recipe == "isomorphic_pruning":
        assert options.calibration_batches == 100 and options.calibration_batch_size == 64
    assert not hasattr(options, "layers") and not hasattr(options, "threads")
    if recipe in ("bn_guided_pruning", "gate_pruning", "group_sparsity", "stability_pruning"):
        assert options.sparse_loss_weight == 1e-4
    if recipe in ("group_sparsity", "stability_pruning"):
        assert options.sparsity_schedule == "cosine"
        assert options.selection_interval_steps == 100
    if recipe == "stability_pruning":
        assert options.max_selection_checks == 11
        assert options.max_selection_checks > options.window + 1
        assert (options.max_selection_checks - 1) * options.selection_interval_steps == 1000
    monkeypatch.setattr(
        sys,
        "argv",
        [
            recipe,
            "--device",
            "cpu",
            "--family_pruning_ratio" if recipe == "isomorphic_pruning" else "--pruning_ratio",
            "0.1",
            "--train_batch_size",
            "16",
            "--val_batch_size",
            "128",
            "--compile_latency",
            "--latency_warmup",
            "0",
            "--latency_repetitions",
            "1",
        ],
    )
    options = module.parse_args()
    assert (options.train_batch_size, options.val_batch_size) == (16, 128)
    assert options.compile_latency
    assert getattr(options, ratio_name) == 0.1
    for removed in (
        "--layers",
        "--threads",
        "--compile",
        "--batch-size",
        "--train-samples",
        "--ratio",
        "--channel_pruning_ratio",
        "--max_macs",
        "--max_params",
        "--strength",
        "--search_steps",
    ):
        monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", removed])
        with pytest.raises(SystemExit) as error:
            module.parse_args()
        assert error.value.code == 2


@pytest.mark.parametrize(
    "recipe", ["bn_guided_pruning", "gate_pruning", "group_sparsity", "stability_pruning"]
)
@pytest.mark.parametrize("weight", ["-1", "nan", "inf"])
def test_workflow_rejects_invalid_sparse_loss_weight(monkeypatch, recipe, weight):
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", "--sparse_loss_weight", weight])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(recipe).parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("recipe", ["group_sparsity", "stability_pruning"])
@pytest.mark.parametrize(
    "flag,value",
    [
        ("--selection_interval_steps", "0"),
        ("--selection_interval_steps", "-1"),
        ("--sparsity_schedule", "invalid"),
    ],
)
def test_workflow_rejects_invalid_sparse_schedule(monkeypatch, recipe, flag, value):
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", flag, value])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(recipe).parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("recipe", WORKFLOWS)
@pytest.mark.parametrize("ratio", ["-0.1", "1", "nan", "inf"])
def test_workflow_rejects_invalid_parameter_ratio(monkeypatch, capsys, recipe, ratio):
    ratio_name = "family_pruning_ratio" if recipe == "isomorphic_pruning" else "pruning_ratio"
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", f"--{ratio_name}", ratio])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(recipe).parse_args()
    assert error.value.code == 2
    assert f"0 <= {ratio_name} < 1" in capsys.readouterr().err


@pytest.mark.parametrize("recipe", WORKFLOWS)
@pytest.mark.parametrize("flag", ["--train_workers", "--val_workers"])
def test_workflow_rejects_negative_worker_counts(monkeypatch, capsys, recipe, flag):
    monkeypatch.setattr(sys, "argv", [recipe, "--device", "cpu", flag, "-1"])
    with pytest.raises(SystemExit) as error:
        importlib.import_module(recipe).parse_args()
    assert error.value.code == 2
    assert f"{flag} must be nonnegative" in capsys.readouterr().err


def test_readme_workflow_commands_match_the_cli(monkeypatch):
    readme = Path(__file__).resolve().parents[2] / "examples/workflows/README.md"
    commands = re.findall(r"^python (\w+\.py) ((?:[^\n]*\\\n)*[^\n]*)", readme.read_text(), re.M)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = set()
    for filename, arguments in commands:
        name = filename.removesuffix(".py")
        module = importlib.import_module(name)
        monkeypatch.setattr(sys, "argv", [filename, *shlex.split(arguments.replace("\\\n", " "))])
        options = module.parse_args()
        assert options.device == "cuda"
        assert options.compile_latency
        assert options.train_samples == options.val_samples == 0
        assert options.val_batch_size == options.train_batch_size == 256
        assert options.val_workers == options.train_workers == 8 and options.seed == 7
        latency_defaults = signature(model_metrics.measure_module_latency).parameters
        assert options.latency_warmup == latency_defaults["warmup"].default
        assert options.latency_repetitions == latency_defaults["repetitions"].default
        assert options.output.name != name  # Verify the full multiline command was parsed.
        seen.add(name)
    assert seen == set(WORKFLOWS)
