"""Workflow measurement adapters use the supplied input and latency settings."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Pruner

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")


import imagenet_data as imagenet
import model_metrics


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("val_batch_size", [None, 7])
@pytest.mark.parametrize("example_batch_size", [1, 2])
def test_measurement_uses_supplied_example_for_complexity_and_latency(
    monkeypatch, compiled, val_batch_size, example_batch_size
):
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 6)).train()
    model[1].eval()
    options = SimpleNamespace(
        device="cpu",
        compile_latency=compiled,
        latency_warmup=2,
        latency_repetitions=4,
    )
    if val_batch_size is not None:
        options.val_batch_size = val_batch_size
    observed = []
    calculate_complexity = model_metrics.calculate_model_complexity

    def complexity(target, inputs):
        assert target is model and inputs is example
        assert not any(module.training for module in model.modules())
        observed.append("complexity")
        return calculate_complexity(target, inputs)

    def latency(target, inputs, **kwargs):
        assert inputs is example
        assert not any(module.training for module in model.modules())
        torch.testing.assert_close(target(inputs), model(inputs))
        assert kwargs == {
            "device": example.device,
            "compile": compiled,
            "warmup": 2,
            "repetitions": 4,
        }
        observed.append("latency")
        return SimpleNamespace(median_ms=1.25)

    def no_compile(*args, **kwargs):
        pytest.fail("Accuracy evaluation and metric setup must not compile the model")

    monkeypatch.setattr(torch, "compile", no_compile)
    monkeypatch.setattr(model_metrics, "calculate_model_complexity", complexity)
    monkeypatch.setattr(model_metrics, "measure_module_latency", latency)
    example = torch.randn(example_batch_size, 3, 2, 2)
    report = model_metrics.measure_model(model, example, options)
    accuracy = imagenet.evaluate(
        model, [(example, torch.zeros(example_batch_size, dtype=torch.long))], "cpu"
    )
    assert observed == ["complexity", "latency"]
    assert report["#MACs"] == example_batch_size * 12 * 6
    assert report["#FLOPs"] == example_batch_size * (2 * 12 * 6 + 6)
    assert report["uncovered_ops"]["flops"] == ()
    assert report["uncovered_ops"]["macs"] == ()
    assert "unsupported_ops" not in report and "unsupported_flop_ops" not in report
    assert report["#Params"] == 12 * 6 + 6
    assert report["input_shape"] == tuple(example.shape)
    assert report["compiled"] is compiled
    assert report["latency_ms"] == 1.25
    assert model.training and not model[1].training
    assert accuracy["samples"] == example_batch_size


@pytest.mark.parametrize("failing", ["complexity", "latency"])
def test_measurement_restores_modes_after_failure(monkeypatch, failing):
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 6)).train()
    model[1].eval()
    options = SimpleNamespace(
        device="cpu", compile_latency=False, latency_warmup=0, latency_repetitions=1
    )

    def fail(*args, **kwargs):
        assert not any(module.training for module in model.modules())
        raise RuntimeError("measurement failed")

    name = "calculate_model_complexity" if failing == "complexity" else "measure_module_latency"
    monkeypatch.setattr(model_metrics, name, fail)
    with pytest.raises(RuntimeError, match="measurement failed"):
        model_metrics.measure_model(model, torch.ones(1, 3, 2, 2), options)
    assert model.training and not model[1].training


@pytest.mark.parametrize("compiled", [False, True])
def test_prisma_measurement_and_compilation_preserve_pruning_lifecycle(
    monkeypatch, compiled, execution_device
):
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).train()
    model[2].eval()
    example = torch.ones(2, 4, device=execution_device)
    # Timing follows the actual input device, even if the CLI label is stale.
    options = SimpleNamespace(
        device="cuda" if execution_device == "cpu" else "cpu",
        compile_latency=compiled,
        latency_warmup=1,
        latency_repetitions=2,
    )
    compile_target = torch.compile
    latency_target = model_metrics.measure_module_latency
    compiled_targets = []

    def compile_eager(target, **kwargs):
        assert not isinstance(target, nn.Module)
        compiled_targets.append(target)
        return compile_target(target, backend="eager", **kwargs)

    def latency(target, inputs, *, device, **kwargs):
        assert inputs is example and device == example.device
        return latency_target(target, inputs, device=device, **kwargs)

    monkeypatch.setattr(torch, "compile", compile_eager)
    monkeypatch.setattr(model_metrics, "measure_module_latency", latency)
    graph = DependencyGraph.build(model, args=(example,))
    parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
    before = model_metrics.measure_model(model, example, options)
    assert before["#Params"] == 44 and before["#MACs"] == 72
    assert before["device"] == str(example.device)
    assert before["latency_ms"] >= 0
    assert model.training and model[0].training and not model[2].training
    graph.validate()
    for parameter, original in zip(model.parameters(), parameters, strict=True):
        torch.testing.assert_close(parameter, original)
    torch.testing.assert_close(example, torch.ones_like(example))

    pruner = Pruner(model, graph=graph)
    compact, _ = pruner.apply(
        pruner.plan_remove((graph.parameter("0.weight").axis(0).select([0, 1]),))
    )
    after = model_metrics.measure_model(compact, example, options)
    assert after["#Params"] == 30 and after["#MACs"] == 48
    assert after["latency_ms"] >= 0
    assert compact is model and model.training and not model[2].training
    assert len(compiled_targets) == (2 if compiled else 0)
    compact(example).sum().backward()
    assert all(parameter.grad is not None for parameter in compact.parameters())
