"""Workflow measurement adapters use the supplied input and latency settings."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

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
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 6))
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

    def complexity(target, inputs, *, device):
        assert target is model and inputs is example
        observed.append("complexity")
        return calculate_complexity(target, inputs, device=device)

    def latency(target, inputs, **kwargs):
        assert target is model and inputs is example
        assert kwargs == {"device": "cpu", "compile": compiled, "warmup": 2, "repetitions": 4}
        observed.append("latency")
        return 1.25

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
    assert report["#Params"] == 12 * 6 + 6
    assert report["input_shape"] == tuple(example.shape)
    assert report["compiled"] is compiled
    assert accuracy["samples"] == example_batch_size
