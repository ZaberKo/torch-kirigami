"""Independent measurement formulas, isolation and public pruning integration."""

import gc
import math

import pytest
import torch
from torch import nn

from tests.support.workflow_fixtures import Transformer, build_space
from torch_kirigami import DependencyGraph
from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency
from torch_kirigami.pruning import Pruner


def test_complexity_linear_conv_and_frozen_shared_parameters(execution_device):
    linear = nn.Linear(4, 5).to(execution_device)
    linear.weight.requires_grad_(False)
    linear.register_parameter("alias", linear.weight)
    value = calculate_model_complexity(linear, torch.ones(2, 3, 4), execution_device)
    assert (value.macs, value.params, value.unsupported_ops) == (2 * 3 * 4 * 5, 4 * 5 + 5, ())
    conv = nn.Conv2d(4, 6, 3, padding=1, groups=2).to(execution_device)
    value = calculate_model_complexity(conv, torch.ones(2, 4, 5, 5), execution_device)
    assert value.macs == 2 * 6 * 5 * 5 * 2 * 3 * 3
    assert value.params == 6 * 2 * 3 * 3 + 6
    assert not value.unsupported_ops


@pytest.mark.parametrize("gated", [False, True])
def test_complexity_includes_attention_and_survives_pruning(gated, execution_device):
    model = Transformer(gated=gated).to(execution_device)
    x = torch.randn(1, 4, 8, device=execution_device)
    pruner, _space = build_space(model, x)
    with torch.inference_mode():
        before = calculate_model_complexity(model, x)
    # Projections/FFN/output + QK and AV; do not derive the reference from a counter.
    assert (
        before.macs
        == 4 * (8 * 16 + 8 * 8 + 8 * 8 + 16 * 8 + 8 * 12 + 12 * 8) + 8 * 3 + 2 * 4 * 4 * 4 * 4
    )
    assert not before.unsupported_ops
    assert model.training
    pruner.graph.validate()
    pruner.apply(
        pruner.plan_remove((pruner.graph.parameter("attn.k.weight").axis(0).select(range(4)),))
    )
    after = calculate_model_complexity(model, x)
    assert (
        after.macs
        == 4 * (8 * 8 + 8 * 4 + 8 * 4 + 8 * 8 + 8 * 12 + 12 * 8) + 8 * 3 + 2 * 2 * 4 * 4 * 4
    )
    assert after.params < before.params and not after.unsupported_ops
    model(x).sum().backward()


def test_complexity_reports_unsupported_work():
    class Inverse(nn.Module):
        def forward(self, x):
            return torch.linalg.inv(x)

    result = calculate_model_complexity(Inverse(), torch.eye(3))
    assert result.unsupported_ops and any("linalg" in op for op in result.unsupported_ops)


@pytest.mark.parametrize("kind", ["complexity", "latency"])
@pytest.mark.parametrize("fail", [False, True])
def test_measurement_restores_modes_buffers_inputs_rng_and_gc(kind, fail, execution_device):
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(3)
            self.drop = nn.Dropout()
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, x, *, extra):
            x.add_(1)
            extra["offset"].add_(1)
            self.counter.add_(1)
            torch.rand(2, device=x.device)
            if fail:
                raise RuntimeError("measurement failed")
            return self.drop(self.bn(x)) + extra["offset"]

    model = Stateful().to(execution_device)
    model.drop.eval()  # Mixed modes must survive both success and failure.
    inputs = torch.zeros(2, 3, device=execution_device)
    offset = torch.zeros_like(inputs)
    modes = [module.training for module in model.modules()]
    buffers = [(name, buf, buf.clone()) for name, buf in model.named_buffers()]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if execution_device == "cuda" else []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        fn = calculate_model_complexity if kind == "complexity" else measure_module_latency
        if fail:
            with pytest.raises(RuntimeError, match="measurement failed"):
                fn(model, inputs, input_kwargs={"extra": {"offset": offset}})
        else:
            fn(model, inputs, input_kwargs={"extra": {"offset": offset}})
        assert not gc.isenabled()
    finally:
        if was_enabled:
            gc.enable()
    assert [module.training for module in model.modules()] == modes
    assert inputs.count_nonzero() == offset.count_nonzero() == 0
    current = dict(model.named_buffers())
    for name, original, reference in buffers:
        assert current[name] is original
        torch.testing.assert_close(original, reference)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    if cuda_rng:
        for before, after in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True):
            torch.testing.assert_close(before, after)


def test_latency_statistics_exclude_compile_and_warmup(monkeypatch):
    class Calls(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x + 1

    compiled = []
    monkeypatch.setattr(
        torch, "compile", lambda target, **kwargs: compiled.append(kwargs) or target
    )
    clock = iter([0.0, 0.001, 1.0, 1.003])
    monkeypatch.setattr("torch_kirigami.measurement.time.perf_counter", lambda: next(clock))
    model = Calls()
    value = measure_module_latency(
        model,
        torch.ones(1),
        compile=True,
        compile_kwargs={"backend": "eager"},
        warmup=3,
        repetitions=2,
    )
    assert value == pytest.approx(2.0)  # Median averages both middle samples.
    assert model.calls == 1 + 3 + 2 and compiled == [{"backend": "eager"}]


def test_actual_compilation_and_remeasurement_after_pruning(execution_device):
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2)).to(execution_device)
    x = torch.ones(2, 4, device=execution_device)
    graph = DependencyGraph.build(model, args=(x,))
    before = calculate_model_complexity(model, x)
    assert before.macs == 72
    latency = measure_module_latency(
        model, x, compile=True, compile_kwargs={"backend": "eager"}, repetitions=2, warmup=1
    )
    assert math.isfinite(latency) and latency >= 0
    graph.validate()
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove((graph.parameter("0.weight").axis(0).select([0, 1]),)))
    assert calculate_model_complexity(model, x).macs == 48
    latency = measure_module_latency(
        model, x, compile=True, compile_kwargs={"backend": "eager"}, repetitions=2, warmup=1
    )
    assert math.isfinite(latency) and latency >= 0 and model.training
    model(x).sum().backward()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"repetitions": 0},
        {"warmup": -1},
        {"repetitions": True},
        {"compile_kwargs": {"backend": "eager"}},
        {"device": "meta"},
    ],
)
def test_latency_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        measure_module_latency(nn.Identity(), torch.ones(1), **kwargs)
