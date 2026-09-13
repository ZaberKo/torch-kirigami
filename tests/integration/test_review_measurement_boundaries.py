"""Measurement preserves input relationships and complete buffer registration state."""

import pytest
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency


@pytest.mark.parametrize("latency", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("change", ["none", "add", "delete", "replace"])
def test_measurement_restores_complete_buffer_tables(latency, fail, change, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", None, persistent=False)
            self.register_buffer("old", torch.ones(2))

        def forward(self, x, *, alias):
            assert x is alias["x"] is alias["again"][0]
            if change == "none":
                self.cache = torch.ones_like(x)
            elif change == "add":
                self.register_buffer("new", torch.ones_like(x), persistent=False)
            elif change == "delete":
                if hasattr(self, "old"):
                    del self.old
            else:
                self.old = torch.zeros_like(x)
            if fail:
                raise RuntimeError("injected failure")
            return x + 1

    model = Model()
    x = torch.ones(2, device="cpu")
    original, modes = model.old, model.training
    kwargs = {"alias": {"x": x, "again": [x]}}

    def measure():
        if latency:
            return measure_module_latency(model, (x,), input_kwargs=kwargs, repetitions=1, warmup=0)
        return calculate_model_complexity(model, (x,), input_kwargs=kwargs)

    if fail:
        with pytest.raises(RuntimeError, match="injected failure"):
            measure()
    else:
        measure()
    assert model.cache is None and model.old is original and model.training == modes
    assert tuple(model._buffers) == ("cache", "old")
    assert model._non_persistent_buffers_set == {"cache"}
    torch.testing.assert_close(x, torch.ones(2, device="cpu"))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_latency_keeps_mha_self_attention_execution_path(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = nn.MultiheadAttention(32, 4, batch_first=True, device=device).eval()
    x = torch.randn(2, 16, 32, device=device)
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as direct:
        model(x, x, x)
    with profile(activities=[ProfilerActivity.CPU]) as measured:
        measure_module_latency(model, (x, x, x), repetitions=1, warmup=0)
    name = "aten::_native_multi_head_attention"
    assert any(e.name == name for e in direct.events())
    assert any(e.name == name for e in measured.events())


@pytest.mark.parametrize("latency", [False, True])
def test_measurement_keeps_views_distinct_and_preserves_shared_storage(latency, execution_device):
    class Model(nn.Module):
        def forward(self, x, view, *, containers):
            assert x is not view and containers[0] is containers[1]
            x.add_(1)
            torch.testing.assert_close(view, torch.ones_like(view))
            return x + view

    x = torch.zeros(4)
    view = x.view(4)
    shared = [x, view]
    arguments = (x, view)
    kwargs = {"containers": [shared, shared]}
    if latency:
        measure_module_latency(
            Model(),
            arguments,
            input_kwargs=kwargs,
            device=execution_device,
            warmup=0,
            repetitions=1,
        )
    else:
        calculate_model_complexity(Model(), arguments, input_kwargs=kwargs, device=execution_device)
    torch.testing.assert_close(x, torch.zeros(4))


def test_measurement_refuses_alias_breaking_device_transfer(execution_device):
    if execution_device != "cuda":
        pytest.skip("Cross-device transfer needs CUDA")

    class Model(nn.Module):
        called = False

        def forward(self, x, y):
            self.called = True
            return x + y

    x = torch.ones(4, device="cpu")
    model = Model()
    with pytest.raises(ValueError, match="storage-sharing"):
        measure_module_latency(model, (x, x.view(4)), device="cuda", warmup=0, repetitions=1)
    assert not model.called and model.training
    torch.testing.assert_close(x, torch.ones(4, device="cpu"))


@pytest.mark.parametrize("latency", [False, True])
@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_measurement_preserves_input_aliases_in_ordinary_attributes(
    latency, view, fail, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cached = torch.zeros(1, 4)
            self.refs = {"cached": [self.cached.view(1, 4) if view else self.cached]}
            self.small = nn.Linear(4, 2, bias=False)
            self.large = nn.Linear(4, 10, bias=False)

        def forward(self, x):
            assert x is self.cached
            if view:
                assert x is not self.refs["cached"][0]
            else:
                assert x is self.refs["cached"][0]
            x.add_(1)
            torch.testing.assert_close(self.refs["cached"][0], x)
            if fail:
                raise RuntimeError("alias failure")
            return self.small(x) if x is self.cached else self.large(x)

    model = Model()
    original, refs = model.cached, model.refs
    before = original.clone()

    def measure():
        if latency:
            return measure_module_latency(model, original, repetitions=1, warmup=0)
        return calculate_model_complexity(model, original)

    if fail:
        with pytest.raises(RuntimeError, match="alias failure"):
            measure()
    else:
        result = measure()
        if not latency:
            assert result.macs == 8
    assert model.cached is original and model.refs is refs and model.training
    torch.testing.assert_close(original, before)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_latency_preserves_cached_mha_identity(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cached = torch.randn(2, 8, 16, device=device)
            self.attn = nn.MultiheadAttention(16, 4, batch_first=True, device=device)

        def forward(self, x):
            return self.attn(x, self.cached, self.cached)

    model = Model().eval()
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as direct:
        model(model.cached)
    with profile(activities=[ProfilerActivity.CPU]) as measured:
        measure_module_latency(model, model.cached, repetitions=1, warmup=0)
    name = "aten::_native_multi_head_attention"
    assert any(e.name == name for e in direct.events())
    assert any(e.name == name for e in measured.events())


def test_measurement_rejects_moving_model_shared_inputs_independently(execution_device):
    if execution_device != "cuda":
        return

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cached = torch.zeros(2, device="cpu")

        def forward(self, x):
            raise AssertionError("Must reject before execution")

    model = Model()
    original = model.cached
    with pytest.raises(ValueError, match="model-shared inputs"):
        measure_module_latency(model, original, device="cuda", repetitions=1, warmup=0)
    assert model.cached is original and model.training


@pytest.mark.parametrize("kind", ["list", "tuple", "dict", "empty"])
@pytest.mark.parametrize("latency", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_measurement_preserves_model_owned_input_containers(kind, latency, fail, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            x = torch.ones(1, 4)
            self.cached = (
                [x]
                if kind == "list"
                else (x,)
                if kind == "tuple"
                else {"x": x}
                if kind == "dict"
                else []
            )
            self.refs = {"alias": self.cached}
            self.small = nn.Linear(4, 2, bias=False)
            self.large = nn.Linear(4, 10, bias=False)

        def forward(self, x):
            assert x is self.cached is self.refs["alias"]
            if fail:
                raise RuntimeError("container failure")
            data = torch.ones(1, 4) if kind == "empty" else x["x"] if kind == "dict" else x[0]
            return self.small(data) if x is self.cached else self.large(data)

    model = Model()
    original, refs = model.cached, model.refs

    def measure():
        if latency:
            return measure_module_latency(model, (original,), repetitions=1, warmup=0)
        return calculate_model_complexity(model, (original,))

    if fail:
        with pytest.raises(RuntimeError, match="container failure"):
            measure()
    else:
        result = measure()
        if not latency:
            assert result.macs == 8
    assert model.cached is original and model.refs is refs
