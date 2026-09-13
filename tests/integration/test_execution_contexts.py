"""Precision, layout and execution contexts must survive lifecycle boundaries."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency
from torch_kirigami.pruning import (
    ExecutionError,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("channels_last", [False, True])
def test_compact_singleton_strides_match_plan_and_checkpoint(
    rank, width, channels_last, execution_device
):
    conv = getattr(nn, f"Conv{rank}d")
    fmt = (
        torch.contiguous_format
        if not channels_last
        else torch.channels_last
        if rank == 2
        else torch.channels_last_3d
    )

    def make():
        return nn.Sequential(
            conv(width, width, 3, padding=1), nn.ReLU(), conv(width, 2, 3, padding=1)
        ).to(memory_format=fmt)

    model = make()
    original = copy.deepcopy(model)
    x = torch.randn((2, width) + (5,) * rank)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan(remove=[graph.parameter("0.weight").axis(0).select([1])])
        .to_dict()
    )
    Pruner(model).apply(plan)
    keep = [i for i in range(width) if i != 1]
    convolution = getattr(F, f"conv{rank}d")
    expected = convolution(
        convolution(x, original[0].weight[keep], original[0].bias[keep], padding=1).relu(),
        original[2].weight[:, keep],
        original[2].bias,
        padding=1,
    )
    torch.testing.assert_close(model(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(make(), stream)
    torch.testing.assert_close(restored(x), expected)
    for instance in (model, restored):
        for state in plan.after.tensors:
            owner, _, field = state.paths[0].rpartition(".")
            assert getattr(instance.get_submodule(owner), field).stride() == state.stride
        instance(x).sum().backward()
        assert all(p.grad is not None for p in instance.parameters())


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("storage", ["attribute", "container", "extra"])
def test_checkpoint_preparation_keeps_all_tensor_state_trainable(
    inference, storage, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3))
            scale = torch.arange(1.0, 4.0)
            self.scale = {"nested": [scale]} if storage == "container" else scale

        def forward(self, x):
            scale = self.scale["nested"][0] if storage == "container" else self.scale
            return self.weight * x * scale

        def get_extra_state(self):
            return self.scale.clone() if storage == "extra" else None

        def set_extra_state(self, value):
            if storage == "extra":
                self.scale = value

    source, target = Model(), Model()
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    with torch.inference_mode(inference):
        load_checkpoint(target, stream)
        assert torch.is_inference_mode_enabled() == inference
    scale = target.scale["nested"][0] if storage == "container" else target.scale
    assert not scale.is_inference() and not target.weight.is_inference()
    target(torch.ones(3)).sum().backward()
    torch.testing.assert_close(target.weight.grad, torch.arange(1.0, 4.0))


@pytest.mark.parametrize("name", ["weight", "weight_extra_state", "_extra_state_weight"])
@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("nested", [False, True])
def test_registered_names_do_not_collide_with_extra_state(name, kind, nested, execution_device):
    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.arange(1.0, 4.0)
            if kind == "parameter":
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
            self.factor = 2

        def forward(self, x):
            return x * getattr(self, name) * self.factor

        def get_extra_state(self):
            return {"factor": self.factor}

        def set_extra_state(self, state):
            self.factor = state["factor"]

    def make():
        return nn.Sequential(Value()) if nested else Value()

    source, target = make(), make()
    owner = source[0] if nested else source
    owner.factor = 3
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    load_checkpoint(target, stream)
    x = torch.ones(3, requires_grad=True)
    torch.testing.assert_close(target(x), source(x))
    target(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.arange(1.0, 4.0) * 3)


@pytest.mark.parametrize("device", ["cpu", "cpu:0", torch.device("cpu:1")])
def test_measurement_accepts_cpu_device_aliases(device):
    model = nn.Linear(3, 2).to(device)
    x = torch.ones(2, 3, device=device)
    assert calculate_model_complexity(model, x, device=device).macs == 12
    assert measure_module_latency(model, x, device=device, warmup=0, repetitions=1) >= 0


class PartialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(4, 8, bias=False) for _ in range(3))

    def forward(self, x):
        q = self.q(x).reshape(x.size(0), x.size(1), 2, -1).transpose(1, 2)
        k = self.k(x).reshape(x.size(0), x.size(1), 2, -1).transpose(1, 2)
        v = self.v(x).reshape(x.size(0), x.size(1), 2, -1).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize("planning_autocast", [False, True])
@pytest.mark.parametrize("low_precision", ["bfloat16", "float16"])
def test_partial_amp_paths_join_with_captured_dtype(
    planning_autocast, low_precision, execution_device
):
    dtype = getattr(torch, low_precision)
    model = PartialAttention()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 4)
    with torch.autocast(execution_device, dtype=dtype):
        graph = DependencyGraph.build(model, args=(x,))
    with torch.autocast(execution_device, dtype=dtype, enabled=planning_autocast):
        plan = PruningPlan.from_dict(
            Pruner(model, graph=graph)
            .plan(remove=[graph.parameter("q.weight").axis(0).select([1, 5])])
            .to_dict()
        )
    Pruner(model).apply(plan)
    assert model.q.out_features == model.k.out_features == 6 and model.v.out_features == 8
    with torch.autocast(execution_device, dtype=dtype):
        keep = [0, 2, 3, 4, 6, 7]
        q, k = (
            F.linear(x, layer.weight[keep]).reshape(2, 3, 2, 3).transpose(1, 2)
            for layer in (reference.q, reference.k)
        )
        v = reference.v(x).reshape(2, 3, 2, 4).transpose(1, 2)
        # Independent explicit attention, including compact Q/K scaling.
        expected = ((q @ k.transpose(-2, -1)) / 3**0.5).softmax(-1) @ v
        actual = model(x)
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
    actual.float().square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(PartialAttention(), stream)
    with torch.autocast(execution_device, dtype=dtype):
        torch.testing.assert_close(restored(x), actual)


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("relocation", ["string", "dict", "device"])
def test_checkpoint_maps_extra_state_only_storage(registered, relocation, execution_device):
    class Model(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.scale = torch.arange(1.0, 3.0, device=device)
            if registered:
                # The extra-state device need not occur in registered state.
                self.weight = nn.Parameter(torch.ones(2, device="cpu"))

        def forward(self, x):
            return x * self.scale

        def get_extra_state(self):
            return {"scale": self.scale}

        def set_extra_state(self, state):
            self.scale = state["scale"]

    source = Model(execution_device)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    source_device = str(source.scale.device)
    relocation = (
        "cpu"
        if relocation == "string"
        else {source_device: "cpu"}
        if relocation == "dict"
        else torch.device("cpu")
    )
    target = Model("cpu")
    with torch.inference_mode():
        assert load_checkpoint(target, stream, map_location=relocation) is target
    x = torch.ones(2, device="cpu", requires_grad=True)
    target(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 2.0], device="cpu"))
    assert target.scale.device.type == "cpu" and not target.scale.is_inference()


def test_checkpoint_callable_relocation_is_rejected_before_reading(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(3.0, device="cpu"))
            self.scale = torch.tensor([1.0, 2.0], device="cpu")

        def forward(self, x):
            return x * self.weight * self.scale

        def get_extra_state(self):
            return {"scale": self.scale}

        def set_extra_state(self, state):
            self.scale = state["scale"]

    source, target = Model(), Model()
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    calls = []

    def relocate(storage, location):
        calls.append((location, storage.nbytes()))
        return storage

    weight, scale = target.weight, target.scale
    with pytest.raises(ExecutionError, match="map_location requires a device"):
        load_checkpoint(target, stream, map_location=relocate)
    assert stream.tell() == 0 and calls == []
    assert target.weight is weight and target.scale is scale
    load_checkpoint(target, stream, map_location={"cpu": "cpu"})
    x = torch.ones(2, device="cpu", requires_grad=True)
    torch.testing.assert_close(target(x), torch.tensor([3.0, 6.0], device="cpu"))
    target(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([3.0, 6.0], device="cpu"))


@pytest.mark.parametrize(
    "overload", ["keyword", "dtype_positional", "tensor_positional", "device_positional"]
)
@pytest.mark.parametrize("copy_output", [False, True])
def test_cast_copy_contract_preserves_native_compact_view_behavior(
    overload, copy_output, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)

        def forward(self, x):
            y = self.a(x)[:, ::2]
            if overload == "keyword":
                y = y.to(copy=copy_output)
            elif overload == "dtype_positional":
                y = y.to(torch.float32, False, copy_output)
            elif overload == "tensor_positional":
                y = y.to(x, False, copy_output)
            else:
                y = y.to(x.device, torch.float32, False, copy_output)
            return y.view(-1)

    model = Model()
    x = torch.randn(2, 4)
    expected = model(x).detach()
    graph = DependencyGraph.build(model, args=(x,))
    old = tuple(model.parameters())
    if copy_output:
        Pruner(model, graph=graph).prune(remove=[graph.parameter("a.weight").axis(0).select([5])])
        torch.testing.assert_close(model(x), expected)
        model(x).sum().backward()
    else:
        with pytest.raises(PlanningError, match="view"):
            Pruner(model, graph=graph).plan(
                remove=[graph.parameter("a.weight").axis(0).select([5])]
            )
        assert all(a is b for a, b in zip(old, model.parameters(), strict=True))


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("count", [0, 3])
@pytest.mark.parametrize("alias", [False, True])
def test_extra_state_registered_identity_checked_even_for_empty_tensors(
    kind, count, alias, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.arange(float(count))
            if kind == "parameter":
                self.value = nn.Parameter(value)
            else:
                self.register_buffer("value", value)
            self.cached = self.value if alias else self.value.detach().clone()

        def forward(self, x):
            return x + self.value.sum() + self.cached.sum()

        def get_extra_state(self):
            return {"nested": [self.cached]}

        def set_extra_state(self, state):
            self.cached = state["nested"][0]

    model = Model()
    original, cached = model.value, model.cached
    before = original.detach().clone()
    stream = io.BytesIO()
    if alias:
        with pytest.raises(ExecutionError, match="registered tensor references"):
            save_checkpoint(model, stream)
        assert not stream.getvalue()
    else:
        save_checkpoint(model, stream)
        stream.seek(0)
        target = Model()
        load_checkpoint(target, stream)
        assert target.cached is not target.value
        torch.testing.assert_close(target(torch.ones(1)), model(torch.ones(1)))
    assert model.value is original and model.cached is cached
    torch.testing.assert_close(model.value, before)
