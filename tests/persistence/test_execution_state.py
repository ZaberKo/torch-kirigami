"""Precision, layout and execution contexts must survive lifecycle boundaries."""

import io

import pytest
import torch
from torch import nn

from torch_kirigami.pruning import (
    ExecutionError,
    load_checkpoint,
    save_checkpoint,
)


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
