"""persistence / checkpoint state contracts."""

import io

import pytest
import torch
from torch import nn

from torch_kirigami.pruning import (
    ExecutionError,
    load_checkpoint,
    save_checkpoint,
)


class ParentState(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Identity()
        self.child.state = {"scale": 1}
        self.drop = nn.Dropout(1)
        self.layers = [self.drop]

    def forward(self, x):
        return self.layers[0](x) * self.child.state["scale"]

    def get_extra_state(self):
        return self.child.state

    def set_extra_state(self, state):
        self.child.state = state
        self.drop = nn.Dropout(1)
        self.layers = [self.drop]


@pytest.mark.parametrize("copy_behavior", ["self", "raise"])
@pytest.mark.parametrize("reject_load", [False, True])
def test_checkpoint_shells_bypass_copy_and_preserve_failure_state(
    copy_behavior, reject_load, execution_device
):
    class CustomCopy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, device=execution_device))
            self.register_buffer("offset", torch.zeros(2, device=execution_device))
            self.child = nn.Identity()
            self.alias = self.child

        def __copy__(self):
            if copy_behavior == "raise":
                raise AssertionError("User copy must not run during loading")
            return self

        def forward(self, x):
            return self.alias(x * self.weight + self.offset)

        def _load_from_state_dict(self, *args, **kwargs):
            if reject_load:
                self.weight.fill_(99)
                raise RuntimeError("Rejected on the isolated shell")
            return super()._load_from_state_dict(*args, **kwargs)

    source, target = CustomCopy(), CustomCopy()
    with torch.no_grad():
        source.weight.fill_(3)
        source.offset.fill_(2)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    weight, offset, child = target.weight, target.offset, target.child
    fields = set(vars(target))
    if reject_load:
        with pytest.raises(ExecutionError, match="Rejected on the isolated shell"):
            load_checkpoint(target, stream)
        assert set(vars(target)) == fields
        assert target.weight is weight and target.offset is offset
        assert target.child is child and target.alias is child
        torch.testing.assert_close(target.weight, torch.ones_like(weight))
        torch.testing.assert_close(target.offset, torch.zeros_like(offset))
    else:
        load_checkpoint(target, stream)
        assert target.alias is target.child
        x = torch.ones(2, device=execution_device)
        torch.testing.assert_close(target(x), source(x))
        target(x).sum().backward()
        torch.testing.assert_close(target.weight.grad, x)


class ExtraState(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(1)
        self.layers = [self.drop]
        self.cache = {}
        self.flag = True

    def forward(self, x):
        return self.layers[0](x)

    def get_extra_state(self):
        return {"drop_cache": not hasattr(self, "cache"), "drop_flag": not hasattr(self, "flag")}

    def set_extra_state(self, state):
        if state["drop_cache"]:
            del self.cache
        if state["drop_flag"]:
            del self.flag


def test_native_extra_state_roundtrip_and_failure_isolation():
    class WithExtra(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.state = {"steps": [0]}

        def forward(self, x):
            return self.fc(x)

        def get_extra_state(self):
            return self.state

        def set_extra_state(self, state):
            self.state = state

    model = WithExtra()
    model.state["steps"].append(5)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(WithExtra(), stream)
    assert restored.state == model.state


def test_parent_extra_state_commits_entire_final_module_graph(execution_device):
    source = ParentState()
    source.child.state = {"scale": 3}
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = ParentState()
    original_drop = target.drop
    restored = load_checkpoint(target, stream)
    assert restored.drop is original_drop and restored.layers[0] is original_drop
    assert restored.child.state == {"scale": 3}
    source.eval()
    restored.eval()
    torch.testing.assert_close(restored(torch.ones(2)), source(torch.ones(2)))


def test_nan_configuration_checkpoint_equality():
    source = nn.Linear(2, 2)
    source.tag = float("nan")
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = nn.Linear(2, 2)
    target.tag = float("nan")
    load_checkpoint(target, stream)
    torch.testing.assert_close(target.weight, source.weight)


def test_conjugate_complex_checkpoint_aliases(execution_device):
    source = nn.Module()
    value = torch.tensor([1 + 2j]).conj()
    source.register_buffer("left", value)
    source.register_buffer("right", value)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = nn.Module()
    initial = torch.zeros(1, dtype=torch.complex64)
    target.register_buffer("left", initial)
    target.register_buffer("right", initial)
    load_checkpoint(target, stream)
    assert target.left is target.right
    torch.testing.assert_close(target.left, value)


def test_cached_buffer_checkpoint_without_extra_state(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1))
            self.refs = {"scale": self.scale}

        def forward(self, x):
            return x * self.refs["scale"]

    source = Model()
    source.scale.fill_(3)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    restored = load_checkpoint(Model(), stream)
    assert restored.refs["scale"] is restored.scale
    torch.testing.assert_close(restored(torch.ones(2)), torch.full((2,), 3.0))


def test_checkpoint_extra_state_deletions_and_module_references(execution_device):
    source = ExtraState()
    del source.cache
    del source.flag
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    restored = load_checkpoint(ExtraState(), stream)
    assert not hasattr(restored, "cache") and not hasattr(restored, "flag")
    assert restored.layers[0] is restored.drop
    source.eval()
    restored.eval()
    torch.testing.assert_close(restored(torch.ones(2)), source(torch.ones(2)))


def test_checkpoint_shared_nan_roundtrip(execution_device):
    module = nn.Linear(2, 2)
    model = nn.ModuleList([module, module])
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    target = nn.Linear(2, 2)
    restored = load_checkpoint(nn.ModuleList([target, target]), stream)
    assert restored[0] is restored[1]
    torch.testing.assert_close(restored[0].weight, module.weight, equal_nan=True)


def test_list_configuration_checkpoint_retains_container_types():
    model = nn.Sequential(nn.AdaptiveAvgPool2d([2, 2]))
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(nn.Sequential(nn.AdaptiveAvgPool2d([2, 2])), stream)
    assert type(restored[0].output_size) is list
    assert restored[0].output_size == [2, 2]


def test_checkpoint_extra_state_transaction_restores_deleted_attributes(monkeypatch):
    source = ExtraState()
    del source.cache
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = ExtraState()
    old_cache, old_layers = target.cache, target.layers
    original = ExtraState.__setattr__

    def fail(self, name, value):
        if self is target and name == "layers":
            raise RuntimeError("Injected commit failure")
        original(self, name, value)

    monkeypatch.setattr(ExtraState, "__setattr__", fail)
    with pytest.raises(ExecutionError, match="Commit failed"):
        load_checkpoint(target, stream)
    assert target.cache is old_cache and target.layers is old_layers
    assert target.layers[0] is target.drop
