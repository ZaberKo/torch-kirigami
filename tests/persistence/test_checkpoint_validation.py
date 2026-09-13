"""persistence / checkpoint validation contracts."""

import io

import pytest
import torch
from torch import nn

from tests.support.models import WithBuffer
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ExecutionError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


class ChangedParameter(nn.Parameter):
    pass


@pytest.mark.parametrize("change", ["storage", "type", "device", "none"])
def test_checkpoint_checks_callback_final_tensor_structure(change, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.b = nn.Parameter(torch.tensor([3.0, 4.0]))

        def get_extra_state(self):
            return None

        def set_extra_state(self, state):
            if change == "storage":
                self.b = nn.Parameter(self.a.detach())
            elif change == "type":
                self.b = ChangedParameter(self.b.detach())
            elif change == "device":
                self.b = nn.Parameter(self.b.to("meta"))

    model = Model()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    old_a, old_b = model.a, model.b
    if change == "none":
        assert load_checkpoint(model, stream, map_location="cpu") is model
        assert model.a.device.type == model.b.device.type == "cpu"
    else:
        with pytest.raises(ExecutionError, match=r"structure|devices"):
            load_checkpoint(model, stream, map_location="cpu")
        assert model.a is old_a and model.b is old_b
    torch.testing.assert_close(model.a.cpu(), torch.tensor([1.0, 2.0], device="cpu"))
    torch.testing.assert_close(model.b.cpu(), torch.tensor([3.0, 4.0], device="cpu"))


def test_checkpoint_rejects_corrupt_alias_values_before_mutation():
    model = WithBuffer()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    payload["state_dict"]["alias.weight"] = payload["state_dict"]["alias.weight"] + 1
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    target = WithBuffer()
    old = target.fc.weight
    with pytest.raises(ExecutionError, match="Conflicting"):
        load_checkpoint(target, stream)
    assert target.fc.weight is old


def test_raw_checkpoint_ignores_custom_state_dict_keys():
    class Renamed(nn.Linear):
        def state_dict(self, *args, **kwargs):
            result = super().state_dict(*args, **kwargs)
            result["saved_weight"] = result.pop("weight")
            return result

        def load_state_dict(self, state, **kwargs):
            state = dict(state)
            state["weight"] = state.pop("saved_weight")
            return super().load_state_dict(state, **kwargs)

    source = Renamed(2, 2)
    source.load_state_dict(source.state_dict())
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    assert set(payload["state_dict"]) == {"weight", "bias"}
    target = Renamed(2, 2)
    stream.seek(0)
    load_checkpoint(target, stream)
    torch.testing.assert_close(target(torch.ones(1, 2)), source(torch.ones(1, 2)))


def test_checkpoint_state_dict_hooks_rejected_at_save():
    model = nn.Linear(2, 2)

    def save_hook(module, state, prefix, metadata):
        state[prefix + "saved_weight"] = state.pop(prefix + "weight")

    def load_hook(module, state, prefix, *args):
        state[prefix + "weight"] = state.pop(prefix + "saved_weight")

    model.register_state_dict_post_hook(save_hook)
    model.register_load_state_dict_pre_hook(load_hook)
    model.load_state_dict(model.state_dict())  # Valid PyTorch usage, outside our schema.
    with pytest.raises(ExecutionError, match="hook"):
        save_checkpoint(model, io.BytesIO())


def test_checkpoint_rejects_internal_overlap_before_writing():
    model = nn.Module()
    model.register_buffer("overlap", torch.ones(1, 3).expand(2, 3))
    stream = io.BytesIO()
    with pytest.raises(ExecutionError, match="overlap"):
        save_checkpoint(model, stream)
    assert stream.getvalue() == b""


@pytest.mark.parametrize(
    "change",
    [
        "replace_module",
        "add_module",
        "delete_module",
        "cycle",
        "parameter",
        "buffer",
        "tensor_values",
        "persistence",
        "forward_hook",
        "data_only",
    ],
)
def test_extra_state_is_data_restoration_without_registered_mutation(change, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Linear(2, 2)
            self.alias = self.child
            self.register_buffer("offset", torch.ones(2), persistent=False)
            self.data = {"scale": 1}

        def forward(self, x):
            return self.alias(x) * self.data["scale"] + self.offset

        def get_extra_state(self):
            return self.data

        def set_extra_state(self, state):
            self.data = state
            if change == "replace_module":
                self.child = nn.Linear(2, 2)
                self.alias = self.child
            elif change == "add_module":
                self.extra = nn.Identity()
            elif change == "delete_module":
                del self.alias
            elif change == "cycle":
                self.child.parent = self
            elif change == "parameter":
                self.child.weight = nn.Parameter(self.child.weight.detach().clone())
            elif change == "buffer":
                self.offset = self.offset.clone()
            elif change == "tensor_values":
                self.offset.add_(5)
            elif change == "persistence":
                self._non_persistent_buffers_set.clear()
            elif change == "forward_hook":
                self.child.register_forward_hook(lambda module, args, output: output * 2)

    source, target = Model(), Model()
    source.data = {"scale": 3}
    with torch.no_grad():
        source.child.weight.fill_(2)
        source.child.bias.fill_(4)
        source.offset.fill_(6)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    child, weight, offset, data = target.child, target.child.weight, target.offset, target.data
    original_weight = weight.detach().clone()
    if change == "data_only":
        assert load_checkpoint(target, stream) is target
        assert target.child is child and target.alias is child
        x = torch.ones(2, 2)
        torch.testing.assert_close(target(x), torch.full((2, 2), 30.0))
        target(x).sum().backward()
        torch.testing.assert_close(target.child.weight.grad, torch.full((2, 2), 6.0))
    else:
        with pytest.raises(ExecutionError, match=r"Extra state changed registered"):
            load_checkpoint(target, stream)
        assert target.child is child and target.alias is child
        assert target.child.weight is weight and target.offset is offset and target.data is data
        assert tuple(target._modules) == ("child", "alias")
        assert not hasattr(target.child, "parent") and not target.child._forward_hooks
        assert target._non_persistent_buffers_set == {"offset"}
        torch.testing.assert_close(weight, original_weight)
        torch.testing.assert_close(offset, torch.ones(2))


@pytest.mark.parametrize("norm", [nn.BatchNorm1d, nn.InstanceNorm1d])
def test_native_normalization_loaders_preserve_compact_checkpoint(norm, execution_device):
    source = nn.Sequential(nn.Conv1d(2, 4, 1), norm(4, affine=True, track_running_stats=True))
    # The closed-form reference below defines expected outputs independently of
    # recipe serialization and native state loading, including the retained order.
    source.eval()
    with torch.no_grad():
        source[0].weight.fill_(2)
        source[0].bias.fill_(1)
        source[1].weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        source[1].bias.fill_(0.5)
        source[1].running_mean.fill_(1)
        source[1].running_var.fill_(4)
    graph = DependencyGraph.build(source, args=(torch.ones(2, 2, 4),))
    Pruner(source, graph=graph).prune(
        remove=[graph.parameter("0.weight").axis(0).select([1])], preserve_io=False
    )
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = nn.Sequential(nn.Conv1d(2, 4, 1), norm(4, affine=True, track_running_stats=True))
    with torch.inference_mode():
        load_checkpoint(target, stream)
    x = torch.ones(2, 2, 4, requires_grad=True)
    expected = (
        torch.tensor([1.0, 3.0, 4.0]).reshape(1, 3, 1) * (4 / (4 + source[1].eps) ** 0.5) + 0.5
    )
    torch.testing.assert_close(target(x), expected.expand(2, 3, 4))
    target(x).sum().backward()
    assert x.grad is not None and not target[1].weight.is_inference()


@pytest.mark.parametrize("defined", ["getter", "setter"])
def test_checkpoint_requires_paired_extra_state_methods(defined, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

    if defined == "getter":
        Model.get_extra_state = lambda self: None
    else:
        Model.set_extra_state = lambda self, state: None
    target = Model()
    old = target.weight
    stream = io.BytesIO(b"unchanged")
    for operation in (save_checkpoint, load_checkpoint):
        with pytest.raises(ExecutionError, match="requires paired"):
            operation(target, stream)
        assert stream.tell() == 0 and stream.getvalue() == b"unchanged"
        assert target.weight is old


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("has_extra", [False, True])
def test_checkpoint_extra_state_cannot_overwrite_raw_tensor_slots(
    kind, has_extra, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            value = torch.tensor([1.0, 2.0])
            if kind == "parameter":
                self.register_parameter("_extra_state", nn.Parameter(value))
            else:
                self.register_buffer("_extra_state", value, persistent=False)

    if has_extra:
        Model.get_extra_state = lambda self: torch.tensor([5.0, 6.0])
        Model.set_extra_state = lambda self, state: None
    source = Model()
    stream = io.BytesIO()
    if has_extra:
        with pytest.raises(ExecutionError, match="collides with a registered tensor"):
            save_checkpoint(source, stream)
        assert stream.getvalue() == b""
        with pytest.raises(ExecutionError, match="collides with a registered tensor"):
            load_checkpoint(source, stream)
        assert stream.tell() == 0
    else:
        save_checkpoint(source, stream)
        stream.seek(0)
        target = load_checkpoint(Model(), stream)
        torch.testing.assert_close(target._extra_state, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(source._extra_state, torch.tensor([1.0, 2.0]))
