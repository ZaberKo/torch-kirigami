"""persistence / checkpoint validation contracts."""

import io

import pytest
import torch
from torch import nn

from tests.support.models import WithBuffer
from torch_kirigami.pruning import (
    ExecutionError,
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


def test_overridden_state_dict_payload_rejected_before_file_write():
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
    with pytest.raises(ExecutionError, match="payload keys"):
        save_checkpoint(source, stream)
    assert not stream.getvalue()


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
