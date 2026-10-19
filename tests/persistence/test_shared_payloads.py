import io

import torch
from torch import nn

from torch_kirigami.pruning import (
    load_checkpoint,
    save_checkpoint,
)


def test_tied_checkpoint_uses_storage_identity_but_checks_independent_payloads(
    monkeypatch, execution_device
):
    model = nn.Linear(4, 4)
    model.alias = model.weight
    calls = []
    original = torch.isnan

    def counted(value):
        calls.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch, "isnan", counted)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    assert not calls
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    payload["state_dict"]["alias"] = payload["state_dict"]["alias"].clone()
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    load_checkpoint(model, stream)
    assert calls
    assert model.alias is model.weight
