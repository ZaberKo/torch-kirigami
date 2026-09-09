"""Object aliases cannot erase parameter/buffer slot kinds in portable structures."""

import io

import pytest
import torch
from torch import nn

from torch_kirigami.pruning import save_checkpoint


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_cross_kind_alias_is_rejected_before_checkpoint_output(
    persistent, nested, execution_device
):
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(2, 2))
    owner = nn.Module() if nested else model
    if nested:
        model.child = owner
    owner.register_buffer("alias", model.weight, persistent=persistent)
    output = io.BytesIO(b"unchanged")
    with pytest.raises(ValueError, match=r"parameter.*buffer|registration kind"):
        save_checkpoint(model, output)
    assert output.getvalue() == b"unchanged"
    assert model.weight is owner.alias
