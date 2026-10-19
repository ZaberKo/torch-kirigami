"""Omitted and explicit arguments must pass through the same isolation boundary."""

import pytest
import torch
from torch import nn

from torch_kirigami import CaptureError, DependencyGraph


@pytest.mark.parametrize("supplied", ["default", "positional", "keyword"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("keyword_only", [False, True])
def test_mutable_default_is_isolated_on_success_and_failure(
    supplied, fail, keyword_only, execution_device
):
    defaults = [-1]

    class Model(nn.Module):
        def forward(self, x, dims=defaults):
            value = x.sum(dims.pop())
            return value.reshape(123) if fail else value

    if keyword_only:

        def forward(self, x, *, dims=defaults):
            value = x.sum(dims.pop())
            return value.reshape(123) if fail else value

        Model.forward = forward
    model = Model()
    x = torch.ones(2, 4)
    args = (x, defaults) if supplied == "positional" and not keyword_only else (x,)
    kwargs = (
        {"dims": defaults}
        if supplied == "keyword" or (supplied == "positional" and keyword_only)
        else {}
    )
    rng = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(CaptureError):
            DependencyGraph.build(model, args=args, kwargs=kwargs)
    else:
        DependencyGraph.build(model, args=args, kwargs=kwargs)
    assert defaults == [-1]
    torch.testing.assert_close(x, torch.ones(2, 4))
    torch.testing.assert_close(torch.get_rng_state(), rng)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("nonleaf", [False, True])
def test_default_tensor_uses_the_same_copy_policy_as_explicit_inputs(
    supplied, nonleaf, execution_device
):
    state = torch.ones(4, requires_grad=nonleaf)
    if nonleaf:
        state = state * 2

    class Model(nn.Module):
        def forward(self, x, scratch=state):
            return x + scratch.zero_()

    model = Model()
    before = state.detach().clone()
    kwargs = {"scratch": state} if supplied else {}
    if nonleaf:
        with pytest.raises(CaptureError, match="copy"):
            DependencyGraph.build(model, args=(torch.ones(4),), kwargs=kwargs)
    else:
        DependencyGraph.build(model, args=(torch.ones(4),), kwargs=kwargs)
    torch.testing.assert_close(state, before)
