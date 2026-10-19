"""Regularizer coordinate reuse preserves live gradients and snapshot validation."""

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, StaleGraphError
from torch_kirigami.pruning import ParameterGroup
from torch_kirigami.sparsity import (
    GroupLasso,
)


def test_cached_regularizer_indices_keep_live_gradients_and_freshness(
    execution_device, monkeypatch
):
    model = nn.Linear(2, 3, bias=False).double().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    axis = graph.parameter("weight").axis(0)
    groups = tuple(ParameterGroup(graph, (axis.select([i]),)) for i in (0, 2))
    with torch.inference_mode():
        regularizer = GroupLasso(groups)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[3.0, 4.0], [9.0, 9.0], [5.0, 12.0]]))

    def forbidden(*args, **kwargs):
        pytest.fail("Repeated regularization must reuse coordinate tensors")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", forbidden)
        value = regularizer()
        value.backward()
    assert value.item() == 18.0
    torch.testing.assert_close(
        model.weight.grad, model.weight.new_tensor([[0.6, 0.8], [0.0, 0.0], [5 / 13, 12 / 13]])
    )
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        regularizer()
    with torch.no_grad():
        model.weight[0, 0] = 3.0
    model.train()
    with pytest.raises(StaleGraphError):
        regularizer()


@pytest.mark.parametrize("change", ["dtype", "configuration", "binding"])
def test_cached_indices_do_not_replace_snapshot_validation(change, execution_device):
    model = nn.Linear(2, 3).float().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    regularizer = GroupLasso((group,))
    old = id(model.weight), model.weight._version
    if change == "dtype":
        model.double()
        assert (id(model.weight), model.weight._version) == old
    elif change == "configuration":
        model.out_features = 99
        assert (id(model.weight), model.weight._version) == old
    else:
        model.weight = nn.Parameter(model.weight.detach().clone())
    with pytest.raises(StaleGraphError):
        regularizer()
