"""operators / pointwise contracts."""

import copy
import operator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.models import Chain
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
)


def test_cast_reference_shape_is_not_a_broadcast_dependency(execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.register_buffer("dtype_reference", torch.zeros(11, dtype=torch.float64))

        def forward(self, x):
            return self.fc(x).type_as(self.dtype_reference).cpu()

    model = Net()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False
    )
    torch.testing.assert_close(model(x), old(x)[:, [0, 2, 3, 4, 5]])


def test_python_comparison_where_keeps_shared_structural_axes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return torch.where(y > 0, y**2, -y)

    model = Net()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False
    )
    torch.testing.assert_close(model(x), original(x)[:, [0, 2, 3, 4, 5]])


@pytest.mark.parametrize("op", [operator.floordiv, operator.mod])
def test_tensor_integer_arithmetic_propagates_channels(op, execution_device):
    class Model(Chain):
        def forward(self, x):
            return self.b(op(self.a(x), 2))

    model = Model()
    x = torch.randn(2, 4)
    kept = [0, 2, 3]
    expected = F.linear(op(model.a(x)[:, kept], 2), model.b.weight[:, kept], model.b.bias)
    graph = DependencyGraph.build(model, args=(x,))
    removal = graph.parameter("a.weight").axis(0).select([1])
    impact = graph.propagate(remove=[removal])
    assert list(impact.selection(graph.parameter("b.weight")).fully_selected_indices(1)) == [1]
    Pruner(model, graph=graph).prune(remove=[removal])
    torch.testing.assert_close(model(x), expected)
