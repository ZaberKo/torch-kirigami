import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
)


@pytest.mark.parametrize("method", ["sin", "cos", "abs", "square", "neg", "clone"])
@pytest.mark.parametrize("functional", [False, True])
def test_allocating_registered_families_support_inplace_consumers(
    method, functional, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            y = self.first(x)
            y = getattr(torch, method)(y) if functional else getattr(y, method)()
            return self.last(y.relu_())

    model = Model()
    x = torch.randn(2, 4)
    keep = [0, 2, 3]
    y = F.linear(x, model.first.weight[keep], model.first.bias[keep])
    expected = F.linear(
        getattr(torch, method)(y).relu(), model.last.weight[:, keep], model.last.bias
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("first.weight").axis(0).select([1])]
        )
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
