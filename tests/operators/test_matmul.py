"""operators / matmul contracts."""

import copy

import pytest
import torch
from torch import nn

from tests.support.graph_helpers import removed
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
)


def test_einsum_and_addmm_independent_reference(execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 3))
            self.fc = nn.Linear(4, 6)
            self.bias = nn.Parameter(torch.randn(3))

        def forward(self, x):
            y = self.fc(x)
            return torch.einsum("...i,ij->...j", y, self.weight), torch.addmm(
                self.bias, y, self.weight
            )

    model = Net()
    old = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("fc.weight").axis(0).select([1, 4])]
        )
    )
    retained = old.fc(x)[:, [0, 2, 3, 5]] @ old.weight[[0, 2, 3, 5]]
    torch.testing.assert_close(model(x), (retained, retained + old.bias))


@pytest.mark.parametrize(
    "a,b,operation",
    [
        ((2, 3, 4), (2, 4, 5), torch.bmm),
        ((3, 4), (4, 5), torch.mm),
        ((2, 3, 4), (1, 4, 5), torch.matmul),
        ((4,), (4, 5), torch.matmul),
        ((3, 4), (4,), torch.matmul),
        ((4,), (4,), torch.matmul),
    ],
)
def test_matmul_contraction_and_batch_broadcast(a, b, operation, execution_device):
    class Model(nn.Module):
        def forward(self, x, y):
            return operation(x, y)

    graph = DependencyGraph.build(Model(), args=(torch.randn(a), torch.randn(b)))
    call = next(c for c in graph.calls() if len(c.inputs) == 2)
    impact = graph.propagate(remove=[call.input().axis(-1).select([1])])
    assert impact.status == "resolved"
    assert removed(impact, call.input(1), len(b) - 2 if len(b) > 1 else 0) == {1}
