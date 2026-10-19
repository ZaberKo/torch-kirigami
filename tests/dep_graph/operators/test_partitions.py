"""operators / partitions contracts."""

import copy

import pytest
import torch
from torch import nn

from tests.support.pruning import build
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)


@pytest.mark.parametrize("legal", [True, False])
def test_split_original_ports(legal):
    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            return self.fc(x).split(4, dim=1)

    model = Split()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    remove = [graph.parameter("fc.weight").axis(0).select([5 if legal else 1])]
    if legal:
        plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(remove)
        pruner.apply(plan)
        a, b = model(x)
        torch.testing.assert_close(a, old(x)[0])
        torch.testing.assert_close(b, old(x)[1][:, :1])
    else:
        with pytest.raises(PlanningError, match="split"):
            Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(remove)


def test_dynamic_split_sizes_need_no_forward_edit():
    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            y = self.fc(x)
            return y.split(y.size(1) // 2, dim=1)

    model = Split()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(
        [graph.parameter("fc.weight").axis(0).select([1, 6])]
    )
    pruner.apply(plan)
    outputs = model(x)
    reference = old(x)
    torch.testing.assert_close(outputs[0], reference[0][:, [0, 2, 3]])
    torch.testing.assert_close(outputs[1], reference[1][:, [0, 1, 3]])
