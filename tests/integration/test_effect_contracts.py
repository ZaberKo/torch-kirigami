"""Capture and execution consume one allocation/effect contract."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Pruner


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("normalization", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_fresh_producers_support_inplace_consumers(
    spatial, functional, normalization, training, execution_device
):
    affine = (lambda a, b: nn.Conv2d(a, b, 1)) if spatial else nn.Linear
    norm = nn.BatchNorm2d if spatial else nn.BatchNorm1d
    call = F.conv2d if spatial else F.linear

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = affine(4, 6)
            self.norm = norm(6) if normalization else nn.Identity()
            self.last = affine(6, 2)

        def forward(self, x):
            y = call(x, self.first.weight, self.first.bias) if functional else self.first(x)
            if normalization:
                y = self.norm(y)
            return self.last(F.relu(y, inplace=True))

    model = Model().train(training)
    reference = copy.deepcopy(model)
    kept = [0, 2, 3, 4, 5]
    reference.first = affine(4, 5)
    reference.last = affine(5, 2)
    if normalization:
        reference.norm = norm(5)
    reference.train(training)
    with torch.no_grad():
        reference.first.weight.copy_(model.first.weight[kept])
        reference.first.bias.copy_(model.first.bias[kept])
        reference.last.weight.copy_(model.last.weight[:, kept])
        reference.last.bias.copy_(model.last.bias)
    x = torch.randn((3, 4, 3, 3) if spatial else (3, 4))
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("first.weight").axis(0).select([1])]
        )
    )
    y, expected = model(x), reference(x)
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.first.weight.grad, reference.first.weight.grad)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(("vector", "bias"), [(False, False), (True, False), (False, True)])
def test_functional_linear_vector_and_scalar_bias(batched, vector, bias, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn((4,) if vector else (3, 4)))
            self.bias = nn.Parameter(torch.tensor(0.5)) if bias else None

        def forward(self, x):
            return F.linear(x, self.w, self.bias)

    model = Model()
    x = torch.randn((2, 4) if batched else (4,))
    expected = F.linear(x[..., [0, 2, 3]], model.w[..., [0, 2, 3]], model.bias)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("w").axis(-1).select([1])]
        )
    )
    torch.testing.assert_close(model(x[..., [0, 2, 3]]), expected)
