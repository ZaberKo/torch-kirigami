"""operators / spatial contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: nn.MaxPool2d(2),
        lambda: nn.AvgPool2d(2),
        lambda: nn.AdaptiveAvgPool2d((2, 2)),
        lambda: nn.AdaptiveMaxPool2d((2, 2)),
        lambda: nn.Upsample(scale_factor=2, mode="nearest"),
        lambda: nn.ReflectionPad2d(1),
        lambda: nn.InstanceNorm2d(6, affine=True),
        lambda: nn.PReLU(6),
    ],
)
def test_channel_family_matches_independent_retained_channels(operation, execution_device):
    model = nn.Sequential(nn.Conv2d(3, 6, 1), operation())
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 6, 6)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("0.weight").axis(0).select([1, 4])]
        )
    )
    torch.testing.assert_close(model(x), original(x)[:, [0, 2, 3, 5]])


def test_padding_shift_cannot_masquerade_as_identity(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv1d(1, 4, 1, bias=False)
            self.b = nn.Conv1d(4, 1, 1, bias=False)

        def forward(self, x):
            return self.b(F.pad(self.a(x), (0, 0, -1, 1)))

    model = Model()
    with torch.no_grad():
        model.a.weight.copy_(torch.arange(1, 5).reshape(4, 1, 1))
        model.b.weight.copy_(torch.arange(1, 5).reshape(1, 4, 1) * 10)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 1, 1),))
    remove = [graph.parameter("a.weight").axis(0).select([1])]
    assert graph.propagate(remove=remove).status != "resolved"
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan_remove(remove)


@pytest.mark.parametrize("module_pad", [False, True])
def test_spatial_padding_keeps_channel_pruning(module_pad, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv1d(2, 4, 1)
            self.pad = nn.ConstantPad1d((1, 2), 0)
            self.out = nn.Conv1d(4, 2, 1)

        def forward(self, x):
            y = self.fc(x)
            return self.out(self.pad(y) if module_pad else F.pad(y, (1, 2)))

    model = Model()
    x = torch.randn(1, 2, 3)
    kept = [0, 2, 3]
    expected = F.conv1d(
        F.pad(model.fc(x)[:, kept], (1, 2)), model.out.weight[:, kept], model.out.bias
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("fc.weight").axis(0).select([1])])
    )
    torch.testing.assert_close(model(x), expected)
