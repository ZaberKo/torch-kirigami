"""operators / blocks contracts."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
)


@pytest.mark.parametrize(
    "operation,remove,keep_out",
    [
        (lambda: nn.GLU(dim=1), [1], [0, 2]),
        (lambda: nn.ChannelShuffle(2), [1], [0, 1, 4, 5]),
        (lambda: nn.PixelShuffle(2), [0], [1, 2]),
    ],
)
def test_block_families_share_coordinate_relations(operation, remove, keep_out, execution_device):
    op = operation()
    channels = 12 if isinstance(op, nn.PixelShuffle) else 6
    model = nn.Sequential(nn.Conv2d(3, channels, 1), op)
    old = copy.deepcopy(model)
    x = torch.randn(2, 3, 3, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("0.weight").axis(0).select(remove)]
        )
    )
    torch.testing.assert_close(model(x), old(x)[:, keep_out])


@pytest.mark.parametrize("channels,groups", [(4, 1), (4, 2), (8, 4), (12, 3)])
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_shuffle_grouped_maps_preserve_compact_coordinates(
    channels, groups, functional, reverse, execution_device
):
    class ShuffleChain(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Conv2d(3, channels, 1)
            self.shuffle = nn.ChannelShuffle(groups)
            self.last = nn.Conv2d(channels, 2, 1)

        def forward(self, x):
            hidden = self.first(x)
            shuffled = (
                torch.nn.functional.channel_shuffle(hidden, groups)
                if functional
                else self.shuffle(hidden)
            )
            return self.last(shuffled)

    model = ShuffleChain().eval()
    old = copy.deepcopy(model)
    x = torch.randn(2, 3, 3, 4)
    width = channels // groups
    keep_in = [i for i in range(channels) if i % width != 0]
    keep_out = list(range(groups, channels))
    # An independently assembled compact chain keeps original row/column order.
    hidden = torch.nn.functional.conv2d(x, old.first.weight[keep_in], old.first.bias[keep_in])
    expected = torch.nn.functional.conv2d(
        torch.nn.functional.channel_shuffle(hidden, groups),
        old.last.weight[:, keep_out],
        old.last.bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    call = next(
        op
        for op in graph.operations()
        if op.node.target in ("shuffle", torch.nn.functional.channel_shuffle)
    )
    assert len(graph.operator_spec(call).relations) == (4 if groups == 1 else 5)
    axis = (
        graph.parameter("last.weight").axis(1)
        if reverse
        else graph.parameter("first.weight").axis(0)
    )
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove((axis.select([0]),)))
    assert model.first.out_channels == channels - groups
    assert model.last.in_channels == channels - groups
    torch.testing.assert_close(model(x), expected)
