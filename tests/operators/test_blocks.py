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
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("0.weight").axis(0).select(remove)], preserve_io=False
    )
    torch.testing.assert_close(model(x), old(x)[:, keep_out])
