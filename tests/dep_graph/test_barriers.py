"""graph / barriers contracts."""

import torch
from torch import nn
from torch.nn import functional as F

from tests.support.models import Chain
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    Pruner,
)


def test_unknown_branch_only_blocks_related_queries():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)

        def forward(self, x):
            return torch.special.gammaln(self.a(x)), self.b(x)

    graph = DependencyGraph.build(Branch(), args=(torch.randn(2, 4),))
    assert graph.diagnostics
    assert (
        graph.propagate(remove=[graph.parameter("a.weight").axis(0).select([1])]).status
        == "unresolved"
    )
    assert (
        graph.propagate(remove=[graph.parameter("b.weight").axis(0).select([1])]).status
        == "resolved"
    )


def test_unused_empty_buffer_does_not_block_hidden_pruning(execution_device):
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 3))
    model.register_buffer("unused", torch.empty(0))
    empty = model.unused
    graph = DependencyGraph.build(model, args=(torch.ones(1, 4),))
    assert graph.propagate(remove=[]).status == "resolved"
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("0.weight").axis(0).select([1])])
    )
    assert model.unused is empty and model[0].out_features == 5


def test_unbound_index_guard_only_blocks_its_dependency_component():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.index = torch.tensor([0, 1])
            self.a = nn.Linear(4, 6)
            self.independent = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))

        def forward(self, x):
            return self.a(x).index_select(1, self.index), self.independent(x)

    model = Model()
    x = torch.randn(2, 4)
    expected = model(x)[0]
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("independent.0.weight").axis(0).select([5])]
        )
    )
    torch.testing.assert_close(model(x)[0], expected)
    assert model.independent[0].out_features == 5


def test_padding_barrier_preserves_independent_branch(execution_device):
    class Model(Chain):
        def forward(self, x):
            return self.b(self.a(x)), F.pad(x, (-1, 1))

    model = Model()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    )
    assert model.b.in_features == 3
    torch.testing.assert_close(model(x)[1], F.pad(x, (-1, 1)))
