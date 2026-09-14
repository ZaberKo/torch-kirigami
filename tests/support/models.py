"""support / models contracts."""

import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    PlanningContext,
)


def _context(model=None):
    model = nn.Linear(4, 6) if model is None else model
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("one", (axis.select([1]),), axis)
    context = PlanningContext(
        graph, graph.operations(), (candidate,), ChannelRatio(0.5), (axis,), ()
    )
    return context, candidate


def chain():
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))


class WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)
        self.last = nn.Linear(6, 2)
        self.alias = self.fc
        self.register_buffer("offset", torch.randn(6), persistent=False)

    def forward(self, x):
        return self.last((self.fc(x) + self.offset).relu())


class CachedWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.cached = [{"weight": self.weight}]
        self.alias = self.cached
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(F.linear(x, self.cached[0]["weight"]))


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)
        self.route = [False]

    def forward(self, x):
        return self.b(x) if self.route[0] else self.b(self.a(x))
