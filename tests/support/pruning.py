"""support / pruning contracts."""

from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    Pruner,
)


def build(model, x, rules=None):
    graph = DependencyGraph.build(model, args=(x,), operators=rules)
    return graph, Pruner(model, graph=graph)
