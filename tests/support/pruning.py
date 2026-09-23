"""support / pruning contracts."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch

from torch_kirigami import (
    DependencyGraph,
    Impact,
)
from torch_kirigami.pruning import (
    Candidate,
    MetricContext,
    PlanningContext,
    Pruner,
    StrategyResult,
)


def build(model, x, rules=None):
    graph = DependencyGraph.build(model, args=(x,), operators=rules)
    return graph, Pruner(model, graph=graph)


@dataclass
class StaticMetric:
    """A test score function intentionally independent of accepted selections."""

    function: Callable[[MetricContext, tuple[Candidate, ...]], Sequence[float] | torch.Tensor]

    def score(
        self, context: MetricContext, candidates: tuple[Candidate, ...], *, selected: Impact
    ) -> Sequence[float] | torch.Tensor:
        return self.function(context, candidates)


@dataclass
class KeyStrategy:
    """A test policy returning only keys, without extra selection diagnostics."""

    function: Callable[[PlanningContext], Iterable[str]]

    def select(self, context: PlanningContext) -> StrategyResult:
        return StrategyResult(tuple(self.function(context)))
