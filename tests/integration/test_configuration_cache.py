"""Equivalent configuration checks are bounded and local to one planning context."""

import pytest
import torch
from torch import fx, nn

from tests.support.pruning import KeyStrategy
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import ChannelRatio, ExecutionError, PlanningError, Pruner


@pytest.mark.parametrize("dependent", [False, True])
def test_same_width_configuration_checks_are_shared(dependent, monkeypatch, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 8)
            self.last = nn.Linear(8, 2)

        def forward(self, x):
            y = self.last(self.first(x))
            return y * self.first.out_features if dependent else y

    model = Model()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    original = fx.Tracer.trace
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(fx.Tracer, "trace", counted)

    @KeyStrategy
    def strategy(context):
        for candidate in context.candidates:
            impact = context.impact(candidate.remove)
            if dependent:
                with pytest.raises(PlanningError, match="structure"):
                    context.compile(impact)
            else:
                context.compile(impact)
        return [] if dependent else [context.candidates[0].key]

    plan = Pruner(model, graph=graph).plan(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.25),
        strategy=strategy,
    )
    # A final independently validated context is retained for accepted changes.
    assert len(calls) == (1 if dependent else 2)
    Pruner(model, graph=graph).apply(plan)
    model(x).sum().backward()


def test_configuration_cache_invalidates_after_tensor_write(monkeypatch, execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    calls = []
    original = fx.Tracer.trace

    def counted(self, *args, **kwargs):
        calls.append(None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(fx.Tracer, "trace", counted)

    @KeyStrategy
    def strategy(context):
        candidate = context.candidates[0]
        impact = context.impact(candidate.remove)
        context.compile(impact)
        with torch.no_grad():
            model[0].weight.add_(1)
        context.compile(impact)
        assert len(calls) == 2
        return []

    # User callbacks must not train while planning; invalidation also prevents
    # stale reuse before the public callback-exit guard reports that violation.
    with pytest.raises(ExecutionError, match="changed since planning"):
        Pruner(model, graph=graph).plan(
            Pruner(model, graph=graph).discover_candidates(),
            budget=ChannelRatio(0.25),
            strategy=strategy,
        )
    assert model[0].weight.shape == (8, 4) and model[1].weight.shape == (2, 8)
