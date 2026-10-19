import pytest
import torch
from torch import nn

from tests.support.pruning import StaticMetric
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
    Magnitude,
    Pruner,
)


@pytest.mark.parametrize("width", [8, 1024])
def test_zero_budget_protected_head_does_not_scan_channels(width, monkeypatch, execution_device):
    model = nn.Linear(4, width)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4),))
    original = graph.propagate
    requests = []

    def counted(*args, **kwargs):
        requests.append(tuple(kwargs["remove"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "propagate", counted)
    plan = Pruner(model, graph=graph).plan(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0),
        strategy=Greedy(Magnitude(), max_trials=0),
    )
    assert len(requests) <= 3 and not any(requests)
    assert not plan.recipes and plan.selection_report.widths == ()
    before = model.weight
    Pruner(model).apply(plan)
    assert model.weight is before


def test_zero_budget_uses_explicit_space_without_rediscovery_or_scoring(
    monkeypatch, execution_device
):
    model = nn.Sequential(nn.Linear(4, 1024), nn.Linear(1024, 2))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4),))

    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()

    def unexpected(*args, **kwargs):
        pytest.fail("Zero-budget planning rediscovered candidates or requested scores")

    monkeypatch.setattr(pruner, "discover_candidates", unexpected)
    plan = pruner.plan(
        space,
        budget=ChannelRatio(0),
        strategy=Greedy(StaticMetric(unexpected)),
    )
    assert plan.selection_report.widths == (1024,) and plan.selection_report.removed == (0,)
    before = tuple(model.parameters())
    Pruner(model).apply(plan)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
