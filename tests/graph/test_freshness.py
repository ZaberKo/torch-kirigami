"""graph / freshness contracts."""

import pytest
import torch
from torch import nn

from tests.support.models import CachedWeight, Chain
from torch_kirigami import (
    DependencyGraph,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    ExecutionError,
    Pruner,
    PruningPlan,
)


def test_static_loop_mode_and_freshness():
    class Static(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 4)
            self.count = 3
            self.enabled = True

        def forward(self, x):
            if self.enabled:
                for _ in range(self.count):
                    x = self.layer(x)
            return x

    model = Static()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    assert len(graph.calls("layer")) == 3
    with torch.no_grad():
        model.layer.weight.add_(1)
    graph.propagate(remove=[])
    model.count = 2
    with pytest.raises(StaleGraphError):
        graph.propagate(remove=[])


def test_training_mode_and_parameter_replacement_invalidate():
    for change in ("mode", "parameter"):
        model = nn.Linear(4, 4)
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
        if change == "mode":
            model.eval()
        else:
            model.weight = nn.Parameter(model.weight.detach().clone())
        with pytest.raises(StaleGraphError):
            graph.propagate(remove=[])


def test_cached_reference_changes_invalidate_plan():
    model = CachedWeight()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan_remove([graph.parameter("weight").axis(0).select([1])])
    model.cached[0]["weight"] = model.weight.detach().clone()
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError):
        Pruner(model).apply(plan)


def test_mixed_reference_container_guards_scalar_neighbors():
    model = CachedWeight()
    model.cached.append({"config": [True]})
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    model.cached[-1]["config"][0] = 1
    with pytest.raises(StaleGraphError):
        graph.validate()


@pytest.mark.parametrize("replacement", [1, 1.0])
def test_scalar_types_are_part_of_configuration_guards(replacement):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)
            self.flag = True

        def forward(self, x):
            return self.b(self.a(x)) if self.flag is True else self.b(x)

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    model.flag = replacement
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError):
        Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))


def test_list_configuration_is_frozen_and_guarded_in_graph_and_plan():
    model = Chain()
    model.config = ([1, [2]], (3,))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    serialized = plan.to_dict()
    model.route[0] = True
    model.config[0][1].append(4)
    assert plan.to_dict() == serialized
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError, match="preconditions"):
        Pruner(model).apply(PruningPlan.from_dict(serialized))
    assert model.b.weight.shape == (2, 4)
