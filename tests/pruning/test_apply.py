"""pruning / apply contracts."""

import copy
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.models import CachedWeight, chain
from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    ChannelRatio,
    ExecutionError,
    Magnitude,
    PlanningError,
    Pruner,
)


def test_prune_wrapper_and_empty_plan():
    model = chain()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    p = Pruner(model, graph=graph)
    plan = p.plan(remove=[])
    assert p.apply(plan)[0] is model
    assert p.apply(plan)[0] is model
    returned, result = p.prune(remove=[graph.parameter("0.weight").axis(0).select([1])])
    assert returned is model and result.plan.analysis.status == "resolved"


def test_manual_chain_readonly_and_apply_inference(execution_device):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    x = torch.randn(2, 4)
    original = copy.deepcopy(model)
    graph, pruner = build(model, x)
    old = model[0].weight
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1, 4])])
    assert model[0].weight is old
    assert model[0].out_features == 6
    with pytest.raises(FrozenInstanceError):
        plan.selected = ()
    with torch.inference_mode():
        returned, result = pruner.apply(plan)
    assert returned is model
    assert result.parameter_map[old] is model[0].weight
    assert model[0].weight.grad is None and not model[0].weight.is_inference()
    keep = [0, 2, 3, 5]
    hidden = F.linear(x, original[0].weight[keep], original[0].bias[keep]).relu()
    reference = F.linear(hidden, original[2].weight[:, keep], original[2].bias)
    torch.testing.assert_close(model(x), reference)
    model(x).sum().backward()
    assert model[0].weight.grad is not None
    with pytest.raises(ExecutionError, match="preconditions"):
        pruner.apply(plan)
    with pytest.raises(StaleGraphError):
        graph.propagate(remove=[])


def test_io_protection_and_mutual_exclusion():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    remove = [graph.parameter("weight").axis(0).select([1])]
    with pytest.raises(PlanningError, match="fixed_axis"):
        pruner.plan(remove=remove)
    with pytest.raises(ValueError, match="mutually"):
        pruner.plan(remove=remove, metric=Magnitude())
    plan = pruner.plan(remove=remove, preserve_io=False)
    pruner.apply(plan)
    assert model.out_features == 5


def test_plan_freshness_owner_and_value_changes():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    plan = pruner.plan(remove=[graph.parameter("weight").axis(0).select([1])], preserve_io=False)
    with torch.no_grad():
        model.weight.add_(1)
    Pruner(model).apply(plan)
    assert model.out_features == 5


def test_allocation_and_commit_failures_restore(monkeypatch):
    from torch_kirigami.pruning import pruner as implementation

    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    graph, pruner = build(model, torch.randn(2, 4))
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1])])
    original = tuple(model.parameters())
    gather = implementation.gather_region

    def fail(*args):
        raise RuntimeError("allocation failure")

    monkeypatch.setattr(implementation, "gather_region", fail)
    with pytest.raises(RuntimeError, match="allocation"):
        pruner.apply(plan)
    assert all(a is b for a, b in zip(original, model.parameters(), strict=True))
    monkeypatch.setattr(implementation, "gather_region", gather)
    setter = nn.Module.__setattr__

    def fail_commit(owner, name, value):
        if owner is model[1] and name == "weight" and value is not original[2]:
            raise RuntimeError("commit failure")
        setter(owner, name, value)

    monkeypatch.setattr(nn.Module, "__setattr__", fail_commit)
    with pytest.raises(ExecutionError, match="restored"):
        pruner.apply(plan)
    assert all(a is b for a, b in zip(original, model.parameters(), strict=True))
    monkeypatch.setattr(nn.Module, "__setattr__", setter)
    pruner.apply(plan)


def test_plan_state_isolation_and_altered_copy_rejected():
    from dataclasses import replace

    model = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Dropout(), nn.Linear(6, 2))
    graph, pruner = build(model, torch.randn(2, 4))
    state = {name: t.clone() for name, t in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1])])
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert all(m.training for m in model.modules())
    for name, t in model.state_dict().items():
        torch.testing.assert_close(t, state[name])
    with pytest.raises(ExecutionError, match="cover"):
        pruner.apply(replace(plan, recipes=()))


def test_cached_reference_commit_failure_rolls_back(monkeypatch):
    model = CachedWeight()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("weight").axis(0).select([1])])
    weight, cached = model.weight, model.cached
    setter = CachedWeight.__setattr__

    def fail(self, name, value):
        if self is model and name == "alias":
            raise RuntimeError("Injected reference commit failure")
        setter(self, name, value)

    monkeypatch.setattr(CachedWeight, "__setattr__", fail)
    with pytest.raises(ExecutionError, match="Commit failed"):
        Pruner(model).apply(plan)
    assert model.weight is weight and model.cached is cached and model.alias is cached
    assert cached[0]["weight"] is weight


def test_final_accepted_compilation_is_revalidated(monkeypatch):
    from torch_kirigami.pruning import planner

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    original = planner.compile_recipes
    counts = {}

    def counted(graph, operations, impact, **kwargs):
        key = tuple((s.tensor.id, s.regions) for s in impact.requested)
        counts[key] = counts.get(key, 0) + 1
        return original(graph, operations, impact, **kwargs)

    monkeypatch.setattr(planner, "compile_recipes", counted)
    plan = Pruner(model, graph=graph).plan(budget=ChannelRatio(0.25), metric=Magnitude())
    # Search results are cached, but the final decision is independently compiled
    # under the original caller premises after the strategy callback returns.
    assert plan.recipes and max(counts.values()) == 2
    assert counts[()] == 1
