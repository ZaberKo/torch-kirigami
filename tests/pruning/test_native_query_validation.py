"""Query batching preserves public freshness checks and extension boundaries."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torch_kirigami.dep_graph.dependency_graph as graph_module
from torch_kirigami import (
    AxisBarrier,
    Balanced,
    DependencyGraph,
    Divisible,
    Fixed,
    IndexSet,
    OperatorRegistry,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
)
from torch_kirigami.sparsity import GroupLasso


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_user_score_hooks_keep_full_checks_inside_greedy(strategy_type):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    metric = Magnitude()

    def custom(context, candidates, *, accepted_impact):
        model[0].out_features = 99
        with pytest.raises(StaleGraphError):
            context.impact(accepted_impact.requested)
        model[0].out_features = 4
        return [float(i) for i in range(len(candidates))]

    metric.score = custom
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(metric))
    pruner.apply(plan)
    assert model[0].out_features == 3


def test_attribute_retracing_suspends_native_query_reuse(monkeypatch):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    original = graph_module._validate_attribute_changes
    calls = []

    def retrace(*args, **kwargs):
        calls.append(1)
        model[0].out_features = 99
        with pytest.raises(StaleGraphError):
            graph.validate()
        model[0].out_features = 4
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_module, "_validate_attribute_changes", retrace)
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude()))
    assert calls
    pruner.apply(plan)
    assert model(torch.randn(2, 3)).shape == (2, 2)


def test_projected_groups_preserve_conflicts_and_reject_stale_mode():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("0.weight").axis(0)
    raw = Pruner(model, graph=graph).discover_candidates(targets=("0",)).candidates
    pruner = Pruner(model, graph=graph, constraints=(Fixed(axis),))
    groups = pruner.parameter_groups((raw[0], raw[0]))
    assert len(groups) == 2 and groups[0].equivalent(groups[1])
    assert pruner.impact((raw[0],)).irreversible
    foreign = DependencyGraph.build(nn.Linear(3, 4).eval(), args=(torch.randn(2, 3),))
    bad_axis = foreign.parameter("weight").axis(0)
    with pytest.raises(ValueError, match="another graph"):
        pruner.parameter_groups((Candidate("foreign", (bad_axis.select([0]),), bad_axis),))
    model.train()
    with pytest.raises(StaleGraphError):
        pruner.parameter_groups(raw)


def test_reusing_a_native_rule_for_custom_forward_keeps_full_checks():
    owner = {}
    observed = []

    class CustomLinear(nn.Linear):
        def forward(self, x):
            if x.device.type == "meta":
                live = owner["model"][0]
                live.out_features = 99
                with pytest.raises(StaleGraphError):
                    owner["graph"].validate()
                live.out_features = 4
                observed.append(1)
            return super().forward(x)

    registry = OperatorRegistry.default()
    registry.register(CustomLinear, registry.modules[nn.Linear])
    model = nn.Sequential(CustomLinear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),), operators=registry)
    owner.update(model=model, graph=graph)
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude()))
    assert observed
    pruner.apply(plan)
    assert model[0].out_features == 3


@pytest.mark.parametrize("max_trials", [0, 10])
def test_native_query_reuse_ends_after_success_or_failure(max_trials):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    if max_trials:
        pruner.plan(
            space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude(), max_trials=max_trials)
        )
    else:
        with pytest.raises(PlanningError):
            pruner.plan(
                space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude(), max_trials=0)
            )
    graph.validate()
    model.train()
    with pytest.raises(StaleGraphError):
        graph.validate()


def test_graph_query_overrides_do_not_inherit_native_validation_reuse(monkeypatch):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    original = graph.propagate
    calls = []

    def query(*args, **kwargs):
        calls.append(1)
        model[0].out_features = 99
        with pytest.raises(StaleGraphError):
            graph.validate()
        model[0].out_features = 4
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "propagate", query)
    pruner.apply(pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude())))
    assert calls and model[0].out_features == 3


def test_nested_budget_extensions_keep_query_time_validation():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    calls = []

    class Scope(str):
        def __eq__(self, other):
            model.train()
            with pytest.raises(StaleGraphError):
                graph.validate()
            model.eval()
            calls.append(1)
            return super().__eq__(other)

        __hash__ = str.__hash__

    budget = ChannelRatio(0.25, scope=Scope("local"))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    pruner.apply(pruner.plan(space, budget=budget, strategy=Greedy(Magnitude())))
    assert calls and model[0].out_features == 3


def test_model_binding_accessors_keep_query_time_validation():
    owner, calls = {}, []

    class Model(nn.Sequential):
        def get_submodule(self, target):
            if owner:
                live = owner["model"]
                live.train()
                with pytest.raises(StaleGraphError):
                    owner["graph"].validate()
                live.eval()
                calls.append(1)
            return super().get_submodule(target)

    model = Model(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    owner.update(model=model, graph=graph)
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude()))
    pruner.apply(plan)
    assert calls and model[0].out_features == 3


@pytest.mark.parametrize("kind", ["fixed", "balanced", "divisible", "barrier"])
def test_shared_projection_checks_preserve_group_and_planning_contracts(kind, execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    constraints = {
        "fixed": Fixed(axis),
        "balanced": Balanced(axis, (IndexSet.span(0, 3), IndexSet.span(3, 6))),
        "divisible": Divisible(axis, 4),
        "barrier": AxisBarrier(axis, "Unknown channel semantics"),
    }
    pruner = Pruner(model, graph=graph, constraints=(constraints[kind],))
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in (0, 3, 0))
    single = pruner.impact((candidates[0],))
    assert single.status == ("conflict" if kind == "fixed" else "unresolved")

    if kind == "barrier":
        assert not single.complete
        with pytest.raises(PlanningError, match="Incomplete parameter group"):
            pruner.parameter_groups(candidates)
    else:
        assert single.complete
        groups = pruner.parameter_groups(candidates)
        assert [group.key for group in groups] == ["0", "3", "0"]
        expected = sum(
            torch.cat(
                (
                    original[0].weight[i],
                    original[0].bias[i : i + 1],
                    original[2].weight[:, i],
                )
            ).norm()
            for i in (0, 3, 0)
        )
        torch.testing.assert_close(GroupLasso(groups)(), expected)
        for group, candidate in zip(groups, candidates, strict=True):
            ordinary = graph.propagate(remove=candidate.remove, constraints=pruner.constraints)
            assert group.selections == ordinary.parameters

    if kind in ("fixed", "barrier"):
        with pytest.raises(PlanningError):
            pruner.plan_remove(candidates[0].remove)
        torch.testing.assert_close(model(x), original(x))
        graph.validate()
        return

    # Two removals resolve either balance or initially invalid divisibility.
    plan = pruner.plan_remove((*candidates[0].remove, *candidates[1].remove))
    pruner.apply(plan)
    keep = [1, 2, 4, 5]
    expected = F.linear(
        F.linear(x, original[0].weight[keep], original[0].bias[keep]).relu(),
        original[2].weight[:, keep],
        original[2].bias,
    )
    torch.testing.assert_close(model(x), expected)


def test_native_group_extraction_keeps_batched_validation(monkeypatch):
    model = nn.Sequential(nn.Linear(3, 64), nn.Linear(64, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    original = graph_module.fingerprint
    calls = []

    def fingerprint(current):
        calls.append(1)
        return original(current)

    monkeypatch.setattr(graph_module, "fingerprint", fingerprint)
    assert len(pruner.parameter_groups(space.candidates)) == 64
    assert 2 <= len(calls) <= 6
    model.train()
    with pytest.raises(StaleGraphError):
        graph.validate()


def test_functional_float_arguments_keep_batched_checks_and_exact_pruning(
    execution_device, monkeypatch
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(3, 16)
            self.output = nn.Linear(16, 2)

        def forward(self, x):
            return self.output(F.leaky_relu(self.hidden(x), negative_slope=0.125))

    model = Model().double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("hidden",))
    fingerprint = graph_module.fingerprint
    calls = []

    def measured(current):
        calls.append(1)
        return fingerprint(current)

    with monkeypatch.context() as patch:
        patch.setattr(graph_module, "fingerprint", measured)
        groups = pruner.parameter_groups(space.candidates)
    assert len(groups) == 16 and 2 <= len(calls) <= 6
    plan = pruner.plan_remove((graph.parameter("hidden.weight").axis(0).select([1]),))
    pruner.apply(plan)
    keep = [i for i in range(16) if i != 1]
    hidden = F.leaky_relu(
        F.linear(x, original.hidden.weight[keep], original.hidden.bias[keep]),
        negative_slope=0.125,
    )
    expected = F.linear(hidden, original.output.weight[:, keep], original.output.bias)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
