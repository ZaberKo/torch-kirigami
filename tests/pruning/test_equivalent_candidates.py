"""Equivalent residual seeds reuse analysis without merging candidate decisions."""

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, Fixed, config
from torch_kirigami.dep_graph.solver import DependencySolver
from torch_kirigami.errors import StaleGraphError
from torch_kirigami.pruning import (
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    Magnitude,
    PlanningContext,
    Pruner,
    PruningPlan,
)
from torch_kirigami.pruning.candidate_checks import CandidateFilter


class ResidualBlock(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.c1 = nn.Conv2d(width, width, 1)
        self.c2 = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return x + self.c2(self.c1(x).relu())


class ResidualCNN(nn.Module):
    def __init__(self, width=4, blocks=2):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width, 2)

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))


def make_pruner(model, x):
    graph = DependencyGraph.build(model, args=(x,))
    return Pruner(model, graph=graph)


def test_filter_reuses_equal_residual_closures_and_latest_space(monkeypatch, execution_device):
    model = ResidualCNN().eval()
    pruner = make_pruner(model, torch.randn(1, 3, 2, 2))
    original = DependencySolver.propagate
    requests = []

    def counted(self, *, remove, constraints=()):
        requests.append(tuple(remove))
        return original(self, remove=remove, constraints=constraints)

    monkeypatch.setattr(DependencySolver, "propagate", counted)
    space = pruner.discover_candidates()
    assert len(space.candidates) == 20  # Five domains; no equivalent candidate is lost.
    assert len(requests) == 4  # Three residual domains, one closure per position.
    requests.clear()
    assert pruner.filter_candidates(space) is space
    assert requests == []
    # A new immutable space has no trusted discovery identity.
    pruner.filter_candidates(replace(space))
    assert len(requests) == 4
    requests.clear()
    protected = Pruner(
        model,
        graph=pruner.graph,
        constraints=(Fixed(pruner.graph.parameter("stem.weight").axis(0)),),
    )
    filtered = protected.filter_candidates(space)
    assert len(filtered.candidates) == 8
    assert len(space.candidates) == 20


def test_context_preserves_original_requests_provenance_and_custom_scores(execution_device):
    model = ResidualCNN().eval()
    pruner = make_pruner(model, torch.randn(1, 3, 2, 2))
    graph = pruner.graph
    space = pruner.discover_candidates()
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.25), space.channel_axes, pruner.constraints
    )
    trunk = tuple(c for c in space.candidates if c.axis.tensor.paths[0] == "stem.weight")
    equivalent = tuple(
        c for c in space.candidates if c.axis.tensor.paths[0] == "blocks.1.c2.weight"
    )
    for candidates in ((trunk[0],), (equivalent[0],), (trunk[0], equivalent[1])):
        seeds = tuple(s for c in candidates for s in c.remove)
        actual = context.impact(seeds)
        expected = graph.propagate(remove=seeds, constraints=pruner.constraints)
        assert actual.requested == seeds
        assert dict(actual.selections) == dict(expected.selections)
        assert actual.diagnostics == expected.diagnostics
        assert actual.requirements == expected.requirements
        reachable = {}
        for seed in seeds:
            previous = reachable.get(seed.tensor.id)
            reachable[seed.tensor.id] = seed if previous is None else previous.union(seed)
        for step in actual.provenance:
            assert not step.source.subtract(reachable[step.source.tensor.id])
            previous = reachable.get(step.target.tensor.id)
            reachable[step.target.tensor.id] = (
                step.target if previous is None else previous.union(step.target)
            )

    class KeyMetric:
        def score(self, context, candidates, *, accepted_impact):
            return tuple(1.0 if c.axis == trunk[0].axis else 2.0 for c in candidates)

    assert context.score(KeyMetric(), (trunk[0], equivalent[0])) == (1.0, 2.0)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("metric_type", [Magnitude, GroupMagnitude])
def test_cached_and_uncached_public_plan_apply_are_identical(
    strategy_type, metric_type, execution_device
):
    torch.manual_seed(7)
    source = ResidualCNN().eval()
    x = torch.randn(2, 3, 2, 2)
    settings = config.as_dict()
    plans, outputs, states = [], [], []
    try:
        for enabled in (False, True):
            config.update(
                candidate_cache_size=settings["candidate_cache_size"] if enabled else 0,
                impact_cache_size=settings["impact_cache_size"] if enabled else 0,
            )
            model = copy.deepcopy(source)
            pruner = make_pruner(model, x)
            space = pruner.discover_candidates()
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            plan = pruner.plan(
                space,
                budget=ChannelRatio(0.25, scope="global"),
                strategy=strategy_type(metric_type()),
            )
            assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
            pruner.apply(PruningPlan.from_dict(plan.to_dict()))
            plans.append(plan)
            outputs.append(model(x))
            states.append(model.state_dict())
        assert plans[0].selected == plans[1].selected
        assert plans[0].selection_report == plans[1].selection_report
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)
    finally:
        config.update(**settings)


def test_custom_constraints_never_reuse_native_rejection_proofs(execution_device):
    class CustomFixed(Fixed):
        pass

    model = ResidualCNN().eval()
    pruner = make_pruner(model, torch.randn(1, 3, 2, 2))
    axis = pruner.graph.parameter("stem.weight").axis(0)
    check = CandidateFilter(pruner.graph, (CustomFixed(axis),))
    assert not check.reuse_equivalent
    assert check.index.canonical_seeds((axis.select([0]),))


def test_rejection_cache_is_bounded_and_checks_freshness(execution_device):
    config.candidate_cache_size = 1
    model = ResidualCNN().eval()
    pruner = make_pruner(model, torch.randn(1, 3, 2, 2))
    space = pruner.discover_candidates()
    candidates = tuple(c for c in space.candidates if c.axis.tensor.paths[0] == "stem.weight")
    check = CandidateFilter(pruner.graph, pruner.constraints)
    assert check.rejections(candidates[0]) == ()
    assert len(check._rejections) == 1
    assert check.rejections(candidates[1]) == ()
    assert len(check._rejections) == 1
    model.train()
    with pytest.raises(StaleGraphError, match=r"changed|stale|mode"):
        check.rejections(candidates[1])
    with pytest.raises(StaleGraphError, match=r"changed|stale|mode"):
        pruner.filter_candidates(space)


def test_incomplete_equal_seeds_keep_original_propagation(monkeypatch, execution_device):
    class UnknownTail(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.second = nn.Linear(4, 2)

        def forward(self, x):
            return self.second(torch.roll(self.first(x), 1, -1))

    model = UnknownTail().eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3),))
    context = PlanningContext(graph, (), ChannelRatio(0.25), (), ())
    axis = graph.parameter("first.weight").axis(0)
    index = context._candidate_filter.index
    equivalent = index.equivalent_axes(axis)
    original = DependencySolver.propagate
    requested = []

    def counted(self, *, remove, constraints=()):
        requested.append(tuple(remove))
        return original(self, remove=remove, constraints=constraints)

    monkeypatch.setattr(DependencySolver, "propagate", counted)
    for target in equivalent:
        seeds = (target.select([0]),)
        impact = context.impact(seeds)
        assert not impact.complete
        assert impact.requested == seeds
        assert requested[-1] == seeds
    # Repeat a noncanonical request after the canonical incomplete cache was filled.
    seeds = (equivalent[-1].select([0]),)
    assert context.impact(seeds).requested == seeds
    assert requested[-1] == seeds


def test_parameter_groups_reuse_residual_analysis_without_deduplication(
    monkeypatch, execution_device
):
    model = ResidualCNN().eval()
    pruner = make_pruner(model, torch.randn(1, 3, 2, 2))
    space = pruner.discover_candidates()
    ordinary = DependencySolver.propagate
    requests = []

    def counted(self, *, remove, constraints=()):
        requests.append(tuple(remove))
        return ordinary(self, remove=remove, constraints=constraints)

    monkeypatch.setattr(DependencySolver, "propagate", counted)
    groups = pruner.parameter_groups(space.candidates)
    assert len(requests) == 4
    assert len(groups) == len(space.candidates) == 20
    assert tuple(group.key for group in groups) == tuple(c.key for c in space.candidates)
    assert len({id(group) for group in groups}) == 20
    for group, candidate in zip(groups, space.candidates, strict=True):
        impact = pruner.graph.propagate(remove=candidate.remove, constraints=pruner.constraints)
        assert group.selections == tuple(sorted(impact.parameters, key=lambda s: s.tensor.paths))
    # A caller filter remains observable once per candidate, in its original order.
    seen = []

    def keep(ref, parameter):
        seen.append(ref)
        return True

    filtered = pruner.parameter_groups(space.candidates, parameter_filter=keep)
    assert [group.selections for group in filtered] == [group.selections for group in groups]
    assert seen == [s.tensor for group in groups for s in group.selections]
