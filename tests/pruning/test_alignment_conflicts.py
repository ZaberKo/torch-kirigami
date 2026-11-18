"""Divisibility on a nonempty axis becomes irreversible below its smallest positive multiple."""

from itertools import combinations

import pytest
import torch
from torch import nn

from torch_kirigami import Balanced, DependencyGraph, Divisible, IndexSet
from torch_kirigami.pruning import (
    DynamicGreedy,
    Granularity,
    Greedy,
    Magnitude,
    ParameterBudget,
    PlanningError,
    Pruner,
)


def test_alignment_conflict_starts_below_the_least_common_multiple():
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("0.weight").axis(0)
    # Factors 2 and 3 require a multiple of 6, so only six or zero rows can remain.
    constraints = (Divisible(axis, 2), Divisible(axis, 3))
    for removed in range(9):
        impact = graph.propagate(remove=(axis.select(range(removed)),), constraints=constraints)
        codes = {d.code for d in impact.diagnostics if d.irreversible}
        retained = 8 - removed
        if retained == 0:
            assert codes == {"empty_axis"}
        elif retained < 6:
            assert codes == {"unreachable_alignment"}
            assert impact.status == "conflict"
        else:
            assert not codes
            assert impact.status == ("resolved" if retained == 6 else "unresolved")


def test_axes_narrower_than_their_factor_fail_before_trials_without_mutation():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph, granularity=Granularity(default=8))
    space = pruner.discover_candidates()
    assert not space.candidates
    reasons = dict(space.exclusions)
    assert "unreachable_alignment" in reasons["domain:0.weight:0"]
    assert "fixed_axis" in reasons["domain:2.weight:0"]
    with pytest.raises(PlanningError, match="empty request is invalid"):
        pruner.plan(space, budget=ParameterBudget(10), strategy=Greedy(Magnitude()))
    graph.validate()
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())


def _two_hidden_layers() -> nn.Module:
    """Return a model whose narrow hidden layer has the smallest L1 channel scores."""
    model = nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 8)
    ).eval()
    with torch.no_grad():
        for index, scale in ((0, 0.01), (2, 0.01), (3, 1.0), (5, 1.0)):
            layer = model[index]
            steps = torch.arange(layer.weight.numel(), dtype=torch.float32)
            layer.weight.copy_(scale * (1 + steps.reshape(layer.weight.shape) / 1000))
            layer.bias.fill_(scale)
    return model


def test_greedy_excludes_unreachable_alignment_seeds_without_trials():
    model = _two_hidden_layers()
    sample = torch.randn(2, 8)
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(default=8))
    space = pruner.discover_candidates(targets=("0", "3"))
    # The narrow layer can lose at most eight channels, so the wide layer must
    # supply the rest. Without the joint proof, each remaining narrow seed
    # exhausts the trial limit while completion scans every other candidate.
    plan = pruner.plan(
        space, budget=ParameterBudget.from_ratio(model, 0.25), strategy=Greedy(Magnitude(p=1), 6)
    )
    report = plan.selection_report
    assert report.after_params <= report.max_params
    assert not report.limit_reached
    narrow = {c.key for c in space.candidates if c.key.startswith("0.weight")}
    selected = set(plan.selected)
    assert len(selected & narrow) == 8
    reasons = dict(report.exclusions)
    assert narrow - selected
    assert all("unreachable_alignment" in reasons[key] for key in narrow - selected)
    pruned, _result = pruner.apply(plan)
    assert pruned[0].out_features == 8
    assert pruned[3].out_features % 8 == 0
    assert pruned(sample).shape == (2, 8)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("nonempty", [True, False])
@pytest.mark.parametrize("factor", [1, 2, 4, 6])
def test_balanced_alignment_conflicts_match_every_superset(factor, nonempty, tied):
    """Report a conflict exactly when no larger removal keeps an aligned balance."""
    model = nn.Sequential(nn.Linear(3, 9), nn.ReLU(), nn.Linear(9, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    rows = graph.parameter("0.weight").axis(0)
    columns = graph.parameter("2.weight").axis(1)
    # Position 8 lies outside both partitions and can keep any count.
    partitions = ((0, 1, 2), (3, 4, 5, 6, 7))
    constraints = (
        Divisible(rows, factor),
        Balanced(
            columns if tied else rows,
            tuple(IndexSet.of(list(p)) for p in partitions),
            nonempty=nonempty,
        ),
    )
    positions = range(9)

    def valid(retained):
        counts = [len(retained & set(p)) for p in partitions]
        return (
            bool(retained)
            and len(retained) % factor == 0
            and len(set(counts)) == 1
            and (not nonempty or min(counts) > 0)
        )

    for size in range(10):
        for removed in map(set, combinations(positions, size)):
            retained = set(positions) - removed
            reachable = any(
                valid(set(kept))
                for count in range(len(retained) + 1)
                for kept in combinations(sorted(retained), count)
            )
            # An emptied axis is reported by `NonEmpty` alone.
            emptied = not retained
            impact = graph.propagate(
                remove=(rows.select(sorted(removed)),), constraints=constraints
            )
            codes = {d.code for d in impact.diagnostics if d.irreversible}
            assert ("unreachable_alignment" in codes) == (not reachable and not emptied), removed


def _grouped(groups: int, wide: int) -> nn.Module:
    """Return a grouped bottleneck whose channels have the smallest L1 scores."""
    model = nn.Sequential(
        nn.Conv2d(8, 16 * groups, 1),
        nn.ReLU(),
        nn.Conv2d(16 * groups, 16 * groups, 3, padding=1, groups=groups),
        nn.ReLU(),
        nn.Conv2d(16 * groups, 8, 1),
        nn.Conv2d(8, wide, 1),
        nn.ReLU(),
        nn.Conv2d(wide, 8, 1),
    ).eval()
    with torch.no_grad():
        for index in (0, 2, 4):
            for parameter in model[index].parameters():
                parameter.mul_(0.001)
    return model


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_balanced_alignment_dead_ends_are_excluded_before_trials(strategy_type):
    """With three groups and factor 8, each group keeps a multiple of 8 channels."""
    torch.manual_seed(0)
    model = _grouped(3, 128)
    sample = torch.randn(1, 8, 4, 4)
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(default=8))
    space = pruner.discover_candidates(targets=("0", "5"))
    budget = ParameterBudget.from_ratio(model, 0.5)
    plan = pruner.plan(space, budget=budget, strategy=strategy_type(Magnitude(p=1), 40))
    report = plan.selection_report
    assert report.after_params <= report.max_params
    assert not report.limit_reached
    narrow = {c.key for c in space.candidates if c.key.startswith("0.weight")}
    assert any(
        key in narrow and "Balanced partitions" in reason for key, reason in report.exclusions
    )
    pruned, _result = pruner.apply(plan)
    assert pruned[0].out_channels == 24
    assert pruned[5].out_channels % 8 == 0
    assert pruned(sample).shape == (1, 8, 4, 4)


class _Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.narrow = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 8))
        self.wide = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.narrow(x), self.wide(x)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("tied", [False, True])
def test_factors_on_identity_equivalent_axes_combine(strategy_type, tied):
    model = _Branches().eval()
    with torch.no_grad():
        for branch, scale in ((model.narrow, 0.01), (model.wide, 1.0)):
            for parameter in branch.parameters():
                parameter.fill_(scale)
    sample = torch.ones(1, 8)
    graph = DependencyGraph.build(model, args=(sample,))
    rows = graph.parameter("narrow.0.weight").axis(0)
    columns = graph.parameter("narrow.2.weight").axis(1)
    # Producer rows and consumer columns share one width of 12, so multiples
    # of 4 and 6 leave only 12 or 0 channels, whichever axis declares them.
    constraints = (Divisible(rows, 4), Divisible(columns if tied else rows, 6))
    pruner = Pruner(model, graph=graph, constraints=constraints)
    space = pruner.discover_candidates(targets=("narrow.0", "wide.0"))
    reasons = dict(space.exclusions)
    assert not [c for c in space.candidates if c.axis == rows]
    assert "unreachable_alignment" in reasons["domain:narrow.0.weight:0"]
    budget = ParameterBudget(sum(p.numel() for p in model.parameters()) - 8 * 17)
    plan = pruner.plan(space, budget=budget, strategy=strategy_type(Magnitude(1), max_trials=8))
    assert plan.selection_report.after_params <= budget.max_params
    pruner.apply(plan)
    assert model.narrow[0].out_features == 12 and model.wide[0].out_features == 56
    assert [tuple(y.shape) for y in model(sample)] == [(1, 8), (1, 8)]
