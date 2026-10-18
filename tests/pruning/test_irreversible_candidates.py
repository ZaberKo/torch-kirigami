"""Irreversible proofs exclude candidates without changing query or score semantics."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import KeyStrategy
from torch_kirigami import DependencyGraph, Diagnostic, Fixed
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    Magnitude,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
)
from torch_kirigami.sparsity import GroupLasso


class SlicedBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.left = nn.Linear(2, 2)
        self.right = nn.Linear(2, 2)

    def forward(self, x):
        hidden = self.stem(x)
        return self.left(hidden[:, :2]), self.right(hidden[:, 2:])


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_partial_protection_filters_before_metrics_and_preserves_sparse_query(
    strategy_type, execution_device
):
    model = SlicedBranches().double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("stem.weight").axis(0)
    pruner = Pruner(
        model, graph=graph, constraints=(Fixed(graph.parameter("left.weight").axis(1)),)
    )
    raw = CandidateSpace(
        tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(4)), (axis,)
    )
    filtered = pruner.filter_candidates(raw)
    assert [c.key for c in filtered.candidates] == ["2", "3"]
    assert filtered.channel_axes == raw.channel_axes
    assert not filtered.excluded_channel_axes
    assert {key for key, _ in filtered.exclusions} == {"0", "1"}

    # The raw query may deliberately regularize protected structure.
    GroupLasso(pruner.parameter_groups(raw.candidates))().backward()
    assert model.left.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    GroupLasso(pruner.parameter_groups(filtered.candidates))().backward()
    assert model.left.weight.grad is None

    seen = []

    class Metric:
        def score(self, context, candidates, *, accepted_impact):
            for candidate in candidates:
                assert not context.impact(
                    (*accepted_impact.requested, *candidate.remove)
                ).irreversible
                seen.append(candidate.key)
            return [float(c.key) for c in candidates]

    plan = pruner.plan(
        raw, budget=ChannelRatio(0.25), strategy=strategy_type(Metric(), max_trials=1)
    )
    assert seen == ["2", "3"]
    assert plan.selected == ("2",)
    assert plan.selection_report.widths == (4,)
    assert plan.selection_report.trials == 1
    assert {key for key, _ in plan.selection_report.exclusions} == {"0", "1"}
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    hidden = F.linear(x, original.stem.weight[[0, 1, 3]], original.stem.bias[[0, 1, 3]])
    expected = (
        original.left(hidden[:, :2]),
        F.linear(hidden[:, 2:], original.right.weight[:, [1]], original.right.bias),
    )
    torch.testing.assert_close(model(x), expected)
    model.zero_grad(set_to_none=True)
    sum(result.sum() for result in model(x)).backward()


@pytest.mark.parametrize("grouped", [False, True])
def test_terminal_domains_are_excluded_without_rebasing_explicit_spaces(grouped):
    model = (
        nn.Sequential(nn.Conv2d(4, 2, 1, groups=2), nn.Conv2d(2, 3, 1))
        if grouped
        else nn.Sequential(nn.Linear(3, 1), nn.Linear(1, 2))
    ).eval()
    x = torch.randn(1, 4, 2, 2) if grouped else torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    candidates = tuple(
        Candidate(str(i), (axis.select([i]),), axis) for i in range(axis.tensor.shape[0])
    )
    assert all(pruner.impact((c,)).irreversible for c in candidates)
    assert len(pruner.parameter_groups(candidates)) == len(candidates)
    discovered = pruner.discover_candidates(targets=("0",))
    assert not discovered.candidates and not discovered.channel_axes
    assert discovered.excluded_channel_axes == (axis,)
    explicit = pruner.filter_candidates(CandidateSpace(candidates, (axis,)))
    assert not explicit.candidates and explicit.channel_axes == (axis,)
    original = copy.deepcopy(model)
    with pytest.raises(PlanningError, match="target not reached"):
        pruner.plan(explicit, budget=ChannelRatio(0.5), strategy=Greedy(Magnitude()))
    torch.testing.assert_close(model(x), original(x))
    graph.validate()


def test_custom_constraints_can_declare_irreversibility_without_registration():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("0.weight").axis(0)

    class KeepFirst:
        refs = (axis.tensor,)

        def check(self, selections):
            selection = selections.get(axis.tensor.id)
            if selection and 0 in selection.fully_selected_indices(0):
                return Diagnostic("keep_first", "Keep position zero", "conflict", irreversible=True)
            return None

    pruner = Pruner(model, graph=graph, constraints=(KeepFirst(),))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 3 and space.channel_axes == (axis,)
    assert len(space.exclusions) == 1
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude()))
    assert 0 not in plan.analysis.selection(axis.tensor).fully_selected_indices(0)
    pruner.apply(plan)
    assert model[0].out_features == 3


def test_changed_scalar_arguments_are_not_irreversible_and_joint_plan_can_execute(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(3, 4)
            self.b = nn.Linear(3, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            a, b = self.a(x), self.b(x)
            return self.head(a.add(a, alpha=a.size(-1) - b.size(-1) + 1))

    model = Model().double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("a", "b"))
    assert len(space.candidates) == 8
    assert all(pruner.impact((c,)).status == "conflict" for c in space.candidates)
    assert not any(pruner.impact((c,)).irreversible for c in space.candidates)
    assert pruner.filter_candidates(space).candidates == space.candidates
    assert len(pruner.parameter_groups(space.candidates)) == 8
    # Default greedy completion remains limited to its supported constraints.
    with pytest.raises(PlanningError, match="target not reached"):
        pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude()))
    torch.testing.assert_close(model(x), original(x))
    strategy = KeyStrategy(lambda context: (context.candidates[0].key, context.candidates[4].key))
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy)
    destination = tmp_path / "joint-plan.pt"
    torch.save(plan.to_dict(), destination)
    plan = PruningPlan.from_dict(torch.load(destination, weights_only=True))
    pruner.apply(plan)
    hidden = F.linear(x, original.a.weight[1:], original.a.bias[1:])
    expected = F.linear(hidden + hidden, original.head.weight[:, 1:], original.head.bias)
    assert model.a.out_features == model.b.out_features == 3
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


def test_fixed_output_width_does_not_protect_input_columns_or_freeze_training():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 5), nn.Linear(5, 2)).double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, constraints=(Fixed(graph.parameter("1.weight").axis(0)),))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 4
    GroupLasso(pruner.parameter_groups(space.candidates))().backward()
    assert model[1].weight.grad.abs().sum() > 0 and model[1].weight.requires_grad
    pruner.apply(pruner.plan_remove(space.candidates[0].remove))
    hidden = F.linear(x, original[0].weight[1:], original[0].bias[1:])
    hidden = F.linear(hidden, original[1].weight[:, 1:], original[1].bias)
    expected = F.linear(hidden, original[2].weight, original[2].bias)
    assert model[1].weight.shape == (5, 3)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_newly_irreversible_additions_do_not_spend_trials_or_change_static_scores(strategy_type):
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    space = pruner.discover_candidates(targets=("0",))
    scored_baselines, contexts = [], []

    class Metric:
        def score(self, context, candidates, *, accepted_impact):
            scored_baselines.append(accepted_impact.requested)
            return [0.0] * len(candidates)

    class ObservedStrategy:
        def select(self, context):
            contexts.append(context)
            return strategy_type(Metric()).select(context)

    with pytest.raises(PlanningError, match="target not reached"):
        pruner.plan(
            space, budget=ChannelCount((0,), space.channel_axes), strategy=ObservedStrategy()
        )
    assert contexts[0].trials == 1
    assert scored_baselines == [()]
    graph.validate()
    assert model[0].out_features == 2


def test_candidate_filter_does_not_change_group_magnitude_normalization_population():
    model = nn.Sequential(nn.Linear(1, 4), nn.Linear(4, 1)).double().eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        model[1].weight.zero_()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 1, dtype=torch.float64),))
    axis = graph.parameter("0.weight").axis(0)

    class KeepFirst:
        refs = (axis.tensor,)

        def check(self, selections):
            selection = selections.get(axis.tensor.id)
            if selection and 0 in selection.fully_selected_indices(0):
                return Diagnostic("keep_first", "Keep zero", "conflict", irreversible=True)
            return None

    pruner = Pruner(model, graph=graph, constraints=(KeepFirst(),))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 3
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.25), space.channel_axes, pruner.constraints
    )
    first = space.candidates[:1]  # Original position 1 has energy 4.
    assert context.score(GroupMagnitude(), first) == pytest.approx((4 / 7.5,))
    accepted = context.impact((axis.select([3]),))
    assert context.score(GroupMagnitude(), first, accepted_impact=accepted) == pytest.approx(
        (4 / (14 / 3),)
    )


def test_nonidentity_filtering_keeps_bounded_propagation_reuse(monkeypatch):
    class Transposed(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            return self.last(self.first(x).transpose(0, 1).transpose(0, 1))

    model = Transposed().double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("first",))
    calls, propagate = [], graph.propagate

    def counted(*args, **kwargs):
        calls.append(1)
        return propagate(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(graph, "propagate", counted)
        plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(GroupMagnitude()))
    # Two candidate passes: pre-strategy filtering and ordinary joint scoring.
    # Repeated rank, metric, and seed checks must reuse the context's cache.
    assert len(calls) <= 2 * len(space.candidates) + 2  # Empty request and final verification.
    assert plan.selection_report.trials == 1
    axis = space.channel_axes[0]
    removed = plan.analysis.selection(axis.tensor).fully_selected_indices(0)
    keep = [i for i in range(4) if i not in removed]
    expected = F.linear(
        F.linear(x, original.first.weight[keep], original.first.bias[keep]),
        original.last.weight[:, keep],
        original.last.bias,
    )
    pruner.apply(plan)
    torch.testing.assert_close(model(x), expected)
