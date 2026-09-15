"""Bounded whole-model search preserves joint effects and executable results."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import Balanced, DependencyGraph, Divisible, Fixed, IndexSet, Region
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Granularity,
    Greedy,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
)


class IndependentBranches(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 32), nn.Linear(32, 2)) for _ in range(8)
        )

    def forward(self, x):
        return sum(block(x) for block in self.blocks)


@pytest.mark.parametrize("scope", ["local", "global"])
def test_whole_model_alignment_completes_without_scanning_unrelated_branches(
    scope, execution_device, monkeypatch
):
    model = IndependentBranches().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path={f"blocks.{i}.0": 8 for i in range(8)}),
    )
    space = pruner.discover_candidates()
    batches = []
    joint_sizes = []
    individual_queries = []
    propagate = graph.propagate

    def counted(**kwargs):
        remove = tuple(kwargs.get("remove", ()))
        if len(remove) > 1:
            joint_sizes.append(len(remove))
        elif remove:
            individual_queries.append(remove[0])
        return propagate(**{**kwargs, "remove": remove})

    monkeypatch.setattr(graph, "propagate", counted)

    def interleaved(context, batch):
        batches.append(tuple(c.key for c in batch))
        return [next(iter(c.remove[0].fully_selected_indices(0))) for c in batch]

    before = tuple(model.parameters())
    plan = pruner.plan(
        space, budget=ChannelRatio(0.25, scope=scope), strategy=Greedy(interleaved, max_trials=8)
    )
    report = plan.selection_report
    assert report.removed == (8,) * 8
    assert report.shortfall == 0 and not report.limit_reached
    assert report.trials == 8  # One aligned batch per branch, not 64 single-channel trials.
    # Check real dependency queries, not just the strategy's reported counter.
    # Scoring queries have one seed; joint queries must already contain whole batches.
    assert set(joint_sizes) == set(range(8, 65, 8))
    assert len(batches) == 1 and set(batches[0]) == {c.key for c in space.candidates}
    # A custom batch larger than the cache must not trigger a second eligibility scan.
    assert len(individual_queries) == len(space.candidates)
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    repeated = pruner.plan(
        space, budget=ChannelRatio(0.25, scope=scope), strategy=Greedy(interleaved, max_trials=8)
    )
    assert repeated.selected == plan.selected and repeated.selection_report == report
    restored = PruningPlan.from_dict(plan.to_dict())
    pruner.apply(restored)
    expected = sum(
        F.linear(F.linear(x, first.weight[8:], first.bias[8:]), last.weight[:, 8:], last.bias)
        for first, last in original.blocks
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize(
    ("width", "factors", "count"), [(64, (8,), 8), (66, (8,), 2), (24, (4, 6), 12)]
)
def test_constraints_construct_noncontiguous_batches_before_first_joint_trial(
    width, factors, count, execution_device
):
    model = nn.Sequential(nn.Linear(3, width), nn.Linear(width, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, constraints=[Divisible(axis, n) for n in factors])
    order = [*range(0, width, 2), *range(1, width, 2)]
    candidates = tuple(
        Candidate(f"channel_{i:03}", (axis.select([p]),)) for i, p in enumerate(order)
    )
    plan = pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=ChannelRatio(count / width),
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=1),
    )
    assert plan.selection_report.removed == (count,)
    assert plan.selection_report.trials == 1 and not plan.selection_report.limit_reached
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = sorted(set(range(width)) - set(order[:count]))
    expected = F.linear(
        F.linear(x, original[0].weight[keep], original[0].bias[keep]),
        original[1].weight[:, keep],
        original[1].bias,
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


def test_one_batch_satisfies_multiple_balance_and_alignment_constraints(execution_device):
    model = nn.Sequential(nn.Linear(3, 12), nn.Linear(12, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    constraints = [
        Balanced(axis, tuple(IndexSet.span(i, i + n) for i in range(0, 12, n))) for n in (6, 4)
    ]
    constraints.append(Divisible(axis, 6))
    pruner = Pruner(model, graph=graph, constraints=constraints)
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=1),
    )
    removed = set(plan.analysis.selection(axis.tensor).fully_selected_indices(0))
    assert len(removed) == 6 and plan.selection_report.trials == 1
    assert not plan.selection_report.limit_reached
    for size in (6, 4):
        assert all(
            len(removed.intersection(range(i, i + size))) == size // 2 for i in range(0, 12, size)
        )
    pruner.apply(plan)
    keep = sorted(set(range(12)) - removed)
    expected = F.linear(
        F.linear(x, original[0].weight[keep], original[0].bias[keep]),
        original[1].weight[:, keep],
        original[1].bias,
    )
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize(
    ("conv", "conv_fn", "spatial"),
    [(nn.Conv1d, F.conv1d, (3,)), (nn.Conv2d, F.conv2d, (3, 3)), (nn.Conv3d, F.conv3d, (2, 2, 2))],
)
@pytest.mark.parametrize("budget", [ChannelRatio(0.5), ParameterBudget(62)])
def test_grouped_convolution_batches_balance_different_local_input_columns(
    conv, conv_fn, spatial, budget, execution_device
):
    model = nn.Sequential(conv(3, 8, 1), conv(8, 8, 1, groups=2), conv(8, 2, 1)).double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, *spatial, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    # Delete local columns 0/1 in the first group and 2/3 in the second.
    order = [0, 1, 6, 7, 2, 3, 4, 5]
    candidates = tuple(Candidate(f"c{i}", (axis.select([p]),)) for i, p in enumerate(order))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 4}))
    plan = pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=budget,
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=1),
    )
    assert len(plan.analysis.selection(axis.tensor).fully_selected_indices(0)) == 4
    assert plan.selection_report.trials == 1
    assert not plan.selection_report.limit_reached
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    hidden = conv_fn(x, original[0].weight[[2, 3, 4, 5]], original[0].bias[[2, 3, 4, 5]])
    middle_weight = torch.cat((original[1].weight[:4, 2:], original[1].weight[4:, :2]))
    middle = conv_fn(hidden, middle_weight, original[1].bias, groups=2)
    expected = conv_fn(middle, original[2].weight, original[2].bias)
    torch.testing.assert_close(model(x), expected)
    assert sum(p.numel() for p in model.parameters()) == 4 * 3 + 4 + 8 * 2 + 8 + 2 * 8 + 2
    model(x).sum().backward()


@pytest.mark.parametrize("budget", [ChannelRatio(0.5), ParameterBudget(26)])
def test_depthwise_candidate_blocks_keep_channel_denominator_and_batch_alignment(
    budget, execution_device
):
    model = nn.Sequential(
        nn.Conv1d(3, 4, 1), nn.Conv1d(4, 8, 1, groups=4), nn.Conv1d(8, 2, 1)
    ).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 3)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"1": 4}))
    space = pruner.discover_candidates(targets=("1",))
    assert len(space.candidates) == 4
    plan = pruner.plan(
        space,
        budget=budget,
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=1),
    )
    axis = graph.parameter("1.weight").axis(0)
    assert len(plan.analysis.selection(axis.tensor).fully_selected_indices(0)) == 4
    if isinstance(budget, ChannelRatio):
        assert plan.selection_report.widths == (8,) and plan.selection_report.removed == (4,)
    assert len(plan.selected) == 2 and plan.selection_report.trials == 1
    pruner.apply(plan)
    hidden = F.conv1d(x, original[0].weight[2:], original[0].bias[2:])
    depthwise = F.conv1d(hidden, original[1].weight[4:], original[1].bias[4:], groups=2)
    expected = F.conv1d(depthwise, original[2].weight[:, 4:], original[2].bias)
    torch.testing.assert_close(model(x), expected)
    assert sum(p.numel() for p in model.parameters()) == 26


def test_custom_constraint_subclass_receives_full_closure_not_axis_projection(execution_device):
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    bias = graph.parameter("0.bias")

    class FullClosureDivisible(Divisible):
        def check(self, selections):
            if selections.get(self.axis.tensor.id):
                assert bias.id in selections
            return super().check(selections)

    pruner = Pruner(model, graph=graph, constraints=[FullClosureDivisible(axis, 2)])
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=2),
    )
    assert plan.selection_report.removed == (2,) and plan.selection_report.trials == 2
    pruner.apply(plan)
    expected = F.linear(
        F.linear(x, original[0].weight[2:], original[0].bias[2:]),
        original[1].weight[:, 2:],
        original[1].bias,
    )
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("limit", [1, 20])
def test_invalid_count_batch_does_not_hide_legal_partner_or_commit_partial_request(
    limit, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.good = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
            self.protected = nn.Linear(3, 4)

        def forward(self, x):
            return self.good(x), self.protected(x)

    model = Model().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("good.0.weight").axis(0)
    protected = graph.parameter("protected.weight").axis(0)
    candidates = (
        Candidate("a_seed", (axis.select([0]),)),
        Candidate("b_bad_partner", (axis.select([1]), protected.select([0]))),
        Candidate("c_good_partner", (axis.select([2]),)),
    )
    pruner = Pruner(model, graph=graph, constraints=[Divisible(axis, 2), Fixed(protected)])
    before = tuple(model.parameters())
    plan = pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=ChannelRatio(0.5),
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=limit),
    )
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    if limit == 1:
        assert not plan.recipes and plan.selection_report.limit_reached
        assert any("fixed_axis" in reason for _, reason in plan.selection_report.exclusions)
        pruner.apply(plan)
        torch.testing.assert_close(model(x), original(x))
    else:
        assert plan.selected == ("a_seed", "c_good_partner")
        assert plan.selection_report.removed == (2,)
        pruner.apply(PruningPlan.from_dict(plan.to_dict()))
        expected = F.linear(
            F.linear(x, original.good[0].weight[[1, 3]], original.good[0].bias[[1, 3]]),
            original.good[1].weight[:, [1, 3]],
            original.good[1].bias,
        )
        torch.testing.assert_close(model(x), (expected, original.protected(x)))
    sum(y.sum() for y in model(x)).backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("ratio", [0.1, 0.2])
def test_completion_falls_back_to_candidate_with_only_joint_axis_effect(ratio, execution_device):
    model = nn.Linear(4, 10).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    weight = graph.parameter("weight")
    axis = weight.axis(0)
    # Neither half of row 0 removes an output channel alone. Only their union
    # does so; an empty single-candidate axis summary must not exclude the partner.
    left = weight.select([Region((IndexSet.of([0]), IndexSet.span(0, 2)))])
    right = weight.select([Region((IndexSet.of([0]), IndexSet.span(2, 4)))])
    assert not graph.propagate(remove=[right]).selection(weight).fully_selected_indices(0)
    candidates = (
        Candidate("a_left", (left,)),
        Candidate("b_right", (right,)),
        Candidate("c_full", (axis.select([1]),)),
    )
    pruner = Pruner(model, graph=graph, preserve_io=False, constraints=[Divisible(axis, 4)])
    space = CandidateSpace(candidates, (axis,))
    strategy = Greedy(lambda context, batch: [0] * len(batch), max_trials=3)
    if ratio == 0.1:
        before = model.weight, model.bias
        with pytest.raises(PlanningError, match="empty request"):
            pruner.plan(space, budget=ChannelRatio(ratio), strategy=strategy)
        assert model.weight is before[0] and model.bias is before[1]
        torch.testing.assert_close(model(x), original(x))
        return
    plan = pruner.plan(space, budget=ChannelRatio(ratio), strategy=strategy)
    assert plan.selected == ("a_left", "c_full", "b_right")
    assert plan.selection_report.removed == (2,) and not plan.selection_report.limit_reached
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    torch.testing.assert_close(model(x), F.linear(x, original.weight[2:], original.bias[2:]))
    model(x).sum().backward()


@pytest.mark.parametrize("scope", ["local", "global"])
def test_budget_prefilter_counts_overlapping_candidates_once(scope, execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(
        Candidate(key, (axis.select(indices),))
        for key, indices in (("a", [0, 1]), ("b", [1, 2]), ("c", [3]))
    )
    pruner = Pruner(model, graph=graph, constraints=[Divisible(axis, 4)])
    plan = pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=ChannelRatio(0.5, scope=scope),
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=3),
    )
    assert plan.selected == ("a", "b", "c")
    assert plan.selection_report.removed == (4,) and not plan.selection_report.limit_reached
    pruner.apply(plan)
    expected = F.linear(
        F.linear(x, original[0].weight[4:], original[0].bias[4:]),
        original[1].weight[:, 4:],
        original[1].bias,
    )
    torch.testing.assert_close(model(x), expected)
