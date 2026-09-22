"""Sparse axis summaries preserve resource checks and executable joint choices."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, Divisible, IndexSet, Region
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Greedy,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
)
from torch_kirigami.pruning.planner import _axis_removals


def test_axis_summary_omits_absent_and_partially_selected_axes() -> None:
    model = nn.Linear(4, 6).eval()
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 4),))
    weight, bias = graph.parameter("weight"), graph.parameter("bias")
    axes = (weight.axis(0), weight.axis(1), bias.axis(0))
    partial = weight.select([Region((IndexSet.of([0]), IndexSet.span(0, 2)))])
    assert _axis_removals(graph.propagate(remove=[partial]), axes) == {}
    complete = graph.propagate(remove=[weight.axis(0).select([1, 3])])
    assert _axis_removals(complete, axes) == {
        weight.axis(0): IndexSet.of([1, 3]),
        bias.axis(0): IndexSet.of([1, 3]),
    }


@pytest.mark.parametrize("cap", [44, 1])
def test_parameter_budget_skips_channel_counting_but_preserves_final_target_and_ownership(
    cap: int, execution_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    other_graph = DependencyGraph.build(original, args=(x,))
    foreign = other_graph.propagate(remove=())
    pruner = Pruner(model, graph=graph)
    parameters = tuple(model.parameters())

    def unexpected_channel_counts(self: PlanningContext, impact: object) -> tuple[int, ...]:
        pytest.fail("An absolute parameter target must not compute channel removal counts")

    def strategy(context: PlanningContext) -> tuple[str, ...]:
        with pytest.raises(ValueError, match="another graph"):
            context.admissible(foreign)
        return Greedy(lambda context, batch: [0] * len(batch))(context)

    monkeypatch.setattr(PlanningContext, "counts", unexpected_channel_counts)
    if cap == 1:
        with pytest.raises(PlanningError, match="Parameter target not reached"):
            pruner.plan(
                pruner.discover_candidates(), budget=ParameterBudget(cap), strategy=strategy
            )
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        torch.testing.assert_close(model(x), original(x))
        graph.validate()
        return
    plan = pruner.plan(pruner.discover_candidates(), budget=ParameterBudget(cap), strategy=strategy)
    assert plan.selection_report.after_params == 44
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    expected = nn.functional.linear(
        nn.functional.linear(x, original[0].weight[2:], original[0].bias[2:]),
        original[1].weight[:, 2:],
        original[1].bias,
    )
    torch.testing.assert_close(model(x), expected)
    assert sum(p.numel() for p in model.parameters()) == 44
    model(x).sum().backward()


@pytest.mark.parametrize("parameter_budget", [False, True])
def test_sparse_multi_axis_candidates_preserve_combination_and_replay(
    parameter_budget: bool, execution_device: str
) -> None:
    model = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 6), nn.Linear(6, 2)).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    first, second = graph.parameter("0.weight").axis(0), graph.parameter("1.weight").axis(0)
    candidates = (
        Candidate("a_first", (first.select([0]),)),
        Candidate("b_second", (second.select([0]),)),
        Candidate("c_both", (first.select([2]), second.select([4]))),
    )
    pruner = Pruner(model, graph=graph, constraints=(Divisible(first, 2), Divisible(second, 2)))
    plan = pruner.plan(
        CandidateSpace(candidates, (first, second)),
        budget=ParameterBudget(46) if parameter_budget else ChannelRatio(1 / 3),
        strategy=Greedy(lambda context, batch: [0] * len(batch), max_trials=1),
    )
    assert plan.selected == ("a_first", "c_both", "b_second")
    assert plan.selection_report.trials == 1
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    first_kept, second_kept = [1, 3, 4, 5], [1, 2, 3, 5]
    hidden = nn.functional.linear(x, original[0].weight[first_kept], original[0].bias[first_kept])
    middle = nn.functional.linear(
        hidden,
        original[1].weight[second_kept][:, first_kept],
        original[1].bias[second_kept],
    )
    expected = nn.functional.linear(middle, original[2].weight[:, second_kept], original[2].bias)
    torch.testing.assert_close(model(x), expected)
    assert sum(p.numel() for p in model.parameters()) == 46
    model(x).sum().backward()
