import copy
import io

import pytest
import torch
from torch import nn

from tests.support.pruning import KeyStrategy, StaticMetric
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    CandidateSpace,
    ChannelCount,
    Greedy,
    Magnitude,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import (
    Constant,
    CumulativeChannelBudget,
    Linear,
    Piecewise,
    Polynomial,
    SelectionWindow,
    selection_similarity,
)


def model_and_space(width=7):
    model = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    return model, graph, Pruner(model, graph=graph).discover_candidates()


@pytest.mark.parametrize("scope", ["local", "global"])
def test_cumulative_integer_rounds_and_checkpoint_resume(scope):
    model, graph, space = model_and_space()
    cumulative = CumulativeChannelBudget(graph, space, scope=scope)
    for ratio, expected_width in ((0.3, 5), (0.5, 4), (0.7, 3)):
        budget = cumulative.budget(graph, space, ratio)
        plan = Pruner(model, graph=graph).plan(
            Pruner(model, graph=graph).discover_candidates(),
            budget=budget,
            strategy=Greedy(Magnitude()),
        )
        # State does not advance merely by planning.
        assert cumulative.budget(graph, space, ratio) == budget
        _, result = Pruner(model, graph=graph).apply(plan)
        graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
        space = Pruner(model, graph=graph).discover_candidates()
        cumulative.update(result, graph, space)
        assert model[0].out_features == expected_width
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored, _, _ = model_and_space()
    load_checkpoint(restored, stream)
    graph2 = DependencyGraph.build(restored, args=(torch.ones(2, 2),))
    space2 = Pruner(restored, graph=graph2).discover_candidates()
    resumed = CumulativeChannelBudget(graph2, space2, scope=scope)
    resumed.load_state_dict(cumulative.state_dict(), graph2, space2)
    assert resumed.budget(graph2, space2, 0.8).counts == (1 if scope == "global" else (1,))
    state = resumed.state_dict()
    bad = copy.deepcopy(state)
    bad["current"][0] += 1
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad, graph2, space2)
    assert resumed.state_dict() == state


def test_underfill_and_unrecorded_structure_do_not_advance_budget():
    model, graph, space = model_and_space()
    cumulative = CumulativeChannelBudget(graph, space)
    budget = cumulative.budget(graph, space, 0.5)
    # A valid alternative strategy deliberately underfills the cap.
    _, result = Pruner(model, graph=graph).prune(
        Pruner(model, graph=graph).discover_candidates(),
        budget=budget,
        strategy=KeyStrategy(lambda ctx: (ctx.candidates[0].key,)),
    )
    new_graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    new = Pruner(model, graph=new_graph).discover_candidates()
    with pytest.raises(ValueError):
        cumulative.budget(new_graph, new, 0.5)
    cumulative.update(result, new_graph, new)
    assert cumulative.budget(new_graph, new, 0.5).counts == (2,)
    state = cumulative.state_dict()
    with pytest.raises(ValueError):
        cumulative.update(result, new_graph, new)
    assert cumulative.state_dict() == state
    with pytest.raises(ValueError):
        cumulative.budget(
            new_graph, CandidateSpace((), (new_graph.parameter("2.weight").axis(1),)), 0.5
        )


def test_count_budget_local_global_constraints_and_validation():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(2, 4), nn.Linear(2, 4)

        def forward(self, x):
            return self.a(x), self.b(x)

    model = Branches()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    axes = (graph.parameter("a.weight").axis(0), graph.parameter("b.weight").axis(0))
    pruner = Pruner(model, graph=graph)
    local = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        CandidateSpace(
            candidates=Pruner(pruner.model, graph=pruner.graph, preserve_io=False)
            .discover_candidates()
            .candidates,
            channel_axes=axes,
        ),
        budget=ChannelCount((1, 2), axes),
        strategy=Greedy(Magnitude()),
    )
    assert local.selection_report.removed == (1, 2)
    global_plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        CandidateSpace(
            candidates=Pruner(pruner.model, graph=pruner.graph, preserve_io=False)
            .discover_candidates()
            .candidates,
            channel_axes=axes,
        ),
        budget=ChannelCount(2, axes, "global"),
        strategy=Greedy(
            StaticMetric(lambda ctx, batch: [0 if c.axis == axes[0] else 1 for c in batch])
        ),
    )
    assert global_plan.selection_report.removed == (2, 0)
    protected = pruner.plan(
        CandidateSpace(candidates=pruner.discover_candidates().candidates, channel_axes=axes),
        budget=ChannelCount((1, 2), axes),
        strategy=Greedy(Magnitude()),
    )
    assert protected.selection_report.removed == (0, 0)
    for counts, scope in [((-1, 1), "local"), ((1,), "local"), (9, "global"), (True, "global")]:
        with pytest.raises(ValueError):
            ChannelCount(counts, axes, scope)


def test_pure_schedules_endpoints_and_invalid_inputs():
    assert [Constant(2)(s) for s in (0, 4)] == [2, 2]
    linear = Linear(0, 1, 4, begin=2)
    assert [linear(s) for s in (0, 2, 3, 4, 9)] == [0, 0, 0.5, 1, 1]
    assert Polynomial(1, 0, 4, power=2)(2) == 0.75
    piecewise = Piecewise(((0, 0.1), (3, 0.5), (8, 1)))
    assert [piecewise(s) for s in (0, 2, 3, 9)] == [0.1, 0.1, 0.5, 1]
    for factory in (
        lambda: Linear(0, 1, 0),
        lambda: Constant(float("nan")),
        lambda: Polynomial(0, 1, 3, power=0),
        lambda: Piecewise(((1, 0),)),
    ):
        with pytest.raises(ValueError):
            factory()
    with pytest.raises(ValueError):
        linear(-1)


def test_selection_identity_window_resume_and_failure_purity():
    a, b = {"a": [0, 1], "b": []}, {"a": [1, 2], "b": []}
    assert selection_similarity(a, b) == pytest.approx(2 / 3)
    window = SelectionWindow(2)
    assert window.update(a) is None
    assert window.update(b) is None
    state = window.state_dict()
    resumed = SelectionWindow(2)
    resumed.load_state_dict(state)
    assert resumed.update(b) == pytest.approx(5 / 6)
    assert window.update(b) == pytest.approx(5 / 6)
    assert window.update(b) == resumed.update(b)
    before = window.state_dict()
    with pytest.raises(ValueError):
        window.update({"different": [1]})
    assert window.state_dict() == before
    bad = copy.deepcopy(before)
    bad["values"] = [float("nan")]
    with pytest.raises(ValueError):
        window.load_state_dict(bad)
    assert window.state_dict() == before
    window.reset()
    assert window.update(a) is None


@pytest.mark.parametrize("scope", ["local", "global"])
def test_cumulative_budget_retains_newly_protected_domain_and_restores(scope):
    model, graph, space = model_and_space(width=3)
    accounting = CumulativeChannelBudget(graph, space, scope=scope)
    pruner = Pruner(model, graph=graph)
    _, result = pruner.prune(
        pruner.discover_candidates(),
        budget=accounting.budget(graph, space, 0.9),
        strategy=Greedy(Magnitude()),
    )
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    compact = Pruner(model, graph=graph).discover_candidates()
    assert model[0].out_features == 1 and not compact.channel_axes
    assert graph.parameter("0.weight").axis(0) in compact.protected_channel_axes
    before = accounting.state_dict()
    # Explicitly dropping a domain has no IO-protection proof.
    with pytest.raises(ValueError, match="domains changed"):
        accounting.update(result, graph, CandidateSpace((), ()))
    assert accounting.state_dict() == before
    accounting.update(result, graph, compact)
    budget = accounting.budget(graph, compact, 0.99)
    assert budget.channel_axes == (graph.parameter("0.weight").axis(0),)
    plan = Pruner(model, graph=graph).plan(
        CandidateSpace(compact.candidates, budget.channel_axes),
        budget=budget,
        strategy=Greedy(Magnitude()),
    )
    assert plan.selection_report.widths == (1,) and plan.selection_report.removed == (0,)
    Pruner(model, graph=graph).apply(plan)
    model(torch.ones(2, 2)).sum().backward()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored, _, _ = model_and_space(width=3)
    load_checkpoint(restored, stream)
    fresh_graph = DependencyGraph.build(restored, args=(torch.ones(2, 2),))
    fresh = Pruner(restored, graph=fresh_graph).discover_candidates()
    resumed = CumulativeChannelBudget(fresh_graph, fresh, scope=scope)
    resumed.load_state_dict(accounting.state_dict(), fresh_graph, fresh)
    assert resumed.state_dict() == accounting.state_dict()
    assert resumed.budget(fresh_graph, fresh, 0.99).counts == (0 if scope == "global" else (0,))


@pytest.mark.parametrize(
    "start,end", [(1e308, -1e308), (-1e308, 1e308), (1.7e308, 1.6e308), (-1.7e308, -1.6e308)]
)
@pytest.mark.parametrize("power", [1, 2, 0.5])
def test_finite_extreme_schedule_endpoints_do_not_overflow(start, end, power):
    schedule = Polynomial(start, end, 12, begin=2, power=power)
    assert schedule(0) == schedule(2) == start
    assert schedule(12) == schedule(13) == end
    # Scale endpoints into a safe independent reference domain before interpolation.
    fraction = 0.5**power
    expected = ((1 - fraction) * (start / 1e308) + fraction * (end / 1e308)) * 1e308
    assert schedule(7) == pytest.approx(expected)
