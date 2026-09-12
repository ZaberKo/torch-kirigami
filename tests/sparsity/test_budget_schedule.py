import copy
import io

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    CandidateSpace,
    ChannelCount,
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
    space = CandidateSpace(DependencyGraph.build(model, args=(torch.ones(2, 2),)))
    return model, space


@pytest.mark.parametrize("scope", ["local", "global"])
def test_cumulative_integer_rounds_and_checkpoint_resume(scope):
    model, space = model_and_space()
    cumulative = CumulativeChannelBudget(space, scope=scope)
    for ratio, expected_width in ((0.3, 5), (0.5, 4), (0.7, 3)):
        budget = cumulative.budget(space, ratio)
        plan = Pruner(model, graph=space.graph).plan(metric=Magnitude(), budget=budget)
        # State does not advance merely by planning.
        assert cumulative.budget(space, ratio) == budget
        _, result = Pruner(model, graph=space.graph).apply(plan)
        space = CandidateSpace(DependencyGraph.build(model, args=(torch.ones(2, 2),)))
        cumulative.update(result, space)
        assert model[0].out_features == expected_width
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored, _ = model_and_space()
    load_checkpoint(restored, stream)
    space2 = CandidateSpace(DependencyGraph.build(restored, args=(torch.ones(2, 2),)))
    resumed = CumulativeChannelBudget(space2, scope=scope)
    resumed.load_state_dict(cumulative.state_dict(), space2)
    assert resumed.budget(space2, 0.8).counts == (1 if scope == "global" else (1,))
    state = resumed.state_dict()
    bad = copy.deepcopy(state)
    bad["current"][0] += 1
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad, space2)
    assert resumed.state_dict() == state


def test_underfill_and_unrecorded_structure_do_not_advance_budget():
    model, space = model_and_space()
    cumulative = CumulativeChannelBudget(space)
    budget = cumulative.budget(space, 0.5)
    # A valid alternative strategy deliberately underfills the cap.
    _, result = Pruner(model, graph=space.graph).prune(
        budget=budget,
        strategy=lambda ctx: (ctx.candidates[0].key,),
    )
    new = CandidateSpace(DependencyGraph.build(model, args=(torch.ones(2, 2),)))
    with pytest.raises(ValueError):
        cumulative.budget(new, 0.5)
    cumulative.update(result, new)
    assert cumulative.budget(new, 0.5).counts == (2,)
    state = cumulative.state_dict()
    with pytest.raises(ValueError):
        cumulative.update(result, new)
    assert cumulative.state_dict() == state
    with pytest.raises(ValueError):
        cumulative.budget(
            CandidateSpace(new.graph, axes=(new.graph.parameter("2.weight").axis(1),)), 0.5
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
    local = pruner.plan(metric=Magnitude(), budget=ChannelCount((1, 2), axes), preserve_io=False)
    assert local.budget.removed == (1, 2)
    global_plan = pruner.plan(
        metric=lambda ctx, batch: [0 if c.axis == axes[0] else 1 for c in batch],
        budget=ChannelCount(2, axes, "global"),
        preserve_io=False,
    )
    assert global_plan.budget.removed == (2, 0)
    protected = pruner.plan(metric=Magnitude(), budget=ChannelCount((1, 2), axes))
    assert protected.budget.removed == (0, 0)
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
    model, space = model_and_space(width=3)
    accounting = CumulativeChannelBudget(space, scope=scope)
    pruner = Pruner(model, graph=space.graph)
    _, result = pruner.prune(metric=Magnitude(), budget=accounting.budget(space, 0.9))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    compact = CandidateSpace(graph)
    assert model[0].out_features == 1 and not compact.axes
    assert graph.parameter("0.weight").axis(0) in compact.protected_axes
    before = accounting.state_dict()
    # Explicitly dropping a domain has no IO-protection proof.
    with pytest.raises(ValueError, match="domains changed"):
        accounting.update(result, CandidateSpace(graph, axes=()))
    assert accounting.state_dict() == before
    accounting.update(result, compact)
    budget = accounting.budget(compact, 0.99)
    assert budget.axes == (graph.parameter("0.weight").axis(0),)
    plan = Pruner(model, graph=graph).plan(metric=Magnitude(), budget=budget)
    assert plan.budget.widths == (1,) and plan.budget.removed == (0,)
    Pruner(model, graph=graph).apply(plan)
    model(torch.ones(2, 2)).sum().backward()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored, _ = model_and_space(width=3)
    load_checkpoint(restored, stream)
    fresh = CandidateSpace(DependencyGraph.build(restored, args=(torch.ones(2, 2),)))
    resumed = CumulativeChannelBudget(fresh, scope=scope)
    resumed.load_state_dict(accounting.state_dict(), fresh)
    assert resumed.state_dict() == accounting.state_dict()
    assert resumed.budget(fresh, 0.99).counts == (0 if scope == "global" else (0,))
