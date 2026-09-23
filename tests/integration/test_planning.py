"""Shared domains, constraint completion, opaque branches and IO budgets interact."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.numerics import _assert_value_and_input_gradient, _initialize
from tests.support.pruning import StaticMetric, build
from torch_kirigami import Balanced, Divisible, IndexSet
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Greedy,
    PlanningError,
    Pruner,
)


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.good = nn.Linear(4, 6, dtype=torch.float64)
        self.alias = self.good
        self.bad = nn.Linear(4, 6, dtype=torch.float64)
        self.out = nn.Linear(6, 2, dtype=torch.float64)

    def forward(self, x):
        return self.out(self.good(x) + self.alias(x)), torch.special.gammaln(self.bad(x).exp())


def test_shared_budget_constraint_completion_preserves_opaque_branch_and_io(execution_device):
    model = _initialize(Branches())
    original = copy.deepcopy(model)
    x = torch.linspace(-0.3, 0.8, 12, dtype=torch.float64).reshape(3, 4)
    graph, pruner = build(model, x)
    axes = {
        name: graph.parameter(f"{name}.weight").axis(0) for name in ("good", "alias", "bad", "out")
    }
    assert axes["good"] == axes["alias"]
    candidates = [
        Candidate(f"{name}:{i}", (axes[name].select([i]),), axes[name])
        for name in ("good", "bad", "out")
        for i in range(6 if name != "out" else 2)
    ]
    batches = []

    @StaticMetric
    def metric(context, batch):
        batches.append(tuple(candidate.key for candidate in batch))

        # Reusing Greedy's eligibility checks must not let a metric bypass
        # completeness when it explicitly scores a different temporary request.
        @StaticMetric
        def forbidden(context, batch):
            pytest.fail("An incomplete temporary candidate must not reach the metric")

        temporary = Candidate("temporary", [axes["bad"].select([0])])
        with pytest.raises(PlanningError, match="Incomplete scoring"):
            context.score(forbidden, [temporary])
        return [0.0] * len(batch)

    constraints = [
        Balanced(axes["good"], (IndexSet.span(0, 3), IndexSet.span(3, 6))),
        Divisible(axes["good"], 2),
    ]
    pruner = Pruner(model, graph=graph, constraints=constraints)
    space = CandidateSpace(candidates, tuple(axes.values()))
    plan = pruner.plan(space, budget=ChannelRatio(0.5, scope="global"), strategy=Greedy(metric))
    repeated = pruner.plan(space, budget=ChannelRatio(0.5, scope="global"), strategy=Greedy(metric))
    assert plan.selected == repeated.selected
    assert plan.selection_report.widths == (6, 6, 2)
    assert plan.selection_report.removed == (4, 0, 0) and plan.selection_report.shortfall == 3
    # IO conflicts are complete analyses and may be scored; the opaque branch
    # has incomplete influence and must be excluded before metric invocation.
    expected_batch = (*[f"good:{i}" for i in range(6)], "out:0", "out:1")
    assert len(batches) == 2 and batches[0] == batches[1] == expected_batch
    assert plan.analysis.status == "resolved"
    removed = set(plan.analysis.selection(axes["good"].tensor).fully_selected_indices(0))
    assert len(removed & {0, 1, 2}) == len(removed & {3, 4, 5}) == 2
    keep = sorted(set(range(6)) - removed)
    before_bad = model.bad.weight
    pruner.apply(plan)
    assert model.bad.weight is before_bad and model.good is model.alias
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    actual, unknown = model(a)
    reference = F.linear(
        2 * F.linear(b, original.good.weight[keep], original.good.bias[keep]),
        original.out.weight[:, keep],
        original.out.bias,
    )
    _assert_value_and_input_gradient(actual, reference, a, b)
    torch.testing.assert_close(unknown, original(x)[1], rtol=0, atol=0)
