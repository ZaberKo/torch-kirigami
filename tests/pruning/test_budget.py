"""pruning / budget contracts."""

import copy
import math
from itertools import pairwise

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import StaticMetric, build
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    PruningPlan,
)
from torch_kirigami.pruning.budget import channel_targets


@pytest.mark.parametrize("scope", ["local", "global"])
@pytest.mark.parametrize("ratio,expected", [(0.29, 29), (0.58, 58), (0.289, 29)])
def test_decimal_channel_ratio_matches_integer_plan_and_compact_reference(
    scope: str, ratio: float, expected: int, execution_device: str
) -> None:
    """Integral decimal targets survive public planning, serialization and apply."""
    hidden = (100,) if scope == "local" else (40, 60)
    widths = (4, *hidden, 3)
    model = nn.Sequential(*(nn.Linear(a, b) for a, b in pairwise(widths))).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    space = pruner.discover_candidates()
    before = tuple(model.parameters())
    strategy = Greedy(StaticMetric(lambda context, batch: [0.0] * len(batch)))
    counts = (100 - expected,) if scope == "local" else 100 - expected
    integer_plan = pruner.plan(
        space, budget=ChannelCount(counts, space.channel_axes, scope), strategy=strategy
    )
    plan = pruner.plan(space, budget=ChannelRatio(ratio, scope), strategy=strategy)
    assert plan.selection_report.targets == (100 - expected,)
    assert sum(plan.selection_report.removed) == expected
    assert plan.selection_report.shortfall == 0
    assert plan.selected == integer_plan.selected
    assert plan.recipes == integer_plan.recipes
    assert plan.attributes == integer_plan.attributes
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    torch.testing.assert_close(model(x), original(x))

    # Compute the Linear chain directly from the original weights and selected
    # seeds, independently of dependency propagation and recipe application.
    registered = {candidate.key: candidate for candidate in space.candidates}
    removed = {axis: set() for axis in space.channel_axes}
    for key in plan.selected:
        candidate = registered[key]
        removed[candidate.axis].update(candidate.remove[0].fully_selected_indices(0))
    kept = [list(range(4))]
    for i, width in enumerate(hidden):
        axis = graph.parameter(f"{i}.weight").axis(0)
        kept.append(sorted(set(range(width)) - removed[axis]))
    kept.append(list(range(3)))
    reference = x
    for i, layer in enumerate(original):
        reference = F.linear(
            reference, layer.weight[kept[i + 1]][:, kept[i]], layer.bias[kept[i + 1]]
        )
    compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert compact is model
    torch.testing.assert_close(compact(x), reference)
    compact(x).sum().backward()
    assert all(parameter.grad is not None for parameter in compact.parameters())


@pytest.mark.parametrize("scope", ["local", "global"])
def test_channel_ratio_floors_exact_decimal_values_without_epsilon(scope: str) -> None:
    """Check every hundredth and neighboring boundaries without allocating tensors."""
    widths = (100,) if scope == "local" else (40, 60)
    for expected in range(100):
        ratio = float(f"0.{expected:02d}")
        assert channel_targets(ChannelRatio(ratio, scope), widths) == (100 - expected,)
    assert channel_targets(ChannelRatio(math.nextafter(0.29, 0.0), scope), widths) == (71,)
    assert channel_targets(ChannelRatio(math.nextafter(0.29, 1.0), scope), widths) == (70,)
    assert channel_targets(ChannelRatio(1 / 3, scope), (6,)) == (4,)
    large = 10**30
    assert channel_targets(ChannelRatio(0.29, scope), (large,)) == (71 * 10**28,)


@pytest.mark.parametrize(
    "ratio", [-0.1, 1, 10**400, float("nan"), float("inf"), True, "0.29", None]
)
def test_channel_ratio_rejects_invalid_values(ratio: object) -> None:
    with pytest.raises(ValueError, match="ratio must be finite"):
        ChannelRatio(ratio)


def test_depthwise_candidate_denominator_and_custom_blocks():
    model = nn.Conv1d(4, 8, 1, groups=4)
    graph, pruner = build(model, torch.randn(2, 4, 5))
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.25),
        strategy=Greedy(Magnitude()),
    )
    assert plan.selection_report.widths == (8,) and plan.selection_report.removed == (2,)
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("pair", (axis.select([0, 1]),))
    with pytest.raises(TypeError):
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
            CandidateSpace(candidates=[candidate], channel_axes=None),
            budget=ChannelRatio(0.25),
            strategy=Greedy(Magnitude()),
        )
    manual = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        CandidateSpace(candidates=[candidate], channel_axes=(axis,)),
        budget=ChannelRatio(0.25),
        strategy=Greedy(Magnitude()),
    )
    assert manual.selection_report.widths == plan.selection_report.widths


def test_global_budget_no_hidden_local_cap():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return self.a(x), self.b(x)

    model = Branches()
    _graph, pruner = build(model, torch.randn(2, 4))

    @StaticMetric
    def metric(ctx, batch):
        return [0 if c.key.startswith("a.") else 100 for c in batch]

    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.25, scope="global"),
        strategy=Greedy(metric),
    )
    assert plan.selection_report.widths == (6, 6)
    assert plan.selection_report.removed == (3, 0)


def test_explicit_protected_axis_keeps_budget_baseline_and_seed_alias_dedup():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    before = tuple(model.parameters())
    with pytest.raises(PlanningError, match=r"remaining=\(6,\), limits=\(3,\)"):
        pruner.plan(
            CandidateSpace(
                candidates=pruner.discover_candidates().candidates, channel_axes=(axis, axis)
            ),
            budget=ChannelRatio(0.5),
            strategy=Greedy(Magnitude()),
        )
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    graph.validate()


def test_unknown_branch_underfill_keeps_denominator():
    class Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return torch.special.gammaln(self.a(x)), self.b(x)

    model = Unknown()
    _graph, pruner = build(model, torch.randn(2, 4))
    pruner = Pruner(model, graph=_graph, preserve_io=False)
    before = tuple(model.parameters())
    with pytest.raises(PlanningError, match=r"remaining=\(6, 3\), limits=\(3, 3\)"):
        pruner.plan(
            pruner.discover_candidates(), budget=ChannelRatio(0.5), strategy=Greedy(Magnitude())
        )
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    _graph.validate()
    # The same frozen global baseline is attainable through the supported branch.
    plan = pruner.plan(
        pruner.discover_candidates(),
        budget=ChannelCount(7, pruner.discover_candidates().channel_axes, "global"),
        strategy=Greedy(Magnitude()),
    )
    assert plan.selection_report.remaining == (6, 1)
    assert any("Incomplete" in reason for _, reason in plan.selection_report.exclusions)
