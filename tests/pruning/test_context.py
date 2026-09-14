"""pruning / context contracts."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.support.models import _context
from torch_kirigami import (
    CandidateAxis,
    DependencyGraph,
    OperatorRegistry,
    OperatorRule,
    Selection,
)
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    WeightTaylor,
)


class FalseyFilter:
    def __bool__(self):
        return False

    def __call__(self, ref, parameter):
        return False


class FalseyStrategy:
    def __bool__(self):
        return False

    def __call__(self, context):
        return ()


def test_cache_hit_validates_full_reference_and_impact_owner():
    context, candidate = _context()
    context.impact(candidate.remove)
    altered = replace(candidate.remove[0].tensor, paths=("wrong",))
    invalid = Candidate("altered", (Selection(altered, candidate.remove[0].regions),))
    with pytest.raises(ValueError, match="altered"):
        context.impact(invalid.remove)
    with pytest.raises(ValueError, match="altered"):
        context.score(Magnitude(), (invalid,))
    context.compile(context.impact(()))
    other, _ = _context()
    with pytest.raises(ValueError, match="another graph"):
        context.compile(other.impact(()))


def test_context_premises_are_readonly():
    context, _ = _context()
    for name in (
        "graph",
        "operations",
        "candidates",
        "budget",
        "channel_axes",
        "constraints",
        "widths",
        "targets",
    ):
        with pytest.raises(AttributeError):
            setattr(context, name, None)
    context.trials = 1
    context.exclusions.append(("one", "test"))
    assert context.report(context.impact(())).trials == 1


@pytest.mark.parametrize("conflict", [None, "key", "block"])
def test_candidate_domains_use_axes_not_keys(conflict):
    registry = OperatorRegistry.default()
    base = registry.modules[nn.Linear]

    class DuplicateDomains(OperatorRule):
        def analyze(self, ctx):
            spec = base.analyze(ctx)
            domain = spec.candidates[0]
            if ctx.module_path == "a":
                twin = CandidateAxis(
                    domain.key if conflict == "key" else "duplicate",
                    ctx.binding("weight").axis(1) if conflict == "key" else domain.axis,
                    2 if conflict == "block" else domain.block_size,
                )
                return replace(spec, candidates=(domain, twin))
            return spec

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)
            self.b = nn.Linear(4, 6)

        def forward(self, x):
            return self.a(x), self.b(x)

    registry.modules[nn.Linear] = DuplicateDomains()
    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),), operators=registry)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    options = {
        "strategy": Greedy(
            lambda ctx, batch: [0 if c.axis.tensor.paths[0] == "b.weight" else 1 for c in batch]
        ),
        "budget": ChannelRatio(0.25, scope="global"),
    }
    if conflict:
        with pytest.raises(PlanningError, match="Conflicting"):
            pruner.discover_candidates()
    else:
        plan = pruner.plan(pruner.discover_candidates(), **options)
        assert plan.selection_report.widths == (6, 6)
        assert plan.selection_report.targets == (3,) and plan.selection_report.removed == (0, 3)


def test_falsey_strategy_and_parameter_filters_are_called():
    model = nn.Linear(4, 6)
    context, candidate = _context(model)
    plan = Pruner(model, graph=context.graph, preserve_io=False).plan(
        Pruner(model, graph=context.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.5),
        strategy=FalseyStrategy(),
    )
    assert plan.selected == ()
    # No gradient is needed because the supplied filter excludes all parameters.
    for metric in (
        Magnitude(parameter_filter=FalseyFilter()),
        WeightTaylor(parameter_filter=FalseyFilter()),
    ):
        assert metric(context, (candidate,)) == [0.0]
