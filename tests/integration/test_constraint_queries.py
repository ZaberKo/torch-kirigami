"""Local constraint indexing preserves exhaustive checks and public planning results."""

import copy
from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    Balanced,
    DependencyGraph,
    Diagnostic,
    Divisible,
    Fixed,
    IndexSet,
    NonEmpty,
    OperatorRegistry,
    OperatorRule,
    Selection,
    StaleGraphError,
)
from torch_kirigami.contracts import Constraint
from torch_kirigami.operation import OperationContext, OperatorSpec
from torch_kirigami.pruning import ChannelRatio, Greedy, Magnitude, PlanningError, Pruner


class Parallel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Sequential(nn.Linear(3, 10), nn.Linear(10, 2))
        self.right = nn.Sequential(nn.Linear(3, 10), nn.Linear(10, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left(x) + self.right(x)


class ExhaustiveGraph(DependencyGraph):
    """Independent reference: evaluate every declared constraint on each closure."""

    def _constraint_diagnostics(
        self, selections: Mapping[str, Selection], extra: tuple[Constraint, ...]
    ) -> tuple[Diagnostic, ...]:
        return tuple(
            diagnostic
            for constraint in (*self.constraints, *extra)
            if (diagnostic := constraint.check(selections)) is not None
        )


@pytest.mark.parametrize("kind", ["parallel", "grouped", "depthwise"])
def test_indexed_queries_match_exhaustive_public_planning(kind, execution_device):
    if kind == "parallel":
        model, x = Parallel().eval(), torch.randn(2, 3)
    else:
        model = nn.Sequential(
            nn.Conv1d(3, 8, 1),
            nn.Conv1d(8, 8, 1, groups=2 if kind == "grouped" else 8),
            nn.Conv1d(8, 2, 1),
        ).eval()
        x = torch.randn(2, 3, 4)
    reference = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    exhaustive = ExhaustiveGraph.build(reference, args=(x,))
    pruner, other = Pruner(model, graph=graph), Pruner(reference, graph=exhaustive)
    plans = [
        item.plan(
            item.discover_candidates(),
            budget=ChannelRatio(0.25),
            strategy=Greedy(Magnitude()),
        )
        for item in (pruner, other)
    ]
    assert plans[0].to_dict() == plans[1].to_dict()
    for item, plan in zip((pruner, other), plans, strict=True):
        item.apply(plan)
    torch.testing.assert_close(model(x), reference(x))
    model(x).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["divisible", "balanced"])
def test_untouched_initial_constraint_failure_survives_until_completed(kind, execution_device):
    registry = OperatorRegistry.default()
    linear = registry.modules[nn.Linear]

    def analyze(context: OperationContext) -> OperatorSpec:
        spec = linear.analyze(context)
        if context.module_path != "left.0":
            return spec
        axis = context.binding("weight").axis(0)
        constraint = (
            Divisible(axis, 4)
            if kind == "divisible"
            else Balanced(axis, (IndexSet.span(0, 4), IndexSet.span(4, 10)))
        )
        return replace(spec, constraints=(*spec.constraints, constraint))

    registry.modules[nn.Linear] = OperatorRule(analyze, lower=linear.lower)
    model = Parallel().eval()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    failure = "indivisible_axis" if kind == "divisible" else "unbalanced_groups"
    unrelated = graph.parameter("right.0.weight").axis(0).select([1])
    for seeds in ((), (unrelated,), ()):
        impact = graph.propagate(remove=seeds)
        expected = tuple(
            dict.fromkeys(
                diagnostic
                for constraint in graph.constraints
                if (diagnostic := constraint.check(impact.selections)) is not None
            )
        )
        assert impact.diagnostics == expected
        assert any(diagnostic.code == failure for diagnostic in impact.diagnostics)
    bindings = tuple(model.parameters())
    with pytest.raises(PlanningError, match=failure):
        pruner.plan_remove([unrelated])
    assert all(a is b for a, b in zip(bindings, model.parameters(), strict=True))
    removed = (0, 1) if kind == "divisible" else (4, 5)
    request = graph.parameter("left.0.weight").axis(0).select(removed)
    assert graph.propagate(remove=[request]).status == "resolved"
    pruner.apply(pruner.plan_remove([request]))
    keep = [index for index in range(10) if index not in removed]
    expected = F.linear(
        F.linear(x, reference.left[0].weight[keep], reference.left[0].bias[keep]),
        reference.left[1].weight[:, keep],
        reference.left[1].bias,
    ) + reference.right(x)
    torch.testing.assert_close(model(x), expected)


def test_custom_constraint_subclasses_run_for_unrelated_queries_and_keep_order(execution_device):
    calls = []
    blocked = [True]
    registry = OperatorRegistry.default()
    linear = registry.modules[nn.Linear]

    class GlobalFixed(Fixed):
        def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
            calls.append("subclass")
            if blocked[0] and selections:
                return Diagnostic(
                    "global_subclass", "Global extension rejects the change", "conflict"
                )
            return None

    class GlobalConstraint:
        refs = ()

        def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
            calls.append("global")
            if blocked[0] and selections:
                return Diagnostic("global_check", "Global extension rejects the change", "conflict")
            return None

    def analyze(context: OperationContext) -> OperatorSpec:
        spec = linear.analyze(context)
        if context.module_path != "left.0":
            return spec
        return replace(
            spec,
            constraints=(
                *spec.constraints,
                GlobalFixed(context.binding("weight").axis(0)),
                GlobalConstraint(),
            ),
        )

    registry.modules[nn.Linear] = OperatorRule(analyze, lower=linear.lower)
    model = Parallel().eval()
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    assert calls == []  # Index preparation must never run an extension callback.
    request = graph.parameter("right.0.weight").axis(0).select([1])
    for seeds in ((request,), (), (request,)):
        impact = graph.propagate(remove=seeds)
        assert [diagnostic.code for diagnostic in impact.diagnostics] == (
            ["global_subclass", "global_check"] if seeds else []
        )
    assert calls == ["subclass", "global"] * 3
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="global_subclass"):
        pruner.plan_remove([request])
    assert model.right[0].out_features == 10
    blocked[0] = False
    pruner.apply(pruner.plan_remove([request]))
    assert model(x).shape == (2, 2)
    previous = tuple(calls)
    with pytest.raises(StaleGraphError):
        graph.propagate(remove=())
    assert tuple(calls) == previous


def test_queries_check_only_touched_builtin_constraints(monkeypatch):
    model = Parallel().eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    original = NonEmpty.check
    checked = []

    def counted(self, selections):
        checked.append(self.axis.tensor.id)
        return original(self, selections)

    monkeypatch.setattr(NonEmpty, "check", counted)
    impact = graph.propagate(remove=[graph.parameter("left.0.weight").axis(0).select([1])])
    assert checked
    assert set(checked) <= set(impact.selections)
    assert len(checked) < sum(type(c) is NonEmpty for c in graph.constraints)
    checked.clear()
    assert graph.propagate(remove=()).status == "resolved"
    assert checked == []


def test_query_constraints_still_validate_unaffected_references():
    graph = DependencyGraph.build(nn.Linear(3, 4), args=(torch.randn(2, 3),))
    altered = replace(graph.parameter("weight"), paths=("wrong",))
    with pytest.raises(ValueError, match="altered metadata"):
        graph.propagate(remove=(), constraints=[Fixed(altered.axis(0))])
