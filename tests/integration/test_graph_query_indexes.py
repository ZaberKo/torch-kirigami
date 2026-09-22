"""Batched seeds and call indexes retain the full public query contract."""

import copy
import random
from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch
from torch import nn

import torch_kirigami.selection as selection_module
from torch_kirigami import (
    DependencyGraph,
    IndexSet,
    OperatorRegistry,
    OperatorRule,
    Region,
    Selection,
    ShapeExpr,
    TensorRef,
)
from torch_kirigami.errors import AnalysisLimitError
from torch_kirigami.operation import OperationContext, OperatorSpec
from torch_kirigami.pruning import Pruner


class SequentialSeedsGraph(DependencyGraph):
    """Use original ordered seed merging as a public planning reference."""

    def _seed_selections(self, requested: tuple[Selection, ...]) -> tuple[Selection, ...]:
        """Retain each supplied seed, including duplicates and empty selections."""
        return requested


class Branched(nn.Module):
    """Join two independently registered branches before an output projection."""

    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 8)
        self.right = nn.Linear(4, 8)
        self.output = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exercise residual alignment and a shared downstream consumer."""
        value = self.left(x) + self.right(x)
        return self.output(value.relu())


class ShapeConsumers(nn.Module):
    """Keep a shape-dependent reduction separate from an independently sliced branch."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.independent = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce layout consumers without propagating removed reduction coordinates."""
        value = self.hidden(x)
        reduced = value.sum(dim=-1).reshape(x.size(0), 1)
        return reduced.contiguous(), self.independent(x)[:, :1]


def exhaustive_affected(
    graph: DependencyGraph, selections: Mapping[str, Selection]
) -> frozenset[str]:
    """Reference the topological FX scan independently of the stored indexes."""
    affected = set()
    expressions = graph.shape_expressions
    for operation in graph.operations():
        inputs = operation.node.all_input_nodes
        direct = (*operation.inputs, *operation.outputs, *operation.bindings.values())
        if (
            any(ref.id in selections for ref in direct)
            or any(node.name in affected for node in inputs)
            or any(
                ref.id in selections
                for node in inputs
                if node.name in expressions
                for ref in expressions[node.name].refs
            )
        ):
            affected.add(operation.node.name)
    return frozenset(affected)


@pytest.mark.parametrize("model_type", [Branched, ShapeConsumers])
def test_affected_operation_index_matches_every_tensor_and_joint_queries(
    model_type: type[nn.Module],
) -> None:
    """Compare indexed reachability with complete topological scans."""
    graph = DependencyGraph.build(model_type().eval(), args=(torch.randn(3, 4),))
    refs = graph.values()
    rng = random.Random(13)
    queries = [(), *[(ref,) for ref in refs], refs]
    queries.extend(tuple(rng.sample(refs, min(5, len(refs)))) for _ in range(20))
    for query in queries:
        # Membership, rather than removed volume, is the activation contract.
        selections = {ref.id: Selection(ref) for ref in query}
        assert graph._affected_operations(selections) == exhaustive_affected(graph, selections)


def test_reduction_keeps_downstream_calls_without_removed_output_coordinates() -> None:
    """Retain downstream value and layout effects after a dimension is reduced."""
    graph = DependencyGraph.build(ShapeConsumers().eval(), args=(torch.randn(3, 4),))
    impact = graph.propagate(remove=[graph.parameter("hidden.weight").axis(0).select([1])])
    affected = graph.affected_operations(impact)
    assert {"hidden", "sum_1", "reshape", "contiguous"} <= affected
    assert "independent" not in affected
    reduction = next(call for call in graph.calls() if call.name == "sum_1")
    assert reduction.output().id not in impact.selections


@pytest.mark.parametrize("nested", [False, True])
def test_shape_expression_subclasses_keep_query_time_truthiness_and_references(
    nested: bool,
) -> None:
    """Do not freeze extension properties, including inside ordinary expressions."""
    enabled = [True]
    references = []

    class Expressions(nn.Module):
        """Expose separate branches connected only by an extension's expression refs."""

        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Linear(4, 8)
            self.right = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Give the right branch a size-expression consumer."""
            left, right = self.left(x), self.right(x)
            return left, right.reshape(x.size(0), 2)

    class DynamicExpression(ShapeExpr):
        """Model extension properties backed by query-time external state."""

        def __bool__(self) -> bool:
            return enabled[0]

        @property
        def refs(self) -> tuple[TensorRef, ...]:
            """Expose the left branch only while the extension is enabled."""
            return tuple(references) if enabled[0] else ()

    registry = OperatorRegistry.default()
    linear, size = registry.modules[nn.Linear], registry.methods["size"]

    def analyze_linear(context: OperationContext) -> OperatorSpec:
        """Capture the left parameter binding while retaining native analysis."""
        if context.module_path == "left":
            references.append(context.binding("weight"))
        return linear.analyze(context)

    def analyze_size(context: OperationContext) -> OperatorSpec:
        """Attach a direct or nested expression with dynamic extension properties."""
        expression = DynamicExpression("constant", 3)
        if nested:
            expression = ShapeExpr("add", args=(expression, ShapeExpr("constant", 0)))
        return replace(size.analyze(context), expression=expression)

    registry.modules[nn.Linear] = OperatorRule(analyze_linear, lower=linear.lower)
    registry.methods["size"] = OperatorRule(analyze_size, lower=size.lower)
    graph = DependencyGraph.build(
        Expressions().eval(), args=(torch.randn(3, 4),), operators=registry
    )
    seed = graph.parameter("left.weight").axis(0).select([1])
    for active in (False, True, False):
        enabled[0] = active
        impact = graph.propagate(remove=[seed])
        affected = graph.affected_operations(impact)
        assert ("reshape" in affected) is active
        assert affected == exhaustive_affected(graph, impact.selections)


@pytest.mark.parametrize("kind", ["rows", "columns", "fragmented", "multi_axis", "regions"])
@pytest.mark.parametrize("reverse", [False, True])
def test_batched_seeds_keep_requests_closure_provenance_and_diagnostics(
    kind: str, reverse: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare complete impacts against ordered merging for varied seed geometries."""
    graph = DependencyGraph.build(Branched().eval(), args=(torch.randn(3, 4),))
    left, right = graph.parameter("left.weight"), graph.parameter("right.weight")
    if kind == "rows":
        seeds = [left.axis(0).select([index]) for index in (1, 3, 1, 5)]
    elif kind == "columns":
        seeds = [left.axis(1).select([index]) for index in (0, 2, 0)]
    elif kind == "fragmented":
        seeds = [left.axis(0).select(indices) for indices in ([1, 5], [3, 7], [1, 3])]
    elif kind == "multi_axis":
        seeds = [left.axis(0).select([1, 3]), left.axis(1).select([0, 2])]
    else:
        region = Region((IndexSet.of([1, 3]), IndexSet.of([0, 2])))
        seeds = [left.select([region]), left.axis(0).select([5])]
    seeds.insert(1, right.axis(0).select([1]))
    seeds.insert(0, Selection(right))
    seeds.append(Selection(left))
    if reverse:
        seeds.reverse()
    actual = graph.propagate(remove=seeds)
    monkeypatch.setattr(graph, "_seed_selections", lambda requested: requested)
    expected = graph.propagate(remove=seeds)
    assert actual.requested == tuple(seeds)
    assert actual == expected


def test_seed_subclasses_keep_incremental_methods() -> None:
    """Extension selections must still receive their subtraction callbacks."""
    graph = DependencyGraph.build(Branched().eval(), args=(torch.randn(3, 4),))
    ref = graph.parameter("left.weight")
    calls = []

    class CheckedSelection(Selection):
        """Observe seed subtraction without changing selected coordinates."""

        def subtract(self, other: Selection) -> Selection:
            """Record the callback before applying the original set difference."""
            calls.append(self)
            return super().subtract(other)

    seed = ref.axis(0).select([1])
    custom = CheckedSelection(ref, seed.regions)
    assert graph._seed_selections((custom, seed)) == (custom, seed)
    graph.propagate(remove=(custom, seed))
    assert calls == [custom]


@pytest.mark.parametrize("foreign_first", [False, True])
@pytest.mark.parametrize("metadata", [False, True])
def test_seed_coalescing_preserves_invalid_reference_and_limit_failure_order(
    monkeypatch: pytest.MonkeyPatch, foreign_first: bool, metadata: bool
) -> None:
    """Report the original first failure when malformed references and limits coexist."""
    graph = DependencyGraph.build(Branched().eval(), args=(torch.randn(3, 4),))
    ref = graph.parameter("left.weight")
    invalid = replace(ref, paths=("altered",)) if metadata else TensorRef("foreign", ref.shape)
    bad = Selection(invalid)
    seeds = [ref.axis(0).select([index]) for index in (0, 2, 4, 1, 3)]
    seeds.insert(0 if foreign_first else len(seeds), bad)
    # The final union would fit one interval, but an earlier prefix exceeds two.
    monkeypatch.setattr(selection_module, "MAX_PARTS", 2)
    expected_error = ValueError if foreign_first else AnalysisLimitError
    with pytest.raises(expected_error):
        graph.propagate(remove=seeds)


def test_seed_coalescing_does_not_hide_intermediate_interval_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compact final union cannot erase an earlier representation-limit failure."""
    graph = DependencyGraph.build(Branched().eval(), args=(torch.randn(3, 4),))
    ref = graph.parameter("left.weight")
    seeds = [ref.axis(0).select([index]) for index in (0, 2, 4, 1, 3)]
    monkeypatch.setattr(selection_module, "MAX_PARTS", 2)
    with pytest.raises(AnalysisLimitError):
        graph.propagate(remove=seeds)


@pytest.mark.parametrize("grouped", [False, True])
def test_public_plans_and_applied_models_match_sequential_seeds(
    grouped: bool, execution_device: str
) -> None:
    """Compare portable plans, applied outputs and backward on CPU and CUDA."""
    if grouped:
        model = nn.Sequential(nn.Conv1d(4, 8, 1, groups=2), nn.Conv1d(8, 2, 1)).eval()
        sample = torch.randn(2, 4, 3)
        path = "0.weight"
    else:
        model, sample, path = Branched().eval(), torch.randn(3, 4), "left.weight"
    model, sample = model.to(execution_device), sample.to(execution_device)
    reference = copy.deepcopy(model)
    graphs = (
        DependencyGraph.build(model, args=(sample,)),
        SequentialSeedsGraph.build(reference, args=(sample,)),
    )
    pruners = [
        Pruner(module, graph=graph)
        for module, graph in zip((model, reference), graphs, strict=True)
    ]
    plans = [
        pruner.plan_remove([graph.parameter(path).axis(0).select([index]) for index in (1, 6)])
        for pruner, graph in zip(pruners, graphs, strict=True)
    ]
    assert plans[0].to_dict() == plans[1].to_dict()
    for pruner, plan in zip(pruners, plans, strict=True):
        pruner.apply(plan)
    torch.testing.assert_close(model(sample), reference(sample), rtol=0, atol=0)
    model(sample).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
