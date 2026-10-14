"""Repeated pure axis maps retain fresh-query scheduling, diagnostics and provenance."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.support.operators import make_rule, with_rule
from torch_kirigami import DependencyGraph, Divisible, Fixed, OperatorRegistry
from torch_kirigami.dep_graph.relations import AxisRelation


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_cached_joint_queries_match_full_recomputation(
    groups: int, execution_device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial groups, whole depthwise groups and constraint repairs retain exact explanations."""
    model = (
        nn.Sequential(nn.Conv2d(4, 8, 1, groups=groups), nn.ReLU(), nn.Conv2d(8, 4, 1))
        .to(execution_device)
        .eval()
    )
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4, 3, 3, device=execution_device),))
    calls = []
    propagate = AxisRelation.propagate

    def counted(relation, source):
        calls.append(relation)
        return propagate(relation, source)

    monkeypatch.setattr(AxisRelation, "propagate", counted)
    axis = graph.parameter("0.weight").axis(0)
    constraints = (Divisible(axis, 2),)
    queries = ([0], [1], [0, 1], [0, 1], [2, 3], [0, 1, 2, 3], [])
    reused = False
    for positions in queries:
        requests = (axis.select(positions),)
        calls.clear()
        cached = graph.propagate(remove=requests, constraints=constraints)
        cached_calls = len(calls)
        cacheable = graph._solver._cacheable_relations
        graph._solver._cacheable_relations = set()
        try:
            calls.clear()
            fresh = graph.propagate(remove=requests, constraints=constraints)
        finally:
            graph._solver._cacheable_relations = cacheable
        assert cached == fresh
        assert cached_calls <= len(calls)
        reused |= cached_calls < len(calls)
    assert reused  # Equality alone would also pass if caching were accidentally disabled.
    # Neither cached maps nor a previously resolved query can bypass new constraints.
    conflict = graph.propagate(remove=[axis.select([0, 1])], constraints=[Fixed(axis)])
    assert conflict.status == "conflict"
    assert any(d.code == "fixed_axis" for d in conflict.diagnostics)
    assert len(graph._solver._relation_outputs) <= 2 * len(graph.relations)


def test_axis_extensions_reexecute_and_limits_remain_incomplete() -> None:
    """Subclass behavior and failed reshape maps are never replaced by cached proofs."""
    calls = []

    class MeasuredRelation(AxisRelation):
        def propagate(self, source):
            calls.append(source)
            return super().propagate(source)

    class Flattened(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x).flatten()

    registry = OperatorRegistry.default()
    native = registry.modules[nn.Linear]

    def analyze(context):
        spec = native.build_spec(context)
        return replace(
            spec,
            relations=tuple(
                MeasuredRelation(r.left, r.right, r.maps, r.reason)
                if type(r) is AxisRelation
                else r
                for r in spec.relations
            ),
        )

    registry = with_rule(registry, nn.Linear, make_rule(analyze))
    graph = DependencyGraph.build(Flattened(), args=(torch.zeros(5_000, 3),), operators=registry)
    request = graph.parameter("fc.weight").axis(0).select([0])
    first = graph.propagate(remove=[request])
    previous = len(calls)
    second = graph.propagate(remove=[request])
    assert len(calls) == 2 * previous
    assert first == second and not second.complete
    assert any(d.code == "analysis_limit" for d in second.diagnostics)
    assert not graph._solver._relation_outputs
