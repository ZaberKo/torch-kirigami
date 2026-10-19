"""graph / queries contracts."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
    Fixed,
    StaleGraphError,
)


def test_metadata_and_capture_context_are_queryable():
    model = nn.Linear(4, 4).double()
    with torch.no_grad():
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4, dtype=torch.float64),))
    facts = graph.metadata(graph.parameter("weight"))
    assert facts.dtype == torch.float64 and facts.stride == (4, 1)
    assert not graph.context["grad_enabled"]
    assert graph.context["torch_version"] == torch.__version__
    assert graph.context["rule_snapshot"]


def test_graph_queries_reject_foreign_and_altered_references():
    graph = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    other = DependencyGraph.build(nn.LayerNorm(4), args=(torch.randn(2, 4),))
    operation, foreign = graph.operations()[0], other.operations()[0]
    assert operation.node.name == foreign.node.name
    assert graph.operator_rule(operation) is not None
    assert graph.operator_spec(operation).relations
    for query in (graph.operator_spec, graph.operator_rule):
        with pytest.raises(ValueError, match="another graph"):
            query(foreign)
        with pytest.raises(ValueError, match="altered"):
            query(replace(operation, module=nn.Linear(4, 4)))
    impact = graph.propagate(remove=[])
    with pytest.raises(ValueError, match="another graph"):
        graph.affected_operations(other.propagate(remove=[]))
    ref = graph.parameter("weight")
    for invalid in (other.parameter("weight"), replace(ref, shape=(5, 4))):
        with pytest.raises(ValueError):
            graph.metadata(invalid)
        with pytest.raises(ValueError):
            graph.propagate(remove=[], constraints=[Fixed(invalid.axis(0))])
        with pytest.raises(ValueError):
            impact.selection(invalid)
    assert not impact.selection(ref)
    with pytest.raises(TypeError):
        impact.tensors[ref.id] = replace(ref, shape=(5, 4))


def test_foreign_graph_selection_rejected():
    a = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    b = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    with pytest.raises(ValueError):
        b.propagate(remove=[a.parameter("weight").axis(0).select([1])])


def test_frozen_parameters_and_no_grad():
    model = nn.Linear(4, 4).requires_grad_(False)
    with torch.no_grad():
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    assert (
        graph.propagate(remove=[graph.parameter("weight").axis(0).select([1])]).status == "resolved"
    )


def test_public_graph_queries_are_isolated_and_alias_aware():
    model = nn.Linear(4, 6)
    graph, _ = build(model, torch.randn(2, 4))
    ref = graph.parameter("weight")
    assert graph.model is model and graph.tensor(ref) is model.weight
    assert graph.bindings(ref) == ((model, "weight"),)
    operations = graph.operations()
    operations[0].kwargs["bad"] = 1
    operations[0].node.target = "bad"
    assert not graph.operations()[0].kwargs
    assert graph.operations()[0].node.target != "bad"
    with pytest.raises(ValueError, match="different"):
        graph.validate(nn.Linear(4, 6))
    graph.invalidate()
    with pytest.raises(StaleGraphError):
        graph.tensor(ref)
