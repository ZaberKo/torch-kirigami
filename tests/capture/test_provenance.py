"""capture / provenance contracts."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.operators.shapes import CallArgumentConstraint


class SizeDivisor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)

    def forward(self, x):
        y = self.fc(x)
        return y / y.size(1)


def test_argument_constraints_do_not_export_capture_state(execution_device):
    model = SizeDivisor().to(execution_device)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4, device=execution_device),))
    request = [graph.parameter("fc.weight").axis(0).select([1])]
    impact = graph.propagate(remove=request)
    assert impact.status == "conflict"
    assert "changed_arguments" in {d.code for d in impact.diagnostics}
    for constraints in (graph.constraints, impact.constraints):
        constraint = next(c for c in constraints if isinstance(c, CallArgumentConstraint))
        assert not hasattr(constraint, "context")
        assert isinstance(constraint.arguments, tuple)
        with pytest.raises(FrozenInstanceError):
            constraint.arguments = ()
    for operation in graph.operations():
        with pytest.raises(AttributeError):
            operation.expressions.clear()
        operation.kwargs.clear()
        operation.bindings.clear()
        operation.node.args = ()
    graph.validate()
    again = graph.propagate(remove=request)
    assert again.status == impact.status
    assert again.diagnostics == impact.diagnostics
