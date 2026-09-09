"""graph / limits contracts."""

import torch
from torch import nn

from torch_kirigami import (
    DependencyGraph,
)


def test_analysis_limit_remains_explicit():
    class Model(nn.Module):
        def forward(self, x):
            return x.flatten()

    graph = DependencyGraph.build(Model(), args=(torch.randn(5000, 2),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([0])])
    assert impact.status == "unresolved"
    assert any(d.code == "analysis_limit" for d in impact.diagnostics)
