"""Declare graph semantics for a fused module through the public rule API.

The example registers an `OperatorRule` on the default
`OperatorRegistry`. Its `OperatorSpec` connects input, output and
weight axes and declares the module attribute that must track a compacted
output. `DependencyGraph.build` captures the module with that registry, and
`propagate` demonstrates the resulting closure. The rule describes supported
structure; it does not rank channels or perform pruning. A custom rule must
declare the relations and attribute requirements its module needs for safe
planning and rewrite.
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AttributeBinding,
    AxisRelation,
    DependencyGraph,
    Operation,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    ShapeExpr,
)


class FusedProjection(nn.Module):
    """Combine a projection and activation behind a single opaque module."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 4))
        self.output_width = 6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the last input dimension and apply GELU."""
        return F.gelu(F.linear(x, self.weight))


class FusedProjectionRule(OperatorRule):
    """Describe the fused module through the same API as built-in operators."""

    def build_spec(self, operation: Operation) -> OperatorSpec:
        """Bind projection axes and the explicit output-width attribute."""
        x, y, weight = operation.inputs[0], operation.outputs[0], operation.bindings["weight"]
        return OperatorSpec(
            relations=(
                AxisRelation.equal(x.axis(-1), weight.axis(1), "fused input"),
                AxisRelation.equal(y.axis(-1), weight.axis(0), "fused output"),
                AxisRelation.equal(x.axis(0), y.axis(0), "fused batch"),
            ),
            requirements=(
                AttributeBinding(
                    f"{operation.module_path}.output_width".lstrip("."),
                    ShapeExpr("dimension", (y, -1)),
                ),
            ),
        )


rules = OperatorRegistry.default().register(FusedProjection, FusedProjectionRule())
graph = DependencyGraph.build(FusedProjection(), args=(torch.randn(2, 4),), operators=rules)
impact = graph.propagate(remove=[graph.parameter("weight").axis(0).select([1, 3])])
assert impact.status == "resolved"
print(graph.explain(impact))
