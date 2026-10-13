"""Declare graph semantics for a fused module through the public rule API.

The example registers an :class:`OperatorRule` on the default
:class:`OperatorRegistry`. Its ``OperatorSpec`` connects input, output and
weight axes and declares the module attribute that must track a compacted
output. ``DependencyGraph.build`` captures the module with that registry, and
``propagate`` demonstrates the resulting closure. The rule describes supported
structure; it does not rank channels or perform pruning. A custom rule must
declare the relations and attribute requirements its module needs for safe
planning and rewrite.
"""

import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRelation,
    DependencyGraph,
    OperationContext,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Requirement,
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


def fused_rule(ctx: OperationContext) -> OperatorSpec:
    """Bind projection axes and the module's explicit output-width attribute."""
    x, y, weight = ctx.inputs[0], ctx.outputs[0], ctx.binding("weight")
    assert ctx.metadata[weight.id].dtype == torch.float32
    return OperatorSpec(
        relations=(
            AxisRelation.equal(x.axis(-1), weight.axis(1), "fused input"),
            AxisRelation.equal(y.axis(-1), weight.axis(0), "fused output"),
            AxisRelation.equal(x.axis(0), y.axis(0), "fused batch"),
        ),
        requirements=(
            Requirement(
                "attribute",
                f"{ctx.module_path}.output_width".lstrip("."),
                (y,),
                "Update the explicitly bound output width",
                (("axis", y.axis(-1)),),
            ),
        ),
    )


rules = OperatorRegistry.default().register(FusedProjection, OperatorRule(fused_rule))
graph = DependencyGraph.build(FusedProjection(), args=(torch.randn(2, 4),), operators=rules)
impact = graph.propagate(remove=[graph.parameter("weight").axis(0).select([1, 3])])
assert impact.status == "resolved"
print(graph.explain(impact))
