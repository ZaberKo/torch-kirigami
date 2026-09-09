"""integration / extensions contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import build
from torch_kirigami import (
    AxisRelation,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Requirement,
)
from torch_kirigami.pruning import (
    AttributeRecipe,
    RewriteResult,
)


def fused_analysis(ctx):
    x, y, w = ctx.inputs[0], ctx.outputs[0], ctx.binding("weight")
    return OperatorSpec(
        (
            AxisRelation.equal(x.axis(1), w.axis(1), "input"),
            AxisRelation.equal(y.axis(1), w.axis(0), "output"),
        ),
        requirements=(
            Requirement(
                "attribute",
                f"{ctx.module_path}.width".lstrip("."),
                (y,),
                "fused width",
                (("axis", y.axis(1)),),
            ),
        ),
    )


class Fused(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 4))
        self.width = 6

    def forward(self, x):
        return F.linear(x, self.weight).reshape(x.size(0), self.width)


def fused_rewrite(ctx):
    op = ctx.operation
    new = ctx.compact_shape(op.outputs[0])[1]
    path = f"{op.module_path}.width".lstrip(".")
    return RewriteResult(
        attributes=(AttributeRecipe(path, op.module.width, new),), handled=ctx.requirements
    )


def fused_function(x):
    return x.relu()


@pytest.mark.parametrize("nested", [True, False])
def test_custom_root_nested_extension(nested, execution_device):
    model = nn.Sequential(Fused(), nn.Linear(6, 2)) if nested else Fused()
    x = torch.randn(2, 4)
    rules = OperatorRegistry.default().register(
        Fused, OperatorRule(fused_analysis, lower=fused_rewrite)
    )
    graph, pruner = build(model, x, rules)
    path = "0.weight" if nested else "weight"
    plan = pruner.plan(remove=[graph.parameter(path).axis(0).select([1, 3])], preserve_io=nested)
    pruner.apply(plan)
    assert model(x).shape == (2, 2 if nested else 4)
    with pytest.raises(ValueError, match="already"):
        rules.register(Fused, OperatorRule(fused_analysis))


def test_custom_function_rule_does_not_require_core_changes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            return fused_function(self.fc(x))

    rules = OperatorRegistry.default().register(
        fused_function,
        OperatorRule(
            lambda c: OperatorSpec(
                (AxisRelation.equal(c.inputs[0].axis(1), c.outputs[0].axis(1), "channels"),)
            )
        ),
    )
    model = Net()
    graph, pruner = build(model, torch.randn(2, 4), rules)
    assert any(c.node.target is fused_function for c in graph.operations())
    pruner.apply(
        pruner.plan(remove=[graph.parameter("fc.weight").axis(0).select([0])], preserve_io=False)
    )
    assert model.fc.out_features == 5
