"""architecture / extensions contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRelation,
    CandidateAxis,
    CaptureError,
    DependencyGraph,
    Fixed,
    IndexSet,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    PermuteRelation,
    Region,
    Requirement,
    ReshapeRelation,
    ShapeExpr,
    TensorRef,
)
from torch_kirigami.operation import PartitionedLayout


def namespace_swap(input, dim=1, *, axis=0):
    return input.transpose(dim, axis)


class Opaque(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x, *, scale=1.0):
        if x.sum() > 0:
            return F.linear(x, self.weight) * scale
        return -F.linear(x, self.weight) * scale


def opaque_rule(ctx):
    x, y, w = ctx.inputs[0], ctx.outputs[0], ctx.binding("weight")
    return OperatorSpec(
        (
            AxisRelation.equal(x.axis(1), w.axis(1), "opaque input"),
            AxisRelation.equal(y.axis(1), w.axis(0), "opaque output"),
            AxisRelation.equal(x.axis(0), y.axis(0), "opaque batch"),
        )
    )


def opaque_function(x):
    return x * 2 if x.sum() > 0 else x * 3


@pytest.mark.parametrize("namespace", ["torch_kirigami_ext", "torchvision", "torchx"])
def test_custom_torch_prefix_does_not_enable_native_aliases(namespace, monkeypatch):
    monkeypatch.setattr(namespace_swap, "__module__", namespace)

    def analyze(ctx):
        dim, axis = ctx.argument("dim", 1, 1), ctx.argument("axis", 2, 0)
        source, output = ctx.inputs[0], ctx.outputs[0]
        dims = list(range(len(source.shape)))
        dims[dim], dims[axis] = dims[axis], dims[dim]
        return OperatorSpec(relations=(PermuteRelation(source, output, tuple(dims)),))

    class Model(nn.Module):
        def forward(self, x):
            return namespace_swap(x, axis=0)

    registry = OperatorRegistry.default().register(namespace_swap, OperatorRule(analyze))
    graph = DependencyGraph.build(Model(), args=(torch.ones(4, 4),), operators=registry)
    call = graph.calls()[0]
    impact = graph.propagate(remove=[call.input().axis(1).select([1])])
    assert tuple(impact.selection(call.output()).fully_selected_indices(0)) == (1,)


@pytest.mark.parametrize(
    "field",
    [
        "relations",
        "constraints",
        "requirements",
        "candidates",
        "alignment_axis",
        "layouts",
        "constants",
        "expression",
        "payload",
    ],
)
def test_extension_spec_rejects_foreign_references(field):
    foreign = TensorRef("foreign", (4, 4))

    def analyze(ctx):
        x = ctx.inputs[0]
        values = {
            "relations": (AxisRelation.equal(x.axis(1), foreign.axis(0)),),
            "constraints": (Fixed(foreign.axis(0)),),
            "requirements": (Requirement("custom", ctx.node.name, (foreign,), ""),),
            "candidates": (CandidateAxis("foreign", foreign.axis(0)),),
            "alignment_axis": (CandidateAxis("axis", x.axis(1), alignment_axis=foreign.axis(0)),),
            "layouts": (PartitionedLayout(foreign, (Region((IndexSet.span(0, 4),) * 2),)),),
            "constants": (foreign,),
            "expression": ShapeExpr("dimension", (foreign, 0)),
            "payload": (
                Requirement("custom", ctx.node.name, (x,), "", (("axis", foreign.axis(0)),)),
            ),
        }
        name = {"payload": "requirements", "alignment_axis": "candidates"}.get(field, field)
        return OperatorSpec(**{name: values[field]})

    registry = OperatorRegistry().register(nn.Linear, OperatorRule(analyze))
    with pytest.raises(ValueError, match="another graph"):
        DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),), operators=registry)


def test_external_root_and_nested_module_use_one_rule_interface():
    rules = OperatorRegistry.default().register(Opaque, OperatorRule(opaque_rule))
    for model, path in ((Opaque(), "weight"), (nn.Sequential(Opaque()), "0.weight")):
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),), operators=rules)
        assert (
            graph.propagate(remove=[graph.parameter(path).axis(0).select([1])]).status == "resolved"
        )
    graph = DependencyGraph.build(
        Opaque(), args=(torch.ones(2, 4),), kwargs={"scale": 2.0}, operators=rules
    )
    assert graph.calls("")[0].output().shape == (2, 4)
    with pytest.raises(ValueError):
        rules.register(Opaque, OperatorRule(opaque_rule))


def test_external_function_capture_is_local():
    class Model(nn.Module):
        def forward(self, x):
            return opaque_function(x)

    rules = OperatorRegistry.default().register(
        opaque_function,
        OperatorRule(lambda c: OperatorSpec((ReshapeRelation(c.inputs[0], c.outputs[0]),))),
    )
    graph = DependencyGraph.build(Model(), args=(torch.ones(2, 4),), operators=rules)
    assert (
        graph.propagate(remove=[graph.calls()[0].input().axis(1).select([1])]).status == "resolved"
    )
    with pytest.raises(CaptureError):
        DependencyGraph.build(Model(), args=(torch.ones(2, 4),))


def test_overridden_linear_subclass_does_not_inherit_rule():
    class Different(nn.Linear):
        def forward(self, x):
            return torch.special.gammaln(F.linear(x, self.weight, self.bias))

    graph = DependencyGraph.build(nn.Sequential(Different(4, 4)), args=(torch.randn(2, 4),))
    assert (
        graph.propagate(remove=[graph.parameter("0.weight").axis(0).select([1])]).status
        == "unresolved"
    )
