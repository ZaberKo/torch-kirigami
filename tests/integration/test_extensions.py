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
    StaleGraphError,
)
from torch_kirigami.pruning import (
    AttributeRecipe,
    PlanningError,
    Pruner,
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
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=nested).plan_remove(
        [graph.parameter(path).axis(0).select([1, 3])]
    )
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
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(
            [graph.parameter("fc.weight").axis(0).select([0])]
        )
    )
    assert model.fc.out_features == 5


@pytest.mark.parametrize("lowering", ["declaration", "callback", "subclass", "instance"])
def test_default_lowering_and_delegating_extensions_preserve_compact_values(
    lowering, execution_device
):
    calls = []

    def lower(ctx):
        calls.append(ctx.operation.node.name)
        return None

    class DelegatingRule(OperatorRule):
        def lower(self, ctx):
            return lower(ctx)

    rule = (
        DelegatingRule(fused_analysis)
        if lowering == "subclass"
        else OperatorRule(fused_analysis, lower=lower if lowering == "callback" else None)
    )
    if lowering == "instance":
        rule.lower = lower
    rules = OperatorRegistry.default().register(Fused, rule)
    model = nn.Sequential(Fused(), nn.Linear(6, 2))
    x = torch.randn(2, 4)
    old = tuple(model.parameters())
    keep = [0, 2, 4, 5]
    expected = F.linear(F.linear(x, model[0].weight[keep]), model[1].weight[:, keep], model[1].bias)
    graph, pruner = build(model, x, rules)
    plan = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1, 3])])
    assert all(before is after for before, after in zip(old, model.parameters(), strict=True))
    assert bool(calls) == (lowering != "declaration")
    pruner.apply(plan)
    assert model[0].width == model[1].in_features == 4
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("lowering", ["callback", "subclass", "instance"])
@pytest.mark.parametrize("outcome", ["return", "reject", "raise"])
def test_lowering_mutations_are_rejected_even_when_callback_raises(lowering, outcome):
    def lower(ctx):
        ctx.operation.module.width += 1
        if outcome == "reject":
            raise PlanningError("injected candidate rejection")
        if outcome == "raise":
            raise RuntimeError("injected callback failure")
        return None

    class MutatingRule(OperatorRule):
        def lower(self, ctx):
            return lower(ctx)

    rule = (
        MutatingRule(fused_analysis)
        if lowering == "subclass"
        else OperatorRule(fused_analysis, lower=lower if lowering == "callback" else None)
    )
    if lowering == "instance":
        rule.lower = lower
    rules = OperatorRegistry.default().register(Fused, rule)
    model = nn.Sequential(Fused(), nn.Linear(6, 2))
    graph, pruner = build(model, torch.randn(2, 4), rules)
    old = tuple(model.parameters())
    with pytest.raises(StaleGraphError):
        pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1, 3])])
    # Invalid extension code changed its own attribute; planning must not also
    # replace any weight or return a usable plan for that modified model.
    assert all(before is after for before, after in zip(old, model.parameters(), strict=True))


@pytest.mark.parametrize("error_type", [PlanningError, RuntimeError])
def test_readonly_lowering_exceptions_keep_their_original_failure(error_type):
    def lower(ctx):
        raise error_type("extension rejected this compaction")

    rules = OperatorRegistry.default().register(Fused, OperatorRule(fused_analysis, lower=lower))
    model = nn.Sequential(Fused(), nn.Linear(6, 2))
    graph, pruner = build(model, torch.randn(2, 4), rules)
    old = tuple(model.parameters())
    with pytest.raises(error_type, match="extension rejected this compaction"):
        pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1, 3])])
    graph.validate()
    assert model[0].width == 6
    assert all(before is after for before, after in zip(old, model.parameters(), strict=True))


def test_default_lowering_does_not_rescan_model_after_every_operation(monkeypatch):
    counts = []
    for depth in (1, 20):
        model = nn.Sequential(nn.Linear(4, 6), *(nn.ReLU() for _ in range(depth)), nn.Linear(6, 2))
        x = torch.randn(2, 4)
        graph, pruner = build(model, x)
        calls = []
        check_fresh = graph._check_fresh

        def counted_check_fresh(_calls=calls, _check_fresh=check_fresh):
            _calls.append(None)
            return _check_fresh()

        # Include implicit scans through tensor() and tensor_bindings(), not
        # only explicit calls to the public validate() method.
        monkeypatch.setattr(graph, "_check_fresh", counted_check_fresh)
        plan = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1, 3])])
        counts.append(len(calls))
        expected = F.linear(
            F.relu(F.linear(x, model[0].weight[[0, 2, 4, 5]], model[0].bias[[0, 2, 4, 5]])),
            model[-1].weight[:, [0, 2, 4, 5]],
            model[-1].bias,
        )
        pruner.apply(plan)
        torch.testing.assert_close(model(x), expected)
    assert counts[1] == counts[0]
