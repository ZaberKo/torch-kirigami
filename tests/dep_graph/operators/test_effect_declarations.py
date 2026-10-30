"""Public operator hooks share descriptors and preserve execution safety."""

import copy
import io
import warnings
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from examples.fused_attention import FusedGQA, FusedGQARule
from torch_kirigami import (
    ArgumentRef,
    AttributeBinding,
    AxisRelation,
    CaptureError,
    DependencyGraph,
    Operation,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Requirement,
    ReshapeRelation,
    ShapeExpr,
    TensorEffects,
    UnsupportedOperation,
)
from torch_kirigami.operator_recipes import OperatorRecipe
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def update_parameter_sequence(*weights):
    """Update each parameter, then return an independently allocated sum."""
    with torch.no_grad():
        for weight in weights:
            weight.add_(1)
    return torch.stack(weights).sum(0)


def update_parameter_container(weights):
    """Accept the same updates through a tuple or dictionary argument."""
    values = tuple(weights.values()) if isinstance(weights, dict) else weights
    return update_parameter_sequence(*values)


@pytest.mark.parametrize("form", ["tuple", "dict", "keyword", "variadic"])
@pytest.mark.parametrize("view", [False, True])
def test_parameter_update_arguments_share_capture_and_planning_resolution(
    form, view, execution_device
):
    """Declared containers work throughout pruning; parameter views stay protected."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Parameter(torch.arange(6.0))
            self.second = nn.Parameter(torch.arange(6.0) * 2)

        def forward(self):
            second = self.second.detach() if view else self.second
            if form == "variadic":
                return update_parameter_sequence(self.first, second)
            weights = (
                {"first": self.first, "second": second} if form == "dict" else (self.first, second)
            )
            if form == "keyword":
                return update_parameter_container(weights=weights)
            return update_parameter_container(weights)

    class Rule(OperatorRule):
        def tensor_effects(self, node, module):
            ref = ArgumentRef("weights", 0, variadic=form == "variadic")
            return TensorEffects(
                writes_inputs=(ref,), outputs_are_fresh=True, mutable_parameters=(ref,)
            )

        def build_spec(self, operation):
            output = operation.outputs[0]
            return OperatorSpec(
                relations=tuple(ReshapeRelation(value, output) for value in operation.inputs)
            )

    target = update_parameter_sequence if form == "variadic" else update_parameter_container
    registry = OperatorRegistry.default().register(target, Rule(evaluate_on_meta=True))
    model = Model()
    original = [p.detach().clone() for p in model.parameters()]
    if view:
        with pytest.raises(CaptureError, match="Parameter/alias write"):
            DependencyGraph.build(model, args=(), operators=registry)
        for parameter, before in zip(model.parameters(), original, strict=True):
            torch.testing.assert_close(parameter, before)
        return
    with pytest.warns(UserWarning, match="parameter value updates") as emitted:
        graph = DependencyGraph.build(model, args=(), operators=registry)
    assert len(emitted) == 1
    assert "Embedding" not in str(emitted[0].message)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    with warnings.catch_warnings(record=True) as emitted:
        plan = pruner.plan_remove([graph.parameter("first").axis(0).select([1, 4])])
    assert not emitted
    for parameter, before in zip(model.parameters(), original, strict=True):
        torch.testing.assert_close(parameter, before + 1)
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3, 5]
    for parameter, before in zip(model.parameters(), original, strict=True):
        torch.testing.assert_close(parameter, before[keep] + 1)
    expected = original[0][keep] + original[1][keep] + 4
    output = model()
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("view", [False, True])
@pytest.mark.parametrize("declared", [False, True])
def test_registered_buffer_writes_require_declared_direct_bindings(
    view, declared, execution_device
):
    """Direct declared writes work without relaxing unknown or aliased writes."""

    class Update(nn.Module):
        def forward(self, value):
            return value.add_(2)

    class Rule(OperatorRule):
        def build_spec(self, operation):
            return OperatorSpec(
                relations=(ReshapeRelation(operation.inputs[0], operation.outputs[0]),)
            )

        def tensor_effects(self, node, module):
            return TensorEffects(writes_inputs=(ArgumentRef("value", 0),) if declared else None)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.arange(8.0).reshape(2, 4))
            self.alias = nn.Identity()
            self.update = Update()
            self.free = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))

        def forward(self, x):
            value = self.alias(self.state) if view else self.state
            return self.update(value), self.free(x)

    model = Model()
    original = model.state.clone()
    identity = model.state
    x = torch.randn(2, 3)
    registry = OperatorRegistry.default().register(Update, Rule(evaluate_on_meta=True))
    with warnings.catch_warnings(record=True) as emitted:
        graph = DependencyGraph.build(model, args=(x,), operators=registry)
    assert not emitted
    assert model.state is identity
    torch.testing.assert_close(model.state, original)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    request = [graph.buffer("state").axis(1).select([1])]
    if view or not declared:
        with pytest.raises(PlanningError, match="in-place alias"):
            pruner.plan_remove(request)
        assert model.state is identity
        torch.testing.assert_close(model.state, original)
        # The unsafe affected request does not prevent independent pruning.
        plan = pruner.plan_remove([graph.parameter("free.0.weight").axis(0).select([1])])
        pruner.apply(plan)
        torch.testing.assert_close(model.state, original)
        assert model.free[0].out_features == 3
        return
    plan = pruner.plan_remove(request)
    torch.testing.assert_close(model.state, original)
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    torch.testing.assert_close(model.state, original[:, [0, 2, 3]])
    result, _ = model(x)
    torch.testing.assert_close(result, original[:, [0, 2, 3]] + 2)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("recipe", [None, "empty", "notes", "reject"])
def test_partial_recipe_preserves_declarations_and_persistence(nested, recipe, execution_device):
    """Partial or absent edits keep the same GQA structure and independent numerical result."""
    calls = []

    class Rule(FusedGQARule):
        def build_recipe(self, operation, spec, impact):
            calls.append(operation.name)
            assert isinstance(operation, Operation) and isinstance(spec, OperatorSpec)
            if recipe == "reject":
                raise UnsupportedOperation("This request needs a different fused kernel")
            if recipe == "empty":
                return OperatorRecipe()
            if recipe == "notes":
                return OperatorRecipe(notes=("No special physical edits needed",))
            return None

    fused = FusedGQA().to(execution_device)
    model = nn.Sequential(fused) if nested else fused
    x = torch.randn(2, 3, 8, device=execution_device)
    original = copy.deepcopy(fused)
    registry = OperatorRegistry.default().register(FusedGQA, Rule())
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    prefix = "0." if nested else ""
    remove = [graph.parameter(prefix + "k.weight").axis(0).select(range(4))]
    before = tuple(model.parameters())
    if recipe == "reject":
        with pytest.raises(PlanningError, match="different fused kernel"):
            pruner.plan_remove(remove)
        assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
        assert fused.q_heads == 4 and fused.kv_heads == 2
        torch.testing.assert_close(model(x), original(x))
        return
    plan = pruner.plan_remove(remove)
    assert calls and all(a is b for a, b in zip(before, model.parameters(), strict=True))
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    # Independent compact attention formula: keep the second KV group and its
    # two query heads, then apply the corresponding output-projection columns.
    q = F.linear(x, original.q.weight[8:]).reshape(2, 3, 2, 4).transpose(1, 2)
    k = F.linear(x, original.k.weight[4:]).reshape(2, 3, 1, 4).transpose(1, 2)
    v = F.linear(x, original.v.weight[4:]).reshape(2, 3, 1, 4).transpose(1, 2)
    attended = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    expected = F.linear(attended.transpose(1, 2).reshape(2, 3, 8), original.out.weight[:, 8:])
    torch.testing.assert_close(model(x), expected)
    assert fused.q_heads == 2 and fused.kv_heads == 1
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    fresh = nn.Sequential(FusedGQA()) if nested else FusedGQA()
    restored = load_checkpoint(fresh.to(execution_device), stream)
    torch.testing.assert_close(restored(x), expected)
    restored(x).sum().backward()
    assert all(p.grad is not None for p in restored.parameters())


class WriteSecond(nn.Module):
    """Modify a designated argument while reading a separate parameter."""

    def forward(self, weight, *, value):
        return value.add_(weight.sum() * 0)


class WriteSecondRule(OperatorRule):
    def build_spec(self, operation):
        x, y = operation.kwargs["value"], operation.outputs[0]
        return OperatorSpec(
            relations=tuple(AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape)))
        )

    def tensor_effects(self, node, module):
        return TensorEffects(writes_inputs=(ArgumentRef("value", 1),))


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.parametrize("meta", [False, True])
def test_write_argument_locator_and_alias_checks_apply_without_meta(fresh, meta, execution_device):
    """Reading a parameter is safe; writing an input alias still needs a fresh producer."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(1))
            self.layer = nn.Linear(3, 4)
            self.op = WriteSecond()

        def forward(self, x):
            value = self.layer(x) if fresh else x
            return self.op(self.weight, value=value)

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    initial = x.clone()
    rules = OperatorRegistry.default().register(WriteSecond, WriteSecondRule(evaluate_on_meta=meta))
    graph = DependencyGraph.build(model, args=(x,), operators=rules)
    torch.testing.assert_close(x, initial)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    if fresh:
        selection = graph.parameter("layer.weight").axis(0).select([1])
        expected = F.linear(x, model.layer.weight[[0, 2, 3]], model.layer.bias[[0, 2, 3]])
        pruner.apply(pruner.plan_remove([selection]))
        torch.testing.assert_close(model(x), expected)
    else:
        selection = graph.operations("op")[0].kwargs["value"].axis(1).select([1])
        before = tuple(model.parameters())
        with pytest.raises(PlanningError, match="in-place alias"):
            pruner.plan_remove([selection])
        assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
        torch.testing.assert_close(x, initial)


def test_opaque_module_parameter_result_cannot_hide_a_write(execution_device):
    class ParameterResult(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 3, device=execution_device))

        def forward(self, x):
            return self.weight

    class Rule(OperatorRule):
        def build_spec(self, operation):
            return OperatorSpec()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.source = ParameterResult()

        def forward(self, x):
            return self.source(x).add_(1)

    model = Model()
    before = model.source.weight.detach().clone()
    registry = OperatorRegistry.default().register(ParameterResult, Rule())
    with pytest.raises(CaptureError, match="Parameter/alias write"):
        DependencyGraph.build(
            model, args=(torch.randn(2, 3, device=execution_device),), operators=registry
        )
    torch.testing.assert_close(model.source.weight, before)


def test_unchanged_attribute_declaration_conflicts_with_another_demand():
    class ConflictingRule(FusedGQARule):
        def build_spec(self, operation):
            spec = super().build_spec(operation)
            unchanged = AttributeBinding("q_heads", ShapeExpr("constant", operation.module.q_heads))
            return replace(spec, requirements=(*spec.requirements, unchanged))

    model = FusedGQA()
    graph = DependencyGraph.build(
        model,
        args=(torch.randn(2, 3, 8),),
        operators=OperatorRegistry.default().register(FusedGQA, ConflictingRule()),
    )
    before = tuple(model.parameters())
    with pytest.raises(PlanningError, match="Shared attribute update disagrees"):
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("k.weight").axis(0).select(range(4))]
        )
    assert model.q_heads == 4
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))


@pytest.mark.parametrize("meta", [False, True])
@pytest.mark.parametrize("fixed_shape", [False, True])
def test_custom_recipe_cannot_skip_declared_forward_check(meta, fixed_shape, execution_device):
    """Declared checks require execution proof even when a custom recipe returns successfully."""

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.size(0), 6 if fixed_shape else -1)

    class Rule(OperatorRule):
        def build_spec(self, operation):
            x, y = operation.inputs[0], operation.outputs[0]
            return OperatorSpec(
                relations=(ReshapeRelation(x, y),),
                requirements=(
                    Requirement(
                        "dimension_transform",
                        operation.name,
                        (x, y),
                        "Verify the original compact call",
                    ),
                ),
            )

        def build_recipe(self, operation, spec, impact):
            return OperatorRecipe()

    model = nn.Sequential(nn.Linear(3, 6), Reshape(), nn.Linear(6, 2)).to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    graph = DependencyGraph.build(
        model,
        args=(x,),
        operators=OperatorRegistry.default().register(Reshape, Rule(evaluate_on_meta=meta)),
    )
    pruner = Pruner(model, graph=graph)
    request = [graph.parameter("0.weight").axis(0).select([1])]
    before = tuple(model.parameters())
    if not meta or fixed_shape:
        with pytest.raises(PlanningError, match=r"evaluate_on_meta|original forward"):
            pruner.plan_remove(request)
        assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
        assert model[0].out_features == 6
    else:
        keep = [0, 2, 3, 4, 5]
        expected = F.linear(
            F.linear(x, model[0].weight[keep], model[0].bias[keep]),
            model[2].weight[:, keep],
            model[2].bias,
        )
        pruner.apply(pruner.plan_remove(request))
        torch.testing.assert_close(model(x), expected)


def test_capture_write_uncertainty_is_distinct_from_local_structural_barrier(execution_device):
    """An explicit read-only contract permits a fixed branch without inventing dependencies."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = nn.Parameter(torch.randn(2, 3))
            self.hidden = nn.Linear(3, 4)
            self.out = nn.Linear(4, 2)

        def forward(self, x):
            return torch.roll(self.constant, 1, -1), self.out(self.hidden(x))

    class ReadOnlyUnknown(OperatorRule):
        def build_spec(self, operation):
            raise UnsupportedOperation("This fixed branch has no structural rule")

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    before = copy.deepcopy(model)
    with pytest.raises(CaptureError, match="Parameter/alias write"):
        DependencyGraph.build(model, args=(x,))
    torch.testing.assert_close(model.constant, before.constant)
    registry = OperatorRegistry.default().register(torch.roll, ReadOnlyUnknown(), opaque=False)
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError):
        pruner.plan_remove([graph.parameter("constant").axis(0).select([0])])
    plan = pruner.plan_remove([graph.parameter("hidden.weight").axis(0).select([1])])
    pruner.apply(plan)
    fixed, output = model(x)
    keep = [0, 2, 3]
    expected = F.linear(
        F.linear(x, before.hidden.weight[keep], before.hidden.bias[keep]),
        before.out.weight[:, keep],
        before.out.bias,
    )
    torch.testing.assert_close(fixed, before(x)[0])
    torch.testing.assert_close(output, expected)
