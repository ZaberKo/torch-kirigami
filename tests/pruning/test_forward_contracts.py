"""Cross-layer contracts that direct operator-rule tests cannot validate."""

import copy
import io
import operator
from threading import Lock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.operators import make_rule
from torch_kirigami import (
    ArgumentRef,
    AttributeBinding,
    AxisRelation,
    DependencyGraph,
    OperatorRegistry,
    OperatorSpec,
    OutputContract,
    Requirement,
    ShapeExpr,
)
from torch_kirigami.pruning import (
    AttributeRecipe,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    Magnitude,
    OperatorRecipe,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


class _PairedWidth(nn.Module):
    def forward(self, x, width):
        if width % 2:
            raise ValueError("paired width must be even")
        return x.clone()


@pytest.mark.parametrize("cast_output", [False, True])
def test_call_argument_exemption_requires_meta_replay(cast_output, execution_device):
    """A declared replay cannot silently discharge a dimension-argument check."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.pair = _PairedWidth()
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            hidden = self.first(x)
            return self.last(self.pair(hidden, hidden.size(-1)))

    def spec(op):
        x, y = op.inputs[0], op.outputs[0]
        return OperatorSpec(
            (AxisRelation.equal(x.axis(0), y.axis(0)), AxisRelation.equal(x.axis(1), y.axis(1))),
            requirements=(
                Requirement(
                    "call_arguments",
                    op.name,
                    (x, y),
                    "Validate paired width",
                    arguments=(ArgumentRef("width", 1),),
                ),
            ),
            contract=OutputContract("cast", copy_output=True) if cast_output else None,
        )

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    old = model.first.weight
    for evaluate, message in (
        (False, "require evaluate_on_meta=True"),
        (True, "paired width must be even"),
    ):
        registry = OperatorRegistry.default().register(
            _PairedWidth, make_rule(spec, evaluate_on_meta=evaluate)
        )
        graph = DependencyGraph.build(model, args=(x,), operators=registry)
        pruner = Pruner(model, graph=graph)
        axis = graph.parameter("first.weight").axis(0)
        with pytest.raises(PlanningError, match=message):
            pruner.plan_remove((axis.select([0]),))
        assert model.first.weight is old and model.first.out_features == 4
    expected = F.linear(model.first(x)[:, 2:], model.last.weight[:, 2:], model.last.bias)
    pruner.apply(pruner.plan_remove((axis.select([0, 1]),)))
    torch.testing.assert_close(model(x), expected)


def test_declared_scalar_result_still_checks_the_original_call(execution_device):
    class Width(nn.Module):
        def forward(self, x):
            width = x.shape[-1]
            if width == 5:
                raise ValueError("width five is unsupported")
            return width + 1 if width == 4 else width

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 6)
            self.width = Width()
            self.last = nn.Linear(6, 2)

        def forward(self, x):
            hidden = self.first(x)
            return self.last(hidden), self.width(hidden)

    def spec(op):
        (x,) = op.inputs
        return OperatorSpec(
            expression=ShapeExpr("dimension", (x, -1)),
            requirements=(Requirement("call_arguments", op.name, (x,), "Check scalar call"),),
        )

    model = Model().eval()
    x = torch.randn(2, 3)
    weight = model.first.weight
    for evaluate, removed, message in (
        (False, [0], "require evaluate_on_meta=True"),
        (True, [0], "width five is unsupported"),
        (True, [0, 1], "scalar result disagrees"),
    ):
        registry = OperatorRegistry.default().register(
            Width, make_rule(spec, evaluate_on_meta=evaluate)
        )
        graph = DependencyGraph.build(model, args=(x,), operators=registry)
        axis = graph.parameter("first.weight").axis(0)
        pruner = Pruner(model, graph=graph)
        with pytest.raises(PlanningError, match=message):
            pruner.plan_remove((axis.select(removed),))
        assert model.first.weight is weight and model.first.out_features == 6
        assert model(x)[1] == 6
    expected = F.linear(model.first(x)[:, 4:], model.last.weight[:, 4:], model.last.bias)
    pruner.apply(pruner.plan_remove((axis.select([0, 1, 2, 3]),)))
    output, width = model(x)
    torch.testing.assert_close(output, expected)
    assert width == 2


def test_compact_module_output_keeps_dictionary_ports(execution_device):
    class NamedOutput(nn.Module):
        def forward(self, x):
            if x.shape[-1] == 3:
                return {"value": x.clone(), "aux": x.square()}
            if x.shape[-1] == 2:
                return {"replacement": x.clone()}
            return {"value": x.clone()}

    def spec(op):
        (x,) = op.inputs
        y = op.result["value"]
        return OperatorSpec(
            tuple(AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape)))
        )

    model = nn.Sequential(nn.Linear(3, 4), NamedOutput()).eval()
    x = torch.randn(2, 3)
    registry = OperatorRegistry.default().register(
        NamedOutput, make_rule(spec, evaluate_on_meta=True)
    )
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    axis = graph.parameter("0.weight").axis(0)
    original = model[0].weight
    for removed in ([0], [0, 1]):
        with pytest.raises(PlanningError, match="changes output keys"):
            pruner.plan_remove((axis.select(removed),))
        assert model[0].weight is original and model[0].out_features == 4
    expected = model(x)["value"][:, 3:]
    pruner.apply(pruner.plan_remove((axis.select([0, 1, 2]),)))
    torch.testing.assert_close(model(x), {"value": expected})


@pytest.mark.parametrize("target", ["wrapper.check.width", "alias.width"])
def test_compact_replay_updates_the_attribute_owner_through_aliases(target, execution_device):
    class WidthCheck(nn.Module):
        def __init__(self):
            super().__init__()
            self.width = 4

        def forward(self, x):
            if x.shape[-1] != self.width:
                raise ValueError("Input does not match the declared width")
            if self.width < 3:
                raise ValueError("At least three features are required")
            return x.clone()

    class Wrapper(nn.Module):
        def __init__(self, check):
            super().__init__()
            self.check = check

        def forward(self, x):
            return self.check(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.alias = WidthCheck()
            self.wrapper = Wrapper(self.alias)
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            return self.last(self.wrapper(self.first(x)))

    def spec(op):
        x, y = op.inputs[0], op.outputs[0]
        return OperatorSpec(
            tuple(AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape))),
            requirements=(AttributeBinding(target, ShapeExpr("dimension", (x, -1))),),
        )

    model = Model().eval()
    x = torch.randn(2, 3)
    registry = OperatorRegistry.default().register(Wrapper, make_rule(spec, evaluate_on_meta=True))
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("first.weight").axis(0)
    original = model.first.weight
    with pytest.raises(PlanningError, match="At least three features are required"):
        pruner.plan_remove((axis.select([0, 1]),))
    assert model.first.weight is original and model.alias.width == 4

    keep = [0, 2, 3]
    expected = F.linear(model.first(x)[:, keep], model.last.weight[:, keep], model.last.bias)
    plan = pruner.plan_remove((axis.select([1]),))
    assert model.alias.width == 4
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert model.wrapper.check is model.alias and model.alias.width == 3
    torch.testing.assert_close(model(x), expected)


def test_attribute_recipes_distinguish_equal_numbers_of_different_types(execution_device):
    class Checked(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = 1

        def forward(self, x):
            if x.shape[-1] < 3:
                raise ValueError("At least three features are required")
            required_type = int if x.shape[-1] == 4 else bool
            if type(self.kind) is not required_type:
                raise ValueError("Incompatible compact configuration type")
            return x.clone()

    def spec(op):
        x, y = op.inputs[0], op.outputs[0]
        return OperatorSpec(
            tuple(AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape)))
        )

    def recipe(op, spec, impact):
        return OperatorRecipe(attributes=(AttributeRecipe("1.kind", op.module.kind, True),))

    def make_model():
        return nn.Sequential(nn.Linear(3, 4), Checked(), nn.Linear(4, 2)).eval()

    model = make_model()
    x = torch.randn(2, 3)
    registry = OperatorRegistry.default().register(
        Checked, make_rule(spec, build_recipe=recipe, evaluate_on_meta=True)
    )
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("0.weight").axis(0)
    original = model[0].weight
    with pytest.raises(PlanningError, match="At least three features are required"):
        pruner.plan_remove((axis.select([0, 1]),))
    assert model[0].weight is original and type(model[1].kind) is int

    keep = [0, 2, 3]
    expected = F.linear(model[0](x)[:, keep], model[2].weight[:, keep], model[2].bias)
    plan = pruner.plan_remove((axis.select([1]),))
    assert type(model[1].kind) is int
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert model[1].kind is True
    torch.testing.assert_close(model(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(make_model(), stream)
    assert restored[1].kind is True
    torch.testing.assert_close(restored(x), expected)


def test_native_metadata_replay_ignores_unrelated_module_state(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Parameter(torch.randn(4, 3))
            self.activation = nn.ReLU()
            self.activation.observer_lock = Lock()
            self.last = nn.Parameter(torch.randn(2, 4))

        def forward(self, x):
            return F.linear(self.activation(F.linear(x, self.first)), self.last)

    model = Model().eval()
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    lock = model.activation.observer_lock
    original = model.first
    axis = graph.parameter("first").axis(0)
    expected = F.linear(F.linear(x, model.first[1:]).relu(), model.last[:, 1:])
    plan = pruner.plan_remove((axis.select([0]),))
    assert model.first is original
    pruner.apply(plan)
    assert model.activation.observer_lock is lock
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("strategy", [Greedy, DynamicGreedy])
def test_zero_compact_split_size_excludes_only_its_branch(strategy, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.sizer = nn.Linear(4, 4)
            self.hidden = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            sections = self.sizer(x).shape[-1] // 4
            return x.split(sections, -1), self.head(self.hidden(x))

    model = Model().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        # The invalid candidates rank first. Hidden row 0 is the cheapest valid choice.
        model.sizer.weight.zero_()
        model.sizer.bias.zero_()
        model.hidden.weight.copy_(torch.arange(1, 5)[:, None].expand(4, 4))
        model.hidden.bias.zero_()
        model.head.weight.fill_(1)
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="split size becomes 0"):
        pruner.plan_remove((graph.parameter("sizer.weight").axis(0).select([0]),))
    torch.testing.assert_close(model.state_dict(), original.state_dict())

    space = pruner.discover_candidates(targets=("sizer", "hidden"))
    plan = pruner.plan(
        space, budget=ChannelRatio(0.125, scope="global"), strategy=strategy(Magnitude())
    )
    assert len(plan.selected) == 1
    assert any("split size becomes 0" in reason for _, reason in plan.selection_report.exclusions)
    pruner.apply(plan)
    assert model.sizer.out_features == 4 and model.hidden.out_features == 3
    parts, prediction = model(x)
    torch.testing.assert_close(parts, x.split(1, -1))
    torch.testing.assert_close(
        prediction,
        F.linear(original.hidden(x)[:, 1:], original.head.weight[:, 1:], original.head.bias),
    )


@pytest.mark.parametrize(
    "consumer", ["invalid_shape", "changed_shape", "slice", "branch", "loop", "scalar"]
)
def test_attribute_consumers_outside_the_pruned_branch_are_checked(consumer, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            y = self.b(self.a(x))
            if consumer in ("invalid_shape", "changed_shape"):
                other = x.reshape(self.a.out_features, -1)
            elif consumer == "slice":
                other = x[:, self.a.out_features - 1]
            elif consumer == "branch":
                other = x + 1 if self.a.out_features == 4 else x - 1
            elif consumer == "loop":
                other = x
                for _ in range(self.a.out_features):
                    other = other + 1
            else:
                other = self.a.out_features
            return y, other

    model = Model()
    x = torch.randn(3 if consumer == "changed_shape" else 2, 4)
    before = copy.deepcopy(model.state_dict())
    expected = model(x)
    graph = DependencyGraph.build(model, args=(x,))
    with pytest.raises(PlanningError, match=r"attribute|structure"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    assert model.a.out_features == 4
    torch.testing.assert_close(model.state_dict(), before)
    torch.testing.assert_close(model(x), expected)
    graph.validate()


@pytest.mark.parametrize(
    "operation",
    [
        operator.and_,
        operator.or_,
        operator.xor,
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
        torch.logical_and,
        torch.logical_or,
        torch.logical_xor,
        pytest.param(lambda a, b: a.bitwise_and(b), id="method-bitwise-and"),
        pytest.param(lambda a, b: a.bitwise_or(b), id="method-bitwise-or"),
        pytest.param(lambda a, b: a.bitwise_xor(b), id="method-bitwise-xor"),
        pytest.param(lambda a, b: a.logical_and(b), id="method-logical-and"),
        pytest.param(lambda a, b: a.logical_or(b), id="method-logical-or"),
        pytest.param(lambda a, b: a.logical_xor(b), id="method-logical-xor"),
    ],
)
@pytest.mark.parametrize("source", ["activation", "parameter"])
def test_python_boolean_operators_run_through_capture_plan_and_backward(
    operation, source, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6, bias=False)

        def forward(self, x):
            if source == "parameter":
                w = self.fc.weight
                return F.linear(x, w * operation(w > 0, w < 1))
            y = self.fc(x)
            return torch.where(operation(y > 0, y < 1), y, 0.0)

    model = Model()
    x = torch.randn(2, 4)
    w = model.fc.weight.detach()[[0, 2, 3, 4, 5]].clone().requires_grad_()
    if source == "parameter":
        expected = F.linear(x, w * operation(w > 0, w < 1))
    else:
        y = F.linear(x, w)
        expected = torch.where(operation(y > 0, y < 1), y, 0.0)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("fc.weight").axis(0).select([1])]
        )
    )
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.fc.weight.grad, w.grad)


@pytest.mark.parametrize("container", [list, tuple, torch.Size])
def test_unflatten_configuration_survives_plan_and_checkpoint(container, execution_device):
    def make():
        return nn.Sequential(
            nn.Linear(4, 6), nn.Unflatten(1, container((2, 3))), nn.Flatten(1), nn.Linear(6, 2)
        )

    model = make()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    kept = [3, 4, 5]
    expected = F.linear(
        F.linear(x, model[0].weight[kept], model[0].bias[kept]),
        model[3].weight[:, kept],
        model[3].bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("0.weight").axis(0).select([0, 1, 2])]
    )
    plan = PruningPlan.from_dict(plan.to_dict())
    assert model[1].unflattened_size == container((2, 3))
    Pruner(model).apply(plan)
    Pruner(original).apply(plan)
    assert type(model[1].unflattened_size) is container
    assert model[1].unflattened_size == container((1, 3))
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(original(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = make()
    load_checkpoint(restored, stream)
    assert type(restored[1].unflattened_size) is container
    torch.testing.assert_close(restored(x), expected)


def test_attribute_barrier_preserves_independent_pruning_and_readonly_planning(
    monkeypatch, execution_device
):
    class Model(nn.Module):
        def __deepcopy__(self, memo):
            raise AssertionError("Configuration verification must not call custom model copying")

        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)
            self.safe = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Linear(6, 2))

        def forward(self, x):
            return self.b(self.a(x)), x[:, self.a.out_features - 1], self.safe(x)

    model = Model().eval()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    state = copy.deepcopy(model.state_dict())
    parameters = tuple(model.parameters())
    buffer = model.safe[1].running_mean
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if execution_device == "cuda" else None

    def forbid_parameter_copy(self, memo):
        raise AssertionError("Planning must not copy model weights")

    monkeypatch.setattr(nn.Parameter, "__deepcopy__", forbid_parameter_copy)
    with pytest.raises(PlanningError, match="structure"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("safe.0.weight").axis(0).select([1])]
    )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert model.safe[1].running_mean is buffer
    torch.testing.assert_close(model.state_dict(), state)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
    untouched = model(x)[:2]
    Pruner(model).apply(plan)
    torch.testing.assert_close(model(x)[:2], untouched)
    assert model.safe[0].out_features == 5
