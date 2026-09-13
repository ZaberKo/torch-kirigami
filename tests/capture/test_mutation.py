"""capture / mutation contracts."""

import operator

import pytest
import torch
from torch import nn
from torch.nn.modules.module import register_module_buffer_registration_hook

from tests.support.pruning import build
from torch_kirigami import (
    CallEffects,
    CaptureError,
    DependencyGraph,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
)
from torch_kirigami.capture import _reject_parameter_writes
from torch_kirigami.operators.effects import native_effects
from torch_kirigami.pruning import (
    PlanningError,
)


def output_writer(input, out=None):
    return torch.add(input, 1, out=out)


@pytest.mark.parametrize("kind", ["contiguous", "function", "module"])
def test_all_known_parameter_alias_mutations_rejected(kind):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            if kind == "contiguous":
                self.weight.contiguous().zero_()
            elif kind == "function":
                torch.relu_(self.weight)
            else:
                self.relu(self.weight)
            return x

    model = Model()
    original = model.weight.detach().clone()
    with pytest.raises(CaptureError, match="write"):
        DependencyGraph.build(model, args=(torch.randn(2, 4),))
    torch.testing.assert_close(model.weight, original)


def test_parameter_clone_is_safe_to_mutate():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            temporary = self.weight.clone()
            temporary.relu_()
            return x @ temporary

    model = Model()
    original = model.weight.detach().clone()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    torch.testing.assert_close(model.weight, original)
    assert (
        graph.propagate(remove=[graph.parameter("weight").axis(1).select([1])]).status == "resolved"
    )


@pytest.mark.parametrize("form", ["native", "keyword", "positional"])
def test_non_none_output_is_rejected_before_execution(form, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("target", torch.ones(3))

        def forward(self, x):
            if form == "native":
                return torch.add(x, 1, out=self.target)
            if form == "keyword":
                return output_writer(x, out=self.target)
            return output_writer(x, self.target)

    registry = OperatorRegistry.default().register(
        output_writer, OperatorRule(lambda ctx: OperatorSpec())
    )
    model = Model()
    original = model.target
    sample = torch.zeros(3)
    with pytest.raises(CaptureError, match="out="):
        DependencyGraph.build(model, args=(sample,), operators=registry)
    assert model.target is original
    torch.testing.assert_close(model.target, torch.ones(3))
    torch.testing.assert_close(sample, torch.zeros(3))


def test_capture_rejects_registration_callbacks_before_touching_state(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.ones(3))
            self.alias = self.b
            self.cached = [self.b]

        def forward(self, x):
            return x + self.b

    model = Model()
    old, cached = model.b, model.cached
    calls = []
    rng = torch.get_rng_state().clone()
    sample = torch.zeros(3)

    def registration(module, name, buffer):
        calls.append(name)
        return buffer + 10

    handle = register_module_buffer_registration_hook(registration)
    try:
        with pytest.raises(CaptureError, match="registration"):
            DependencyGraph.build(model, args=(sample,))
    finally:
        handle.remove()
    assert calls == []
    assert model.b is model.alias is cached[0] is old and model.cached is cached
    torch.testing.assert_close(model.b, torch.ones(3))
    torch.testing.assert_close(torch.get_rng_state(), rng)


@pytest.mark.parametrize("unsafe", [True, False])
def test_inplace_single_consumer_proof(unsafe):
    class Inplace(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.out = nn.Linear(6, 2)

        def forward(self, x):
            y = self.fc(x)
            if unsafe:
                z = y + 1
                return self.out(y.relu_() + z)
            return self.out(y.relu_())

    model = Inplace()
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    remove = [graph.parameter("fc.weight").axis(0).select([1])]
    if unsafe:
        with pytest.raises(PlanningError, match="in-place"):
            pruner.plan(remove=remove)
    else:
        plan = pruner.plan(remove=remove)
        pruner.apply(plan)
        assert model(x).shape == (2, 2)


@pytest.mark.parametrize(
    "target",
    [
        operator.iadd,
        operator.iand,
        operator.setitem,
        "__iadd__",
        "__setitem__",
        "add_",
        torch.relu_,
    ],
)
def test_shared_effect_detection_still_rejects_real_parameter_writes(target):
    graph = torch.fx.Graph()
    weight = graph.get_attr("weight")
    args = (weight, 0, 1) if target in (operator.setitem, "__setitem__") else (weight, 0)
    node = graph.create_node(
        "call_method" if isinstance(target, str) else "call_function", target, args
    )
    graph.output(weight)
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(3))
    gm = torch.fx.GraphModule(model, graph)
    assert native_effects(node, None, fresh_output=False).mutates_input
    with pytest.raises(CaptureError, match="write"):
        _reject_parameter_writes(gm, OperatorRegistry.default(), set())
    torch.testing.assert_close(model.weight, torch.ones(3))


@pytest.mark.parametrize("fresh_output", [False, True])
def test_capture_queries_each_call_effect_once(fresh_output, execution_device):
    class Copy(nn.Module):
        def forward(self, x):
            return x.clone()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(16.0).reshape(4, 4) - 8)
            self.copy = Copy()

        def forward(self, x):
            return x @ self.copy(self.weight).relu_()

    calls = []

    def effects(node, module):
        calls.append((node.name, module))
        return CallEffects(fresh_output=fresh_output)

    registry = OperatorRegistry.default().register(
        Copy, OperatorRule(lambda ctx: OperatorSpec(), effects=effects)
    )
    model = Model()
    original = model.weight.detach().clone()
    sample = torch.ones(2, 4)
    if fresh_output:
        graph = DependencyGraph.build(model, args=(sample,), operators=registry)
        assert calls == [(graph.calls("copy")[0].name, model.copy)]
        torch.testing.assert_close(model(sample), sample @ original.clamp_min(0))
    else:
        with pytest.raises(CaptureError, match="Parameter/alias write"):
            DependencyGraph.build(model, args=(sample,), operators=registry)
        assert len(calls) == 1 and calls[0][1] is model.copy
    torch.testing.assert_close(model.weight, original)
    assert model.weight.grad is None
