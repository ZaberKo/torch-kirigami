"""Eager tensor decisions and constant premises across build/plan/apply."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import CaptureError, DependencyGraph, StaleGraphError
from torch_kirigami.pruning import ExecutionError, PlanningError, Pruner, PruningPlan


class Routed(nn.Module):
    def __init__(self, decision, registered=True):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(4, 6), nn.Linear(6, 2), nn.Linear(6, 2)
        if registered:
            self.register_buffer("route", torch.zeros(2))
        else:
            self.route = torch.zeros(2)
        self.decision = decision

    def forward(self, x):
        y = self.a(x)
        if self.decision == "item":
            branch = self.route[0].item() == 0
        elif self.decision == "bool":
            branch = bool(self.route.sum() == 0)
        elif self.decision == "equal":
            branch = torch.equal(self.route, torch.zeros_like(self.route))
        elif self.decision == "tolist":
            branch = self.route.tolist()[0] == 0
        elif self.decision == "nonzero_length":
            branch = len(self.route.nonzero()) == 0
        elif self.decision == "metadata":
            branch = self.route.shape[0] == 2
        else:
            branch = True
        return self.b(y) if branch else self.c(y)


@pytest.mark.parametrize("registered", [True, False])
@pytest.mark.parametrize("decision", ["item", "bool", "equal", "tolist", "nonzero_length"])
def test_eager_data_decisions_fail_before_a_partial_graph(decision, registered, execution_device):
    model = Routed(decision, registered)
    x = torch.randn(2, 4)
    old, sample = copy.deepcopy(model.state_dict()), x.clone()
    rng = torch.random.get_rng_state().clone()
    route = model.route
    with pytest.raises(CaptureError, match="Unrecorded Tensor data/derived-metadata"):
        DependencyGraph.build(model, args=(x,))
    assert model.route is route and torch.count_nonzero(route) == 0
    torch.testing.assert_close(x, sample)
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, old[name])
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("registered", [True, False])
@pytest.mark.parametrize("decision", ["metadata", "config"])
def test_guarded_metadata_and_python_configuration_allow_pruning(
    decision, registered, execution_device
):
    model = Routed(decision, registered)
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan_remove([graph.parameter("a.weight").axis(0).select([1])])
        .to_dict()
    )
    Pruner(model).apply(plan)
    expected = original.a(x)
    expected[:, 1] = 0
    torch.testing.assert_close(model(x), original.b(expected))
    model(x).sum().backward()


class Scaled(nn.Module):
    def __init__(self, width=1, container=False):
        super().__init__()
        self.a, self.b = nn.Linear(4, 6), nn.Linear(6, 2)
        self.container = container
        self.scale = {"nested": [torch.ones(width)]} if container else torch.ones(width)

    def forward(self, x):
        scale = self.scale["nested"][0] if self.container else self.scale
        return self.b(self.a(x) * scale)


@pytest.mark.parametrize("container", [True, False])
@pytest.mark.parametrize("change", ["shape", "dtype", "stride", "requires_grad"])
def test_ordinary_tensor_premises_guard_live_and_portable_plans(
    container, change, execution_device
):
    # Non-contiguous singleton constants exercise stride without changing broadcasting.
    model = Scaled(container=container)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan_remove([graph.parameter("a.weight").axis(0).select([1])])
        .to_dict()
    )
    target = copy.deepcopy(model)
    replacement = {
        "shape": lambda: torch.ones(6),
        "dtype": lambda: torch.ones(1, dtype=torch.float64),
        "stride": lambda: torch.empty_strided((1,), (2,)).fill_(1),
        "requires_grad": lambda: torch.ones(1, requires_grad=True),
    }[change]()
    for instance in (target, model):
        if container:
            instance.scale["nested"][0] = replacement
        else:
            instance.scale = replacement
    old = tuple(target.parameters())
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError):
        Pruner(target).apply(plan)
    assert all(a is b for a, b in zip(old, target.parameters(), strict=True))


@pytest.mark.parametrize("container", [True, False])
def test_ordinary_constant_values_remain_constructor_owned(container, execution_device):
    model = Scaled(container=container)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan_remove([graph.parameter("a.weight").axis(0).select([1])])
        .to_dict()
    )
    target = copy.deepcopy(model)
    constant = target.scale["nested"][0] if container else target.scale
    constant.fill_(3)
    reference = target.a(x)
    reference[:, 1] = 0
    expected = target.b(reference * 3)
    Pruner(target).apply(plan)
    torch.testing.assert_close(target(x), expected)
    target(x).sum().backward()


@pytest.mark.parametrize("query", ["shape", "numel", "length"])
def test_eager_buffer_metadata_blocks_only_its_affected_component(query, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = nn.Linear(4, 6), nn.Linear(6, 2), nn.Linear(6, 2)
            self.independent = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
            self.register_buffer("scale", torch.ones(6))

        def forward(self, x):
            y = self.a(x) * self.scale
            width = (
                self.scale.shape[0]
                if query == "shape"
                else self.scale.numel()
                if query == "numel"
                else len(self.scale)
            )
            return (self.b(y) if width == 6 else self.c(y)), self.independent(x)

    model = Model()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    old = tuple(model.parameters())
    with pytest.raises(PlanningError, match="metadata"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    assert all(a is b for a, b in zip(old, model.parameters(), strict=True))
    expected = model(x)[0].detach()
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("independent.0.weight").axis(0).select([1])]
        )
    )
    torch.testing.assert_close(model(x)[0], expected)
    sum(y.sum() for y in model(x)).backward()


@pytest.mark.parametrize("derived", ["nonzero", "view", "sum"])
def test_eager_tensor_folding_cannot_lose_its_source(derived, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(6, 2)
            self.register_buffer("route", torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

        def forward(self, x):
            value = (
                self.route.nonzero().flatten()
                if derived == "nonzero"
                else self.route.view(-1)
                if derived == "view"
                else self.route.sum()
            )
            return self.b(self.a(x) * value)

    model = Model()
    before = model.route.clone()
    with pytest.raises(CaptureError, match="folded"):
        DependencyGraph.build(model, args=(torch.ones(2, 4),))
    torch.testing.assert_close(model.route, before)


@pytest.mark.parametrize("field", ["grad", "grad_fn", "is_leaf"])
def test_runtime_tensor_properties_are_not_structural_metadata(field, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 2)
            self.register_buffer("route", torch.ones(1, requires_grad=True))

        def forward(self, x):
            return self.a(x) if getattr(self.route, field) is None else self.a(x) + 1

    with pytest.raises(CaptureError, match="runtime-state"):
        DependencyGraph.build(Model(), args=(torch.ones(2, 4),))


@pytest.mark.parametrize(
    "values",
    [
        (torch.float32, torch.float64),
        (torch.device("cpu"), torch.device("cpu:0")),
        (torch.contiguous_format, torch.channels_last),
        (torch.strided, torch.sparse_coo),
    ],
)
@pytest.mark.parametrize("container", [False, True])
def test_native_torch_configuration_is_guarded_and_serialized(values, container, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.c = nn.Linear(4, 6), nn.Linear(6, 2), nn.Linear(6, 2)
            self.option = [values[0]] if container else values[0]

        def forward(self, x):
            option = self.option[0] if container else self.option
            return self.b(self.a(x)) if option == values[0] else self.c(self.a(x))

    model = Model()
    x = torch.ones(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan_remove([graph.parameter("a.weight").axis(0).select([1])])
        .to_dict()
    )
    compatible = copy.deepcopy(model)
    Pruner(compatible).apply(plan)
    compatible(x).sum().backward()
    model.option = [values[1]] if container else values[1]
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError):
        Pruner(model).apply(plan)


@pytest.mark.parametrize("container", [False, True])
@pytest.mark.parametrize("fails_later", [False, True])
def test_proxy_dependent_ordinary_constant_writes_never_reach_original_state(
    container, fails_later, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = [torch.zeros(2)] if container else torch.zeros(2)
            self.a = nn.Linear(2, 2)

        def forward(self, x):
            value = self.scale[0] if container else self.scale
            value.add_(x.size(0))
            y = self.a(x) + value
            return y.reshape(99) if fails_later else y

    model = Model()
    value = model.scale[0] if container else model.scale
    original = value.clone()
    with pytest.raises(CaptureError, match="Unisolated Tensor constant write"):
        DependencyGraph.build(model, args=(torch.ones(1, 2),))
    torch.testing.assert_close(value, original)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("key_kind", ["tensor", "tuple", "module"])
def test_ordinary_dictionary_keys_cannot_hide_live_bindings(key_kind, execution_device):
    class Model(nn.Module):
        def __init__(self, valid):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 4))
            self.out = nn.Linear(6, 2)
            self.valid = valid
            key = (
                self.weight
                if key_kind == "tensor"
                else (self.weight,)
                if key_kind == "tuple"
                else self.out
            )
            self.refs = {"weight": self.weight} if valid else {key: "weight"}

        def forward(self, x):
            if self.valid:
                weight = self.refs["weight"]
            else:
                key = next(iter(self.refs))
                weight = (
                    key if key_kind == "tensor" else key[0] if key_kind == "tuple" else self.weight
                )
            return self.out(torch.nn.functional.linear(x, weight))

    bad = Model(False)
    x = torch.randn(2, 4)
    original = bad.weight
    keys = tuple(bad.refs)
    assert bad(x).shape == (2, 2)
    with pytest.raises(CaptureError, match="Unsupported key"):
        DependencyGraph.build(bad, args=(x,))
    assert bad.weight is original and all(a is b for a, b in zip(keys, bad.refs, strict=True))
    good = Model(True)
    old_weight, old_out, old_bias = (
        good.weight.detach().clone(),
        good.out.weight.detach().clone(),
        good.out.bias.detach().clone(),
    )
    graph = DependencyGraph.build(good, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(good, graph=graph)
        .plan_remove([graph.parameter("weight").axis(0).select([1])])
        .to_dict()
    )
    Pruner(good).apply(plan)
    assert good.refs["weight"] is good.weight
    keep = [0, 2, 3, 4, 5]
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(x, old_weight[keep]), old_out[:, keep], old_bias
    )
    torch.testing.assert_close(good(x), expected)
    good(x).sum().backward()


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("nan_component", ["real", "imag"])
def test_read_only_complex_nan_buffers_preserve_capture_and_independent_pruning(
    conjugate, nan_component, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(6, 2)
            value = (
                complex(float("nan"), 1) if nan_component == "real" else complex(1, float("nan"))
            )
            buffer = torch.full((4,), value, dtype=torch.complex64)
            self.register_buffer("offset", buffer.conj() if conjugate else buffer)

        def forward(self, x):
            return self.b(self.a(x)), x + self.offset

    model = Model()
    original = model.offset
    x = torch.randn(2, 4)
    expected = x + model.offset
    graph = DependencyGraph.build(model, args=(x,))
    assert model.offset is original and model.offset.is_conj() == conjugate
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan_remove([graph.parameter("a.weight").axis(0).select([1])])
        .to_dict()
    )
    Pruner(model).apply(plan)
    torch.testing.assert_close(model(x)[1], expected, equal_nan=True)
    model(x)[0].sum().backward()
    assert model.offset is original
