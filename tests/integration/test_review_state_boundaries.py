"""State ownership, guards and callback semantics across public persistence APIs."""

import copy
import io
from datetime import date

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, StaleGraphError
from torch_kirigami.pruning import (
    ExecutionError,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("change", ["value", "insert", "delete", "order"])
def test_dictionary_configuration_is_guarded_through_plan_and_checkpoint(
    nested, change, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(2, 4)
            self.a = nn.Linear(4, 1)
            self.b = nn.Linear(4, 1)
            config = {"branch": "a", "flag": True}
            self.config = [{"inner": config}] if nested else config

        def forward(self, x):
            config = self.config[0]["inner"] if nested else self.config
            y = self.first(x)
            return self.a(y) if config["branch"] == "a" else self.b(y)

    source = Model()
    graph = DependencyGraph.build(source, args=(torch.ones(2, 2),))
    plan = PruningPlan.from_dict(
        Pruner(source, graph=graph)
        .plan(remove=[graph.parameter("first.weight").axis(0).select([0])])
        .to_dict()
    )
    changed = copy.deepcopy(source)
    config = changed.config[0]["inner"] if nested else changed.config
    if change == "value":
        config["branch"] = "b"
    elif change == "insert":
        config["extra"] = 1
    elif change == "delete":
        del config["flag"]
    else:
        flag = config.pop("branch")
        config["branch"] = flag
    old = changed.first.weight
    with pytest.raises(ExecutionError, match=r"configuration|preconditions"):
        Pruner(changed).apply(plan)
    assert changed.first.weight is old
    source.config = changed.config
    with pytest.raises(StaleGraphError):
        graph.validate()
    original = Model()
    Pruner(original).apply(plan)
    stream = io.BytesIO()
    save_checkpoint(original, stream)
    stream.seek(0)
    target = Model()
    load_checkpoint(target, stream)
    x = torch.ones(2, 2)
    torch.testing.assert_close(original(x), target(x))
    target(x).sum().backward()


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_ordinary_tensor_and_parent_references_keep_registration_categories(
    checkpoint, fail, execution_device
):
    class Model(nn.Module):
        fail_commit = False

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            object.__setattr__(self, "cached", self.weight)
            self.child = nn.Identity()
            object.__setattr__(self.child, "parent", self)

        def __setattr__(self, name, value):
            if name == "_kirigami_structure" and self.fail_commit:
                raise RuntimeError("injected commit failure")
            super().__setattr__(name, value)

        def forward(self, x):
            return self.cached * x

    model = Model()
    original = model.weight
    graph = DependencyGraph.build(model, args=(torch.ones(4),))
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("weight").axis(0).select([0])], preserve_io=False
    )
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    Model.fail_commit = fail

    def apply():
        return load_checkpoint(model, stream) if checkpoint else Pruner(model).apply(plan)

    if fail:
        with pytest.raises(ExecutionError, match="Commit failed"):
            apply()
        assert model.weight is original
    else:
        apply()
    assert model.cached is model.weight
    assert tuple(model._parameters) == ("weight",)
    assert not model.child._modules and model.child.parent is model
    model.eval()
    Model.fail_commit = False
    save_checkpoint(model, io.BytesIO())
    model(torch.ones_like(model.weight)).sum().backward()


@pytest.mark.parametrize("fail", [False, True])
def test_slot_extra_state_restores_values_and_gradients(fail, execution_device):
    class Model(nn.Module):
        __slots__ = ("scale",)
        fail_commit = False

        def __setattr__(self, name, value):
            if name == "_kirigami_structure" and self.fail_commit:
                raise RuntimeError("injected failure")
            super().__setattr__(name, value)

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.scale = 2

        def forward(self, x):
            return self.weight * x * self.scale

        def get_extra_state(self):
            return {"scale": self.scale}

        def set_extra_state(self, state):
            self.scale = state["scale"]

    source, target = Model(), Model()
    source.scale = 3
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    old = target.weight
    Model.fail_commit = fail
    if fail:
        with pytest.raises(ExecutionError, match="Commit failed"):
            load_checkpoint(target, stream)
        assert target.scale == 2 and target.weight is old
        Model.fail_commit = False
        stream.seek(0)
    load_checkpoint(target, stream)
    x = torch.ones(2)
    torch.testing.assert_close(source(x), target(x))
    target(x).sum().backward()
    torch.testing.assert_close(target.weight.grad, torch.full((2,), 3.0))


@pytest.mark.parametrize("hook", ["gradient", "post_accumulate"])
def test_loading_refuses_gradient_hooks_before_replacement(hook, execution_device):
    target = nn.Linear(2, 2)
    stream = io.BytesIO()
    save_checkpoint(target, stream)
    stream.seek(0)
    if hook == "gradient":
        handle = target.weight.register_hook(lambda grad: grad * 0)
    else:

        def clear_grad(parameter):
            parameter.grad.zero_()

        handle = target.weight.register_post_accumulate_grad_hook(clear_grad)
    old = target.weight
    with pytest.raises(ExecutionError, match="gradient hooks"):
        load_checkpoint(target, stream)
    assert target.weight is old
    target(torch.ones(1, 2)).sum().backward()
    assert not target.weight.grad.any()
    handle.remove()
    stream.seek(0)
    load_checkpoint(target, stream)
    handle = target.weight.register_hook(lambda grad: grad * 0)
    target(torch.ones(1, 2)).sum().backward()
    assert not target.weight.grad.any()
    handle.remove()


def test_raw_checkpoint_bypasses_constructor_dependent_state_codecs(execution_device):
    class Delta(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def forward(self, x):
            return x * self.weight

        def state_dict(self, *args, **kwargs):
            state = super().state_dict(*args, **kwargs)
            state["weight"] = state["weight"] - 1
            return state

        def _load_from_state_dict(self, state, prefix, *args, **kwargs):
            state = dict(state)
            state[prefix + "weight"] = state[prefix + "weight"] + self.weight
            return super()._load_from_state_dict(state, prefix, *args, **kwargs)

    source, native, target = Delta(), Delta(), Delta()
    with torch.no_grad():
        source.weight.fill_(3)
    native.load_state_dict(source.state_dict())
    torch.testing.assert_close(native.weight, source.weight)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    torch.testing.assert_close(payload["state_dict"]["weight"], torch.full((2,), 3.0))
    stream.seek(0)
    assert load_checkpoint(target, stream) is target
    torch.testing.assert_close(target.weight, native.weight)
    target(torch.ones(2)).sum().backward()
    torch.testing.assert_close(target.weight.grad, torch.ones(2))


@pytest.mark.parametrize("state", ["date", "parameter", "view"])
def test_unrestorable_extra_state_is_rejected_before_writing(state, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def get_extra_state(self):
            value = (
                date(2026, 1, 1)
                if state == "date"
                else self.weight
                if state == "parameter"
                else self.weight.detach()
            )
            return {"nested": [value]}

        def set_extra_state(self, state):
            self.extra = state

    stream = io.BytesIO(b"keep existing file")
    with pytest.raises(ExecutionError, match="extra state"):
        save_checkpoint(Model(), stream)
    assert stream.getvalue() == b"keep existing file"


@pytest.mark.parametrize("when", ["before_plan", "after_plan", "unaffected"])
def test_pruning_never_discards_hooks_on_replaced_parameters(when, execution_device):
    model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2))
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    pruner = Pruner(model, graph=graph)
    request = [graph.parameter("0.weight").axis(0).select([0])]
    old = model[0].weight
    if when == "before_plan":
        handle = old.register_hook(lambda grad: grad * 0)
        with pytest.raises(PlanningError, match="gradient hooks"):
            pruner.plan(remove=request)
    else:
        plan = pruner.plan(remove=request)
        parameter = model[1].bias if when == "unaffected" else old
        handle = parameter.register_hook(lambda grad: grad * 0)
        if when == "after_plan":
            with pytest.raises(ExecutionError, match="gradient hooks"):
                pruner.apply(plan)
        else:
            pruner.apply(plan)
            assert model[1].bias is parameter
    assert (model[0].weight is old) == (when != "unaffected")
    model(torch.ones(2, 2)).sum().backward()
    assert not (model[1].bias if when == "unaffected" else old).grad.any()
    handle.remove()


@pytest.mark.parametrize(
    "method", ["state_dict", "load_state_dict", "_save_to_state_dict", "_load_from_state_dict"]
)
@pytest.mark.parametrize("placement", ["class", "instance", "spoofed_callable"])
def test_custom_state_codecs_are_not_executed_for_raw_checkpoints(
    method, placement, monkeypatch, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            object.__setattr__(self, "cached", self.weight)

        def forward(self, x):
            return self.cached * x

    source, target = Model(), Model()
    with torch.no_grad():
        source.weight.copy_(torch.arange(4.0))
    graph = DependencyGraph.build(source, args=(torch.ones(4),))
    Pruner(source, graph=graph).prune(
        remove=[graph.parameter("weight").axis(0).select([1])], preserve_io=False
    )
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    restored = load_checkpoint(Model(), stream)
    assert restored.cached is restored.weight and not restored._load_state_dict_pre_hooks
    torch.testing.assert_close(restored(torch.ones(3)), torch.tensor([0.0, 2.0, 3.0]))
    restored(torch.ones(3)).sum().backward()
    torch.testing.assert_close(restored.weight.grad, torch.ones(3))

    calls = []

    def custom(self, *args, **kwargs):
        calls.append(self)
        raise AssertionError("Unsupported user codec must not run")

    # The format never dispatches codecs, regardless of their claimed provenance.
    custom.__module__ = "torch.nn.modules.module"
    if placement == "class":
        monkeypatch.setattr(Model, method, custom)
    elif placement == "instance":
        monkeypatch.setattr(target, method, custom.__get__(target, Model))
    else:

        class Spoofed:
            def __init__(self):
                self.__self__ = target
                self.__func__ = getattr(nn.Module, method)

            def __call__(self, *args, **kwargs):
                return custom(target, *args, **kwargs)

        monkeypatch.setattr(target, method, Spoofed())
    stream.seek(0)
    assert load_checkpoint(target, stream) is target
    assert target.cached is target.weight and target.weight.shape == (3,)
    torch.testing.assert_close(target(torch.ones(3)), torch.tensor([0.0, 2.0, 3.0]))
    destination = io.BytesIO()
    save_checkpoint(target, destination)
    destination.seek(0)
    payload = torch.load(destination, weights_only=True)
    assert set(payload["state_dict"]) == {"weight"}
    torch.testing.assert_close(payload["state_dict"]["weight"], torch.tensor([0.0, 2.0, 3.0]))
    assert calls == [] and not target._load_state_dict_pre_hooks


@pytest.mark.parametrize("payload", ["tensor_attribute", "metadata", "cycle", "deep", "clone"])
def test_checkpoint_data_tree_is_closed_before_save(payload, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))
            self.extra = {"value": 3}

        def forward(self, x):
            return x * self.weight

        def get_extra_state(self):
            return self.extra

        def set_extra_state(self, state):
            self.extra = state

    model = Model()
    if payload == "metadata":
        model._version = date(2026, 1, 1)
    if payload in ("tensor_attribute", "clone"):
        model.extra = model.weight.detach().clone()
        if payload == "tensor_attribute":
            model.extra.hidden = date(2026, 1, 1)
    elif payload == "cycle":
        model.extra["cycle"] = model.extra
    elif payload == "deep":
        for _ in range(60):
            model.extra = [model.extra]
    stream = io.BytesIO(b"unchanged")
    if payload not in ("clone", "metadata"):
        with pytest.raises(ExecutionError, match=r"extra state|tensor payload"):
            save_checkpoint(model, stream)
        assert stream.getvalue() == b"unchanged"
    else:
        save_checkpoint(model, stream)
        stream.seek(0)
        target = load_checkpoint(Model(), stream)
        torch.testing.assert_close(target.extra, model.extra)
        assert target.extra is not target.weight
        target(torch.ones(2)).sum().backward()
