import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph, StaleGraphError
from torch_kirigami.pruning import (
    ChannelRatio,
    ExecutionError,
    Magnitude,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


class CachedWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.cached = [{"weight": self.weight}]
        self.alias = self.cached
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        return self.out(F.linear(x, self.cached[0]["weight"]))


@pytest.mark.parametrize("fail", [False, True])
def test_container_buffer_isolation_success_and_failure(fail, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.zeros(1))
            self.cached = [self.counter]
            self.alias = self.cached

        def forward(self, x):
            self.cached[0].add_(1)
            if fail:
                raise RuntimeError("stop after write")
            return x + self.cached[0]

    model = Model()
    before, container = model.counter, model.cached
    if fail:
        with pytest.raises(CaptureError):
            DependencyGraph.build(model, args=(torch.ones(2),))
    else:
        DependencyGraph.build(model, args=(torch.ones(2),))
    assert model.counter is before and before.item() == 0
    assert model.cached is container and model.alias is container and container[0] is before


def test_cached_parameter_plan_apply_and_checkpoint(execution_device):
    model = CachedWeight()
    x = torch.randn(2, 4)
    keep = [0, 2, 3]
    expected = F.linear(F.linear(x, model.weight[keep]), model.out.weight[:, keep], model.out.bias)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("weight").axis(0).select([1])])
    portable = PruningPlan.from_dict(plan.to_dict())
    Pruner(model).apply(portable)
    assert model.cached is model.alias
    assert model.cached[0]["weight"] is model.weight
    torch.testing.assert_close(model(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(CachedWeight(), stream)
    assert restored.cached is restored.alias and restored.cached[0]["weight"] is restored.weight
    torch.testing.assert_close(restored(x), expected)


def test_cached_reference_changes_invalidate_plan():
    model = CachedWeight()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("weight").axis(0).select([1])])
    model.cached[0]["weight"] = model.weight.detach().clone()
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError):
        Pruner(model).apply(plan)


def test_mixed_reference_container_guards_scalar_neighbors():
    model = CachedWeight()
    model.cached.append({"config": [True]})
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    model.cached[-1]["config"][0] = 1
    with pytest.raises(StaleGraphError):
        graph.validate()


def test_cached_storage_view_rejected_before_execution():
    model = nn.Module()
    model.register_buffer("counter", torch.zeros(2))
    model.view_cache = [model.counter[:1]]
    with pytest.raises(CaptureError, match="separate view"):
        DependencyGraph.build(model, args=())
    assert not model.counter.any()


class ParentState(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.Identity()
        self.child.state = {"scale": 1}
        self.drop = nn.Dropout(1)
        self.layers = [self.drop]

    def forward(self, x):
        return self.layers[0](x) * self.child.state["scale"]

    def get_extra_state(self):
        return self.child.state

    def set_extra_state(self, state):
        self.child.state = state
        self.drop = nn.Dropout(1)
        self.layers = [self.drop]


def test_parent_extra_state_commits_entire_final_module_graph(execution_device):
    source = ParentState()
    source.child.state = {"scale": 3}
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = ParentState()
    original_drop = target.drop
    restored = load_checkpoint(target, stream)
    assert restored.drop is original_drop and restored.layers[0] is original_drop
    assert restored.child.state == {"scale": 3}
    source.eval()
    restored.eval()
    torch.testing.assert_close(restored(torch.ones(2)), source(torch.ones(2)))


@pytest.mark.parametrize("replacement", [1, 1.0])
def test_scalar_types_are_part_of_configuration_guards(replacement):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)
            self.flag = True

        def forward(self, x):
            return self.b(self.a(x)) if self.flag is True else self.b(x)

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    model.flag = replacement
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError):
        Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))


def test_nan_configuration_checkpoint_equality():
    source = nn.Linear(2, 2)
    source.tag = float("nan")
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = nn.Linear(2, 2)
    target.tag = float("nan")
    load_checkpoint(target, stream)
    torch.testing.assert_close(target.weight, source.weight)


def test_channels_last_weight_layout_is_validated_before_execution(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 3).to(memory_format=torch.channels_last)

        def forward(self, x):
            y = self.conv(x)
            return y.permute(0, 2, 3, 1).view(-1, y.size(1))

    model = Model()
    x = torch.randn(2, 3, 6, 6)
    expected = model(x)[:, [0, 2, 3, 5]]
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("conv.weight").axis(0).select([1, 4])], preserve_io=False
    )
    assert (
        next(r for r in plan.recipes if r.tensor.paths == ("conv.weight",)).memory_format
        == "channels_last"
    )
    Pruner(model).apply(plan)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("mode", ["swapaxes", "inplace", "narrow"])
def test_named_arguments_and_negative_narrow(mode, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)
            self.relu = nn.ReLU(inplace=True)
            self.b = nn.Linear(6, 2)

        def forward(self, x):
            y = self.a(x)
            if mode == "swapaxes":
                y = torch.swapaxes(input=y, axis0=0, axis1=1)
                y = y.swapaxes(axis0=0, axis1=1)
            elif mode == "inplace":
                y = self.relu(input=y)
            else:
                y = torch.narrow(y, dim=0, start=-1, length=1)
            return self.b(y)

    model = Model()
    x = torch.randn(2, 4)
    keep = [0, 2, 3, 4, 5]
    y = model.a(x)[:, keep]
    if mode == "inplace":
        y = y.relu()
    elif mode == "narrow":
        y = y[-1:]
    expected = F.linear(y, model.b.weight[:, keep], model.b.bias)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("a.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)


def test_unpool_keyword_order_and_expand_template_dependencies(execution_device):
    class Unpool(nn.Module):
        def forward(self, values, indices):
            return F.max_unpool1d(indices=indices, input=values, kernel_size=2)

    values, indices = F.max_pool1d(torch.randn(1, 4, 6), 2, return_indices=True)
    graph = DependencyGraph.build(Unpool(), args=(values, indices))
    refs = graph.interfaces()
    source = next(
        r
        for r in refs
        if graph.metadata(r).dtype == values.dtype and r.shape == tuple(values.shape)
    )
    index_ref = next(r for r in refs if graph.metadata(r).dtype == torch.int64)
    impact = graph.propagate(remove=[source.axis(1).select([1])])
    assert list(impact.selection(index_ref).project(1)) == [1]

    class Expand(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 1)
            self.b = nn.Linear(4, 6)
            self.c = nn.Linear(6, 2)

        def forward(self, x):
            return self.c(self.a(x).expand_as(self.b(x)))

    model = Expand()
    x = torch.randn(2, 4)
    expected = F.linear(model.a(x).expand(2, 5), model.c.weight[:, [0, 2, 3, 4, 5]], model.c.bias)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("b.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)


def test_conjugate_complex_checkpoint_aliases(execution_device):
    source = nn.Module()
    value = torch.tensor([1 + 2j]).conj()
    source.register_buffer("left", value)
    source.register_buffer("right", value)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = nn.Module()
    initial = torch.zeros(1, dtype=torch.complex64)
    target.register_buffer("left", initial)
    target.register_buffer("right", initial)
    load_checkpoint(target, stream)
    assert target.left is target.right
    torch.testing.assert_close(target.left, value)


def test_overridden_state_dict_payload_rejected_before_file_write():
    class Renamed(nn.Linear):
        def state_dict(self, *args, **kwargs):
            result = super().state_dict(*args, **kwargs)
            result["saved_weight"] = result.pop("weight")
            return result

        def load_state_dict(self, state, **kwargs):
            state = dict(state)
            state["weight"] = state.pop("saved_weight")
            return super().load_state_dict(state, **kwargs)

    source = Renamed(2, 2)
    source.load_state_dict(source.state_dict())
    stream = io.BytesIO()
    with pytest.raises(ExecutionError, match="payload keys"):
        save_checkpoint(source, stream)
    assert not stream.getvalue()


@pytest.mark.parametrize("pre", [False, True])
def test_global_forward_hooks_rejected_at_build_and_apply(pre):
    from torch.nn.modules import module as runtime

    model = CachedWeight()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("weight").axis(0).select([1])])
    register = (
        runtime.register_module_forward_pre_hook if pre else runtime.register_module_forward_hook
    )
    handle = register(lambda *args: None)
    try:
        with pytest.raises(CaptureError, match="global"):
            DependencyGraph.build(model, args=(torch.randn(2, 4),))
        with pytest.raises(ExecutionError, match="global"):
            Pruner(model).apply(plan)
    finally:
        handle.remove()


def test_cached_reference_commit_failure_rolls_back(monkeypatch):
    model = CachedWeight()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("weight").axis(0).select([1])])
    weight, cached = model.weight, model.cached
    setter = CachedWeight.__setattr__

    def fail(self, name, value):
        if self is model and name == "alias":
            raise RuntimeError("Injected reference commit failure")
        setter(self, name, value)

    monkeypatch.setattr(CachedWeight, "__setattr__", fail)
    with pytest.raises(ExecutionError, match="Commit failed"):
        Pruner(model).apply(plan)
    assert model.weight is weight and model.cached is cached and model.alias is cached
    assert cached[0]["weight"] is weight


def test_cached_buffer_checkpoint_without_extra_state(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.ones(1))
            self.refs = {"scale": self.scale}

        def forward(self, x):
            return x * self.refs["scale"]

    source = Model()
    source.scale.fill_(3)
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    restored = load_checkpoint(Model(), stream)
    assert restored.refs["scale"] is restored.scale
    torch.testing.assert_close(restored(torch.ones(2)), torch.full((2,), 3.0))


def test_default_zero_budget_avoids_candidate_scoring(monkeypatch):
    model = nn.Sequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    calls = []
    original = DependencyGraph.propagate

    def counted(self, **kwargs):
        calls.append(1)
        return original(self, **kwargs)

    def forbidden_metric(context, batch):
        raise AssertionError("Zero budget must not score candidates")

    monkeypatch.setattr(DependencyGraph, "propagate", counted)
    plan = Pruner(model, graph=graph).plan(budget=ChannelRatio(0), metric=forbidden_metric)
    assert not plan.recipes and len(calls) <= 5


def test_final_accepted_compilation_is_reused(monkeypatch):
    from torch_kirigami.pruning import planner

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    original = planner.compile_recipes
    counts = {}

    def counted(graph, operations, impact):
        key = tuple((s.tensor.id, s.regions) for s in impact.requested)
        counts[key] = counts.get(key, 0) + 1
        return original(graph, operations, impact)

    monkeypatch.setattr(planner, "compile_recipes", counted)
    plan = Pruner(model, graph=graph).plan(budget=ChannelRatio(0.25), metric=Magnitude())
    assert plan.recipes and max(counts.values()) == 1
