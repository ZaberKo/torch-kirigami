import io
import operator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph, StaleGraphError
from torch_kirigami.pruning import (
    ChannelRatio,
    ExecutionError,
    Greedy,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.selection import Selection, TensorRef


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)
        self.route = [False]

    def forward(self, x):
        return self.b(x) if self.route[0] else self.b(self.a(x))


def test_padding_shift_cannot_masquerade_as_identity(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv1d(1, 4, 1, bias=False)
            self.b = nn.Conv1d(4, 1, 1, bias=False)

        def forward(self, x):
            return self.b(F.pad(self.a(x), (0, 0, -1, 1)))

    model = Model()
    with torch.no_grad():
        model.a.weight.copy_(torch.arange(1, 5).reshape(4, 1, 1))
        model.b.weight.copy_(torch.arange(1, 5).reshape(1, 4, 1) * 10)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 1, 1),))
    remove = [graph.parameter("a.weight").axis(0).select([1])]
    assert graph.propagate(remove=remove).status != "resolved"
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan(remove=remove)


@pytest.mark.parametrize("where", ["root", "nested", "leaf"])
@pytest.mark.parametrize("pre", [False, True])
def test_forward_hooks_rejected_before_capture(where, pre):
    model = nn.Sequential(Chain())
    target = {"root": model, "nested": model[0], "leaf": model[0].a}[where]
    calls = []
    if pre:
        target.register_forward_pre_hook(lambda *args: calls.append(True))
    else:
        target.register_forward_hook(lambda *args: calls.append(True))
    with pytest.raises(CaptureError, match="hook"):
        DependencyGraph.build(model, args=(torch.randn(2, 4),))
    assert not calls


@pytest.mark.parametrize("where", ["root", "leaf"])
def test_apply_rechecks_added_hooks(where):
    model = Chain()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    (model if where == "root" else model.a).register_forward_hook(lambda *args: None)
    old = model.a.weight
    with pytest.raises(ExecutionError, match="hook"):
        Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    assert model.a.weight is old


def test_list_configuration_is_frozen_and_guarded_in_graph_and_plan():
    model = Chain()
    model.config = ([1, [2]], (3,))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    serialized = plan.to_dict()
    model.route[0] = True
    model.config[0][1].append(4)
    assert plan.to_dict() == serialized
    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(ExecutionError, match="preconditions"):
        Pruner(model).apply(PruningPlan.from_dict(serialized))
    assert model.b.weight.shape == (2, 4)


@pytest.mark.parametrize("method", [False, True])
def test_narrow_keywords_match_positional_coordinates(method, execution_device):
    class Model(Chain):
        def forward(self, x):
            y = self.a(x)
            y = (
                y.narrow(dim=0, start=0, length=1)
                if method
                else torch.narrow(y, dim=0, start=0, length=1)
            )
            return self.b(y)

    model = Model()
    x = torch.randn(2, 4)
    kept = [0, 2, 3]
    expected = F.linear(
        F.linear(x[:1], model.a.weight[kept], model.a.bias[kept]),
        model.b.weight[:, kept],
        model.b.bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("a.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("mask_first", [False, True])
def test_mha_keyword_order_does_not_reclassify_value(mask_first, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

        def forward(self, q, k, v, mask):
            if mask_first:
                return self.attn(key_padding_mask=mask, query=q, key=k, value=v)[0]
            return self.attn(query=q, key=k, value=v, key_padding_mask=mask)[0]

    model = Model()
    args = (*[torch.randn(2, 3, 8) for _ in range(3)], torch.zeros(2, 3, dtype=torch.bool))
    graph = DependencyGraph.build(model, args=args)
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("attn.out_proj.weight").axis(0).select([1, 5])],
        preserve_io=False,
    )
    assert plan.analysis.status == "resolved"


@pytest.mark.parametrize("op", [operator.floordiv, operator.mod])
def test_tensor_integer_arithmetic_propagates_channels(op, execution_device):
    class Model(Chain):
        def forward(self, x):
            return self.b(op(self.a(x), 2))

    model = Model()
    x = torch.randn(2, 4)
    kept = [0, 2, 3]
    expected = F.linear(op(model.a(x)[:, kept], 2), model.b.weight[:, kept], model.b.bias)
    graph = DependencyGraph.build(model, args=(x,))
    removal = graph.parameter("a.weight").axis(0).select([1])
    impact = graph.propagate(remove=[removal])
    assert list(impact.selection(graph.parameter("b.weight")).project(1)) == [1]
    Pruner(model, graph=graph).prune(remove=[removal])
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("intermediate", [False, True])
def test_zero_element_metadata_rejected(intermediate):
    class Model(Chain):
        def forward(self, x):
            return self.b(self.a(x[:0] if intermediate else x))

    with pytest.raises(CaptureError, match=r"[Zz]ero|empty"):
        DependencyGraph.build(Model(), args=(torch.empty(2 if intermediate else 0, 4),))


def test_empty_selection_preserves_zero_shape():
    ref = TensorRef("empty", (3, 0))
    selection = Selection(ref)
    assert not selection.project(0)
    assert selection.compact_shape() == (3, 0)


class ExtraState(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(1)
        self.layers = [self.drop]
        self.cache = {}
        self.flag = True

    def forward(self, x):
        return self.layers[0](x)

    def get_extra_state(self):
        return {"drop_cache": not hasattr(self, "cache"), "drop_flag": not hasattr(self, "flag")}

    def set_extra_state(self, state):
        if state["drop_cache"]:
            del self.cache
        if state["drop_flag"]:
            del self.flag


def test_checkpoint_extra_state_deletions_and_module_references(execution_device):
    source = ExtraState()
    del source.cache
    del source.flag
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    restored = load_checkpoint(ExtraState(), stream)
    assert not hasattr(restored, "cache") and not hasattr(restored, "flag")
    assert restored.layers[0] is restored.drop
    source.eval()
    restored.eval()
    torch.testing.assert_close(restored(torch.ones(2)), source(torch.ones(2)))


def test_checkpoint_state_dict_hooks_rejected_at_save():
    model = nn.Linear(2, 2)

    def save_hook(module, state, prefix, metadata):
        state[prefix + "saved_weight"] = state.pop(prefix + "weight")

    def load_hook(module, state, prefix, *args):
        state[prefix + "weight"] = state.pop(prefix + "saved_weight")

    model.register_state_dict_post_hook(save_hook)
    model.register_load_state_dict_pre_hook(load_hook)
    model.load_state_dict(model.state_dict())  # Valid PyTorch usage, outside our schema.
    with pytest.raises(ExecutionError, match="hook"):
        save_checkpoint(model, io.BytesIO())


def test_checkpoint_shared_nan_roundtrip(execution_device):
    module = nn.Linear(2, 2)
    model = nn.ModuleList([module, module])
    with torch.no_grad():
        module.weight[0, 0] = float("nan")
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    target = nn.Linear(2, 2)
    restored = load_checkpoint(nn.ModuleList([target, target]), stream)
    assert restored[0] is restored[1]
    torch.testing.assert_close(restored[0].weight, module.weight, equal_nan=True)


def test_checkpoint_rejects_internal_overlap_before_writing():
    model = nn.Module()
    model.register_buffer("overlap", torch.ones(1, 3).expand(2, 3))
    stream = io.BytesIO()
    with pytest.raises(ExecutionError, match="overlap"):
        save_checkpoint(model, stream)
    assert stream.getvalue() == b""


def test_padding_barrier_preserves_independent_branch(execution_device):
    class Model(Chain):
        def forward(self, x):
            return self.b(self.a(x)), F.pad(x, (-1, 1))

    model = Model()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("a.weight").axis(0).select([1])])
    assert model.b.in_features == 3
    torch.testing.assert_close(model(x)[1], F.pad(x, (-1, 1)))


@pytest.mark.parametrize("module_pad", [False, True])
def test_spatial_padding_keeps_channel_pruning(module_pad, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv1d(2, 4, 1)
            self.pad = nn.ConstantPad1d((1, 2), 0)
            self.out = nn.Conv1d(4, 2, 1)

        def forward(self, x):
            y = self.fc(x)
            return self.out(self.pad(y) if module_pad else F.pad(y, (1, 2)))

    model = Model()
    x = torch.randn(1, 2, 3)
    kept = [0, 2, 3]
    expected = F.conv1d(
        F.pad(model.fc(x)[:, kept], (1, 2)), model.out.weight[:, kept], model.out.bias
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("fc.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)


def test_list_configuration_checkpoint_retains_container_types():
    model = nn.Sequential(nn.AdaptiveAvgPool2d([2, 2]))
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(nn.Sequential(nn.AdaptiveAvgPool2d([2, 2])), stream)
    assert type(restored[0].output_size) is list
    assert restored[0].output_size == [2, 2]


def test_checkpoint_extra_state_transaction_restores_deleted_attributes(monkeypatch):
    source = ExtraState()
    del source.cache
    stream = io.BytesIO()
    save_checkpoint(source, stream)
    stream.seek(0)
    target = ExtraState()
    old_cache, old_layers = target.cache, target.layers
    original = ExtraState.__setattr__

    def fail(self, name, value):
        if self is target and name == "layers":
            raise RuntimeError("Injected commit failure")
        original(self, name, value)

    monkeypatch.setattr(ExtraState, "__setattr__", fail)
    with pytest.raises(ExecutionError, match="Commit failed"):
        load_checkpoint(target, stream)
    assert target.cache is old_cache and target.layers is old_layers
    assert target.layers[0] is target.drop


def test_zero_trial_strategy_skips_scoring_but_validates_empty(monkeypatch):
    model = Chain()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    calls = []

    def metric(context, batch):
        calls.append(batch)
        raise AssertionError("No score should be needed")

    plan = Pruner(model, graph=graph).plan(
        budget=ChannelRatio(0.5), metric=metric, strategy=Greedy(max_trials=0)
    )
    assert not calls and not plan.recipes and plan.budget.limit_reached


def test_implicit_causal_attention_protects_token_coordinates(execution_device):
    class Model(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    model = Model()
    q, k = torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 4, 2)
    v = torch.arange(4.0).reshape(1, 1, 4, 1) * 10
    # Removing a query prefix would regenerate the triangle with wrong positions.
    torch.testing.assert_close(
        model(q, k, v)[..., 1:, :].flatten(), torch.tensor([5.0, 10.0, 15.0])
    )
    torch.testing.assert_close(model(q[..., 1:, :], k, v).flatten(), torch.tensor([0.0, 5.0, 10.0]))
    graph = DependencyGraph.build(model, args=(q, k, v))
    query = next(r for r in graph.interfaces() if r.shape == tuple(q.shape))
    with pytest.raises(PlanningError, match="causal"):
        Pruner(model, graph=graph).plan(remove=[query.axis(-2).select([0])], preserve_io=False)
    plan = Pruner(model, graph=graph).plan(remove=[query.axis(-1).select([0])], preserve_io=False)
    assert plan.analysis.status == "resolved"
