import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph


def removed(impact, tensor, dim):
    return set(impact.selection(tensor).project(dim))


@pytest.mark.parametrize(
    "module,shape,axis",
    [
        (nn.BatchNorm1d(6), (3, 6), 1),
        (nn.BatchNorm2d(6), (3, 6, 4, 4), 1),
        (nn.BatchNorm3d(6), (3, 6, 2, 3, 4), 1),
        (nn.LayerNorm(6), (2, 3, 6), 2),
        (nn.LayerNorm((3, 6)), (2, 3, 6), 2),
        (nn.GroupNorm(2, 6), (2, 6, 3, 3), 1),
    ],
)
def test_normalization_rules(module, shape, axis, execution_device):
    module = module.to(execution_device)
    graph = DependencyGraph.build(module, args=(torch.randn(shape),))
    selection = [0, 3] if isinstance(module, nn.GroupNorm) else [1, 4]
    impact = graph.propagate(remove=[graph.calls("")[0].input().axis(axis).select(selection)])
    assert impact.status == "resolved"
    assert removed(impact, graph.parameter("weight"), len(module.weight.shape) - 1) == set(
        selection
    )
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        assert removed(impact, graph.buffer("running_mean"), 0) == set(selection)
    assert impact.requirements


@pytest.mark.parametrize("kind", ["linear", "conv", "batch", "layer", "group"])
def test_functional_forms_with_keyword_arguments(kind, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "linear":
                self.weight = nn.Parameter(torch.randn(6, 4))
            elif kind == "conv":
                self.weight = nn.Parameter(torch.randn(6, 4, 1))
            else:
                self.weight = nn.Parameter(torch.randn(4))

        def forward(self, x):
            if kind == "linear":
                return F.linear(input=x, weight=self.weight)
            if kind == "conv":
                return F.conv1d(input=x, weight=self.weight)
            if kind == "batch":
                return F.batch_norm(
                    input=x, running_mean=None, running_var=None, weight=self.weight, training=True
                )
            if kind == "layer":
                return F.layer_norm(input=x, normalized_shape=(4,), weight=self.weight)
            return F.group_norm(input=x, num_groups=2, weight=self.weight)

    x = torch.randn(3, 4, 5) if kind == "conv" else torch.randn(3, 4)
    graph = DependencyGraph.build(Model(), args=(x,))
    impact = graph.propagate(remove=[graph.parameter("weight").axis(0).select([0, 2])])
    assert impact.status == "resolved"


@pytest.mark.parametrize(
    "a,b,operation",
    [
        ((2, 3, 4), (2, 4, 5), torch.bmm),
        ((3, 4), (4, 5), torch.mm),
        ((2, 3, 4), (1, 4, 5), torch.matmul),
        ((4,), (4, 5), torch.matmul),
        ((3, 4), (4,), torch.matmul),
        ((4,), (4,), torch.matmul),
    ],
)
def test_matmul_contraction_and_batch_broadcast(a, b, operation, execution_device):
    class Model(nn.Module):
        def forward(self, x, y):
            return operation(x, y)

    graph = DependencyGraph.build(Model(), args=(torch.randn(a), torch.randn(b)))
    call = next(c for c in graph.calls() if len(c.inputs) == 2)
    impact = graph.propagate(remove=[call.input().axis(-1).select([1])])
    assert impact.status == "resolved"
    assert removed(impact, call.input(1), len(b) - 2 if len(b) > 1 else 0) == {1}


def test_broadcast_does_not_delete_scalar_operand():
    class Model(nn.Module):
        def forward(self, x, bias):
            return x + bias

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 4, 3), torch.randn(1, 4, 1)))
    call = graph.calls()[0]
    result = graph.propagate(remove=[call.output().axis(0).select([0])])
    assert not result.selection(call.input(1))
    result = graph.propagate(remove=[call.output().axis(1).select([1])])
    assert removed(result, call.input(1), 1) == {1}


def test_cat_split_slice_permute_chain(execution_device):
    class Model(nn.Module):
        def forward(self, x, y):
            z = torch.cat([x, y], dim=1)
            a, b = torch.split(z, [4, 6], dim=1)
            return a.transpose(0, 1), b[:, 1:5:2]

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 4), torch.randn(2, 6)))
    inputs = [v for v in graph.values() if v.kind == "input"]
    impact = graph.propagate(remove=[inputs[1].axis(1).select([1])])
    assert impact.status == "resolved"
    sliced = next(c for c in graph.calls() if c.name == "getitem_2")
    assert removed(impact, sliced.output(), 1) == {0}
    assert not impact.selection(inputs[0])
    assert any(r.kind == "slice_arguments" for r in impact.requirements)


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_reduction_distinguishes_reduced_and_retained_axes(method):
    class Model(nn.Module):
        def forward(self, x):
            return getattr(x, method)(dim=2)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 4, 6),))
    call = graph.calls()[0]
    reduced = graph.propagate(remove=[call.input().axis(2).select([1])])
    assert reduced.status == "resolved"
    assert not reduced.selection(call.output())
    kept = graph.propagate(remove=[call.input().axis(1).select([1])])
    assert removed(kept, call.output(), 1) == {1}
    assert any(r.kind == "reduction_domain" for r in reduced.requirements)


def test_same_shape_different_reshape_provenance():
    class A(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], 8)

    class B(nn.Module):
        def forward(self, x):
            return x.reshape(2, -1)

    results = []
    for model in (A(), B()):
        graph = DependencyGraph.build(model, args=(torch.randn(2, 8),))
        input_ = next(v for v in graph.values() if v.kind == "input")
        impact = graph.propagate(remove=[input_.axis(1).select([1, 3])])
        assert impact.status == "resolved"
        results.append(next(r for r in impact.requirements if r.kind == "shape_arguments"))
    assert dict(results[0].data)["expression"] != dict(results[1].data)["expression"]
    assert all(r.kind != "attribute" for r in results)


def test_shape_arithmetic_and_squeeze_unsqueeze():
    class Model(nn.Module):
        def forward(self, x):
            return x.reshape(x.size(0), x.size(1) // 2, 2).unsqueeze(1).squeeze(1)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 8),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([2, 3])])
    assert impact.status == "resolved"
    assert any(e.kind == "floordiv" for e in graph.shape_expressions.values())


def test_nonrectangular_reshape_is_unresolved():
    class Model(nn.Module):
        def forward(self, x):
            return x.reshape(2, 4, 2)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 8),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    result = graph.propagate(remove=[input_.axis(1).select([0])])
    assert result.status == "unresolved"
    assert any(d.code == "unsupported_layout" for d in result.diagnostics)


class Attention(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(8, heads * 2 * 3)
        self.proj = nn.Linear(heads * 2, 8)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = (
            (self.qkv(x).reshape(b, t, 3, self.heads, 2) * 1.0).permute(2, 0, 3, 1, 4).unbind(0)
        )
        scores = (q @ k.transpose(-2, -1)) * (2**-0.5)
        y = scores.softmax(-1) @ v
        return self.proj(y.transpose(1, 2).reshape(b, t, self.heads * 2))


def test_attention_composition_and_independent_head_mask_reference(execution_device):
    torch.manual_seed(7)
    model = Attention()
    x = torch.randn(2, 3, 8)
    graph = DependencyGraph.build(model, args=(x,))
    impact = graph.propagate(remove=[graph.parameter("qkv.weight").axis(0).select([2, 3])])
    assert impact.status == "resolved", graph.explain(impact)
    assert removed(impact, graph.parameter("qkv.weight"), 0) == {2, 3, 10, 11, 18, 19}
    assert removed(impact, graph.parameter("proj.weight"), 1) == {2, 3}
    compact = Attention(heads=3)
    rows = [i for i in range(24) if i not in {2, 3, 10, 11, 18, 19}]
    with torch.no_grad():
        compact.qkv.weight.copy_(model.qkv.weight[rows])
        compact.qkv.bias.copy_(model.qkv.bias[rows])
        compact.proj.weight.copy_(model.proj.weight[:, [0, 1, 4, 5, 6, 7]])
        compact.proj.bias.copy_(model.proj.bias)
        q, k, v = model.qkv(x).reshape(2, 3, 3, 4, 2).permute(2, 0, 3, 1, 4).unbind(0)
        heads = ((q @ k.transpose(-2, -1)) * (2**-0.5)).softmax(-1) @ v
        heads[:, 1] = 0
        reference = model.proj(heads.transpose(1, 2).reshape(2, 3, 8))
    torch.testing.assert_close(compact(x), reference)


def test_groupnorm_compact_reference_recomputes_normalization(execution_device):
    torch.manual_seed(9)
    model = nn.GroupNorm(2, 6)
    x = torch.randn(2, 6, 3, 3)
    graph = DependencyGraph.build(model, args=(x,))
    impact = graph.propagate(remove=[graph.parameter("weight").axis(0).select([0, 4])])
    assert impact.status == "resolved"
    keep = [1, 2, 3, 5]
    compact = nn.GroupNorm(2, 4)
    with torch.no_grad():
        compact.weight.copy_(model.weight[keep])
        compact.bias.copy_(model.bias[keep])
    retained = x[:, keep].reshape(2, 2, 2, 3, 3)
    mean = retained.mean((2, 3, 4), keepdim=True)
    variance = retained.var((2, 3, 4), unbiased=False, keepdim=True)
    normalized = ((retained - mean) / (variance + model.eps).sqrt()).reshape(2, 4, 3, 3)
    reference = (
        normalized * model.weight[keep][None, :, None, None] + model.bias[keep][None, :, None, None]
    )
    torch.testing.assert_close(compact(x[:, keep]), reference)
