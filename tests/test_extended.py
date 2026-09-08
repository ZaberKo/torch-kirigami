import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import (
    ChannelRatio,
    Magnitude,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize(
    "cls,shape",
    [
        (nn.ConvTranspose1d, (2, 6, 5)),
        (nn.ConvTranspose2d, (2, 6, 4, 5)),
        (nn.ConvTranspose3d, (2, 6, 3, 4, 5)),
    ],
)
def test_grouped_transpose_independent_input_and_output_positions(cls, shape, execution_device):
    model = cls(6, 8, 1, groups=2)
    old = copy.deepcopy(model)
    x = torch.randn(shape)
    graph = DependencyGraph.build(model, args=(x,))
    call = graph.calls("")[0]
    Pruner(model, graph=graph).prune(
        remove=[call.input().axis(1).select([0, 4]), call.output().axis(1).select([1, 6])],
        preserve_io=False,
    )
    weight = torch.cat((old.weight[[1, 2]][:, [0, 2, 3]], old.weight[[3, 5]][:, [0, 1, 3]]))
    keep_out = [0, 2, 3, 4, 5, 7]
    func = {
        nn.ConvTranspose1d: F.conv_transpose1d,
        nn.ConvTranspose2d: F.conv_transpose2d,
        nn.ConvTranspose3d: F.conv_transpose3d,
    }[cls]
    reference = func(x[:, [1, 2, 3, 5]], weight, old.bias[keep_out], groups=2)
    torch.testing.assert_close(model(x[:, [1, 2, 3, 5]]), reference)
    assert model.in_channels == 4 and model.out_channels == 6


@pytest.mark.parametrize(
    "operation",
    [
        lambda: nn.MaxPool2d(2),
        lambda: nn.AvgPool2d(2),
        lambda: nn.AdaptiveAvgPool2d((2, 2)),
        lambda: nn.AdaptiveMaxPool2d((2, 2)),
        lambda: nn.Upsample(scale_factor=2, mode="nearest"),
        lambda: nn.ReflectionPad2d(1),
        lambda: nn.InstanceNorm2d(6, affine=True),
        lambda: nn.PReLU(6),
    ],
)
def test_channel_family_matches_independent_retained_channels(operation, execution_device):
    model = nn.Sequential(nn.Conv2d(3, 6, 1), operation())
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 6, 6)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("0.weight").axis(0).select([1, 4])], preserve_io=False
    )
    torch.testing.assert_close(model(x), original(x)[:, [0, 2, 3, 5]])


def test_embedding_uses_feature_axis_budget_and_rejects_parameter_writes(execution_device):
    model = nn.Sequential(nn.Embedding(12, 6), nn.Linear(6, 2))
    x = torch.tensor([[1, 3, 6], [2, 4, 5]])
    graph = DependencyGraph.build(model, args=(x,))
    _, result = Pruner(model, graph=graph).prune(metric=Magnitude(), budget=ChannelRatio(0.34))
    assert result.plan.budget.widths == (6,) and result.plan.budget.removed == (2,)
    assert model[0].embedding_dim == 4 and model[0].num_embeddings == 12
    model(x).sum().backward()
    bad = nn.Embedding(12, 6, max_norm=1)
    original = bad.weight.detach().clone()
    with pytest.raises(CaptureError, match="max_norm"):
        DependencyGraph.build(bad, args=(x,))
    torch.testing.assert_close(bad.weight, original)


def test_rms_norm_compact_domain_reference(execution_device):
    model = nn.RMSNorm(6, eps=1e-5)
    x = torch.randn(2, 6)
    old = model.weight.detach().clone()
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("weight").axis(0).select([1, 4])], preserve_io=False
    )
    kept = x[:, [0, 2, 3, 5]]
    expected = kept * torch.rsqrt(kept.square().mean(-1, keepdim=True) + 1e-5) * old[[0, 2, 3, 5]]
    torch.testing.assert_close(model(kept), expected)


@pytest.mark.parametrize(
    "operation,remove,keep_out",
    [
        (lambda: nn.GLU(dim=1), [1], [0, 2]),
        (lambda: nn.ChannelShuffle(2), [1], [0, 1, 4, 5]),
        (lambda: nn.PixelShuffle(2), [0], [1, 2]),
    ],
)
def test_block_families_share_coordinate_relations(operation, remove, keep_out, execution_device):
    op = operation()
    channels = 12 if isinstance(op, nn.PixelShuffle) else 6
    model = nn.Sequential(nn.Conv2d(3, channels, 1), op)
    old = copy.deepcopy(model)
    x = torch.randn(2, 3, 3, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("0.weight").axis(0).select(remove)], preserve_io=False
    )
    torch.testing.assert_close(model(x), old(x)[:, keep_out])


@pytest.mark.parametrize(
    "mode", ["repeat", "interleave", "tile", "stack", "chunk", "narrow", "unfold"]
)
def test_axis_family_compaction(mode, execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 1)

        def forward(self, x):
            y = self.conv(x)
            if mode == "repeat":
                return y.repeat(1, 2, 1, 1)
            if mode == "interleave":
                return y.repeat_interleave(2, dim=1)
            if mode == "tile":
                return torch.tile(y, (1, 2, 1, 1))
            if mode == "stack":
                return torch.stack([y, y], dim=2)
            if mode == "chunk":
                return y.chunk(2, dim=1)
            if mode == "narrow":
                return y.narrow(2, 0, 2)
            return F.unfold(y, 2)

    model = Net()
    x = torch.randn(2, 3, 4, 4)
    old = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("conv.weight").axis(0).select([1, 4])], preserve_io=False
    )
    if mode == "chunk":
        a, b = old(x)
        expected = (a[:, [0, 2]], b[:, [0, 2]])
    elif mode in ("repeat", "tile"):
        expected = old(x)[:, [0, 2, 3, 5, 6, 8, 9, 11]]
    elif mode == "interleave":
        expected = old(x)[:, [0, 1, 4, 5, 6, 7, 10, 11]]
    elif mode == "unfold":
        expected = F.unfold(old.conv(x)[:, [0, 2, 3, 5]], 2)
    else:
        expected = old(x)[:, [0, 2, 3, 5]]
    torch.testing.assert_close(model(x), expected)


def test_sdpa_gqa_group_and_multiplier_pruning(execution_device):
    class Net(nn.Module):
        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    for remove in ([0], [0, 2]):
        q, k, v = torch.randn(2, 4, 3, 4), torch.randn(2, 2, 5, 4), torch.randn(2, 2, 5, 6)
        graph = DependencyGraph.build(Net(), args=(q, k, v))
        call = next(c for c in graph.calls() if len(c.inputs) == 3)
        seed = (
            call.inputs[1].axis(1).select(remove)
            if remove == [0]
            else call.inputs[0].axis(1).select(remove)
        )
        model, _ = Pruner(graph.model, graph=graph).prune(remove=[seed], preserve_io=False)
        qs = q[:, [2, 3]] if remove == [0] else q[:, [1, 3]]
        ks, vs = (k[:, [1]], v[:, [1]]) if remove == [0] else (k, v)
        keys = ks.repeat_interleave(qs.shape[1] // ks.shape[1], dim=1)
        values = vs.repeat_interleave(qs.shape[1] // vs.shape[1], dim=1)
        logits = qs @ keys.transpose(-2, -1) / qs.shape[-1] ** 0.5
        expected = logits.softmax(-1) @ values
        torch.testing.assert_close(model(qs, ks, vs), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("separate", [False, True])
def test_mha_width_checkpoint_and_compact_reference(separate, execution_device):
    kwargs = {"kdim": 5, "vdim": 7} if separate else {}
    model = nn.MultiheadAttention(8, 2, batch_first=True, **kwargs)
    old = copy.deepcopy(model)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 4, kwargs.get("kdim", 8))
    v = torch.randn(2, 4, kwargs.get("vdim", 8))
    graph = DependencyGraph.build(model, args=(q, k, v))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("out_proj.weight").axis(0).select([1, 5])], preserve_io=False
    )
    keep = [0, 2, 3, 4, 6, 7]
    qs, ks, vs = q[..., keep], k if separate else k[..., keep], v if separate else v[..., keep]
    projections = []
    for i, inp in enumerate((qs, ks, vs)):
        weight = (
            (old.q_proj_weight, old.k_proj_weight, old.v_proj_weight)[i]
            if separate
            else old.in_proj_weight[i * 8 : (i + 1) * 8]
        )
        weight = weight[keep]
        if i == 0 or not separate:
            weight = weight[:, keep]
        bias = old.in_proj_bias[i * 8 : (i + 1) * 8][keep]
        projected = F.linear(inp, weight, bias).reshape(2, inp.shape[1], 2, 3).transpose(1, 2)
        projections.append(projected)
    a, b, c = projections
    expected_weights = (a @ b.transpose(-1, -2) / 3**0.5).softmax(-1)
    hidden = (expected_weights @ c).transpose(1, 2).reshape(2, 3, 6)
    expected = F.linear(hidden, old.out_proj.weight[keep][:, keep], old.out_proj.bias[keep])
    output, weights = model(qs, ks, vs)
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weights, expected_weights.mean(1), rtol=1e-4, atol=1e-5)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(nn.MultiheadAttention(8, 2, batch_first=True, **kwargs), stream)
    torch.testing.assert_close(restored(qs, ks, vs), (output, weights))


def test_static_index_values_and_coordinate_guards():
    from torch_kirigami import StaleGraphError

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.register_buffer("indices", torch.tensor([0, 2]))

        def forward(self, x):
            return self.fc(x).index_select(1, self.indices)

    model = Net()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    p = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="coordinates"):
        p.plan(remove=[graph.parameter("fc.weight").axis(0).select([1])])
    plan = p.plan(remove=[graph.parameter("fc.weight").axis(0).select([4, 5])])
    p.apply(plan)
    torch.testing.assert_close(model(x), old(x))
    graph = DependencyGraph.build(model, args=(x,))
    model.indices[0] = 1
    with pytest.raises(StaleGraphError, match="constant"):
        graph.propagate(remove=[])


def test_einsum_and_addmm_independent_reference(execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 3))
            self.fc = nn.Linear(4, 6)
            self.bias = nn.Parameter(torch.randn(3))

        def forward(self, x):
            y = self.fc(x)
            return torch.einsum("...i,ij->...j", y, self.weight), torch.addmm(
                self.bias, y, self.weight
            )

    model = Net()
    old = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("fc.weight").axis(0).select([1, 4])])
    retained = old.fc(x)[:, [0, 2, 3, 5]] @ old.weight[[0, 2, 3, 5]]
    torch.testing.assert_close(model(x), (retained, retained + old.bias))


def test_transpose_output_protection_uses_logical_domain():
    model = nn.Sequential(nn.Conv1d(3, 6, 1), nn.ConvTranspose1d(6, 8, 1, groups=2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3, 4),))
    model, result = Pruner(model, graph=graph).prune(metric=Magnitude(), budget=ChannelRatio(0.34))
    assert result.plan.budget.widths == (6,)
    assert model[0].out_channels == 4 and model[1].out_channels == 8


def test_cast_reference_shape_is_not_a_broadcast_dependency(execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.register_buffer("dtype_reference", torch.zeros(11, dtype=torch.float64))

        def forward(self, x):
            return self.fc(x).type_as(self.dtype_reference).cpu()

    model = Net()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False
    )
    torch.testing.assert_close(model(x), old(x)[:, [0, 2, 3, 4, 5]])


def test_python_comparison_where_keeps_shared_structural_axes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return torch.where(y > 0, y**2, -y)

    model = Net()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False
    )
    torch.testing.assert_close(model(x), original(x)[:, [0, 2, 3, 4, 5]])
