"""operators / attention contracts."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


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
        model, _ = Pruner(graph.model, graph=graph, preserve_io=False).apply(
            Pruner(graph.model, graph=graph, preserve_io=False).plan_remove([seed])
        )
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
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("out_proj.weight").axis(0).select([1, 5])]
        )
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
    plan = Pruner(model, graph=graph, preserve_io=False).plan_remove(
        [graph.parameter("attn.out_proj.weight").axis(0).select([1, 5])]
    )
    assert plan.analysis.status == "resolved"


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
        Pruner(model, graph=graph, preserve_io=False).plan_remove([query.axis(-2).select([0])])
    plan = Pruner(model, graph=graph, preserve_io=False).plan_remove([query.axis(-1).select([0])])
    assert plan.analysis.status == "resolved"
