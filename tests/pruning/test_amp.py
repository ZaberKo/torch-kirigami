"""Precision, layout and execution contexts must survive lifecycle boundaries."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


class PartialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(4, 8, bias=False) for _ in range(3))

    def forward(self, x):
        q = self.q(x).reshape(x.size(0), x.size(1), 2, -1).transpose(1, 2)
        k = self.k(x).reshape(x.size(0), x.size(1), 2, -1).transpose(1, 2)
        v = self.v(x).reshape(x.size(0), x.size(1), 2, -1).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v)


@pytest.mark.parametrize("planning_autocast", [False, True])
@pytest.mark.parametrize("low_precision", ["bfloat16", "float16"])
def test_partial_amp_paths_join_with_captured_dtype(
    planning_autocast, low_precision, execution_device
):
    dtype = getattr(torch, low_precision)
    model = PartialAttention()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 4)
    with torch.autocast(execution_device, dtype=dtype):
        graph = DependencyGraph.build(model, args=(x,))
    with torch.autocast(execution_device, dtype=dtype, enabled=planning_autocast):
        plan = PruningPlan.from_dict(
            Pruner(model, graph=graph)
            .plan_remove([graph.parameter("q.weight").axis(0).select([1, 5])])
            .to_dict()
        )
    Pruner(model).apply(plan)
    assert model.q.out_features == model.k.out_features == 6 and model.v.out_features == 8
    with torch.autocast(execution_device, dtype=dtype):
        keep = [0, 2, 3, 4, 6, 7]
        q, k = (
            F.linear(x, layer.weight[keep]).reshape(2, 3, 2, 3).transpose(1, 2)
            for layer in (reference.q, reference.k)
        )
        v = reference.v(x).reshape(2, 3, 2, 4).transpose(1, 2)
        # Independent explicit attention, including compact Q/K scaling.
        expected = ((q @ k.transpose(-2, -1)) / 3**0.5).softmax(-1) @ v
        actual = model(x)
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
    actual.float().square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(PartialAttention(), stream)
    with torch.autocast(execution_device, dtype=dtype):
        torch.testing.assert_close(restored(x), actual)
