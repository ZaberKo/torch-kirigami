"""Patch-model width compaction keeps unchanged batch and token axes symbolic."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph, config
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


class PatchHead(nn.Module):
    def __init__(self, fixed_width=False):
        super().__init__()
        self.proj = nn.Linear(16, 128)
        self.head = nn.Linear(128 * 42, 4)
        self.fixed_width = fixed_width

    def forward(self, x):
        patches = x.unfold(1, 16, 8).permute(0, 2, 1, 3).reshape(-1, 42, 16)
        z = self.proj(patches)
        z = z.reshape(x.size(0), 7, 42, 128 if self.fixed_width else z.size(-1))
        return self.head(z.permute(0, 1, 3, 2).flatten(2))


@pytest.mark.parametrize("fixed_width", [False, True])
def test_patch_head_large_batch_pruning_and_forward_preconditions(fixed_width, execution_device):
    model = PatchHead(fixed_width).eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(128, 344, 7)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    assert config.max_parts == 4096
    removal = graph.parameter("proj.weight").axis(0).select(range(8))
    impact = graph.propagate(remove=[removal])
    assert impact.complete and impact.status == "resolved"
    pruner = Pruner(model, graph=graph)
    if fixed_width:
        with pytest.raises(PlanningError, match=r"shape|forward|reshape"):
            pruner.plan_remove([removal])
        assert model.proj.out_features == 128
        torch.testing.assert_close(model.proj.weight, dense.proj.weight)
        return
    plan = pruner.plan_remove([removal])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert model.proj.out_features == 120 and model.head.in_features == 120 * 42
    # Explicit slices supply patches independently of Tensor.unfold semantics.
    patches = torch.stack([sample[:, i : i + 16] for i in range(0, 329, 8)], dim=2)
    patches = patches.permute(0, 3, 2, 1)
    projected = F.linear(patches, dense.proj.weight[8:], dense.proj.bias[8:])
    features = projected.transpose(-1, -2).reshape(128, 7, 120 * 42)
    expected = F.linear(features, dense.head.weight[:, 8 * 42 :], dense.head.bias)
    torch.testing.assert_close(model(sample), expected)
    model(sample).sum().backward()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(PatchHead(), stream)
    torch.testing.assert_close(restored(sample), expected)
