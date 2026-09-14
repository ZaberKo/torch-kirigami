"""operators / embedding contracts."""

import pytest
import torch
from torch import nn

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
    Magnitude,
    Pruner,
)


def test_embedding_uses_feature_axis_budget_and_rejects_parameter_writes(execution_device):
    model = nn.Sequential(nn.Embedding(12, 6), nn.Linear(6, 2))
    x = torch.tensor([[1, 3, 6], [2, 4, 5]])
    graph = DependencyGraph.build(model, args=(x,))
    _, result = Pruner(model, graph=graph).prune(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.34),
        strategy=Greedy(Magnitude()),
    )
    assert result.plan.selection_report.widths == (6,) and result.plan.selection_report.removed == (
        2,
    )
    assert model[0].embedding_dim == 4 and model[0].num_embeddings == 12
    model(x).sum().backward()
    bad = nn.Embedding(12, 6, max_norm=1)
    original = bad.weight.detach().clone()
    with pytest.raises(CaptureError, match="max_norm"):
        DependencyGraph.build(bad, args=(x,))
    torch.testing.assert_close(bad.weight, original)
