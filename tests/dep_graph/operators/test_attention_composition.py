"""integration / attention contracts."""

import torch
from torch import nn

from tests.support.graph_helpers import removed
from torch_kirigami import DependencyGraph


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
