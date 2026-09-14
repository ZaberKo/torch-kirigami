"""Run with: python examples/dependency.py"""

import torch
from torch import nn

from torch_kirigami import DependencyGraph

model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
impact = graph.propagate(remove=[graph.parameter("0.weight").axis(0).select([1, 4])])
assert impact.status == "resolved"
assert set(impact.selection(graph.parameter("2.weight")).fully_selected_indices(1)) == {1, 4}
print(graph.explain(impact))
