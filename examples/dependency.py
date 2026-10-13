"""Build and query a structural dependency graph for a small MLP.

The example captures the model with :meth:`DependencyGraph.build`, removes two
output coordinates from the first Linear weight, then inspects the propagated
impact on the next Linear layer. ``propagate`` computes structural closure and
``explain`` reports it; neither API chooses an importance policy nor mutates the
model. This illustrates a dependency-analysis idea related to DepGraph,
"DepGraph: Towards Any Structural Pruning" (CVPR 2023), without reproducing
that paper's method: https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html

Run with: python examples/dependency.py
"""

import torch
from torch import nn

from torch_kirigami import DependencyGraph

model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
impact = graph.propagate(remove=[graph.parameter("0.weight").axis(0).select([1, 4])])
assert impact.status == "resolved"
assert set(impact.selection(graph.parameter("2.weight")).fully_selected_indices(1)) == {1, 4}
print(graph.explain(impact))
