"""Check the example projection rule against the module's forward computation."""

import pytest
import torch

from examples.custom_rule import FusedProjection, FusedProjectionRule
from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import Pruner


@pytest.mark.parametrize("shape", [(4,), (2, 3, 4)])
def test_projection_preserves_each_leading_axis(shape, execution_device):
    model = FusedProjection().eval()
    x = torch.randn(shape)
    reference = model(x).detach()
    rules = OperatorRegistry.default().register(FusedProjection, FusedProjectionRule())
    graph = DependencyGraph.build(model, args=(x,), operators=rules)
    inputs, outputs = graph.interfaces()
    pruner = Pruner(model, graph=graph, preserve_io=False)

    leading_dims = range(x.ndim - 1)
    plan = pruner.plan_remove(
        [
            graph.parameter("weight").axis(0).select([1]),
            *(inputs.axis(dim).select([0]) for dim in leading_dims),
        ]
    )
    removed = plan.analysis.selection(outputs)
    for dim in leading_dims:
        assert tuple(removed.fully_selected_indices(dim)) == (0,)
    assert tuple(removed.fully_selected_indices(-1)) == (1,)

    compact, _ = pruner.apply(plan)
    keep = (slice(1, None),) * len(leading_dims) + (slice(None),)
    assert compact.output_width == 5
    torch.testing.assert_close(compact(x[keep]), reference[keep][..., [0, 2, 3, 4, 5]])
