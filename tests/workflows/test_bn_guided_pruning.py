"""BN-scale scoring, alias deduplication, and public structural pruning."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    PlanningContext,
    PlanningError,
    Pruner,
)

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")


import bn_guided_pruning


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_bn_scale_metric_uses_explicit_scales_and_public_pruning(
    strategy_type, execution_device, monkeypatch
):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2)).double().eval()
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([4.0, -1.0, 3.0, 2.0]))
        model[0].weight[1].fill_(1000)
        model[1].bias.fill_(1000)
    model.register_parameter("scale_alias", model[1].weight)
    x = torch.randn(2, 3, dtype=torch.float64)
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    axis = graph.parameter("0.weight").axis(0)
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.25), space.channel_axes, pruner.constraints
    )
    metric = bn_guided_pruning.BNScaleMagnitude(
        (graph.parameter("1.weight"), graph.parameter("scale_alias"))
    )
    transfers = []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor, *args, **kwargs):
        transfers.append(tensor)
        return original_cpu(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", counted_cpu)
        assert context.score(metric, space.candidates) == (4.0, 1.0, 3.0, 2.0)
    # Read the aliased scale vector once for the batch, not once per candidate.
    # The default-device context can dispatch the same cpu() call twice.
    assert len({id(tensor) for tensor in transfers}) == 1
    assert transfers[0].shape == (4,)
    block = Candidate("block", (axis.select([0, 1]), axis.select([1])), axis)
    accepted = context.impact((axis.select([0]),))
    assert context.score(metric, (block,), accepted_impact=accepted) == (1.0,)
    covered = context.impact(block.remove)
    assert context.score(metric, (block,), accepted_impact=covered) == (0.0,)

    # An unrelated parameter cannot substitute for missing BN evidence.
    wrong_metric = bn_guided_pruning.BNScaleMagnitude((graph.parameter("2.bias"),))
    with pytest.raises(PlanningError, match="No BN scale"):
        pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(wrong_metric))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)
    graph.validate()

    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(metric))
    compact, _ = pruner.apply(plan)
    keep = [0, 2, 3]
    hidden = F.linear(x, original[0].weight[keep], original[0].bias[keep])
    hidden = F.batch_norm(
        hidden,
        original[1].running_mean[keep],
        original[1].running_var[keep],
        original[1].weight[keep],
        original[1].bias[keep],
        eps=original[1].eps,
    )
    reference = F.linear(hidden, original[2].weight[:, keep], original[2].bias)
    torch.testing.assert_close(compact(x), reference)
    assert compact.scale_alias is compact[1].weight
    compact(x).sum().backward()


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_bn_scale_metric_sums_distinct_branches_without_recounting_aliases(
    strategy_type, execution_device
):
    class BranchedBatchNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(3, 4)
            self.left = nn.BatchNorm1d(4)
            self.right = nn.BatchNorm1d(4)
            self.output = nn.Linear(4, 2)
            self.register_parameter("left_scale_alias", self.left.weight)

        def forward(self, inputs):
            hidden = self.hidden(inputs)
            return self.output(self.left(hidden) + self.right(hidden))

    model = BranchedBatchNorm().double().eval()
    with torch.no_grad():
        model.left.weight.copy_(torch.tensor([1.0, 3.0, 6.0, 4.0]))
        model.right.weight.copy_(torch.tensor([-8.0, -0.5, 0.25, -2.0]))
        model.left.bias.fill_(100)
        model.right.bias.fill_(-90)
        model.hidden.weight[1].fill_(50)
    original = copy.deepcopy(model)
    x = torch.randn(3, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("hidden",))
    axis = graph.parameter("hidden.weight").axis(0)
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.25), space.channel_axes, pruner.constraints
    )
    metric = bn_guided_pruning.BNScaleMagnitude(
        tuple(graph.parameter(path) for path in ("left.weight", "right.weight", "left_scale_alias"))
    )
    assert context.score(metric, space.candidates) == (9.0, 3.5, 6.25, 6.0)
    block = Candidate("overlapping_block", (axis.select([1, 2]), axis.select([2])), axis)
    assert context.score(metric, (block,)) == (9.75,)
    assert context.score(metric, (block,), accepted_impact=context.impact((axis.select([1]),))) == (
        6.25,
    )

    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(metric))
    assert tuple(plan.analysis.selection(axis.tensor).fully_selected_indices(0)) == (1,)
    compact, _ = pruner.apply(plan)
    keep = [0, 2, 3]
    hidden = F.linear(x, original.hidden.weight[keep], original.hidden.bias[keep])
    branches = [
        F.batch_norm(
            hidden,
            branch.running_mean[keep],
            branch.running_var[keep],
            branch.weight[keep],
            branch.bias[keep],
            eps=branch.eps,
        )
        for branch in (original.left, original.right)
    ]
    reference = F.linear(
        branches[0] + branches[1], original.output.weight[:, keep], original.output.bias
    )
    torch.testing.assert_close(compact(x), reference)
    assert compact.left_scale_alias is compact.left.weight
    assert compact.left.num_features == compact.right.num_features == 3
    compact(x).sum().backward()
