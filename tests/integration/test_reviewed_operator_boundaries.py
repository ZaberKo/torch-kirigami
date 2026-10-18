"""Public pruning regressions for degenerate groups and extension defaults."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    DependencyGraph,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    ReshapeRelation,
)
from torch_kirigami.pruning import ChannelRatio, Greedy, Magnitude, Pruner, PruningPlan


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
def test_single_input_channel_offers_independent_output_candidates(
    dimension, transposed, execution_device
):
    conv = getattr(nn, f"Conv{dimension}d")
    first = getattr(nn, f"ConvTranspose{dimension}d") if transposed else conv
    model = nn.Sequential(first(1, 8, 3, padding=1), nn.ReLU(), conv(8, 4, 1)).double().eval()
    original = copy.deepcopy(model)
    x = torch.randn((2, 1, *((4,) * dimension)), dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 8 and not space.exclusions
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude()))
    axis = 1 if transposed else 0
    removed = plan.analysis.selection(graph.parameter("0.weight")).fully_selected_indices(axis)
    keep = [i for i in range(8) if i not in removed]
    weight = original[0].weight[:, keep] if transposed else original[0].weight[keep]
    operation = getattr(F, f"conv_transpose{dimension}d" if transposed else f"conv{dimension}d")
    hidden = operation(x, weight, original[0].bias[keep], padding=1).relu()
    expected = getattr(F, f"conv{dimension}d")(
        hidden, original[2].weight[:, keep], original[2].bias
    )
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert model[0].groups == 1 and model[0].out_channels == 6
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


def test_rule_constructor_can_rely_on_inherited_validation_defaults(execution_device):
    class Half(nn.Module):
        def forward(self, x):
            return x * 0.5

    class HalfRule(OperatorRule):
        def __init__(self):
            self.label = "custom configuration"

        def build_spec(self, operation):
            return OperatorSpec(
                relations=(ReshapeRelation(operation.inputs[0], operation.outputs[0]),)
            )

    registry = OperatorRegistry.default().register(Half, HalfRule())
    model = nn.Sequential(nn.Linear(3, 4), Half(), nn.Linear(4, 2)).double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove((graph.parameter("0.weight").axis(0).select([0]),))
    pruner.apply(plan)
    expected = F.linear(
        F.linear(x, original[0].weight[1:], original[0].bias[1:]) * 0.5,
        original[2].weight[:, 1:],
        original[2].bias,
    )
    torch.testing.assert_close(model(x), expected)


def test_true_depthwise_candidates_still_remove_complete_input_groups(execution_device):
    model = (
        nn.Sequential(
            nn.Conv2d(3, 2, 1), nn.Conv2d(2, 8, 3, padding=1, groups=2), nn.Conv2d(8, 2, 1)
        )
        .double()
        .eval()
    )
    original = copy.deepcopy(model)
    x = torch.randn(1, 3, 4, 4, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("1",))
    assert len(space.candidates) == 2
    assert all(len(c.remove[0].fully_selected_indices(0)) == 4 for c in space.candidates)
    plan = pruner.plan_remove(space.candidates[0].remove)
    pruner.apply(plan)
    hidden = F.conv2d(x, original[0].weight[1:], original[0].bias[1:])
    hidden = F.conv2d(hidden, original[1].weight[4:], original[1].bias[4:], padding=1)
    expected = F.conv2d(hidden, original[2].weight[:, 4:], original[2].bias)
    assert model[1].groups == 1 and model[1].out_channels == 4
    torch.testing.assert_close(model(x), expected)
    rebuilt = DependencyGraph.build(model, args=(x,))
    assert len(Pruner(model, graph=rebuilt).discover_candidates(targets=("1",)).candidates) == 4
