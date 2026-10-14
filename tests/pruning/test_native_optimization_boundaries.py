"""Native fast paths must preserve nested extension behavior and IO protection."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import (
    AxisRelation,
    DependencyGraph,
    Divisible,
    IndexSet,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Region,
    Selection,
)
from torch_kirigami.dep_graph.operation import CallEffects
from torch_kirigami.dep_graph.relations import AxisPort, BlockMap
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
)
from torch_kirigami.pruning.ranking import IndependentRanking


class ReferenceMagnitude(GroupMagnitude):
    """Use ordinary joint scoring rather than the native ranking optimization."""


@pytest.mark.parametrize("boundary", ["constraint_factor", "selection", "region", "indices"])
@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_nested_inputs_keep_generic_scoring_and_public_execution(
    boundary, strategy_type, execution_device
):
    """Normalization cannot turn an extension-bearing request into a native proof."""

    class ExtendedFactor(int):
        def __rmod__(self, remaining):
            return remaining % int(self)

    class ExtendedSelection(Selection):
        """Accepted selection subclasses must still cross the generic boundary."""

    class ExtendedRegion(Region):
        """Region identity alone does not prove ordinary region operations."""

    class ExtendedIndices(IndexSet):
        """Index set extensions remain extensions inside a native Region."""

    source = nn.Sequential(nn.Linear(2, 4, bias=False), nn.Linear(4, 1, bias=False)).double()
    with torch.no_grad():
        source[0].weight.copy_(torch.arange(1, 5).reshape(4, 1).expand(4, 2))
        source[1].weight.fill_(1)
    x = torch.tensor([[1.0, 2.0], [-1.0, 2.0]], dtype=torch.float64)
    plans = []
    for metric in (GroupMagnitude(), ReferenceMagnitude()):
        model = copy.deepcopy(source)
        graph = DependencyGraph.build(model, args=(x,))
        axis = graph.parameter("0.weight").axis(0)
        constraints = (
            (Divisible(axis, ExtendedFactor(2)),) if boundary == "constraint_factor" else ()
        )
        pruner = Pruner(model, graph=graph, constraints=constraints)
        candidates = []
        for position in range(4):
            selection = axis.select([position])
            region = selection.regions[0]
            if boundary == "selection":
                selection = ExtendedSelection(selection.tensor, selection.regions)
            elif boundary == "region":
                selection = Selection(selection.tensor, (ExtendedRegion(region.axes),))
            elif boundary == "indices":
                axes = (ExtendedIndices(region.axes[0].intervals), *region.axes[1:])
                selection = Selection(selection.tensor, (Region(axes),))
            candidates.append(Candidate(str(position), (selection,), axis))
        space = CandidateSpace(tuple(candidates), (axis,))
        context = PlanningContext(
            graph, space.candidates, ChannelRatio(0.25), space.channel_axes, pruner.constraints
        )
        assert IndependentRanking.build(context, metric, context.channel_axes) is None
        original_parameters = tuple(model.parameters())
        plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(metric))
        assert all(a is b for a, b in zip(original_parameters, model.parameters(), strict=True))
        expected_keys = ("0", "1") if boundary == "constraint_factor" else ("0",)
        assert plan.selected == expected_keys
        plans.append(plan)
        if boundary == "constraint_factor":
            executable = PruningPlan.from_dict(plan.to_dict())
        else:
            # The portable codec deliberately accepts exact records only. Raw
            # request extensions may be analyzed and applied in the current process.
            with pytest.raises(TypeError, match="Nonportable plan value"):
                plan.to_dict()
            assert all(a is b for a, b in zip(original_parameters, model.parameters(), strict=True))
            executable = plan
        pruner.apply(executable)
        retained_sum = 7 if boundary == "constraint_factor" else 9
        torch.testing.assert_close(model(x), retained_sum * x.sum(1, keepdim=True))
        model(x).sum().backward()
        assert all(parameter.grad is not None for parameter in model.parameters())
    assert plans[0].selected == plans[1].selected
    assert plans[0].selection_report == plans[1].selection_report


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_discovery_does_not_protect_an_unused_input_channel(strategy_type, execution_device):
    """A partial dependency cannot mark every producer channel as an external output."""

    class CroppedCount(int):
        def __mul__(self, other):
            return int(self) * other - 1

    class FirstThreeWithZero(nn.Module):
        """Ignore the last input channel and emit a fixed-width four-channel output."""

        def forward(self, x):
            return torch.cat((x[:, :3], torch.zeros_like(x[:, :1])), dim=1)

    def analyze(context):
        x, y = context.inputs[0], context.outputs[0]
        return OperatorSpec(
            relations=(
                AxisRelation.equal(x.axis(0), y.axis(0)),
                AxisRelation(
                    AxisPort(x.axis(1)),
                    AxisPort(y.axis(1)),
                    (BlockMap(0, 0, CroppedCount(y.shape[1])),),
                ),
            )
        )

    registry = OperatorRegistry.default().register(
        FirstThreeWithZero,
        OperatorRule(
            analyze,
            evaluate_on_meta=True,
            effects=lambda *args: CallEffects(fresh_output=True),
        ),
    )
    model = nn.Sequential(nn.Linear(2, 4, bias=False), FirstThreeWithZero()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1, 1], [2, 2], [3, 3], [0, 0]]))
    x = torch.tensor([[1.0, 2.0], [-1.0, 2.0]], dtype=torch.float64)
    expected = x.sum(1, keepdim=True) * torch.tensor([1, 2, 3, 0])
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("0.weight").axis(0)
    original_parameter = model[0].weight
    with pytest.raises(PlanningError, match="fixed_axis"):
        pruner.plan_remove((axis.select([0]),))
    assert model[0].weight is original_parameter
    torch.testing.assert_close(model(x), expected)
    space = pruner.discover_candidates(targets=("0",))
    assert space.channel_axes == (axis,)
    assert not space.protected_channel_axes
    assert len(space.candidates) == 4
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(GroupMagnitude()))
    assert plan.selected == ("0.weight:0:000000000003",)
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert model[0].out_features == 3
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert model[0].weight.grad is not None
