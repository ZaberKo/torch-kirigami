"""Fixed protection affects discovery without treating all conflicts as permanent."""

import copy
from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import KeyStrategy
from torch_kirigami import (
    AxisRelation,
    CandidateAxis,
    DependencyGraph,
    Diagnostic,
    Divisible,
    Fixed,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
)
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    PlanningError,
    Pruner,
    unique_groups,
)
from torch_kirigami.sparsity import GroupLasso


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 8)
        self.blocks = nn.Sequential(ResidualBlock(), ResidualBlock())
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_fixed_residual_domains_are_excluded_before_sparse_training_and_planning(
    strategy_type, execution_device
):
    model = ResidualMLP().double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    fixed = Fixed(graph.parameter("stem.weight").axis(0))
    pruner = Pruner(model, graph=graph, constraints=(fixed,))
    space = pruner.discover_candidates(targets=("blocks.*.mlp*",))
    exact = pruner.discover_candidates(targets=("blocks.*.mlp.0",))
    assert space.candidates == exact.candidates
    assert len(space.candidates) == 32
    assert [a.tensor.shape[a.dim] for a in space.channel_axes] == [16, 16]
    assert {a.tensor.paths[0] for a in space.excluded_channel_axes} == {
        "blocks.0.mlp.2.weight",
        "blocks.1.mlp.2.weight",
    }
    assert len(space.exclusions) == 2
    assert all("fixed_axis" in reason for _, reason in space.exclusions)
    assert all(pruner.impact((c,)).status == "resolved" for c in space.candidates)

    groups = unique_groups(pruner.parameter_groups(space.candidates))
    protected = graph.parameter("stem.weight")
    assert all(s.tensor != protected for group in groups for s in group.selections)
    model.zero_grad(set_to_none=True)
    GroupLasso(groups)().backward()
    assert model.stem.weight.grad is None
    assert all(block.mlp[0].weight.grad is not None for block in model.blocks)

    # An explicitly retained protected budget axis must not silently disappear.
    explicit = CandidateSpace(
        space.candidates,
        (*space.channel_axes, *space.excluded_channel_axes),
        space.excluded_channel_axes,
        space.exclusions,
    )
    with pytest.raises(PlanningError, match="target not reached"):
        pruner.plan(explicit, budget=ChannelRatio(0.25), strategy=strategy_type(GroupMagnitude()))
    torch.testing.assert_close(model(x), original(x))
    graph.validate()

    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(GroupMagnitude()))
    assert plan.selection_report.widths == (16, 16)
    assert plan.selection_report.remaining == (12, 12)
    assert len(plan.selection_report.exclusions) >= 2
    compact, _ = pruner.apply(plan)
    expected = original.stem(x)
    for index, block in enumerate(original.blocks):
        axis = space.channel_axes[index]
        removed = plan.analysis.selection(axis.tensor).fully_selected_indices(0)
        keep = [i for i in range(16) if i not in removed]
        normalized = F.layer_norm(
            expected, (8,), block.norm.weight, block.norm.bias, block.norm.eps
        )
        hidden = F.gelu(F.linear(normalized, block.mlp[0].weight[keep], block.mlp[0].bias[keep]))
        expected = expected + F.linear(hidden, block.mlp[2].weight[:, keep], block.mlp[2].bias)
    expected = original.head(expected)
    torch.testing.assert_close(compact(x), expected)
    torch.testing.assert_close(compact.stem.weight, original.stem.weight)
    compact.zero_grad(set_to_none=True)
    compact(x).sum().backward()


def test_fixed_protection_uses_propagation_when_axis_widths_differ():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Flatten(1), nn.Linear(16, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3, 2, 2),))
    pruner = Pruner(
        model,
        graph=graph,
        preserve_io=False,
        constraints=(Fixed(graph.parameter("2.weight").axis(1)),),
    )
    space = pruner.discover_candidates(targets=("0",))
    assert not space.candidates and not space.channel_axes
    assert space.excluded_channel_axes == (graph.parameter("0.weight").axis(0),)
    assert len(space.exclusions) == 1


def test_partial_fixed_protection_does_not_remove_the_whole_domain():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Linear(3, 4)
            self.left = nn.Linear(2, 2)
            self.right = nn.Linear(2, 2)

        def forward(self, x):
            hidden = self.stem(x)
            return self.left(hidden[:, :2]), self.right(hidden[:, 2:])

    model = Model().eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3),))
    pruner = Pruner(
        model, graph=graph, constraints=(Fixed(graph.parameter("left.weight").axis(1)),)
    )
    space = pruner.discover_candidates(targets=("stem",))
    assert len(space.candidates) == 2
    assert not space.excluded_channel_axes
    assert space.channel_axes == (graph.parameter("stem.weight").axis(0),)
    assert all(pruner.impact((c,)).status == "resolved" for c in space.candidates)
    assert len(space.exclusions) == 2
    # Extraction remains a raw query for explicit infeasible requests.
    axis = graph.parameter("stem.weight").axis(0)
    raw = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(4))
    assert len(pruner.parameter_groups(raw)) == 4


def test_discovery_respects_fixed_constraints_declared_by_operator_rules():
    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 3))

        def forward(self, x):
            return F.linear(x, self.weight)

    class Rule(OperatorRule):
        def build_spec(self, operation):
            weight = operation.bindings["weight"]
            return OperatorSpec(
                relations=(
                    AxisRelation.equal(operation.inputs[0].axis(-1), weight.axis(1)),
                    AxisRelation.equal(operation.outputs[0].axis(-1), weight.axis(0)),
                ),
                constraints=(Fixed(weight.axis(0)),),
                candidates=(CandidateAxis("weight:0", weight.axis(0)),),
            )

    model = Projection().eval()
    graph = DependencyGraph.build(
        model,
        args=(torch.randn(2, 3),),
        operators=OperatorRegistry.default().register(Projection, Rule()),
    )
    pruner = Pruner(model, graph=graph, preserve_io=False)
    space = pruner.discover_candidates()
    assert not space.candidates and not space.channel_axes
    assert space.excluded_channel_axes == (graph.parameter("weight").axis(0),)
    with pytest.raises(PlanningError, match="fixed_axis"):
        pruner.plan_remove((graph.parameter("weight").axis(0).select([0]),))


def test_fixed_subclasses_do_not_inherit_the_native_exclusion_proof():
    class AdvisoryFixed(Fixed):
        def check(self, selections):
            return None

    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3),))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, constraints=(AdvisoryFixed(axis),))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 4 and not space.excluded_channel_axes
    pruner.apply(pruner.plan_remove((axis.select([0]),)))
    assert model[0].out_features == model[1].in_features == 3


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_discovery_keeps_grouped_candidates_that_need_joint_completion(strategy_type):
    model = nn.Sequential(nn.Conv2d(4, 6, 1, groups=2), nn.Conv2d(6, 2, 1)).double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(1, 4, 2, 2, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, constraints=(Divisible(axis, 2),))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 6 and not space.excluded_channel_axes
    assert all(pruner.impact((c,)).status == "unresolved" for c in space.candidates)
    assert len(pruner.parameter_groups(space.candidates)) == 6
    plan = pruner.plan(space, budget=ChannelRatio(1 / 3), strategy=strategy_type(GroupMagnitude()))
    removed = plan.analysis.selection(axis.tensor).fully_selected_indices(0)
    assert len(removed) == 2
    assert sum(i < 3 for i in removed) == 1
    keep = [i for i in range(6) if i not in removed]
    pruner.apply(plan)
    expected = F.conv2d(x, original[0].weight[keep], original[0].bias[keep], groups=2)
    expected = F.conv2d(expected, original[1].weight[:, keep], original[1].bias)
    torch.testing.assert_close(model(x), expected)


def test_custom_conflict_is_not_a_proof_that_every_larger_request_conflicts():
    @dataclass
    class PairConstraint:
        axis: object

        @property
        def refs(self):
            return (self.axis.tensor,)

        def check(self, selections):
            selection = selections.get(self.axis.tensor.id)
            if selection and len(selection.fully_selected_indices(self.axis.dim)) % 2:
                return Diagnostic("custom_pair", "Choose an even number", "conflict")
            return None

    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3),))
    axis = graph.parameter("0.weight").axis(0)
    pruner = Pruner(model, graph=graph, constraints=(PairConstraint(axis),))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 4
    assert all(pruner.impact((c,)).status == "conflict" for c in space.candidates)
    assert pruner.impact(space.candidates[:2]).status == "resolved"
    strategy = KeyStrategy(lambda context: (c.key for c in context.candidates[:2]))
    pruner.apply(pruner.plan(space, budget=ChannelRatio(0.5), strategy=strategy))
    assert model[0].out_features == model[1].in_features == 2
