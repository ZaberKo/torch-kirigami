"""Alignment configuration through discovery, planning, persistence and execution."""

import copy
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRelation,
    CandidateAxis,
    DependencyGraph,
    Divisible,
    NonEmpty,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
)
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    Granularity,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    PruningPlan,
)
from torch_kirigami.sparsity import CumulativeChannelBudget


@pytest.mark.parametrize("factor", [True, False, 0, -1, 8.0, "8", None])
@pytest.mark.parametrize("field", ["default", "by_type", "by_path"])
def test_invalid_factors_are_rejected(factor, field):
    value = factor if field == "default" else {nn.Linear if field == "by_type" else "0": factor}
    with pytest.raises(ValueError, match="positive integers"):
        Granularity(**{field: value})


def test_configuration_is_a_snapshot_and_exact_paths_have_no_glob_semantics():
    types, paths = {nn.Linear: 8}, {"0": 4}
    config = Granularity(by_type=types, by_path=paths)
    types[nn.Linear], paths["0"] = 16, 32
    assert config.by_type[nn.Linear] == 8 and config.by_path["0"] == 4
    with pytest.raises(TypeError):
        config.by_path["0"] = 2
    with pytest.raises(FrozenInstanceError):
        config.default = 2
    for pattern in ("encoder.*", "encoder.?", "encoder.[01]"):
        with pytest.raises(ValueError, match="exact"):
            Granularity(by_path={pattern: 8})
    with pytest.raises(TypeError, match="Module types"):
        Granularity(by_type={"Linear": 8})


def test_nonadjacent_alignment_roundtrip_and_independent_compact_reference(execution_device):
    model = nn.Sequential(nn.Linear(4, 10), nn.Linear(10, 2)).double()
    original = copy.deepcopy(model)
    x = torch.randn(3, 4, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 4}))
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 10  # Alignment does not package adjacent channels.

    def score(context, batch):
        return [0 if set(c.remove[0].fully_selected_indices(0)) <= {1, 7} else 1 for c in batch]

    bindings = tuple(model.parameters())
    plan = pruner.plan(space, budget=ChannelRatio(0.2), strategy=Greedy(score))
    assert plan.selection_report.removed == (2,)
    assert "multiple of 4 (path" in plan.explain()
    assert all(p is before for p, before in zip(model.parameters(), bindings, strict=True))
    for p, before in zip(model.parameters(), original.parameters(), strict=True):
        torch.testing.assert_close(p, before)
    restored_plan = PruningPlan.from_dict(plan.to_dict())
    # A static plan applies without retaining its config, candidate space or metric.
    with torch.inference_mode():
        compact, result = Pruner(model).apply(restored_plan)
    assert compact is model and model[0].out_features == model[1].in_features == 8
    keep = [i for i in range(10) if i not in (1, 7)]
    actual_input, reference_input = x.clone().requires_grad_(), x.clone().requires_grad_()
    actual = model(actual_input)
    reference = F.linear(
        F.linear(reference_input, original[0].weight[keep], original[0].bias[keep]),
        original[1].weight[:, keep],
        original[1].bias,
    )
    torch.testing.assert_close(actual, reference)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    assert all(p.is_leaf and not p.is_inference() for p in result.parameter_map.values())


def test_alignment_cannot_exceed_budget_or_be_bypassed_by_manual_or_custom_strategy():
    model = nn.Sequential(nn.Linear(4, 10), nn.Linear(10, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 4}))
    space = pruner.discover_candidates()
    before = tuple(model.parameters())
    with pytest.raises(PlanningError, match="empty request"):
        pruner.plan(space, budget=ChannelRatio(0.1), strategy=Greedy(Magnitude()))
    with pytest.raises(PlanningError, match="indivisible"):
        pruner.plan_remove(space.candidates[0].remove)
    with pytest.raises(PlanningError, match="indivisible"):
        pruner.plan(space, budget=ChannelRatio(0.2), strategy=lambda ctx: [ctx.candidates[0].key])
    assert all(p is old for p, old in zip(model.parameters(), before, strict=True))
    plan = pruner.plan_remove((graph.parameter("0.weight").axis(0).select([0, 2]),))
    pruner.apply(plan)
    assert model[0].out_features == 8


def test_budget_underfill_and_protected_unchanged_axis():
    model = nn.Sequential(nn.Linear(4, 64), nn.Linear(64, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"0": 8}))
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ChannelRatio(0.2), strategy=Greedy(Magnitude())
    )
    assert plan.selection_report.targets == (12,) and plan.selection_report.removed == (8,)
    assert plan.selection_report.shortfall == 4
    protected = Pruner(model, graph=graph, granularity=Granularity(by_type={nn.Linear: 8}))
    with pytest.raises(PlanningError, match="indivisible"):
        protected.plan_remove(())  # The unchanged output width 2 is not divisible by 8.


def test_exact_type_path_precedence_root_and_unused_type_notes():
    class SpecialLinear(nn.Linear):
        pass

    model = nn.Sequential(nn.Linear(4, 16), SpecialLinear(16, 12), nn.Linear(12, 2))
    registry = OperatorRegistry.default()
    registry.register(SpecialLinear, registry.modules[nn.Linear])
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),), operators=registry)
    config = Granularity(default=2, by_type={nn.Linear: 8, nn.Conv2d: 4}, by_path={"0": 4, "2": 1})
    pruner = Pruner(model, graph=graph, granularity=config)
    aligned = {c.axis: c.factor for c in pruner.constraints if isinstance(c, Divisible)}
    assert aligned == {
        graph.parameter("0.weight").axis(0): 4,
        graph.parameter("1.weight").axis(0): 2,
    }
    assert "Conv2d: no declared axes matched" in pruner.plan_remove(()).explain()
    root = nn.Linear(4, 8)
    root_graph = DependencyGraph.build(root, args=(torch.randn(2, 4),))
    root_pruner = Pruner(
        root, graph=root_graph, preserve_io=False, granularity=Granularity(by_path={"": 4})
    )
    root_pruner.apply(
        root_pruner.plan_remove((root_graph.parameter("weight").axis(0).select([0, 2, 4, 6]),))
    )
    assert root.out_features == 4
    for path in ("missing", "", "0.weight"):
        with pytest.raises(PlanningError, match=r"path|declares no"):
            Pruner(model, graph=graph, granularity=Granularity(by_path={path: 8}))


@pytest.mark.parametrize("shared_module", [False, True])
def test_aliases_and_shared_weights_have_distinct_configuration_rules(shared_module):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 16, bias=False)
            self.b = self.a if shared_module else nn.Linear(4, 16, bias=False)
            self.b.weight = self.a.weight
            self.out = nn.Linear(16, 2)

        def forward(self, x):
            return self.out(self.a(x) + self.b(x))

    model = Shared()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    config = Granularity(by_path={"a": 4, "b": 8})
    if shared_module:
        with pytest.raises(PlanningError, match="aliases"):
            Pruner(model, graph=graph, granularity=config)
        config = Granularity(by_path={"b": 8})
    pruner = Pruner(model, graph=graph, granularity=config)
    space = pruner.discover_candidates()
    assert len(space.candidates) == 16
    plan = pruner.plan(space, budget=ChannelRatio(0.5), strategy=Greedy(Magnitude()))
    pruner.apply(plan)
    assert model.a.weight is model.b.weight and model.a.out_features == model.b.out_features == 8


def test_space_is_explicit_immutable_and_checks_graph_identity_and_axis_order():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 8))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    space = pruner.discover_candidates()
    with pytest.raises(FrozenInstanceError):
        space.candidates = ()
    with pytest.raises(TypeError):
        CandidateSpace()
    with pytest.raises(ValueError, match=r"match.*order"):
        pruner.plan(
            space,
            budget=ChannelCount((1, 1), space.channel_axes[::-1]),
            strategy=Greedy(Magnitude()),
        )
    other_graph = DependencyGraph.build(copy.deepcopy(model), args=(torch.randn(2, 4),))
    foreign = Pruner(other_graph.model, graph=other_graph, preserve_io=False)
    with pytest.raises(ValueError, match="another graph"):
        foreign.plan(space, budget=ChannelRatio(0.2), strategy=Greedy(Magnitude()))
    with pytest.raises(ValueError, match="another graph"):
        CumulativeChannelBudget(other_graph, space)
    axis = space.channel_axes[0]
    custom = CandidateSpace([Candidate("pair", (axis.select([0, 3]),))], [axis, axis])
    plan = pruner.plan(custom, budget=ChannelRatio(0.25), strategy=lambda ctx: ["pair"])
    assert plan.selection_report.widths == (8,) and plan.selection_report.removed == (2,)


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_grouped_convolution_alignment_keeps_existing_partition_constraints(
    dimension, execution_device
):
    conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[dimension - 1]
    function = (F.conv1d, F.conv2d, F.conv3d)[dimension - 1]
    model = nn.Sequential(conv(4, 12, 1, groups=2), conv(12, 2, 1)).double()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4, *((3,) * dimension), dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(
        model, graph=graph, granularity=Granularity(by_type={conv: 4}, by_path={"1": 1})
    )
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ChannelRatio(0.34), strategy=Greedy(Magnitude())
    )
    removed = set(plan.analysis.selection(graph.parameter("0.weight")).fully_selected_indices(0))
    assert len(removed) == 4 and len(removed & set(range(6))) == 2
    keep = sorted(set(range(12)) - removed)
    expected = function(
        function(x, original[0].weight[keep], original[0].bias[keep], groups=2),
        original[1].weight[:, keep],
        original[1].bias,
    )
    pruner.apply(plan)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


def test_depthwise_blocks_are_not_redefined_by_alignment():
    model = nn.Sequential(nn.Conv1d(4, 12, 1, groups=4), nn.Conv1d(12, 2, 1))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4, 3),))
    pruner = Pruner(
        model, graph=graph, preserve_io=False, granularity=Granularity(by_path={"0": 4})
    )
    space = pruner.discover_candidates(targets=("0",))
    assert len(space.candidates) == 4
    assert all(len(c.remove[0].fully_selected_indices(0)) == 3 for c in space.candidates)
    plan = pruner.plan(space, budget=ChannelRatio(0.5), strategy=Greedy(Magnitude()))
    # No positive deletion of complete multiplier-3 groups can leave a positive
    # width divisible by four. Alignment does not invent multiplier shrinking.
    assert not plan.recipes and plan.selection_report.shortfall == 6


def test_input_axis_constraints_combine_with_output_granularity():
    model = nn.Linear(4, 8)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    # Input alignment can be specified precisely without enlarging Granularity.
    input_axis = graph.interfaces()[0].axis(1)
    pruner = Pruner(
        model,
        graph=graph,
        preserve_io=False,
        constraints=(Divisible(input_axis, 2),),
        granularity=Granularity(default=4),
    )
    output = graph.parameter("weight").axis(0)
    parameter_input = graph.parameter("weight").axis(1)
    with pytest.raises(PlanningError, match="indivisible"):
        pruner.plan_remove((output.select([0, 2, 4, 6]), parameter_input.select([0])))
    plan = pruner.plan_remove((output.select([0, 2, 4, 6]), parameter_input.select([0, 2])))
    pruner.apply(plan)
    assert model.weight.shape == (4, 2)


def test_fused_module_alignment_applies_to_all_declared_axes(execution_device):
    class TwoProjections(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(8, 4))
            self.b = nn.Parameter(torch.randn(12, 4))

        def forward(self, x):
            return F.linear(x, self.a), F.linear(x, self.b)

    def analyze(ctx):
        domains, relations, constraints = [], [], []
        for name, output in zip(("a", "b"), ctx.output, strict=True):
            weight = ctx.binding(name)
            domains.append(CandidateAxis(name, weight.axis(0)))
            relations.extend(
                (
                    AxisRelation.equal(ctx.inputs[0].axis(1), weight.axis(1)),
                    AxisRelation.equal(weight.axis(0), output.axis(1)),
                )
            )
            constraints.append(NonEmpty(weight.axis(0)))
        return OperatorSpec(
            candidates=tuple(domains), relations=tuple(relations), constraints=tuple(constraints)
        )

    model = TwoProjections()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    registry = OperatorRegistry.default().register(TwoProjections, OperatorRule(analyze=analyze))
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(
        model, graph=graph, preserve_io=False, granularity=Granularity(by_type={TwoProjections: 4})
    )
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ChannelRatio(0.5), strategy=Greedy(Magnitude())
    )
    assert plan.selection_report.removed == (4, 4)
    expected = []
    for name in ("a", "b"):
        weight = getattr(original, name)
        removed = set(plan.analysis.selection(graph.parameter(name)).fully_selected_indices(0))
        keep = sorted(set(range(weight.shape[0])) - removed)
        expected.append(F.linear(x, weight[keep]))
    pruner.apply(plan)
    for actual, reference in zip(model(x), expected, strict=True):
        torch.testing.assert_close(actual, reference)
    sum(y.sum() for y in model(x)).backward()
