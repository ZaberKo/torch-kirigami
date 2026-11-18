"""Analysis limits stay with one request, whatever the cache settings and request order."""

import gc
import itertools
import random
import weakref
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from torch_kirigami import (
    AnalysisLimitError,
    AxisPort,
    AxisRef,
    DependencyGraph,
    Diagnostic,
    IndexSet,
    LayoutConstraint,
    Region,
    Selection,
    config,
)
from torch_kirigami.dep_graph.identity_axes import IdentityAxisIndex
from torch_kirigami.dep_graph.selection import canonical_boxes
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    Magnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    StrategyResult,
    WeightTaylor,
    parameter_regions,
    ranking,
)

# None keeps the default capacity.
CACHE_SIZES = [0, None]


def _set_impact_cache(size):
    if size is not None:
        config.impact_cache_size = size


class TokenBranches(nn.Module):
    """Linear layers on rank-3 inputs, whose output channels are the last of three axes."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 6)
        self.b = nn.Linear(2, 6)

    def forward(self, x):
        return self.a(x).relu().sum(-1), self.b(x).relu().sum(-1)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_identity_slices_span_different_tensor_dimensions(strategy_type):
    """Weight rows and activation channels share one class at different dimensions."""
    torch.manual_seed(17)
    model = TokenBranches().eval()
    x = torch.randn(2, 3, 2)
    expected = []
    for layer in (model.a, model.b):
        energy = layer.weight.detach().square().sum(1) + layer.bias.detach().square()
        kept = sorted(energy.argsort()[2:].tolist())
        weight, bias = layer.weight.detach()[kept], layer.bias.detach()[kept]
        expected.append((x @ weight.T + bias).relu().sum(-1))
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    plan = pruner.plan(
        space, budget=ChannelCount((4, 4), space.channel_axes), strategy=strategy_type(Magnitude())
    )
    assert len(plan.selected) == 4
    pruner.apply(plan)
    assert model.a.out_features == model.b.out_features == 4
    for output, reference in zip(model(x), expected, strict=True):
        torch.testing.assert_close(output, reference)


@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_interleaved_regions_beyond_the_limit_score_and_plan(impact_cache_size):
    """Row groups with more column intervals than `max_parts` still score exactly."""
    _set_impact_cache(impact_cache_size)
    torch.manual_seed(0)
    width = 2 * config.max_parts + 1
    model = nn.Linear(width, 3, bias=False).double().eval()
    model.weight.grad = torch.randn_like(model.weight)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, width, dtype=torch.float64),))
    ref = graph.parameter("weight")
    axis = ref.axis(0)
    interleaved = Candidate(
        "interleaved",
        (
            ref.select(
                (
                    Region((IndexSet.of([0, 1]), IndexSet.of(range(0, width, 4)))),
                    Region((IndexSet.of([1, 2]), IndexSet.of(range(2, width, 4)))),
                )
            ),
        ),
    )
    safe = Candidate("safe", (axis.select([0]),), axis)
    mask = torch.zeros(3, width, dtype=torch.bool)
    mask[0:2, 0::4] = True
    mask[1:3, 2::4] = True
    values, gradients = model.weight.detach()[mask], model.weight.grad[mask]
    references = (
        (Magnitude(1), values.abs().sum()),
        (Magnitude(2), values.square().sum().sqrt()),
        (WeightTaylor(), (values * gradients).abs().sum()),
    )
    context = PlanningContext(graph, (interleaved, safe), ParameterBudget(0), (axis,), ())
    for metric, reference in references:
        (score,) = context.score(metric, (interleaved,))
        assert score == pytest.approx(reference.item(), rel=1e-12)
    regions = parameter_regions(context, interleaved, accepted_impact=context.impact(()))
    assert regions == interleaved.remove
    pruner = Pruner(model, graph=graph, preserve_io=False)
    space = CandidateSpace((interleaved, safe), (axis,))
    for strategy in (Greedy(Magnitude(1)), DynamicGreedy(Magnitude(1))):
        plan = pruner.plan(space, budget=ParameterBudget(2 * width), strategy=strategy)
        assert plan.selected == ("safe",)
    pruner.apply(plan)
    assert model.weight.shape == (2, width)


class Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.arange(1.0, 19.0).reshape(9, 2))

    def forward(self, x):
        return self.p + x


def _mask(selections, shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    for selection in selections:
        for region in selection.regions:
            mask[torch.meshgrid(*(torch.tensor(list(a)) for a in region.axes), indexing="ij")] = 1
    return mask


@pytest.mark.parametrize("case", ["regrouped", "difference"])
@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_additional_regions_beyond_the_limit_score_exactly(case, impact_cache_size):
    """Grouping rows by their columns, or the difference itself, can exceed `max_parts`."""
    config.max_parts = 3
    _set_impact_cache(impact_cache_size)
    model = Table().double().eval()
    model.p.grad = torch.ones_like(model.p)
    graph = DependencyGraph.build(model, args=(torch.zeros(9, 2, dtype=torch.float64),))
    p = graph.parameter("p")

    def selection(pairs):
        return Selection(
            p, tuple(Region((IndexSet.of(rows), IndexSet.of([column]))) for rows, column in pairs)
        )

    expected = torch.zeros(9, 2, dtype=torch.bool)
    if case == "regrouped":
        # Rows 2, 4, 6 and 8 keep only column 0: four separate intervals.
        accepted = selection([([2, 4, 5, 8], 1)])
        seeds = (selection([([2, 3, 4, 6, 7, 8], 0), ([3, 4, 7, 8], 1)]),)
        expected[[2, 3, 4, 6, 7, 8], 0] = True
        expected[[3, 7], 1] = True
    else:
        # The additional rows 0, 2, 4, 6 and 7 form four intervals.
        rows = p.axis(0)
        accepted = rows.select([1, 3])
        seeds = (rows.select([2, 4, 6, 7]), rows.select([0, 4, 7]))
        expected[[0, 2, 4, 6, 7]] = True
    candidate = Candidate("partial", seeds)
    context = PlanningContext(graph, (candidate,), ParameterBudget(0), (), ())
    accepted_impact = context.impact((accepted,))
    assert accepted_impact.complete
    assert context.impact((accepted, *seeds)).complete
    values = model.p.detach()[expected]
    references = (
        (Magnitude(1), values.sum()),
        (Magnitude(2), values.square().sum().sqrt()),
        (WeightTaylor(), values.sum()),
    )
    for metric, reference in references:
        (score,) = context.score(metric, (candidate,), accepted_impact=accepted_impact)
        assert score == pytest.approx(reference.item(), rel=1e-12)
    if case == "regrouped":
        regions = parameter_regions(context, candidate, accepted_impact=accepted_impact)
        assert torch.equal(_mask(regions, (9, 2)), expected)
    else:
        with pytest.raises(AnalysisLimitError):
            parameter_regions(context, candidate, accepted_impact=accepted_impact)


def _fibers(points, dim):
    """Describe points by boxes whose intervals run along one axis only."""
    groups = {}
    for point in points:
        groups.setdefault(point[:dim] + point[dim + 1 :], []).append(point[dim])
    boxes = []
    for rest, positions in groups.items():
        # Built by hand: `IndexSet` would apply the `max_parts` limit.
        intervals = []
        for i in sorted(positions):
            if intervals and intervals[-1][1] == i:
                intervals[-1] = (intervals[-1][0], i + 1)
            else:
                intervals.append((i, i + 1))
        axes = [((i, i + 1),) for i in rest]
        axes.insert(dim, tuple(intervals))
        boxes.append(tuple(axes))
    return boxes


def _positions(box):
    return set(itertools.product(*(tuple(i for a, b in axis for i in range(a, b)) for axis in box)))


def test_canonical_boxes_depend_only_on_coordinates():
    """Different descriptions of one position set give one exact disjoint decomposition."""
    config.max_parts = 2  # Boxes are not limited by `max_parts`.
    rng = random.Random(3)
    for shape in ((4, 3), (3, 2, 3), (1, 9)):
        coordinates = list(itertools.product(*map(range, shape)))
        for _ in range(150):
            wanted = {c for c in coordinates if rng.random() < 0.5}
            excluded = {c for c in coordinates if c not in wanted and rng.random() < 0.5}
            extra = {c for c in excluded if rng.random() < 0.5}
            results = set()
            for plus_dim, minus_dim in itertools.product(range(len(shape)), repeat=2):
                plus = _fibers(wanted | extra, plus_dim)
                rng.shuffle(plus)
                boxes = canonical_boxes(plus, _fibers(excluded, minus_dim))
                covered = [_positions(box) for box in boxes]
                assert sum(map(len, covered)) == len(wanted)
                assert set().union(*covered) == wanted
                for box in boxes:
                    for intervals in box:
                        assert all(a < b for a, b in intervals)
                        assert all(b < c for (_, b), (c, _) in itertools.pairwise(intervals))
                results.add(boxes)
            assert len(results) == 1
            assert canonical_boxes(_fibers(wanted, 0)) == results.pop()


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.b = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))

    def forward(self, x):
        return self.a(x), self.b(x)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_seed_limits_exclude_one_candidate_and_planning_continues(strategy_type, impact_cache_size):
    """Direct queries raise for unmergeable seeds; automatic planning records them."""
    config.max_parts = 2
    _set_impact_cache(impact_cache_size)
    torch.manual_seed(2)
    model = Branches().eval()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    rows = graph.parameter("a.0.weight").axis(0)
    safe_axis = graph.parameter("b.0.weight").axis(0)
    merged = Candidate("merged", tuple(rows.select([i]) for i in (0, 2, 4)))
    # Seeds on different tensors of one identity class merge only during
    # propagation, where the limit is an incomplete diagnostic.
    relu = next(ref for ref in graph.values() if ref.id.endswith(":value:a_1:0"))
    spread = Candidate(
        "spread",
        (
            rows.select([0]),
            graph.parameter("a.0.bias").axis(0).select([2]),
            relu.axis(1).select([4]),
        ),
    )
    safe = Candidate("safe", (safe_axis.select([0]),), safe_axis)
    pruner = Pruner(model, graph=graph)
    with pytest.raises(AnalysisLimitError):
        graph.propagate(remove=merged.remove)
    with pytest.raises(AnalysisLimitError):
        pruner.impact((merged,))
    with pytest.raises(AnalysisLimitError):
        pruner.plan_remove(merged.remove)
    canonical = IdentityAxisIndex(graph.relations).canonical_seeds(spread.remove)
    with pytest.raises(AnalysisLimitError):
        graph.propagate(remove=canonical, constraints=pruner.constraints)
    direct = graph.propagate(remove=spread.remove, constraints=pruner.constraints)
    assert not direct.complete
    context = PlanningContext(graph, (merged, spread, safe), ParameterBudget(0), (), ())
    for _ in range(2):
        assert not context.impact(merged.remove).complete
        assert context.impact(spread.remove).diagnostics == direct.diagnostics
    space = CandidateSpace((merged, spread, safe), (rows, safe_axis))
    budget = ParameterBudget(sum(p.numel() for p in model.parameters()) - 7)
    plan = pruner.plan(space, budget=budget, strategy=strategy_type(Magnitude()))
    assert plan.selected == ("safe",)
    exclusions = dict(plan.selection_report.exclusions)
    assert "analysis_limit" in exclusions["merged"] and "analysis_limit" in exclusions["spread"]
    expected = (x @ model.b[0].weight[1:].T + model.b[0].bias[1:]) @ model.b[1].weight[:, 1:].T
    expected = expected + model.b[1].bias
    pruner.apply(plan)
    assert model.b[0].out_features == 2
    torch.testing.assert_close(model(x)[1], expected.detach())


class Broadcasts(nn.Module):
    """Two broadcast additions, so slices on three axes of `a` meet `b`."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(3, 3, 3))
        self.b = nn.Parameter(torch.zeros(3, 3, 3))
        self.tail = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.a[0, 1:, :2] = 0.01
            self.a[1, 1:, :2] = 0.1
            self.tail.weight.fill_(1.0)

    def forward(self, x, y):
        return (x + self.a).relu() + self.b, self.tail(y)


def _broadcasts(max_parts):
    config.max_parts = max_parts
    model = Broadcasts()
    args = (torch.ones(3, 3, 3), torch.ones(1, 3))
    graph = DependencyGraph.build(model, args=args)
    values = {ref.id.split(":")[-2]: ref for ref in graph.values() if ":value:" in ref.id}
    return model, args, graph, values


def _equivalent_requests(graph, values):
    """Return two requests for the same slices on three broadcast axes."""
    seeds = (
        values["add"].axis(0).select([2]),
        graph.parameter("a").axis(1).select([1]),
        values["add_1"].axis(1).select([0]),
        values["add"].axis(2).select([1]),
    )
    canonical = IdentityAxisIndex(graph.relations).canonical_seeds(seeds)
    assert graph.propagate(remove=canonical).complete
    assert graph.propagate(remove=seeds).complete
    return seeds, canonical


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_proved_broadcast_closure_preserves_plan_and_output(strategy_type, impact_cache_size):
    """A complete coordinate proof also supports independent final verification."""
    _set_impact_cache(impact_cache_size)
    model, args, graph, values = _broadcasts(3)
    a = graph.parameter("a")
    tail = graph.parameter("tail.weight").axis(0)
    seeds, canonical = _equivalent_requests(graph, values)
    limited = Candidate("limited", seeds)
    safe = Candidate("safe", (tail.select([0]),), tail)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    assert graph.propagate(remove=canonical, constraints=pruner.constraints).complete
    direct = graph.propagate(remove=limited.remove, constraints=pruner.constraints)
    assert direct.complete
    expected_mask = torch.ones(3, 3, 3, dtype=torch.bool)
    expected_mask[:2, 2:, ::2] = False
    assert torch.equal(_mask((direct.selection(a),), a.shape), expected_mask)
    space = CandidateSpace((limited, safe), (a.axis(0), tail))
    budget = ParameterBudget(20)
    context = PlanningContext(
        graph, space.candidates, budget, space.channel_axes, pruner.constraints
    )
    assert context.impact(canonical).selections == context.impact(seeds).selections
    expected = model(*args)[0][:2, 2:, ::2].detach()
    plan = pruner.plan(space, budget=budget, strategy=strategy_type(Magnitude(1), max_trials=2))
    assert "limited" in plan.selected
    pruner.apply(plan)
    torch.testing.assert_close(model(args[0][:2, 2:, ::2], args[1])[0], expected)


def _same_closure_requests(graph, values):
    """Return equivalent accepted requests and a conditional slice."""
    tail = graph.parameter("b").axis(2).select([0])
    good = (values["add_1"].axis(1).select([2]), values["relu"].axis(1).select([1]), tail)
    bad = (values["relu"].axis(1).select([2]), values["add_1"].axis(1).select([1]), tail)
    blocked = graph.parameter("a").axis(0).select([0])
    closures = [graph.propagate(remove=request) for request in (good, bad)]
    assert all(closure.complete for closure in closures)
    # Even the region decompositions agree, not only the selected coordinates.
    assert [
        [(ref, selection.regions) for ref, selection in closure.selections.items()]
        for closure in closures
    ] == [[(ref, selection.regions) for ref, selection in closures[0].selections.items()]] * 2
    assert graph.propagate(remove=(*good, blocked)).complete
    assert graph.propagate(remove=(*bad, blocked)).complete
    return good, bad, blocked


@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_parameter_groups_preserve_order_for_proved_broadcasts(impact_cache_size):
    """Parameter selections agree for both descriptions of the proved closure."""
    _set_impact_cache(impact_cache_size)
    model, _args, graph, values = _broadcasts(3)
    seeds, canonical = _equivalent_requests(graph, values)
    complete = Candidate("complete", canonical)
    limited = Candidate("limited", seeds)
    pruner = Pruner(model, graph=graph, preserve_io=False)

    def by_tensor(selections):
        return {selection.tensor.id: selection for selection in selections}

    expected = by_tensor(graph.propagate(remove=complete.remove).parameters)
    (alone,) = pruner.parameter_groups((complete,))
    assert by_tensor(alone.selections) == expected
    for candidates in ((complete, limited), (limited, complete), (limited,)):
        groups = pruner.parameter_groups(candidates)
        assert [group.key for group in groups] == [c.key for c in candidates]
        assert all(by_tensor(group.selections) == expected for group in groups)
    twice = pruner.parameter_groups((complete, Candidate("again", complete.remove)))
    assert [by_tensor(group.selections) for group in twice] == [expected, expected]


@pytest.mark.parametrize("candidate_cache_size", CACHE_SIZES)
def test_filter_reports_proved_broadcast_conflicts(candidate_cache_size):
    """Equivalent whole-tensor deletions receive the same irreversible diagnostic."""
    if candidate_cache_size is not None:
        config.candidate_cache_size = candidate_cache_size
    model, _args, graph, values = _broadcasts(3)
    a, b = graph.parameter("a"), graph.parameter("b")
    seeds = (
        a.axis(0).select([0]),
        a.axis(1).select([1]),
        values["relu"].axis(1).select([2]),
        b.axis(1).select([0]),
    )
    canonical = IdentityAxisIndex(graph.relations).canonical_seeds(seeds)
    pruner = Pruner(model, graph=graph)
    assert graph.propagate(remove=canonical, constraints=pruner.constraints).complete
    assert graph.propagate(remove=seeds, constraints=pruner.constraints).complete
    complete, limited = Candidate("complete", canonical), Candidate("limited", seeds)

    def exclusions(candidates):
        space = pruner.filter_candidates(CandidateSpace(candidates, ()))
        return dict(space.exclusions)

    alone = exclusions((limited,))["limited"]
    assert "empty_axis" in alone
    assert "empty_axis" in exclusions((complete,))["complete"]
    for candidates in ((complete, limited), (limited, complete)):
        assert exclusions(candidates)["limited"] == alone


@pytest.mark.parametrize("normalization_cache_size", CACHE_SIZES)
@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_normalization_agrees_for_proved_broadcasts(impact_cache_size, normalization_cache_size):
    """Singleton population scores use the proved conditional closure."""
    _set_impact_cache(impact_cache_size)
    if normalization_cache_size is not None:
        config.normalization_cache_size = normalization_cache_size
    _model, _args, graph, values = _broadcasts(3)
    good, bad, blocked = _same_closure_requests(graph, values)
    axis = graph.parameter("a").axis(0)
    candidate = Candidate("single", (blocked,), axis)
    context = PlanningContext(graph, (candidate,), ParameterBudget(0), (axis,), ())
    accepted = {name: context.impact(seeds) for name, seeds in (("good", good), ("bad", bad))}
    assert accepted["good"].selections == accepted["bad"].selections
    assert all(impact.complete for impact in accepted.values())
    metric = GroupMagnitude()
    reference = GroupMagnitude().score(context, (candidate,), accepted_impact=accepted["good"])
    for name in ("good", "bad", "good", "bad"):
        assert metric.score(context, (candidate,), accepted_impact=accepted[name]) == reference


def test_conditional_norms_after_removal_through_flatten(monkeypatch):
    """Removing convolution channels reaches the next weight through a flatten."""
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1, bias=False),
        nn.Flatten(1),
        nn.Linear(16, 128, bias=False),
        nn.ReLU(),
        nn.Linear(128, 2, bias=False),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(12.0).reshape(4, 3, 1, 1) * 1e-4)
        model[2].weight.fill_(1.0)
        model[2].weight[:, :4].fill_(1e-4)
        model[4].weight.fill_(1.0)
    x = torch.ones(2, 3, 2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0", "2"))
    budget = ChannelCount((3, 127), space.channel_axes)
    conditional = []
    original = ranking._position_norms

    def counted(metric, members, bindings, positions, base=None):
        positions = tuple(positions)
        if base is not None:
            conditional.append(len(positions))
        return original(metric, members, bindings, positions, base)

    monkeypatch.setattr(ranking, "_position_norms", counted)
    plan = pruner.plan(space, budget=budget, strategy=DynamicGreedy(Magnitude()))
    assert conditional == [128]  # The other branch changed four input columns of this weight.
    monkeypatch.setattr(ranking, "_position_norms", original)

    class OrdinaryMagnitude(Magnitude):
        pass

    ordinary = pruner.plan(space, budget=budget, strategy=DynamicGreedy(OrdinaryMagnitude()))
    assert plan.selected == ordinary.selected
    assert plan.selection_report == ordinary.selection_report
    pruned, _ = pruner.apply(plan)
    assert pruned[0].weight.shape == (3, 3, 1, 1) and pruned[2].weight.shape == (127, 12)
    assert pruned(x).shape == (2, 2)


class ScopedTable(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 10))
        self.safe = nn.Linear(3, 4)

    def forward(self, x, y):
        return (self.weight + x).relu(), self.safe(y)


@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_equal_closures_pass_partitioned_layouts_alike(impact_cache_size):
    """Equal closures with different region decompositions check one partitioned layout alike."""
    config.max_parts = 4
    _set_impact_cache(impact_cache_size)
    torch.manual_seed(4)
    model = ScopedTable().eval()
    args = (torch.randn(4, 10), torch.randn(2, 3))
    graph = DependencyGraph.build(model, args=args)
    weight = graph.parameter("weight")
    x = next(ref for ref in graph.values() if ref.kind == "input" and ref.shape == weight.shape)
    scoped = Region((IndexSet.of([2]), IndexSet(((0, 3), (4, 5), (6, 7), (8, 9)))))
    layout = LayoutConstraint(
        weight,
        (
            AxisPort(weight.axis(0), Region((IndexSet.span(2, 4), IndexSet.span(0, 10)))),
            AxisPort(weight.axis(1), Region((IndexSet.span(0, 2), IndexSet.span(0, 10)))),
            AxisPort(weight.axis(0), scoped),
        ),
    )
    good = Candidate("good", (x.axis(1).select([1]), weight.axis(0).select([2, 3])))
    bad = Candidate("bad", (weight.axis(1).select([1]), x.axis(0).select([2, 3])))
    safe_axis = graph.parameter("safe.weight").axis(0)
    safe = Candidate("safe", (safe_axis.select([0]),), safe_axis)
    pruner = Pruner(model, graph=graph, preserve_io=False, constraints=(layout,))
    direct = [graph.propagate(remove=c.remove, constraints=pruner.constraints) for c in (good, bad)]
    assert direct[0].selection(weight) == direct[1].selection(weight)
    assert direct[0].selection(weight).regions != direct[1].selection(weight).regions
    assert direct[0].complete and direct[1].complete
    assert direct[0].diagnostics == direct[1].diagnostics
    space = CandidateSpace((good, bad, safe), (safe_axis,))
    budget = ParameterBudget(sum(p.numel() for p in model.parameters()) - 4)
    context = PlanningContext(graph, space.candidates, budget, (), pruner.constraints)
    context.impact(good.remove)
    assert context.impact(bad.remove).diagnostics == direct[1].diagnostics
    for order in ((good, bad), (bad, good)):
        groups = pruner.parameter_groups(order)
        assert [group.key for group in groups] == [c.key for c in order]

    class PickBad:
        def select(self, context):
            context.impact(good.remove)
            assert context.impact(bad.remove).complete
            return StrategyResult(keys=("bad",))

    plan = pruner.plan(space, budget=budget, strategy=PickBad())
    columns = [0, *range(2, 10)]
    inputs = (args[0][:2, columns], args[1])
    expected = ((1.0 + inputs[0]).relu(), model.safe(args[1]).detach())
    pruner.apply(plan)
    for output, reference in zip(model(*inputs), expected, strict=True):
        torch.testing.assert_close(output.detach(), reference)


@dataclass(frozen=True)
class UnprovedPosition:
    """Report one complete position of an axis as unproved.

    Attributes:
        axis: Axis whose position this constraint cannot prove.
        position: The unproved position.
    """

    axis: AxisRef
    position: int

    @property
    def refs(self):
        return (self.axis.tensor,)

    def check(self, selections):
        selection = selections.get(self.axis.tensor.id)
        if selection and self.position in selection.fully_selected_indices(self.axis.dim):
            return Diagnostic("opaque", "Unproved position", complete=False)
        return None


def _branch_without_first_channel(model, x):
    b = model.b
    hidden = x @ b[0].weight[1:].T + b[0].bias[1:]
    return (hidden @ b[1].weight[:, 1:].T + b[1].bias).detach()


@pytest.mark.parametrize("case", ["unproved", "nonfinite"])
@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_unscorable_normalization_excludes_only_its_axis(case, strategy_type, impact_cache_size):
    """A complete candidate whose axis cannot be normalized is excluded alone."""
    _set_impact_cache(impact_cache_size)
    torch.manual_seed(6)
    model = Branches().eval()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    rows = graph.parameter("a.0.weight").axis(0)
    other = graph.parameter("b.0.weight").axis(0)
    constraints = ()
    if case == "unproved":
        constraints = (UnprovedPosition(rows, 7),)
    else:
        with torch.no_grad():
            model.a[0].weight[7, 0] = torch.inf
    pruner = Pruner(model, graph=graph, constraints=constraints)
    first = Candidate("a", (rows.select([1]),), rows)
    safe = Candidate("b", (other.select([0]),), other)
    assert pruner.impact((first,)).complete
    space = CandidateSpace((first, safe), (rows, other))
    budget = ParameterBudget(sum(p.numel() for p in model.parameters()) - 7)
    plan = pruner.plan(space, budget=budget, strategy=strategy_type(GroupMagnitude(1)))
    assert plan.selected == ("b",)
    reason = {"unproved": "Unproved position", "nonfinite": "nonfinite"}[case]
    assert reason in dict(plan.selection_report.exclusions)["a"]
    expected = _branch_without_first_channel(model, x)
    pruner.apply(plan)
    torch.testing.assert_close(model(x)[1], expected)


class BatchOnlyFailure(Magnitude):
    """Break batch independence: fail only for batches of several candidates."""

    def score(self, context, candidates, *, accepted_impact):
        if len(candidates) > 1:
            raise PlanningError("Batch-only failure")
        return super().score(context, candidates, accepted_impact=accepted_impact)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_batch_failures_without_a_failing_candidate_are_raised(strategy_type):
    model = Branches().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 4),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("a.0", "b.0"))
    with pytest.raises(PlanningError, match="Batch-only failure"):
        pruner.plan(
            space,
            budget=ChannelCount((6, 2), space.channel_axes),
            strategy=strategy_type(BatchOnlyFailure()),
        )


class ZeroedBranch(nn.Module):
    """Two branches; rows 1 and 5 of branch `a` carry no weight."""

    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(4, 7, bias=False), nn.Linear(7, 2, bias=False))
        self.b = nn.Sequential(nn.Linear(4, 7, bias=False), nn.Linear(7, 2, bias=False))
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(1.0)
            self.a[0].weight[[1, 5]] = 0.0
            self.a[1].weight[:, [1, 5]] = 0.0

    def forward(self, x):
        return self.a(x), self.b(x)


@pytest.mark.parametrize("metric_type", [GroupMagnitude, Magnitude])
@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_group_scores_merge_seeds_with_the_accepted_slices(metric_type, impact_cache_size):
    """Slices that exceed the limit alone are scored once the accepted slices fill the gaps."""
    config.max_parts = 3
    _set_impact_cache(impact_cache_size)
    model = ZeroedBranch().eval()
    x = torch.ones(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    rows = graph.parameter("a.0.weight").axis(0)
    other = graph.parameter("b.0.weight").axis(0)
    initial = Candidate("initial", (rows.select([1, 5]),), rows)
    fragmented = Candidate("fragmented", tuple(rows.select([i]) for i in (0, 2, 4, 6)), rows)
    safe = Candidate("safe", (other.select([0, 1, 2, 3]),), other)
    pruner = Pruner(model, graph=graph)
    assert pruner.impact((initial, fragmented)).complete
    space = CandidateSpace((initial, fragmented, safe), (rows, other))
    plan = pruner.plan(space, budget=ParameterBudget(48), strategy=DynamicGreedy(metric_type(1)))
    assert plan.selected == ("initial", "fragmented")
    expected = (x @ model.a[0].weight[[3]].T @ model.a[1].weight[:, [3]].T).detach()
    pruner.apply(plan)
    assert model.a[0].out_features == 1 and model.b[0].out_features == 7
    torch.testing.assert_close(model(x)[0], expected)


def _counted_propagation(monkeypatch):
    """Count propagations and the most impacts alive at once."""
    original = DependencyGraph.propagate
    live = {}
    stats = {"calls": 0, "peak": 0}

    def counted(self, *args, **kwargs):
        impact = original(self, *args, **kwargs)
        index = stats["calls"]
        stats["calls"] += 1
        live[index] = weakref.ref(impact, lambda _: live.pop(index))
        stats["peak"] = max(stats["peak"], len(live))
        return impact

    monkeypatch.setattr(DependencyGraph, "propagate", counted)
    return stats


def _flattened(width):
    model = nn.Sequential(
        nn.Conv2d(3, width, 1), nn.ReLU(), nn.Flatten(1), nn.Linear(4 * width, 3)
    ).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3, 2, 2),))
    return Pruner(model, graph=graph), graph


@pytest.mark.parametrize("impact_cache_size", CACHE_SIZES)
def test_parameter_groups_analyze_equal_requests_once(monkeypatch, impact_cache_size):
    _set_impact_cache(impact_cache_size)
    pruner, graph = _flattened(64)
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(k), (axis.select([0]),), axis) for k in range(64))
    expected = graph.propagate(remove=candidates[0].remove).parameters
    stats = _counted_propagation(monkeypatch)
    groups = pruner.parameter_groups(candidates)
    assert stats["calls"] == 1
    assert all(group.selections == expected for group in groups)
    assert [group.key for group in groups] == [c.key for c in candidates]


def test_parameter_groups_keep_no_impacts(monkeypatch):
    pruner, graph = _flattened(64)
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(64))
    gc.collect()
    stats = _counted_propagation(monkeypatch)
    groups = pruner.parameter_groups(candidates)
    assert stats["calls"] == 64 and stats["peak"] <= 2
    assert [group.key for group in groups] == [c.key for c in candidates]


@pytest.mark.parametrize("impact_cache_size", [0, 1, None])
def test_parameter_groups_share_equivalent_requests_within_the_cache(
    monkeypatch, impact_cache_size
):
    """Equivalent requests share one complete analysis, up to `impact_cache_size` of them."""
    _set_impact_cache(impact_cache_size)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 3)).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3),))
    pruner = Pruner(model, graph=graph)
    bias = graph.parameter("0.bias").axis(0)
    hidden = next(ref for ref in graph.values() if ref.shape == (1, 4)).axis(1)
    # Weight rows take the projection shortcut, so these seeds need propagation.
    candidates = (
        Candidate("b0", (bias.select([0]),)),
        Candidate("b1", (bias.select([1]),)),
        Candidate("h1", (hidden.select([1]),)),
        Candidate("h0", (hidden.select([0]),)),
    )
    expected = [graph.propagate(remove=c.remove).parameters for c in candidates]
    stats = _counted_propagation(monkeypatch)
    groups = pruner.parameter_groups(candidates)
    # A single cache entry still holds `b1` for `h1`, but no longer `b0` for `h0`.
    assert stats["calls"] == {0: 4, 1: 3, None: 2}[impact_cache_size]
    assert [group.selections for group in groups] == expected


@pytest.mark.parametrize("column", [False, True])
@pytest.mark.parametrize("scoped_first", [False, True])
def test_redundant_partition_does_not_limit_coverage(column, scoped_first):
    """A scoped port may cover more intervals than a materialized axis could store."""
    config.max_parts = 5
    model = nn.Linear(3, 20, bias=False).double().eval()
    x = torch.randn(2, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    weight = graph.parameter("weight")
    rows = [i for i in range(20) if i % 4 != 3]
    first, middle, last = rows[:5], rows[5:10], rows[10:]
    # Tile the same rectangle with five pieces that pairwise coalescing cannot merge.
    pieces = tuple(
        Region((IndexSet.of(band), IndexSet.of(cols)))
        for band, cols in (
            (first, (0, 1)),
            (first + middle, (2,)),
            (last, (1, 2)),
            (middle + last, (0,)),
            (middle, (1,)),
        )
    )
    assert weight.select(pieces) == weight.axis(0).select(rows)
    scope = Region((IndexSet(((1, 5), (6, 9), (10, 13), (14, 17), (18, 20))), IndexSet.span(0, 3)))
    scoped = AxisPort(weight.axis(0), scope)
    full = (AxisPort(weight.axis(0)), AxisPort(weight.axis(1)))
    ports = (scoped, *full) if scoped_first else (*full, scoped)
    pruner = Pruner(
        model, graph=graph, preserve_io=False, constraints=(LayoutConstraint(weight, ports),)
    )
    remove = (
        (weight.axis(0).select(rows), weight.axis(1).select([0]))
        if column
        else (weight.select(pieces),)
    )
    plan = pruner.plan_remove(remove)
    kept_inputs = x[:, 1:] if column else x
    weights = model.weight.detach()[3::4, 1:] if column else model.weight.detach()[3::4]
    expected = kept_inputs @ weights.T
    pruner.apply(plan)
    torch.testing.assert_close(model(kept_inputs), expected)


def test_single_slice_cache_replays_operator_reasons_from_the_requested_seed():
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    context = PlanningContext(graph, (), ParameterBudget(0), (), ())
    first = graph.parameter("0.weight").axis(0).select([1])
    second = graph.parameter("3.weight").axis(1).select([1])
    assert context.impact((first,)).complete
    cached = context.impact((second,))
    assert cached == graph.propagate(remove=(second,))
    known = {second.tensor.id: second}
    for step in cached.provenance:
        assert step.reason in {r.reason for r in graph.relations}
        assert step.source.issubset(known[step.source.tensor.id])
        previous = known.get(step.target.tensor.id, Selection(step.target.tensor))
        assert step.target.subtract(previous) == step.target
        known[step.target.tensor.id] = previous.union(step.target)
    assert known == cached.selections


@pytest.mark.parametrize("limit", [2, 3, 4])
def test_complete_broadcast_closures_match_coordinate_reference(limit):
    """Every successful proof must select exactly the forced cells on every branch tensor."""
    _model, _args, graph, values = _broadcasts(limit)
    a = graph.parameter("a")
    seeds = (
        values["add_1"].axis(0).select([2]),
        values["add"].axis(1).select([0]),
        values["x"].axis(2).select([2]),
        a.axis(0).select([0]),
    )
    mask = torch.ones(3, 3, 3, dtype=torch.bool)
    mask[1, 1:, :2] = False
    for order in itertools.permutations(seeds):
        impact = graph.propagate(remove=order)
        if not impact.complete:
            assert limit == 2 and any(d.code == "analysis_limit" for d in impact.diagnostics)
            continue
        refs = (a, graph.parameter("b"), *(values[k] for k in ("add", "add_1", "x", "relu")))
        assert set(impact.selections) == {ref.id for ref in refs}
        for ref in refs:
            assert torch.equal(_mask((impact.selection(ref),), ref.shape), mask)


def test_native_cache_preserves_extension_requests_with_identical_coordinates():
    class CustomSelection(Selection):
        """Use ordinary propagation for this extension's original request."""

    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3),))
    context = PlanningContext(graph, (), ParameterBudget(0), (), ())
    seed = graph.parameter("0.bias").axis(0).select([1])
    custom = CustomSelection(seed.tensor, seed.regions)
    for request in (custom, seed, custom):
        impact = context.impact((request,))
        assert impact.requested[0] is request
        assert impact == graph.propagate(remove=(request,))
