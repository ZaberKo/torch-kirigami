import copy
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    Balanced,
    DependencyGraph,
    Divisible,
    IndexSet,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    ExecutionError,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    WeightTaylor,
)


def build(model, x):
    graph = DependencyGraph.build(model, args=(x,))
    return graph, Pruner(model, graph=graph)


def test_manual_chain_readonly_and_apply_inference(execution_device):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    x = torch.randn(2, 4)
    original = copy.deepcopy(model)
    graph, pruner = build(model, x)
    old = model[0].weight
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1, 4])])
    assert model[0].weight is old
    assert model[0].out_features == 6
    with pytest.raises(FrozenInstanceError):
        plan.selected = ()
    with torch.inference_mode():
        returned, result = pruner.apply(plan)
    assert returned is model
    assert result.parameter_map[old] is model[0].weight
    assert model[0].weight.grad is None and not model[0].weight.is_inference()
    keep = [0, 2, 3, 5]
    hidden = F.linear(x, original[0].weight[keep], original[0].bias[keep]).relu()
    reference = F.linear(hidden, original[2].weight[:, keep], original[2].bias)
    torch.testing.assert_close(model(x), reference)
    model(x).sum().backward()
    assert model[0].weight.grad is not None
    with pytest.raises(ExecutionError, match="preconditions"):
        pruner.apply(plan)
    with pytest.raises(StaleGraphError):
        graph.propagate(remove=[])


def test_io_protection_and_mutual_exclusion():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    remove = [graph.parameter("weight").axis(0).select([1])]
    with pytest.raises(PlanningError, match="fixed_axis"):
        pruner.plan(remove=remove)
    with pytest.raises(ValueError, match="mutually"):
        pruner.plan(remove=remove, metric=Magnitude())
    plan = pruner.plan(remove=remove, preserve_io=False)
    pruner.apply(plan)
    assert model.out_features == 5


@pytest.mark.parametrize(
    "conv,shape", [(nn.Conv1d, (2, 6, 5)), (nn.Conv2d, (2, 6, 4, 5)), (nn.Conv3d, (2, 6, 3, 4, 5))]
)
def test_grouped_rows_and_different_local_columns(conv, shape, execution_device):
    model = conv(6, 6, 1, groups=2)
    x = torch.randn(shape)
    original = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = pruner.plan(
        remove=[
            graph.calls("")[0].input().axis(1).select([0, 4]),
            graph.parameter("weight").axis(0).select([1, 5]),
        ],
        preserve_io=False,
    )
    weight_recipe = next(
        r for r in plan.recipes if r.tensor.paths == graph.parameter("weight").paths
    )
    assert len(weight_recipe.segments) == 2
    pruner.apply(plan)
    reference = conv(4, 4, 1, groups=2)
    with torch.no_grad():
        reference.weight.copy_(
            torch.cat([original.weight[[0, 2]][:, [1, 2]], original.weight[[3, 4]][:, [0, 2]]])
        )
        reference.bias.copy_(original.bias[[0, 2, 3, 4]])
    compact_x = x[:, [1, 2, 3, 5]]
    torch.testing.assert_close(model(compact_x), reference(compact_x))
    assert model.in_channels == model.out_channels == 4


@pytest.mark.parametrize("whole", [True, False])
def test_depthwise_groups_and_multiplier(whole, execution_device):
    model = nn.Conv2d(4, 8, 1, groups=4)
    x = torch.randn(2, 4, 3, 3)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    indices = [2, 3] if whole else [1, 2, 5, 6]
    plan = pruner.plan(
        remove=[graph.parameter("weight").axis(0).select(indices)], preserve_io=False
    )
    pruner.apply(plan)
    assert model.groups == (3 if whole else 4)
    keep = [i for i in range(8) if i not in indices]
    compact_x = x[:, [0, 2, 3]] if whole else x
    reference = F.conv2d(compact_x, old.weight[keep], old.bias[keep], groups=model.groups)
    torch.testing.assert_close(model(compact_x), reference)


def test_shared_module_parameters_and_aliases(execution_device):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)
            self.alias = self.a
            self.b = nn.Linear(4, 6)
            self.b.weight = self.a.weight
            self.out = nn.Linear(6, 2)

        def forward(self, x):
            return self.out(self.a(x) + self.alias(x) + self.b(x))

    model = Shared()
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    old = model.a.weight
    plan = pruner.plan(remove=[graph.parameter("a.weight").axis(0).select([1, 3])])
    _returned, result = pruner.apply(plan)
    assert model.a.weight is model.alias.weight is model.b.weight
    assert len([p for p in result.parameter_map if p is old]) == 1
    assert model(x).shape == (2, 2)


def test_greedy_initial_invalid_divisibility_and_limit():
    model = nn.Linear(4, 10)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    kwargs = {"metric": Magnitude(), "preserve_io": False, "constraints": [Divisible(axis, 4)]}
    plan = pruner.plan(budget=ChannelRatio(0.2), **kwargs)
    assert plan.budget.removed == (2,)
    with pytest.raises(PlanningError, match="empty request"):
        pruner.plan(budget=ChannelRatio(0.1), **kwargs)
    plan = pruner.plan(
        metric=Magnitude(),
        budget=ChannelRatio(0.4),
        preserve_io=False,
        strategy=Greedy(max_trials=1),
    )
    assert plan.budget.limit_reached
    assert plan.analysis.status == "resolved"
    assert plan.budget.removed == (1,)


def test_balanced_completion_multiple_constraints_and_stable_ties():
    model = nn.Linear(4, 12, bias=False)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    constraints = [
        Balanced(axis, tuple(IndexSet.span(i, i + n) for i in range(0, 12, n))) for n in (6, 4)
    ]

    def metric(ctx, batch):
        return [0.0] * len(batch)

    kwargs = {
        "metric": metric,
        "budget": ChannelRatio(0.5),
        "constraints": constraints,
        "preserve_io": False,
    }
    a, b = pruner.plan(**kwargs), pruner.plan(**kwargs)
    assert a.selected == b.selected
    assert a.analysis.status == "resolved"
    assert 0 <= sum(a.budget.removed) <= 6
    manual = pruner.plan(
        remove=[axis.select([0, 1, 4, 6, 8, 9])], constraints=constraints, preserve_io=False
    )
    assert manual.analysis.status == "resolved"
    assert a.budget.trials <= 10_000


def test_depthwise_candidate_denominator_and_custom_blocks():
    model = nn.Conv1d(4, 8, 1, groups=4)
    graph, pruner = build(model, torch.randn(2, 4, 5))
    plan = pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.25), preserve_io=False)
    assert plan.budget.widths == (8,) and plan.budget.removed == (2,)
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("pair", (axis.select([0, 1]),))
    with pytest.raises(ValueError, match="explicit"):
        pruner.plan(
            candidates=[candidate], metric=Magnitude(), budget=ChannelRatio(0.25), preserve_io=False
        )
    manual = pruner.plan(
        candidates=[candidate],
        metric=Magnitude(),
        budget=ChannelRatio(0.25, axes=(axis,)),
        preserve_io=False,
    )
    assert manual.budget.widths == plan.budget.widths


@pytest.mark.parametrize(
    "metric",
    [Magnitude(1), Magnitude(2), WeightTaylor("elementwise_abs"), WeightTaylor("joint_abs")],
)
def test_metric_union_formula_and_precision(metric, execution_device):
    model = nn.Linear(4, 5, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(-10, 10, dtype=torch.float64).reshape(5, 4))
    model.weight.grad = torch.linspace(-2, 2, 20, dtype=torch.float64).reshape(5, 4)
    graph, pruner = build(model, torch.randn(2, 4, dtype=torch.float64))
    ref = graph.parameter("weight")
    candidate = Candidate("joint", (ref.axis(0).select([1]), ref.axis(1).select([2])))

    def strategy(ctx):
        score = ctx.score([candidate])[0]
        mask = torch.zeros_like(model.weight, dtype=torch.bool)
        mask[1] = True
        mask[:, 2] = True
        if isinstance(metric, Magnitude):
            expected = (
                model.weight[mask].abs().sum()
                if metric.p == 1
                else model.weight[mask].square().sum().sqrt()
            )
        else:
            terms = (model.weight * model.weight.grad)[mask]
            expected = terms.abs().sum() if metric.mode == "elementwise_abs" else terms.sum().abs()
        assert score == pytest.approx(expected.item(), rel=1e-12)
        return ["joint"]

    pruner.plan(
        candidates=[candidate],
        metric=metric,
        strategy=strategy,
        budget=ChannelRatio(0.5, axes=(ref.axis(0),)),
        preserve_io=False,
    )


def test_metric_errors_and_nonadditive_custom_scoring():
    model = nn.Linear(4, 6)
    _graph, pruner = build(model, torch.randn(2, 4))
    common = {"budget": ChannelRatio(0.4), "preserve_io": False}
    with pytest.raises(PlanningError, match="gradients"):
        pruner.plan(metric=WeightTaylor(), **common)
    for metric in (lambda c, b: [float("nan")] * len(b), lambda c, b: [1]):
        with pytest.raises(PlanningError, match=r"nonfinite|length"):
            pruner.plan(metric=metric, **common)
    seen = []

    def metric(ctx, batch):
        seen.extend(len(c.remove) for c in batch)
        return [float(len(c.remove) ** 2) for c in batch]

    def strategy(ctx):
        a, b = ctx.candidates[:2]
        joint = Candidate("temporary", (*a.remove, *b.remove))
        assert ctx.score([joint]) == (4.0,)
        return [a.key, b.key]

    plan = pruner.plan(metric=metric, strategy=strategy, **common)
    assert seen == [2] and len(plan.selected) == 2
    with pytest.raises(PlanningError, match="unregistered"):
        pruner.plan(strategy=lambda c: ["bad"], **common)


def test_global_budget_no_hidden_local_cap():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return self.a(x), self.b(x)

    model = Branches()
    _graph, pruner = build(model, torch.randn(2, 4))

    def metric(ctx, batch):
        return [0 if c.key.startswith("a.") else 100 for c in batch]

    plan = pruner.plan(metric=metric, budget=ChannelRatio(0.25, scope="global"), preserve_io=False)
    assert plan.budget.widths == (6, 6)
    assert plan.budget.removed == (3, 0)


def test_plan_freshness_owner_and_value_changes():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    plan = pruner.plan(remove=[graph.parameter("weight").axis(0).select([1])], preserve_io=False)
    with torch.no_grad():
        model.weight.add_(1)
    Pruner(model).apply(plan)
    assert model.out_features == 5


def test_allocation_and_commit_failures_restore(monkeypatch):
    from torch_kirigami.pruning import pruner as implementation

    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    graph, pruner = build(model, torch.randn(2, 4))
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1])])
    original = tuple(model.parameters())
    gather = implementation.gather_region

    def fail(*args):
        raise RuntimeError("allocation failure")

    monkeypatch.setattr(implementation, "gather_region", fail)
    with pytest.raises(RuntimeError, match="allocation"):
        pruner.apply(plan)
    assert all(a is b for a, b in zip(original, model.parameters(), strict=True))
    monkeypatch.setattr(implementation, "gather_region", gather)
    setter = nn.Module.__setattr__

    def fail_commit(owner, name, value):
        if owner is model[1] and name == "weight" and value is not original[2]:
            raise RuntimeError("commit failure")
        setter(owner, name, value)

    monkeypatch.setattr(nn.Module, "__setattr__", fail_commit)
    with pytest.raises(ExecutionError, match="restored"):
        pruner.apply(plan)
    assert all(a is b for a, b in zip(original, model.parameters(), strict=True))
    monkeypatch.setattr(nn.Module, "__setattr__", setter)
    pruner.apply(plan)


def test_automatic_group_balance_selects_different_local_positions():
    model = nn.Conv1d(6, 6, 1, groups=2)
    graph, pruner = build(model, torch.randn(2, 6, 4))

    def metric(ctx, batch):
        ranking = {0: 0, 4: 1, 1: 2, 2: 3, 3: 4, 5: 5}
        return [ranking[int(c.key.rsplit(":", 1)[1])] for c in batch]

    plan = pruner.plan(metric=metric, budget=ChannelRatio(0.34), preserve_io=False)
    assert set(plan.analysis.selection(graph.parameter("weight")).project(0)) == {0, 4}
    assert plan.budget.removed == (2,)


def test_joint_rows_columns_greedy_grouped_chain(execution_device):
    model = nn.Sequential(nn.Conv1d(4, 6, 1), nn.Conv1d(6, 6, 1, groups=2), nn.Conv1d(6, 2, 1))
    x = torch.randn(2, 4, 5)
    _graph, pruner = build(model, x)

    def metric(ctx, batch):
        return [
            (0 if c.key.startswith("0.") else 10)
            + (0 if int(c.key.rsplit(":", 1)[1]) in (0, 4) else 2)
            for c in batch
        ]

    plan = pruner.plan(metric=metric, budget=ChannelRatio(0.34))
    assert plan.budget.removed == (2, 2)
    pruner.apply(plan)
    model(x).sum().backward()
    assert model[1].weight.shape == (4, 2, 1)


def test_plan_state_isolation_and_altered_copy_rejected():
    from dataclasses import replace

    model = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Dropout(), nn.Linear(6, 2))
    graph, pruner = build(model, torch.randn(2, 4))
    state = {name: t.clone() for name, t in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1])])
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert all(m.training for m in model.modules())
    for name, t in model.state_dict().items():
        torch.testing.assert_close(t, state[name])
    with pytest.raises(ExecutionError, match="cover"):
        pruner.apply(replace(plan, recipes=()))


def test_rebuild_explicit_second_round_and_all_old_plans_stale(execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    a = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1])])
    b = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([2])])
    pruner.apply(a)
    with pytest.raises(ExecutionError):
        pruner.apply(b)
    graph, pruner = build(model, x)
    pruner.apply(pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.2)))
    assert model[0].out_features == 6
    assert model(x).shape == (2, 2)


def test_matmul_cat_residual_and_fixed_loop_execution(execution_device):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 6))
            self.b = nn.Parameter(torch.randn(12, 2))
            self.layers = nn.ModuleList([nn.Linear(6, 6) for _ in range(2)])

        def forward(self, x):
            y = x @ self.a
            for layer in self.layers:
                y = y + layer(y)
            return torch.cat((y, y), dim=1) @ self.b

    model = Net()
    old = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    plan = pruner.plan(remove=[graph.parameter("a").axis(1).select([1, 4])])
    pruner.apply(plan)
    keep = [0, 2, 3, 5]
    y = x @ old.a[:, keep]
    for layer in old.layers:
        y = y + F.linear(y, layer.weight[keep][:, keep], layer.bias[keep])
    reference = torch.cat((y, y), dim=1) @ old.b[[0, 2, 3, 5, 6, 8, 9, 11]]
    torch.testing.assert_close(model(x), reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_magnitude_low_precision_accumulation_and_filter(dtype):
    model = nn.Linear(4, 6).to(dtype)
    with torch.no_grad():
        model.weight.fill_(200)
        model.bias.fill_(10)
    graph, pruner = build(model, torch.randn(2, 4, dtype=dtype))
    axis = graph.parameter("weight").axis(0)
    candidates = [Candidate("one", (axis.select([0]),))]

    def strategy(ctx):
        expected = (4 * 200**2 + 10**2) ** 0.5
        assert ctx.score(ctx.candidates)[0] == pytest.approx(expected)
        filtered = Magnitude(parameter_filter=lambda ref, param: ref.paths[0] == "weight")
        assert filtered(ctx, ctx.candidates) == [400.0]
        return ["one"]

    pruner.plan(
        candidates=candidates,
        strategy=strategy,
        metric=Magnitude(),
        budget=ChannelRatio(0.2, axes=(axis,)),
        preserve_io=False,
    )


def test_explicit_protected_axis_keeps_budget_baseline_and_seed_alias_dedup():
    model = nn.Linear(4, 6)
    graph, pruner = build(model, torch.randn(2, 4))
    axis = graph.parameter("weight").axis(0)
    plan = pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.5, axes=(axis, axis)))
    assert plan.budget.widths == (6,) and plan.budget.shortfall == 3
    assert not plan.recipes
    pruner.apply(plan)
    graph.validate(model)
