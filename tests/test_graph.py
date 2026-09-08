import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRelation,
    Balanced,
    CaptureError,
    DependencyGraph,
    Fixed,
    IndexSet,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    ReshapeRelation,
    StaleGraphError,
)


def indices(impact, tensor, dim):
    return set(impact.selection(tensor).project(dim))


def test_linear_chain_and_joint_requests(execution_device):
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    a = graph.parameter("0.weight").axis(0).select([1])
    b = graph.parameter("2.weight").axis(1).select([4])
    impact = graph.propagate(remove=[a, b])
    assert impact.status == "resolved"
    assert indices(impact, graph.parameter("0.bias"), 0) == {1, 4}
    assert indices(impact, graph.parameter("2.weight"), 1) == {1, 4}
    assert impact.selections == graph.propagate(remove=[b, a, a]).selections
    assert impact.selections == graph.propagate(remove=list(impact.selections.values())).selections
    assert "via" in graph.explain(impact)
    protected = graph.propagate(remove=[a], constraints=[Fixed(graph.parameter("0.bias").axis(0))])
    assert protected.status == "conflict"


def test_shared_module_paths_and_calls(execution_device):
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(4, 4)
            self.right = self.left

        def forward(self, x):
            return self.left(x) + self.right(x)

    graph = DependencyGraph.build(Shared(), args=(torch.randn(2, 4),))
    assert graph.parameter("left.weight") is graph.parameter("right.weight")
    assert len(graph.calls("left")) == len(graph.calls("right")) == 2
    result = graph.propagate(remove=[graph.parameter("left.weight").axis(0).select([2])])
    assert result.status == "resolved"
    assert all(indices(result, c.output(), 1) == {2} for c in graph.calls("left"))


def test_residual_and_shared_weight_cycle_reaches_fixed_point(execution_device):
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4, bias=False)
            self.b = nn.Linear(4, 4, bias=False)
            self.b.weight = self.a.weight

        def forward(self, x):
            return self.a(x) + self.b(x)

    graph = DependencyGraph.build(Residual(), args=(torch.randn(2, 4),))
    impact = graph.propagate(remove=[graph.calls("a")[0].output().axis(1).select([1, 3])])
    assert impact.status == "resolved"
    assert indices(impact, graph.calls("b")[0].output(), 1) == {1, 3}


@pytest.mark.parametrize(
    "conv,shape",
    [
        (nn.Conv1d, (2, 6, 5)),
        (nn.Conv2d, (2, 6, 4, 5)),
        (nn.Conv3d, (2, 6, 3, 4, 5)),
    ],
)
def test_partitioned_group_convolution_and_numerical_oracle(conv, shape, execution_device):
    torch.manual_seed(1)
    model = conv(6, 4, 1, groups=2)
    x = torch.randn(shape)
    graph = DependencyGraph.build(model, args=(x,))
    impact = graph.propagate(remove=[graph.calls("")[0].input().axis(1).select([0, 4])])
    assert impact.status == "resolved"
    regions = impact.selection(graph.parameter("weight")).regions
    assert len(regions) == 2
    assert tuple(set(r.axes[0]) for r in regions) == ({0, 1}, {2, 3})
    assert tuple(set(r.axes[1]) for r in regions) == ({0}, {1})
    compact = conv(4, 4, 1, groups=2)
    with torch.no_grad():
        compact.weight.copy_(torch.cat((model.weight[:2, [1, 2]], model.weight[2:, [0, 2]])))
        compact.bias.copy_(model.bias)
    reference_input = x.clone()
    reference_input[:, [0, 4]] = 0
    torch.testing.assert_close(compact(x[:, [1, 2, 3, 5]]), model(reference_input))
    assert any(r.kind == "partitioned_compaction" for r in impact.requirements)


def test_unequal_group_choices_are_not_silently_completed():
    graph = DependencyGraph.build(nn.Conv2d(6, 4, 1, groups=2), args=(torch.randn(2, 6, 3, 3),))
    request = graph.calls("")[0].input().axis(1).select([0])
    impact = graph.propagate(remove=[request])
    assert impact.status == "unresolved"
    assert indices(impact, request.tensor, 1) == {0}
    assert any(d.code == "unbalanced_groups" for d in impact.diagnostics)


def test_multiple_partition_constraints():
    graph = DependencyGraph.build(nn.Linear(12, 12), args=(torch.randn(2, 12),))
    axis = graph.parameter("weight").axis(0)
    constraints = [
        Balanced(axis, tuple(IndexSet.span(i, i + size) for i in range(0, 12, size)))
        for size in (6, 4)
    ]
    impact = graph.propagate(remove=[axis.select([0, 4, 8])], constraints=constraints)
    assert impact.status == "unresolved"
    assert any(d.code == "unbalanced_groups" for d in impact.diagnostics)


def test_depthwise_input_removal_forces_whole_output_blocks(execution_device):
    graph = DependencyGraph.build(nn.Conv2d(4, 8, 1, groups=4), args=(torch.randn(2, 4, 3, 3),))
    impact = graph.propagate(remove=[graph.calls("")[0].input().axis(1).select([1])])
    assert impact.status == "resolved"
    assert indices(impact, graph.parameter("weight"), 0) == {2, 3}
    assert indices(impact, graph.calls("")[0].input(), 1) == {1}
    assert any(r.target == "groups" for r in impact.requirements)


def test_unknown_branch_only_blocks_related_queries():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 4), nn.Linear(4, 4)

        def forward(self, x):
            return torch.special.gammaln(self.a(x)), self.b(x)

    graph = DependencyGraph.build(Branch(), args=(torch.randn(2, 4),))
    assert graph.diagnostics
    assert (
        graph.propagate(remove=[graph.parameter("a.weight").axis(0).select([1])]).status
        == "unresolved"
    )
    assert (
        graph.propagate(remove=[graph.parameter("b.weight").axis(0).select([1])]).status
        == "resolved"
    )


def test_distinct_parameters_sharing_storage_are_not_merged():
    class SharedStorage(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 4))
            self.b = nn.Parameter(self.a.detach())

        def forward(self, x):
            return F.linear(x, self.a) + F.linear(x, self.b)

    graph = DependencyGraph.build(SharedStorage(), args=(torch.randn(2, 4),))
    assert graph.parameter("a") is not graph.parameter("b")
    impact = graph.propagate(remove=[graph.parameter("a").axis(0).select([1])])
    assert any(d.code == "storage_alias" for d in impact.diagnostics)


@pytest.mark.parametrize("kind", ["data", "shape", "loop"])
def test_dynamic_python_control_flow_fails(kind):
    class Dynamic(nn.Module):
        def forward(self, x):
            if kind == "data":
                return x if x.sum() > 0 else -x
            if kind == "shape":
                return x if x.shape[0] > 2 else -x
            while x.sum() > 0:
                x = x - 1
            return x

    with pytest.raises(CaptureError):
        DependencyGraph.build(Dynamic(), args=(torch.ones(3, 4),))


def test_static_loop_mode_and_freshness():
    class Static(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 4)
            self.count = 3
            self.enabled = True

        def forward(self, x):
            if self.enabled:
                for _ in range(self.count):
                    x = self.layer(x)
            return x

    model = Static()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    assert len(graph.calls("layer")) == 3
    with torch.no_grad():
        model.layer.weight.add_(1)
    graph.propagate(remove=[])
    model.count = 2
    with pytest.raises(StaleGraphError):
        graph.propagate(remove=[])


def test_training_mode_and_parameter_replacement_invalidate():
    for change in ("mode", "parameter"):
        model = nn.Linear(4, 4)
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
        if change == "mode":
            model.eval()
        else:
            model.weight = nn.Parameter(model.weight.detach().clone())
        with pytest.raises(StaleGraphError):
            graph.propagate(remove=[])


def test_foreign_graph_selection_rejected():
    a = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    b = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    with pytest.raises(ValueError):
        b.propagate(remove=[a.parameter("weight").axis(0).select([1])])


def test_frozen_parameters_and_no_grad():
    model = nn.Linear(4, 4).requires_grad_(False)
    with torch.no_grad():
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    assert (
        graph.propagate(remove=[graph.parameter("weight").axis(0).select([1])]).status == "resolved"
    )


class Opaque(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(self, x, *, scale=1.0):
        if x.sum() > 0:
            return F.linear(x, self.weight) * scale
        return -F.linear(x, self.weight) * scale


def opaque_rule(ctx):
    x, y, w = ctx.inputs[0], ctx.outputs[0], ctx.parameter("weight")
    return OperatorSpec(
        (
            AxisRelation.equal(x.axis(1), w.axis(1), "opaque input"),
            AxisRelation.equal(y.axis(1), w.axis(0), "opaque output"),
            AxisRelation.equal(x.axis(0), y.axis(0), "opaque batch"),
        )
    )


def test_external_root_and_nested_module_use_one_rule_interface():
    rules = OperatorRegistry.default().register(Opaque, OperatorRule(opaque_rule))
    for model, path in ((Opaque(), "weight"), (nn.Sequential(Opaque()), "0.weight")):
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),), operators=rules)
        assert (
            graph.propagate(remove=[graph.parameter(path).axis(0).select([1])]).status == "resolved"
        )
    graph = DependencyGraph.build(
        Opaque(), args=(torch.ones(2, 4),), kwargs={"scale": 2.0}, operators=rules
    )
    assert graph.calls("")[0].output().shape == (2, 4)
    with pytest.raises(ValueError):
        rules.register(Opaque, OperatorRule(opaque_rule))


def opaque_function(x):
    return x * 2 if x.sum() > 0 else x * 3


def test_external_function_capture_is_local():
    class Model(nn.Module):
        def forward(self, x):
            return opaque_function(x)

    rules = OperatorRegistry.default().register(
        opaque_function,
        OperatorRule(lambda c: OperatorSpec((ReshapeRelation(c.inputs[0], c.outputs[0]),))),
    )
    graph = DependencyGraph.build(Model(), args=(torch.ones(2, 4),), operators=rules)
    assert (
        graph.propagate(remove=[graph.calls()[0].input().axis(1).select([1])]).status == "resolved"
    )
    with pytest.raises(CaptureError):
        DependencyGraph.build(Model(), args=(torch.ones(2, 4),))


def test_overridden_linear_subclass_does_not_inherit_rule():
    class Different(nn.Linear):
        def forward(self, x):
            return torch.special.gammaln(F.linear(x, self.weight, self.bias))

    graph = DependencyGraph.build(nn.Sequential(Different(4, 4)), args=(torch.randn(2, 4),))
    assert (
        graph.propagate(remove=[graph.parameter("0.weight").axis(0).select([1])]).status
        == "unresolved"
    )
