import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRelation,
    DependencyGraph,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Requirement,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    AttributeRecipe,
    ChannelRatio,
    Magnitude,
    PlanningError,
    Pruner,
    RewriteResult,
)


def build(model, x, rules=None):
    graph = DependencyGraph.build(model, args=(x,), operators=rules)
    return graph, Pruner(model, graph=graph)


@pytest.mark.parametrize("dynamic", [True, False])
def test_reshape_original_size_provenance(dynamic, execution_device):
    class Reshape(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return y.reshape(y.size(0), y.size(1) if dynamic else 6)

    model = Reshape()
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    request = [graph.parameter("fc.weight").axis(0).select([1, 3])]
    if not dynamic:
        with pytest.raises(PlanningError, match="original forward"):
            pruner.plan(remove=request, preserve_io=False)
    else:
        plan = pruner.plan(remove=request, preserve_io=False)
        pruner.apply(plan)
        assert model(x).shape == (2, 4)


def test_view_noncontiguous_parameter_is_rejected():
    class View(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(16.0).reshape(4, 4).t())

        def forward(self, x):
            return self.weight.t().view(-1) + x

    model = View()
    graph, pruner = build(model, torch.zeros(()))
    with pytest.raises(PlanningError, match=r"view.*original forward"):
        pruner.plan(remove=[graph.parameter("weight").axis(0).select([0])], preserve_io=False)


@pytest.mark.parametrize("wrong", [False, True])
def test_slice_same_shape_wrong_coordinates(wrong):
    class Slice(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return y[:, 1:5], y

    model = Slice()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    # Removing 0 leaves the old slice output width unchanged, yet shifts which
    # original columns [1:5] selects. Removing 5 leaves that slice unchanged.
    remove = [graph.parameter("fc.weight").axis(0).select([0 if wrong else 5])]
    if wrong:
        with pytest.raises(PlanningError, match="correspondence"):
            pruner.plan(remove=remove, preserve_io=False)
    else:
        plan = pruner.plan(remove=remove, preserve_io=False)
        pruner.apply(plan)
        torch.testing.assert_close(model(x)[0], old(x)[0])


@pytest.mark.parametrize("legal", [True, False])
def test_split_original_ports(legal):
    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            return self.fc(x).split(4, dim=1)

    model = Split()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    remove = [graph.parameter("fc.weight").axis(0).select([5 if legal else 1])]
    if legal:
        plan = pruner.plan(remove=remove, preserve_io=False)
        pruner.apply(plan)
        a, b = model(x)
        torch.testing.assert_close(a, old(x)[0])
        torch.testing.assert_close(b, old(x)[1][:, :1])
    else:
        with pytest.raises(PlanningError, match="split"):
            pruner.plan(remove=remove, preserve_io=False)


def test_squeeze_new_singleton_rejected():
    class Squeeze(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x).squeeze()

    model = Squeeze()
    graph, pruner = build(model, torch.randn(2, 4))
    with pytest.raises(PlanningError, match="compact shape"):
        pruner.plan(remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False)


@pytest.mark.parametrize("unsafe", [True, False])
def test_inplace_single_consumer_proof(unsafe):
    class Inplace(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.out = nn.Linear(6, 2)

        def forward(self, x):
            y = self.fc(x)
            if unsafe:
                z = y + 1
                return self.out(y.relu_() + z)
            return self.out(y.relu_())

    model = Inplace()
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    remove = [graph.parameter("fc.weight").axis(0).select([1])]
    if unsafe:
        with pytest.raises(PlanningError, match="in-place"):
            pruner.plan(remove=remove)
    else:
        plan = pruner.plan(remove=remove)
        pruner.apply(plan)
        assert model(x).shape == (2, 2)


@pytest.mark.parametrize("kind", ["batch", "layer", "group", "softmax"])
def test_normalization_compact_domain_reference(kind, execution_device):
    norm = {
        "batch": lambda: nn.BatchNorm1d(6),
        "layer": lambda: nn.LayerNorm(6),
        "group": lambda: nn.GroupNorm(2, 6),
        "softmax": lambda: nn.Softmax(dim=1),
    }[kind]()
    model = nn.Sequential(nn.Linear(4, 6), norm, nn.Linear(6, 2)).eval()
    x = torch.randn(3, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = pruner.plan(remove=[graph.parameter("0.weight").axis(0).select([1, 4])])
    pruner.apply(plan)
    keep = [0, 2, 3, 5]
    h = F.linear(x, old[0].weight[keep], old[0].bias[keep])
    if kind == "batch":
        h = F.batch_norm(
            h,
            old[1].running_mean[keep],
            old[1].running_var[keep],
            old[1].weight[keep],
            old[1].bias[keep],
            training=False,
            eps=old[1].eps,
        )
    elif kind == "layer":
        h = F.layer_norm(h, (4,), old[1].weight[keep], old[1].bias[keep], old[1].eps)
        assert model[1].normalized_shape == (4,)
    elif kind == "group":
        h = F.group_norm(h, 2, old[1].weight[keep], old[1].bias[keep], old[1].eps)
        assert model[1].num_channels == 4
    else:
        h = h.softmax(dim=1)
    reference = F.linear(h, old[2].weight[:, keep], old[2].bias)
    torch.testing.assert_close(model(x), reference)


def test_unknown_branch_underfill_keeps_denominator():
    class Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(4, 6)

        def forward(self, x):
            return torch.special.gammaln(self.a(x)), self.b(x)

    model = Unknown()
    _graph, pruner = build(model, torch.randn(2, 4))
    plan = pruner.plan(metric=Magnitude(), budget=ChannelRatio(0.5), preserve_io=False)
    assert plan.budget.widths == (6, 6)
    assert plan.budget.removed == (0, 3)
    assert plan.budget.shortfall == 3
    assert any("Incomplete" in reason for _, reason in plan.budget.exclusions)


class Fused(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 4))
        self.width = 6

    def forward(self, x):
        return F.linear(x, self.weight).reshape(x.size(0), self.width)


def fused_analysis(ctx):
    x, y, w = ctx.inputs[0], ctx.outputs[0], ctx.parameter("weight")
    return OperatorSpec(
        (
            AxisRelation.equal(x.axis(1), w.axis(1), "input"),
            AxisRelation.equal(y.axis(1), w.axis(0), "output"),
        ),
        requirements=(
            Requirement(
                "attribute",
                f"{ctx.module_path}.width".lstrip("."),
                (y,),
                "fused width",
                (("axis", y.axis(1)),),
            ),
        ),
    )


def fused_rewrite(ctx):
    op = ctx.operation
    new = ctx.shape(op.outputs[0])[1]
    path = f"{op.module_path}.width".lstrip(".")
    return RewriteResult(
        attributes=(AttributeRecipe(path, op.module.width, new),), handled=ctx.requirements
    )


@pytest.mark.parametrize("nested", [True, False])
def test_custom_root_nested_extension(nested, execution_device):
    model = nn.Sequential(Fused(), nn.Linear(6, 2)) if nested else Fused()
    x = torch.randn(2, 4)
    rules = OperatorRegistry.default().register(
        Fused, OperatorRule(fused_analysis, lower=fused_rewrite)
    )
    graph, pruner = build(model, x, rules)
    path = "0.weight" if nested else "weight"
    plan = pruner.plan(remove=[graph.parameter(path).axis(0).select([1, 3])], preserve_io=nested)
    pruner.apply(plan)
    assert model(x).shape == (2, 2 if nested else 4)
    with pytest.raises(ValueError, match="already"):
        rules.register(Fused, OperatorRule(fused_analysis))


def fused_function(x):
    return x.relu()


def test_custom_function_rule_does_not_require_core_changes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            return fused_function(self.fc(x))

    rules = OperatorRegistry.default().register(
        fused_function,
        OperatorRule(
            lambda c: OperatorSpec(
                (AxisRelation.equal(c.inputs[0].axis(1), c.outputs[0].axis(1), "channels"),)
            )
        ),
    )
    model = Net()
    graph, pruner = build(model, torch.randn(2, 4), rules)
    assert any(c.node.target is fused_function for c in graph.operations())
    pruner.apply(
        pruner.plan(remove=[graph.parameter("fc.weight").axis(0).select([0])], preserve_io=False)
    )
    assert model.fc.out_features == 5


def test_public_graph_queries_are_isolated_and_alias_aware():
    model = nn.Linear(4, 6)
    graph, _ = build(model, torch.randn(2, 4))
    ref = graph.parameter("weight")
    assert graph.model is model and graph.tensor(ref) is model.weight
    assert graph.bindings(ref) == ((model, "weight"),)
    operations = graph.operations()
    operations[0].kwargs["bad"] = 1
    operations[0].node.target = "bad"
    assert not graph.operations()[0].kwargs
    assert graph.operations()[0].node.target != "bad"
    with pytest.raises(ValueError, match="different"):
        graph.validate(nn.Linear(4, 6))
    graph.invalidate()
    with pytest.raises(StaleGraphError):
        graph.tensor(ref)


@pytest.mark.parametrize("dynamic", [True, False])
def test_functional_layernorm_argument_proof(dynamic, execution_device):
    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.scale = nn.Parameter(torch.randn(6))

        def forward(self, x):
            y = self.fc(x)
            return F.layer_norm(y, (y.size(1) if dynamic else 6,), self.scale)

    model = Functional()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    request = [graph.parameter("fc.weight").axis(0).select([1, 4])]
    if dynamic:
        plan = pruner.plan(remove=request, preserve_io=False)
        pruner.apply(plan)
        keep = [0, 2, 3, 5]
        y = F.linear(x, old.fc.weight[keep], old.fc.bias[keep])
        torch.testing.assert_close(model(x), F.layer_norm(y, (4,), old.scale[keep]))
    else:
        with pytest.raises(PlanningError, match="functional argument"):
            pruner.plan(remove=request, preserve_io=False)


@pytest.mark.parametrize("remove", [[0], [5]])
def test_dimension_based_integer_index_tracks_original_coordinate(remove):
    class Index(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return y[:, y.size(1) - 1], y

    model = Index()
    graph, pruner = build(model, torch.randn(2, 4))
    # Removing 0 preserves the old last column after evaluating size()-1 anew.
    # Removing the captured last output is blocked by the scalar/empty constraint.
    if remove == [0]:
        plan = pruner.plan(
            remove=[graph.parameter("fc.weight").axis(0).select(remove)], preserve_io=False
        )
        pruner.apply(plan)
        assert model.fc.out_features == 5
    else:
        with pytest.raises(PlanningError):
            pruner.plan(
                remove=[graph.parameter("fc.weight").axis(0).select(remove)], preserve_io=False
            )


def test_dynamic_split_sizes_need_no_forward_edit():
    class Split(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)

        def forward(self, x):
            y = self.fc(x)
            return y.split(y.size(1) // 2, dim=1)

    model = Split()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = pruner.plan(
        remove=[graph.parameter("fc.weight").axis(0).select([1, 6])], preserve_io=False
    )
    pruner.apply(plan)
    outputs = model(x)
    reference = old(x)
    torch.testing.assert_close(outputs[0], reference[0][:, [0, 2, 3]])
    torch.testing.assert_close(outputs[1], reference[1][:, [0, 1, 3]])


def test_unflatten_bound_tuple_and_unbind_ports():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.unflatten = nn.Unflatten(1, (2, 3))

        def forward(self, x):
            return self.unflatten(self.fc(x)).unbind(1)

    model = Net()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = pruner.plan(
        remove=[graph.parameter("fc.weight").axis(0).select([1, 4])], preserve_io=False
    )
    pruner.apply(plan)
    assert model.unflatten.unflattened_size == (2, 2)
    for actual, reference in zip(model(x), old(x), strict=True):
        torch.testing.assert_close(actual, reference[:, [0, 2]])


def test_coordinate_mapping_compares_order_not_shape():
    from torch_kirigami import IndexSet, Region, TensorRef
    from torch_kirigami.pruning import TensorRecipe
    from torch_kirigami.pruning.rewrite import _same_mapping

    ref = TensorRef("test", (4, 2), "parameter", ("weight",))
    a = Region((IndexSet.span(0, 2), IndexSet.span(0, 2)))
    b = Region((IndexSet.span(2, 4), IndexSet.span(0, 2)))
    first = TensorRecipe(ref, (a, b))
    second = TensorRecipe(ref, (b, a))
    whole = TensorRecipe(ref, (Region((IndexSet.span(0, 4), IndexSet.span(0, 2))),))
    assert first.shape == second.shape
    assert not _same_mapping(first, second)
    assert _same_mapping(first, whole)


def test_unregistered_tensor_constant_cannot_be_silently_compacted():
    class Constant(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.fixed_weight = torch.randn(2, 6)

        def forward(self, x):
            return F.linear(self.fc(x), self.fixed_weight)

    model = Constant()
    graph, pruner = build(model, torch.randn(2, 4))
    assert graph.constants()
    with pytest.raises(PlanningError, match="Unregistered captured constant"):
        pruner.plan(remove=[graph.parameter("fc.weight").axis(0).select([1])])


def test_forward_hook_needs_explicit_execution_semantics():
    model = nn.Linear(4, 6)
    model.register_forward_hook(lambda module, args, output: output.flip(-1))
    graph, pruner = build(model, torch.randn(2, 4))
    with pytest.raises(PlanningError, match="forward-hook"):
        pruner.plan(remove=[graph.parameter("weight").axis(0).select([1])], preserve_io=False)
