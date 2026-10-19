"""operators / indexing contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.models import Chain
from tests.support.numerics import _assert_value_and_input_gradient, _initialize, _sample
from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    ExecutionError,
    PlanningError,
    Pruner,
    PruningPlan,
)


@pytest.mark.parametrize("method", [False, True])
def test_scalar_index_select_keeps_value_and_guards_index(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.register_buffer("index", torch.tensor([0]))

        def forward(self, x):
            value = self.fc(x).sum()
            return (
                value.index_select(0, self.index)
                if method
                else torch.index_select(value, -1, self.index)
            )

    model = Model()
    x = torch.randn(2, 4)
    expected = F.linear(x, model.fc.weight[[0, 2, 3, 4, 5]], model.fc.bias[[0, 2, 3, 4, 5]]).sum()
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("fc.weight").axis(0).select([1])])
    )
    torch.testing.assert_close(model(x), expected)


def test_static_index_values_and_coordinate_guards():
    from torch_kirigami import StaleGraphError

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.register_buffer("indices", torch.tensor([0, 2]))

        def forward(self, x):
            return self.fc(x).index_select(1, self.indices)

    model = Net()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    p = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="coordinates"):
        p.plan_remove([graph.parameter("fc.weight").axis(0).select([1])])
    plan = p.plan_remove([graph.parameter("fc.weight").axis(0).select([4, 5])])
    p.apply(plan)
    torch.testing.assert_close(model(x), old(x))
    graph = DependencyGraph.build(model, args=(x,))
    model.indices[0] = 1
    with pytest.raises(StaleGraphError, match="constant"):
        graph.propagate(remove=[])


@pytest.mark.parametrize("registered", [False, True])
def test_structural_indices_require_portable_value_guards(registered):
    class Model(nn.Module):
        def __init__(self, index):
            super().__init__()
            self.a, self.b = nn.Linear(4, 6), nn.Linear(2, 3)
            if registered:
                self.register_buffer("index", torch.tensor(index))
            else:
                self.index = torch.tensor(index)

        def forward(self, x):
            return self.b(self.a(x).index_select(1, self.index))

    model = Model([0, 1])
    graph = DependencyGraph.build(model, args=(torch.ones(1, 4),))
    pruner = Pruner(model, graph=graph)
    remove = [graph.parameter("a.weight").axis(0).select([5])]
    if not registered:
        with pytest.raises(PlanningError, match=r"constant|buffer"):
            pruner.plan_remove(remove)
        return
    plan = PruningPlan.from_dict(pruner.plan_remove(remove).to_dict())
    with pytest.raises(ExecutionError, match="preconditions"):
        Pruner(Model([4, 5])).apply(plan)
    model.index.copy_(torch.tensor([4, 5]))
    with pytest.raises(StaleGraphError):
        graph.validate()


@pytest.mark.parametrize("alias", [False, True])
def test_index_select_alias_with_registered_constant(alias, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.register_buffer("index", torch.tensor([0, 2]))

        def forward(self, x):
            y = self.fc(x)
            return (
                y.index_select(axis=1, index=self.index)
                if alias
                else y.index_select(dim=1, index=self.index)
            )

    model = Model()
    x = torch.randn(2, 4)
    expected = model(x)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("fc.weight").axis(0).select([5])])
    )
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize(
    "form",
    [
        "slice_positive",
        "slice_negative",
        "narrow_positive",
        "narrow_negative",
        "index_select_function",
        "index_select_method",
    ],
)
@pytest.mark.parametrize("removed_channel", [0, 5])
def test_static_indexing_accepts_only_original_coordinate_preserving_compaction(
    form, removed_channel, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.producer = nn.Linear(3, 6)
            self.consumer = nn.Linear(3, 2)
            self.register_buffer("indices", torch.tensor([2, 3, 4]))

        def forward(self, x):
            y = self.producer(x)
            if form == "slice_positive":
                selected = y[:, 2:5]
            elif form == "slice_negative":
                selected = y[:, -4:-1]
            elif form == "narrow_positive":
                selected = torch.narrow(length=3, start=2, input=y, dim=-1)
            elif form == "narrow_negative":
                selected = y.narrow(length=3, start=-4, dim=-1)
            elif form == "index_select_function":
                selected = torch.index_select(index=self.indices, dim=-1, input=y)
            else:
                selected = y.index_select(index=self.indices, dim=-1)
            return self.consumer(selected)

    model = _initialize(Model().double())
    original = copy.deepcopy(model)
    sample = _sample(True)
    reference_input = sample.detach().clone().requires_grad_()
    hidden = F.linear(reference_input, original.producer.weight, original.producer.bias)
    expected = F.linear(hidden[:, [2, 3, 4]], original.consumer.weight, original.consumer.bias)
    torch.testing.assert_close(model(sample), expected)
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph)
    remove = [graph.parameter("producer.weight").axis(0).select([removed_channel])]
    safe = removed_channel == (0 if form.endswith("negative") else 5)
    if safe:
        pruner.apply(pruner.plan_remove(remove))
    else:
        # In both unsafe cases the compact forward still has every expected shape.
        # Only checking which original coordinates it uses can reveal the error.
        wrong_columns = [3, 4, 5] if removed_channel == 0 else [1, 2, 3]
        wrong = F.linear(hidden[:, wrong_columns], original.consumer.weight, original.consumer.bias)
        assert wrong.shape == expected.shape and not torch.allclose(wrong, expected)
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        with pytest.raises(PlanningError, match="coordinate"):
            pruner.plan_remove(remove)
        assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
        assert model.producer.out_features == 6
    _assert_value_and_input_gradient(model(sample), expected, sample, reference_input)


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
            Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(remove)
    else:
        plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(remove)
        pruner.apply(plan)
        torch.testing.assert_close(model(x)[0], old(x)[0])


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
        plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(
            [graph.parameter("fc.weight").axis(0).select(remove)]
        )
        pruner.apply(plan)
        assert model.fc.out_features == 5
    else:
        with pytest.raises(PlanningError):
            Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(
                [graph.parameter("fc.weight").axis(0).select(remove)]
            )


def test_coordinate_mapping_compares_order_not_shape():
    from torch_kirigami import IndexSet, Region, TensorRef
    from torch_kirigami.pruning import TensorRecipe
    from torch_kirigami.pruning.recipes import same_mapping

    ref = TensorRef("test", (4, 2), "parameter", ("weight",))
    a = Region((IndexSet.span(0, 2), IndexSet.span(0, 2)))
    b = Region((IndexSet.span(2, 4), IndexSet.span(0, 2)))
    first = TensorRecipe(ref, (a, b))
    second = TensorRecipe(ref, (b, a))
    whole = TensorRecipe(ref, (Region((IndexSet.span(0, 4), IndexSet.span(0, 2))),))
    assert first.shape == second.shape
    assert not same_mapping(first, second)
    assert same_mapping(first, whole)


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
        pruner.plan_remove([graph.parameter("fc.weight").axis(0).select([1])])


@pytest.mark.parametrize("method", [False, True])
def test_narrow_keywords_match_positional_coordinates(method, execution_device):
    class Model(Chain):
        def forward(self, x):
            y = self.a(x)
            y = (
                y.narrow(dim=0, start=0, length=1)
                if method
                else torch.narrow(y, dim=0, start=0, length=1)
            )
            return self.b(y)

    model = Model()
    x = torch.randn(2, 4)
    kept = [0, 2, 3]
    expected = F.linear(
        F.linear(x[:1], model.a.weight[kept], model.a.bias[kept]),
        model.b.weight[:, kept],
        model.b.bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    )
    torch.testing.assert_close(model(x), expected)
