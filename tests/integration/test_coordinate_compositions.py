"""integration / coordinate compositions contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.graph_helpers import removed
from tests.support.numerics import _assert_value_and_input_gradient, _initialize, _sample
from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    Pruner,
)

_SWAP_FORMS = (
    "transpose_function_positional",
    "transpose_function_keyword",
    "transpose_method_keyword",
    "swapaxes_function_keyword",
    "swapdims_method_positional",
    "permute_function_keyword",
    "permute_method_variadic",
)


def _swap_features(y, form, negative):
    first, second = (-2, -1) if negative else (1, 2)
    if form == "transpose_function_positional":
        return torch.transpose(y, first, second)
    if form == "transpose_function_keyword":
        return torch.transpose(dim1=second, input=y, dim0=first)
    if form == "transpose_method_keyword":
        return y.transpose(dim1=second, dim0=first)
    if form == "swapaxes_function_keyword":
        return torch.swapaxes(axis1=second, input=y, axis0=first)
    if form == "swapdims_method_positional":
        return y.swapdims(first, second)
    dims = (0, second, first)
    if form == "permute_function_keyword":
        return torch.permute(dims=dims, input=y)
    return y.permute(*dims)


_PARTITION_FORMS = (
    "split_function_positional",
    "split_function_keyword",
    "split_method_alias",
    "chunk_function_keyword",
    "chunk_method_positional",
)


def _partition(y, form, dim):
    if form == "split_function_positional":
        return torch.split(y, y.size(dim) // 2, dim)
    if form == "split_function_keyword":
        return torch.split(dim=dim, split_size_or_sections=y.size(dim) // 2, tensor=y)
    if form == "split_method_alias":
        return y.split(split_size=y.size(dim) // 2, dim=dim)
    if form == "chunk_function_keyword":
        return torch.chunk(chunks=2, dim=dim, input=y)
    return y.chunk(2, dim)


@pytest.mark.parametrize("form", _SWAP_FORMS)
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_reshape_permutation_composition_preserves_original_coordinates(
    form, negative, noncontiguous, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.producer = nn.Linear(3, 6)
            self.consumer = nn.Linear(6, 2)

        def forward(self, x):
            y = self.producer(x)
            y = y.reshape(y.size(0), 2, -1)
            return self.consumer(_swap_features(y, form, negative).flatten(1))

    model = _initialize(Model().double())
    original = copy.deepcopy(model)
    sample = _sample(noncontiguous)
    reference_input = sample.detach().clone().requires_grad_()
    # Validate the native invocation first, so invalid API spellings are not library bugs.
    model(sample)
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("producer.weight").axis(0).select([1, 4])])
    # Original transpose+flatten order is [0, 3, 1, 4, 2, 5].
    consumer_columns = [0, 1, 4, 5]
    assert tuple(
        plan.analysis.selection(graph.parameter("consumer.weight")).fully_selected_indices(1)
    ) == (2, 3)
    pruner.apply(plan)

    hidden = F.linear(
        reference_input,
        original.producer.weight[[0, 2, 3, 5]],
        original.producer.bias[[0, 2, 3, 5]],
    )
    reordered = torch.stack((hidden[:, 0], hidden[:, 2], hidden[:, 1], hidden[:, 3]), dim=1)
    expected = F.linear(
        reordered, original.consumer.weight[:, consumer_columns], original.consumer.bias
    )
    _assert_value_and_input_gradient(model(sample), expected, sample, reference_input)


@pytest.mark.parametrize("form", _PARTITION_FORMS)
@pytest.mark.parametrize("remove", [(0, 3), (1, 5)])
@pytest.mark.parametrize("negative", [False, True])
def test_partition_reordering_joint_pruning_matches_retained_coordinate_reference(
    form, remove, negative, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.producer = nn.Linear(3, 6)
            self.consumer = nn.Linear(6, 2)

        def forward(self, x):
            first, second = _partition(self.producer(x), form, -1 if negative else 1)
            # Both branches use a channel-specific permutation and a distinct scale.
            return self.consumer(torch.cat(tensors=(second * 3, first * 2), dim=-1))

    model = _initialize(Model().double())
    original = copy.deepcopy(model)
    sample = _sample(True)
    reference_input = sample.detach().clone().requires_grad_()
    # Validate the native invocation first, so invalid API spellings are not library bugs.
    model(sample)
    graph = DependencyGraph.build(model, args=(sample,))
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("producer.weight").axis(0).select(remove)]
    )
    original_order = [3, 4, 5, 0, 1, 2]
    keep_columns = [
        column for column, channel in enumerate(original_order) if channel not in remove
    ]
    assert tuple(
        plan.analysis.selection(graph.parameter("consumer.weight")).fully_selected_indices(1)
    ) == tuple(column for column in range(6) if column not in keep_columns)
    Pruner(model).apply(plan)

    hidden = F.linear(reference_input, original.producer.weight, original.producer.bias)
    retained = torch.stack(
        [hidden[:, original_order[column]] * (3 if column < 3 else 2) for column in keep_columns],
        dim=1,
    )
    expected = F.linear(retained, original.consumer.weight[:, keep_columns], original.consumer.bias)
    _assert_value_and_input_gradient(model(sample), expected, sample, reference_input)


def test_cat_split_slice_permute_chain(execution_device):
    class Model(nn.Module):
        def forward(self, x, y):
            z = torch.cat([x, y], dim=1)
            a, b = torch.split(z, [4, 6], dim=1)
            return a.transpose(0, 1), b[:, 1:5:2]

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 4), torch.randn(2, 6)))
    inputs = [v for v in graph.values() if v.kind == "input"]
    impact = graph.propagate(remove=[inputs[1].axis(1).select([1])])
    assert impact.status == "resolved"
    sliced = next(c for c in graph.calls() if c.name == "getitem_2")
    assert removed(impact, sliced.output(), 1) == {0}
    assert not impact.selection(inputs[0])
    assert any(r.kind == "slice_arguments" for r in impact.requirements)


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
    plan = pruner.plan_remove([graph.parameter("a").axis(1).select([1, 4])])
    pruner.apply(plan)
    keep = [0, 2, 3, 5]
    y = x @ old.a[:, keep]
    for layer in old.layers:
        y = y + F.linear(y, layer.weight[keep][:, keep], layer.bias[keep])
    reference = torch.cat((y, y), dim=1) @ old.b[[0, 2, 3, 5, 6, 8, 9, 11]]
    torch.testing.assert_close(model(x), reference)
