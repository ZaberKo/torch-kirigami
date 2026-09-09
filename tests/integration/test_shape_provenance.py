"""integration / shape provenance contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    DependencyGraph,
    OperatorRegistry,
)
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)


def controlled_conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # FX 2.6 cannot pass scalar Proxies through conv1d's native argument parser.
    # Its public leaf-function hook exercises the same builtin semantics without
    # requiring another tracer or pretending the native spelling is supported.
    return F.conv1d(input, weight, bias, stride, padding, dilation, groups)


def conv_registry():
    registry = OperatorRegistry.default()
    registry.register(controlled_conv, registry.functions[F.conv1d])
    return registry


@pytest.mark.parametrize("form", ["size", "keyword", "shape", "negative", "arithmetic"])
@pytest.mark.parametrize("constant_selector", [False, True])
def test_nested_dimension_selector_is_guarded(form, constant_selector, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv1d(3, 2, 1)

        def forward(self, x, z, table):
            c = self.fc(z).size(1)
            if constant_selector:
                c = c // 3 + 1
            if form == "size":
                index = x.size(c)
            elif form == "keyword":
                index = x.size(dim=c)
            elif form == "shape":
                index = x.shape[c]
            elif form == "negative":
                index = x.size(c - 3)
            else:
                index = x.size(c) + 1
            return table[index]

    model = Model()
    args = (torch.ones(1, 3, 4), torch.ones(1, 3, 1), torch.arange(8.0))
    expected = model(*args)
    graph = DependencyGraph.build(model, args=args)
    pruner = Pruner(model, graph=graph)
    remove = [graph.parameter("fc.weight").axis(0).select([0])]
    if constant_selector:
        pruner.prune(remove=remove)
        torch.testing.assert_close(model(*args), expected)
    else:
        with pytest.raises(PlanningError, match="argument"):
            pruner.plan(remove=remove)
        torch.testing.assert_close(model(*args), expected)


@pytest.mark.parametrize("operation", ["getitem", "narrow", "unbind", "conv"])
def test_size_only_consumer_cannot_change_coordinates(operation, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv1d(3, 2, 1)
            self.weight = nn.Parameter(torch.tensor([[[1.0, 2.0, 3.0]]]))

        def forward(self, x, z):
            c = self.fc(z).size(1)
            if operation == "getitem":
                return x[c]
            if operation == "narrow":
                return x.narrow(0, c, 1)
            if operation == "unbind":
                return x.unbind(c)
            return controlled_conv(x, self.weight, stride=c, padding=c)

    x = torch.arange(8.0).reshape(2, 2, 2) if operation == "unbind" else torch.arange(4.0)
    if operation == "conv":
        x = torch.arange(3.0).reshape(1, 1, 3)
    model = Model()
    graph = DependencyGraph.build(model, args=(x, torch.ones(1, 3, 1)), operators=conv_registry())
    with pytest.raises(PlanningError, match=r"argument|coordinate"):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("fc.weight").axis(0).select([0])])


@pytest.mark.parametrize("shared_expression", [False, True])
def test_allowed_groups_does_not_allow_stride_or_padding(shared_expression, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv1d(3, 2, 1)
            self.w = nn.Parameter(torch.arange(12.0).reshape(4, 1, 3))

        def forward(self, x, z):
            c = self.fc(z).size(1)
            return controlled_conv(
                x, self.w, stride=c, padding=c, groups=c if shared_expression else x.size(1)
            )

    model = Model()
    graph = DependencyGraph.build(
        model,
        args=(torch.arange(6.0).reshape(1, 2, 3), torch.ones(1, 3, 1)),
        operators=conv_registry(),
    )
    remove = [
        graph.parameter("fc.weight").axis(0).select([0]),
        graph.parameter("w").axis(0).select([0, 1]),
    ]
    with pytest.raises(PlanningError, match="argument"):
        Pruner(model, graph=graph).plan(remove=remove, preserve_io=False)


def test_checked_groups_parameter_can_change_without_changing_spatial_arguments(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.arange(12.0).reshape(4, 1, 3))

        def forward(self, x):
            return controlled_conv(x, self.w, groups=x.size(1))

    model = Model()
    x = torch.arange(10.0).reshape(1, 2, 5)
    expected = F.conv1d(x[:, 1:], model.w[2:], groups=1)
    graph = DependencyGraph.build(model, args=(x,), operators=conv_registry())
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("w").axis(0).select([0, 1])], preserve_io=False
    )
    torch.testing.assert_close(model(x[:, 1:]), expected)


def test_size_only_consumer_that_keeps_coordinates_is_supported(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x, data):
            y = self.fc(x)
            return data.narrow(0, y.size(1) // 4, 1)

    model = Model()
    x, data = torch.randn(2, 4), torch.arange(8.0)
    expected = model(x, data)
    graph = DependencyGraph.build(model, args=(x, data))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("fc.weight").axis(0).select([5])])
    torch.testing.assert_close(model(x, data), expected)


@pytest.mark.parametrize("reshape", [False, True])
def test_size_consumers_on_unchanged_tensors_are_checked(reshape):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 3)

        def forward(self, x):
            y = self.fc(x)
            other = x.reshape(y.size(1) - 1, -1) if reshape else x.sum(dim=y.size(1) - 2)
            return y.sum(), other

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 2),))
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("fc.weight").axis(0).select([0])])


def test_unpool_keyword_order_and_expand_template_dependencies(execution_device):
    class Unpool(nn.Module):
        def forward(self, values, indices):
            return F.max_unpool1d(indices=indices, input=values, kernel_size=2)

    values, indices = F.max_pool1d(torch.randn(1, 4, 6), 2, return_indices=True)
    graph = DependencyGraph.build(Unpool(), args=(values, indices))
    refs = graph.interfaces()
    source = next(
        r
        for r in refs
        if graph.metadata(r).dtype == values.dtype and r.shape == tuple(values.shape)
    )
    index_ref = next(r for r in refs if graph.metadata(r).dtype == torch.int64)
    impact = graph.propagate(remove=[source.axis(1).select([1])])
    assert list(impact.selection(index_ref).fully_selected_indices(1)) == [1]

    class Expand(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 1)
            self.b = nn.Linear(4, 6)
            self.c = nn.Linear(6, 2)

        def forward(self, x):
            return self.c(self.a(x).expand_as(self.b(x)))

    model = Expand()
    x = torch.randn(2, 4)
    expected = F.linear(model.a(x).expand(2, 5), model.c.weight[:, [0, 2, 3, 4, 5]], model.c.bias)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("b.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)
