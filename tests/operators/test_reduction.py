"""operators / reduction contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.graph_helpers import removed
from tests.support.numerics import _assert_value_and_input_gradient, _initialize, _sample
from torch_kirigami import (
    DependencyGraph,
    IndexSet,
)
from torch_kirigami.pruning import (
    Pruner,
)


def test_softmax_reports_reduced_domain():
    graph = DependencyGraph.build(nn.Softmax(dim=1), args=(torch.randn(2, 4),))
    call = graph.calls("")[0]
    impact = graph.propagate(remove=[call.input().axis(1).select([1])])
    assert impact.status == "resolved"
    assert any(r.kind == "reduction_domain" for r in impact.requirements)


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("keepdims", [False, True])
def test_alias_reduction_joint_channels(method, mean, keepdims, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6, bias=False)
            self.b = nn.Linear(6, 3, bias=False)

        def forward(self, x):
            y = self.a(x)
            if method:
                y = y.mean(axis=0, keepdims=keepdims) if mean else y.sum(axis=0, keepdims=keepdims)
            else:
                y = (
                    torch.mean(y, axis=0, keepdims=keepdims)
                    if mean
                    else torch.sum(y, axis=0, keepdims=keepdims)
                )
            return self.b(y)

    model = Model()
    x = torch.arange(24.0).reshape(6, 4)
    with torch.no_grad():
        model.a.weight.copy_(torch.arange(24.0).reshape(6, 4))
        model.b.weight.copy_(torch.arange(18.0).reshape(3, 6))
    keep = [0, 3, 4, 5]
    hidden = F.linear(x, model.a.weight[keep])
    hidden = hidden.mean(dim=0, keepdim=keepdims) if mean else hidden.sum(dim=0, keepdim=keepdims)
    expected = F.linear(hidden, model.b.weight[:, keep])
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan_remove(
        [
            graph.parameter("a.weight").axis(0).select([1]),
            graph.parameter("b.weight").axis(1).select([2]),
        ]
    )
    assert plan.analysis.selection(graph.parameter("a.weight")).fully_selected_indices(
        0
    ) == IndexSet.of([1, 2])
    Pruner(model).apply(plan)
    assert model.a.weight.shape == (4, 4) and model.b.weight.shape == (3, 4)
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("kind", ["sum", "mean", "amax"])
@pytest.mark.parametrize("dims", [(), []])
@pytest.mark.parametrize("keep", [False, True])
def test_empty_reduction_dimensions_are_all_axes(kind, dims, keep, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            return getattr(torch, kind)(self.fc(x), dim=dims, keepdim=keep)

    model = Model()
    x = torch.randn(2, 4)
    expected = getattr(torch, kind)(
        F.linear(x, model.fc.weight[[0, 2, 3, 4, 5]], model.fc.bias[[0, 2, 3, 4, 5]]),
        keepdim=keep,
        dim=(0, 1),
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove([graph.parameter("fc.weight").axis(0).select([1])])
    )
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize(
    "operation",
    [lambda x: x.sum(dim=0), lambda x: x.transpose(0, 0), lambda x: torch.softmax(x, dim=0)],
)
def test_scalar_tensor_operations_do_not_divide_by_rank(operation, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return operation(x)

    graph = DependencyGraph.build(Model(), args=(torch.tensor(3.0),))
    assert graph.propagate(remove=[]).status == "resolved"


@pytest.mark.parametrize("operation", ["sum", "mean"])
@pytest.mark.parametrize("form", ["function_keyword", "method_positional", "method_alias"])
@pytest.mark.parametrize("domain", ["channels", "sequence", "all"])
def test_reduction_api_forms_recompute_retained_domain_with_independent_arithmetic(
    operation, form, domain, execution_device
):
    dim = {"channels": -1, "sequence": -2, "all": ()}[domain]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.producer = nn.Linear(4, 6)

        def forward(self, x):
            y = self.producer(x)
            if form == "function_keyword":
                return getattr(torch, operation)(keepdim=False, dim=dim, input=y)
            if form == "method_positional":
                return getattr(y, operation)(dim, False)
            return getattr(y, operation)(axis=dim, keepdims=False)

    model = _initialize(Model().double())
    original = copy.deepcopy(model)
    sample = _sample(True, sequence=True)
    reference_input = sample.detach().clone().requires_grad_()
    # Validate the native invocation first, so invalid API spellings are not library bugs.
    model(sample)
    graph = DependencyGraph.build(model, args=(sample,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("producer.weight").axis(0).select([1, 4])]
        )
    )
    hidden = F.linear(
        reference_input,
        original.producer.weight[[0, 2, 3, 5]],
        original.producer.bias[[0, 2, 3, 5]],
    )
    # Explicit additions avoid deriving the reference axes from the operator rule.
    if domain == "channels":
        expected = hidden[:, :, 0] + hidden[:, :, 1] + hidden[:, :, 2] + hidden[:, :, 3]
        divisor = 4
    elif domain == "sequence":
        expected = hidden[:, 0, :] + hidden[:, 1, :] + hidden[:, 2, :]
        divisor = 3
    else:
        expected = sum(hidden[b, t, c] for b in range(2) for t in range(3) for c in range(4))
        divisor = 24
    if operation == "mean":
        expected = expected / divisor
    _assert_value_and_input_gradient(model(sample), expected, sample, reference_input)


@pytest.mark.parametrize("method", ["sum", "mean"])
def test_reduction_distinguishes_reduced_and_retained_axes(method):
    class Model(nn.Module):
        def forward(self, x):
            return getattr(x, method)(dim=2)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 4, 6),))
    call = graph.calls()[0]
    reduced = graph.propagate(remove=[call.input().axis(2).select([1])])
    assert reduced.status == "resolved"
    assert not reduced.selection(call.output())
    kept = graph.propagate(remove=[call.input().axis(1).select([1])])
    assert removed(kept, call.output(), 1) == {1}
    assert any(r.kind == "reduction_domain" for r in reduced.requirements)
