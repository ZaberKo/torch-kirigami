"""Native argument defaults reuse coordinate rules through public planning."""

import copy
import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan


@pytest.mark.parametrize("equation", ["ij,jk", "...ij,...jk", "Zb,bA"])
def test_einsum_implicit_output_compacts_contracted_axis(equation, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(3, 4, bias=False)
            self.right = nn.Parameter(torch.randn(4, 5))

        def forward(self, x):
            return torch.einsum(equation, self.left(x), self.right)

    model = Model()
    original = copy.deepcopy(model)
    x = torch.randn(2, 2, 3) if "..." in equation else torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("left.weight").axis(0).select([1])])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3]
    expected = F.linear(x, original.left.weight[keep]) @ original.right[keep]
    if equation == "Zb,bA":
        expected = expected.T
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


def test_unknown_operator_keeps_diagnostics_without_warning(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fixed = nn.Linear(3, 3, bias=False)
            self.free = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))

        def forward(self, x):
            return torch.special.i0(self.fixed(x)), self.free(x)

    model = Model()
    original = copy.deepcopy(model)
    x = torch.randn(3, 3)
    with warnings.catch_warnings(record=True) as emitted:
        graph = DependencyGraph.build(model, args=(x,))
    assert not emitted
    assert graph.diagnostics
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="No semantics"):
        pruner.plan_remove([graph.parameter("fixed.weight").axis(0).select([1])])
    torch.testing.assert_close(model.fixed.weight, original.fixed.weight)
    pruner.apply(pruner.plan_remove([graph.parameter("free.0.weight").axis(0).select([1])]))
    hidden = original.free[0](x)[..., [0, 2, 3]]
    expected = F.linear(hidden, original.free[1].weight[:, [0, 2, 3]], original.free[1].bias)
    actual = model(x)
    torch.testing.assert_close(actual[0], original(x)[0])
    torch.testing.assert_close(actual[1], expected)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("form", ["module", "function", "log_module", "log_function"])
def test_softmax_implicit_dimension(rank, form, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(3, 4)
            self.softmax = nn.LogSoftmax() if form.startswith("log") else nn.Softmax()
            self.right = nn.Linear(4, 2)

        def forward(self, x):
            y = self.left(x)
            if form == "function":
                y = F.softmax(y)
            elif form == "log_function":
                y = F.log_softmax(y)
            else:
                y = self.softmax(y)
            return self.right(y)

    model = Model()
    original = copy.deepcopy(model)
    x = torch.randn(*(2,) * (rank - 1), 3)
    with pytest.warns(UserWarning, match="Implicit dimension"):
        graph = DependencyGraph.build(model, args=(x,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Implicit dimension")
        plan = pruner.plan_remove([graph.parameter("left.weight").axis(0).select([1])])
        pruner.apply(PruningPlan.from_dict(plan.to_dict()))
        actual = model(x)
    y = F.linear(x, original.left.weight[[0, 2, 3]], original.left.bias[[0, 2, 3]])
    dim = 0 if rank in (1, 3) else 1
    y = y.log_softmax(dim) if form.startswith("log") else y.softmax(dim)
    expected = F.linear(y, original.right.weight[:, [0, 2, 3]], original.right.bias)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
