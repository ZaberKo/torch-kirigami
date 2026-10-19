"""Diagonal coordinates and alias safety through public pruning APIs."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan


@pytest.mark.parametrize("equation", ["ii->i", "ii->", "ii", "iii->i", "...ii->...i", "ijij->ij"])
@pytest.mark.parametrize("seed", ["first_axis", "second_axis", "output"])
def test_diagonal_coordinates_and_compact_values(equation, seed, execution_device):
    shapes = {"iii->i": (4, 4, 4), "...ii->...i": (2, 4, 4), "ijij->ij": (4, 3, 4, 3)}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(shapes.get(equation, (4, 4))))

        def forward(self):
            return torch.einsum(equation, self.weight)

    model = Model()
    original = model.weight.detach().clone()
    graph = DependencyGraph.build(model)
    assert not graph.diagnostics
    ref = graph.parameter("weight")
    dims = (1, 2) if "..." in equation else (0, 2) if equation == "ijij->ij" else (0, 1)
    if seed == "output" and equation not in ("ii", "ii->"):
        output = graph.operations()[0].outputs[0]
        request = output.axis(1 if "..." in equation else 0).select([1])
    else:
        request = ref.axis(dims[seed == "second_axis"]).select([1])
    impact = graph.propagate(remove=[request])
    assert impact.status == "resolved"
    coupled = (0, 1, 2) if equation == "iii->i" else dims
    for dim in coupled:
        assert list(impact.selection(ref).fully_selected_indices(dim)) == [1]
    # Repeated requests and request ordering do not change the closure.
    reverse = graph.propagate(remove=[ref.axis(dims[1]).select([1]), request, request])
    assert reverse.selections == impact.selections
    pruner = Pruner(model, graph=graph, preserve_io=False)
    plan = pruner.plan_remove([request])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = torch.tensor([0, 2, 3])
    if equation == "iii->i":
        expected = original[keep, keep, keep]
    elif equation == "...ii->...i":
        expected = original[:, keep, keep]
    elif equation == "ijij->ij":
        j = torch.arange(3)
        expected = original[keep[:, None], j, keep[:, None], j]
    else:
        expected = original[keep, keep]
        if equation in ("ii", "ii->"):
            expected = expected.sum()
    torch.testing.assert_close(model(), expected)
    model().sum().backward()
    assert model.weight.grad is not None


@pytest.mark.parametrize("contracted", [False, True])
def test_diagonal_broadcast_and_other_consumers(contracted, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))
            self.scale = nn.Parameter(torch.randn(4, 3) if contracted else torch.randn(1))

        def forward(self, x):
            equation = "ii,ij->j" if contracted else "ii,i->i"
            return torch.einsum(equation, self.weight, self.scale), x @ self.weight

    model = Model()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    request = graph.parameter("weight").axis(0).select([1])
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan_remove([request])
    torch.testing.assert_close(model.weight, original.weight)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    pruner.apply(pruner.plan_remove([request]))
    keep = [0, 2, 3]
    diagonal, product = model(x[:, keep])
    expected = (
        original.weight[keep, keep] @ original.scale[keep]
        if contracted
        else original.weight[keep, keep] * original.scale
    )
    torch.testing.assert_close(diagonal, expected)
    torch.testing.assert_close(product, x[:, keep] @ original.weight[keep][:, keep])
    assert model.scale.shape == ((3, 3) if contracted else (1,))


@pytest.mark.parametrize("equation", ["ii->i", "ij->ji", "ij->ij"])
@pytest.mark.parametrize("clone", [False, True])
def test_einsum_views_cannot_hide_parameter_writes(equation, clone, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(-8.0, 8.0).reshape(4, 4))

        def forward(self):
            y = torch.einsum(equation, self.weight)
            if clone:
                y = y.clone()
            return y.relu_()

    model = Model()
    original = model.weight.detach().clone()
    with torch.no_grad():
        if not clone:
            with pytest.raises(CaptureError, match="Parameter/alias write"):
                DependencyGraph.build(model)
            torch.testing.assert_close(model.weight, original)
            return
        graph = DependencyGraph.build(model)
    torch.testing.assert_close(model.weight, original)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    pruner.apply(pruner.plan_remove([graph.parameter("weight").axis(0).select([1])]))
    kept = original[[0, 2, 3]]
    if equation == "ii->i":
        expected = original[[0, 2, 3], [0, 2, 3]]
    else:
        expected = kept.T if equation == "ij->ji" else kept
    torch.testing.assert_close(model(), expected.clamp_min(0))
    model().sum().backward()


@pytest.mark.parametrize("materialize", ["view", "reshape", "contiguous"])
@pytest.mark.parametrize("padded", [False, True])
def test_diagonal_followed_by_flatten_respects_strides(materialize, padded, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            # The padded batch stride equals four diagonal steps. The original
            # diagonal can flatten, but ordinary compaction destroys that equality.
            weight = (
                torch.empty_strided((2, 4, 4), (20, 4, 1)).normal_()
                if padded
                else torch.randn(1, 4, 4)
            )
            self.weight = nn.Parameter(weight)

        def forward(self):
            diagonal = torch.einsum("bii->bi", self.weight)
            if materialize == "reshape":
                return diagonal.reshape(-1)
            if materialize == "contiguous":
                diagonal = diagonal.contiguous()
            return diagonal.view(-1)

    model = Model()
    original = model.weight.detach().clone()
    graph = DependencyGraph.build(model)
    pruner = Pruner(model, graph=graph, preserve_io=False)
    request = [graph.parameter("weight").axis(1).select([1])]
    if padded and materialize == "view":
        identity = model.weight
        with pytest.raises(PlanningError, match="not proved executable"):
            pruner.plan_remove(request)
        assert model.weight is identity
        torch.testing.assert_close(model.weight, original)
        # Removing a complete batch instead is still valid after the rejection.
        pruner.apply(pruner.plan_remove([graph.parameter("weight").axis(0).select([1])]))
        indices = torch.arange(4)
        torch.testing.assert_close(model(), original[0, indices, indices])
    else:
        pruner.apply(pruner.plan_remove(request))
        torch.testing.assert_close(model(), original[:, [0, 2, 3], [0, 2, 3]].reshape(-1))


@pytest.mark.parametrize("operand_list", [False, True])
def test_einsum_products_preserve_safe_inplace_consumers(operand_list, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Parameter(torch.randn(4, 4))
            self.right = nn.Parameter(torch.randn(4, 3))

        def forward(self):
            if operand_list:
                y = torch.einsum("ij,jk->ik", [self.left, self.right])
            else:
                y = torch.einsum("ij,jk->ik", self.left, self.right)
            return y.relu_()

    model = Model()
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model)
    torch.testing.assert_close(model.left, original.left)
    torch.testing.assert_close(model.right, original.right)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("left").axis(1).select([1])]))
    expected = (original.left[:, [0, 2, 3]] @ original.right[[0, 2, 3]]).clamp_min(0)
    torch.testing.assert_close(model(), expected)
    model().sum().backward()
