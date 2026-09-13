"""Value and layout effects survive coordinate-neutral edges of the FX graph."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import ChannelRatio, Magnitude, PlanningError, Pruner


@pytest.mark.parametrize("method", ["view", "reshape", "flatten", "contiguous_view"])
def test_layout_changes_reach_coordinate_neutral_consumers(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(24.0).reshape(3, 2, 4).transpose(0, 1))

        def forward(self, x):
            y = self.weight[:, 0, :]
            if method == "contiguous_view":
                y = y.contiguous()
                return y.view(-1) + x
            return (y.flatten() if method == "flatten" else getattr(y, method)(-1)) + x

    model = Model()
    x = torch.zeros(8)
    expected = model(x).detach()
    graph = DependencyGraph.build(model, args=(x,))
    before = model.weight
    pruner = Pruner(model, graph=graph)
    remove = [graph.parameter("weight").axis(1).select([2])]
    if method == "view":
        with pytest.raises(PlanningError, match=r"view|stride|executable"):
            pruner.plan(remove=remove)
        assert model.weight is before and model.weight.stride() == (4, 8, 1)
    else:
        pruner.prune(remove=remove)
        torch.testing.assert_close(model(x), expected)
        model(x).sum().backward()
        assert model.weight.grad is not None


@pytest.mark.parametrize("reduction", ["sum", "mean", "amax"])
@pytest.mark.parametrize("cast", ["long", "int"])
def test_unknown_value_consumer_blocks_reduced_ancestors_but_not_independent_branch(
    cast, reduction, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Linear(1, 4, bias=False)
            self.independent = nn.Sequential(nn.Linear(1, 4), nn.Linear(4, 1))
            with torch.no_grad():
                self.score.weight.fill_(1)

        def forward(self, x):
            repeats = getattr(getattr(self.score(x), reduction)(), cast)()
            return torch.repeat_interleave(x, repeats, dim=0), self.independent(x)

    model = Model()
    x = torch.ones(1)
    graph = DependencyGraph.build(model, args=(x,))
    selection = graph.parameter("score.weight").axis(0).select([0])
    assert not graph.propagate(remove=[selection]).complete
    before = model.score.weight
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan(remove=[selection])
    assert model.score.weight is before
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("independent.0.weight").axis(0).select([0])]
    )
    torch.testing.assert_close(model(x)[0], torch.ones(4 if reduction == "sum" else 1))
    model(x)[1].sum().backward()


@pytest.mark.parametrize("mutates", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_integer_read_requires_no_intervening_writes(mutates, alias, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))
            self.register_buffer("idx", torch.tensor([0]))

        def forward(self, x):
            idx = self.idx.view(-1) if alias else self.idx
            if mutates:
                idx.add_(x.size(0))
            y = torch.index_select(self.weight, 0, self.idx)
            if mutates:
                idx.add_(0 - x.size(0))
            return y

    model = Model()
    before = model.weight
    x = torch.zeros(1)
    graph = DependencyGraph.build(model, args=(x,))
    request = [graph.parameter("weight").axis(0).select([1])]
    if mutates:
        with pytest.raises(PlanningError):
            Pruner(model, graph=graph).plan(remove=request)
        assert model.weight is before
        torch.testing.assert_close(model(x), torch.tensor([[20.0]]))
    else:
        Pruner(model, graph=graph).prune(remove=request)
        torch.testing.assert_close(model(x), torch.tensor([[10.0]]))
    assert model.idx.item() == 0


@pytest.mark.parametrize("rank_form", ["dim", "ndim"])
def test_rank_spellings_have_identical_provenance(rank_form, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4)
            self.last = nn.Linear(4, 2)

        def forward(self, x):
            y = self.first(x)
            rank = y.ndim if rank_form == "ndim" else y.dim()
            return self.last(y.reshape(-1, y.size(rank - 1)))

    model = Model()
    x = torch.randn(2, 4)
    keep = [0, 2, 3]
    expected = F.linear(
        F.linear(x, model.first.weight[keep], model.first.bias[keep]),
        model.last.weight[:, keep],
        model.last.bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("first.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
def test_reused_transposed_convolution_has_one_budget_domain(dimension, groups, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = getattr(nn, f"ConvTranspose{dimension}d")(4, 6, 1, groups=groups)
            self.out = getattr(nn, f"Conv{dimension}d")(6, 2, 1)

        def forward(self, x):
            return self.out(self.up(x) + self.up(x))

    model = Model()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, *((3,) * dimension))
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(metric=Magnitude(), budget=ChannelRatio(0.5))
    assert plan.budget.widths == (6,) and plan.budget.removed == (3 if groups == 1 else 2,)
    Pruner(model, graph=graph).apply(plan)
    output = next(op.outputs[0] for op in graph.operations() if op.module_path == "up")
    removed = plan.analysis.selection(output).fully_selected_indices(1)
    keep = [i for i in range(6) if i not in removed]
    expected = getattr(F, f"conv{dimension}d")(
        reference.up(x)[:, keep] * 2, reference.out.weight[:, keep], reference.out.bias
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
