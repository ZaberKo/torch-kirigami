"""Public pruning contracts for function and method dimension transforms."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan


@pytest.mark.parametrize("form", ["function", "method", "function_kwargs", "method_kwargs"])
@pytest.mark.parametrize("sizes_kind", ["inferred", "derived", "fixed"])
def test_unflatten_original_forward_and_restore(form, sizes_kind, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 8)
            self.num_heads = 2

        def forward(self, x):
            y = self.fc(x).transpose(0, 1)
            if sizes_kind == "inferred":
                sizes = (-1, 4)
            elif sizes_kind == "derived":
                sizes = (y.size(-1) // 4, 4)
            else:
                sizes = (self.num_heads, 4)
            if form == "function":
                return torch.unflatten(y, -1, sizes)
            if form == "function_kwargs":
                return torch.unflatten(input=y, dim=-1, sizes=sizes)
            if form == "method_kwargs":
                return y.unflatten(dim=-1, sizes=sizes)
            return y.unflatten(-1, sizes)

    model = Model().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 4)
    if sizes_kind == "derived" and form.startswith("function"):
        # Some supported PyTorch releases reject Proxy factors in torch.unflatten
        # before FX dispatch. Check native FX itself rather than adding a tracer.
        try:
            torch.fx.symbolic_trace(model)
        except TypeError as error:
            assert "unflatten" in str(error) and "sizes" in str(error)
            with pytest.raises(CaptureError, match="unflatten"):
                DependencyGraph.build(model, args=(x,))
            torch.testing.assert_close(model(x), original(x))
            return
    graph = DependencyGraph.build(model, args=(x,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph, preserve_io=False)
    request = [graph.parameter("fc.weight").axis(0).select([4, 5, 6, 7])]
    if sizes_kind == "fixed":
        weight = model.fc.weight
        with pytest.raises(PlanningError):
            pruner.plan_remove(request)
        assert model.fc.weight is weight
        assert model.num_heads == 2
        torch.testing.assert_close(model(x), original(x))
        # A valid change outside the split axis remains available after rejection.
        plan = pruner.plan_remove([graph.parameter("fc.weight").axis(1).select([1])])
        pruner.apply(plan)
        masked = x.clone()
        masked[..., 1] = 0
        torch.testing.assert_close(model(x[..., [0, 2, 3]]), original(masked))
        return
    plan = pruner.plan_remove(request)
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    # Independent compact reference uses a sliced affine operation and reshape.
    expected = torch.nn.functional.linear(x, original.fc.weight[:4], original.fc.bias[:4])
    expected = expected.transpose(0, 1).reshape(3, 2, 1, 4)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert model.fc.weight.grad is not None
    assert model.num_heads == 2  # No inferred binding to a Python integer attribute.


@pytest.mark.parametrize("source", [0, -1, ()])
def test_movedim_scalar_identity(source, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return torch.movedim(x, source, source)

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.tensor(2.0),))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([]))
    torch.testing.assert_close(model(torch.tensor(2.0)), torch.tensor(2.0))


@pytest.mark.parametrize("name", ["movedim", "moveaxis"])
@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_movedim_compaction(name, method, multiple, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            source = (-1, 0) if multiple else -1
            destination = (0, 2) if multiple else 0
            if method:
                return getattr(y, name)(source=source, destination=destination)
            return getattr(torch, name)(y, source=source, destination=destination)

    model = Model().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 4)
    graph = DependencyGraph.build(model, args=(x,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph, preserve_io=False)
    plan = pruner.plan_remove([graph.parameter("fc.weight").axis(0).select([1, 4])])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    y = original.fc(x)[..., [0, 2, 3, 5]]
    expected = y.permute(2, 1, 0) if multiple else y.permute(2, 0, 1)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert model.fc.weight.grad is not None
