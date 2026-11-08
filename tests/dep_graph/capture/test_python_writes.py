"""Preserve Python Tensor writes and validate their storage dependencies."""

import operator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan


def augment(value: torch.Tensor, form: str) -> torch.Tensor:
    """Exercise the Python syntax, including forms that use integer Tensors."""
    if form == "add":
        value += 2
    elif form == "sub":
        value -= 2
    elif form == "mul":
        value *= 2
    elif form == "truediv":
        value /= 2
    elif form == "floordiv":
        value //= 2
    elif form == "mod":
        value %= 2
    elif form == "pow":
        value **= 2
    elif form == "and_":
        value &= 2
    elif form == "or_":
        value |= 2
    elif form == "xor":
        value ^= 2
    elif form == "lshift":
        value <<= 2
    elif form == "rshift":
        value >>= 2
    return value


@pytest.mark.parametrize(
    ("form", "integer"),
    [
        ("add", False),
        ("sub", False),
        ("mul", False),
        ("truediv", False),
        ("floordiv", False),
        ("mod", False),
        ("pow", False),
        ("and_", True),
        ("or_", True),
        ("xor", True),
        ("lshift", True),
        ("rshift", True),
    ],
)
def test_augmented_assignment_preserves_alias_reads_after_compaction(
    form: str, integer: bool, execution_device: str
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 6)
            self.head = nn.Linear(6, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.first(x)
            if integer:
                y = y.to(dtype=torch.int64, copy=True)
            augment(y, form)
            return self.head(y.float())

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    keep = [0, 2, 3, 5]
    y = F.linear(x, model.first.weight[keep], model.first.bias[keep])
    expected = F.linear(
        getattr(operator, form)(y.long() if integer else y, 2).float(),
        model.head.weight[:, keep],
        model.head.bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    assert any(op.effects.returns_input for op in graph.operations())
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("first.weight").axis(0).select([1, 4])])
    Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    torch.testing.assert_close(model(x), expected)


def test_augmented_matmul_uses_the_tensor_fallback_and_scalar_arithmetic(
    execution_device: str,
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 6)
            self.weight = nn.Parameter(torch.randn(6, 4))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y = self.first(x)
            width = y.size(-1)
            width -= 1
            result = y
            result @= self.weight
            return result, y[..., :width].sum(dim=-1)

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(x, model.first.weight[keep], model.first.bias[keep])
    expected = hidden @ model.weight[keep], hidden[..., :-1].sum(dim=-1)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("first.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("identity", [False, True])
def test_later_index_sees_dependencies_of_write_through_another_name(
    identity: bool, execution_device: str
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(1, 3)
            self.identity = nn.Identity()
            with torch.no_grad():
                self.first.weight.zero_()
                self.first.bias.copy_(torch.tensor([-1.0, 1.0, 1.0]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            alias = self.identity(y) if identity else y
            alias += self.first(x).sum()
            return y[..., 0][y[..., 0] > 0]

    model = Model().to(execution_device)
    sample = torch.tensor([[-1.0], [1.0]], device=execution_device)
    before = model(sample).detach().clone()
    assert before.tolist() == [1.0, 3.0]
    graph = DependencyGraph.build(model, args=(sample,))
    selection = graph.parameter("first.weight").axis(0).select([1])
    impact = graph.propagate(remove=[selection])
    assert any(diagnostic.node == "getitem_2" for diagnostic in impact.diagnostics)
    with pytest.raises(PlanningError, match="Runtime index Tensors"):
        Pruner(model, graph=graph).plan_remove([selection])
    torch.testing.assert_close(model(sample), before)


def test_consecutive_writes_can_reuse_the_same_tensor_name(execution_device: str) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.first(x)
            y.add_(1)
            y.mul_(2)
            return self.head(y)

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    keep = [0, 2, 3]
    hidden = F.linear(x, model.first.weight[keep], model.first.bias[keep])
    expected = F.linear((hidden + 1) * 2, model.head.weight[:, keep], model.head.bias)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("first.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize("form", ["syntax", "operator", "attribute"])
def test_augmentation_rejects_writes_through_unproved_aliases(
    form: str, execution_device: str
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 2)
            self.head = nn.Linear(2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.first(x)
            if form == "attribute":
                view = y.T
                view += 10
            else:
                view = y.transpose(0, 1).reshape(-1)
                if form == "syntax":
                    view += 10
                else:
                    operator.iadd(view, 10)
            return self.head(y)

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    graph = DependencyGraph.build(model, args=(x,))
    assert any(op.node.target is operator.iadd for op in graph.operations())
    bindings = tuple(model.parameters())
    before = tuple(parameter.detach().clone() for parameter in bindings)
    with pytest.raises(PlanningError, match="in-place alias/consumer"):
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("first.weight").axis(0).select([1])]
        )
    assert all(a is b for a, b in zip(bindings, model.parameters(), strict=True))
    torch.testing.assert_close(tuple(model.parameters()), before)


@pytest.mark.parametrize("form", ["augment", "index", "data", "delete"])
def test_python_writes_protect_the_original_parameter(form: str, execution_device: str) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.weight
            if form == "augment":
                y += 2
            elif form == "index":
                y[..., 1:] = x
            elif form == "data":
                y.data = x
            else:
                del y[0]
            return y + x.sum()

    model = Model().to(execution_device)
    parameter = model.weight
    before = parameter.detach().clone()
    x = torch.tensor([3.0, 4.0], device=execution_device)
    with pytest.raises(CaptureError, match=r"write|assignment|deletion"):
        DependencyGraph.build(model, args=(x,))
    assert model.weight is parameter
    torch.testing.assert_close(parameter, before)


@pytest.mark.parametrize("form", ["setitem", "augmented", "scatter"])
def test_unsupported_indexed_write_is_local(form: str, execution_device: str) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bad = nn.Linear(3, 4)
            self.good = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
            self.value = nn.Parameter(torch.tensor(2.0))
            self.register_buffer("index", torch.zeros(2, 1, dtype=torch.long))

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y = self.bad(x)
            if form == "setitem":
                y[..., :1] = self.value
            elif form == "augmented":
                y[..., :1] += self.value
            else:
                y.scatter_(-1, self.index, self.value.expand_as(self.index))
            return y, self.good(x)

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    before = model(x)[0].detach().clone()
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    writer = "scatter_" if form == "scatter" else "setitem"
    with pytest.raises(
        PlanningError, match=f"Indexed writes have no compact coordinate rule for {writer}"
    ):
        pruner.plan_remove([graph.parameter("bad.weight").axis(0).select([1])])
    pruner.apply(pruner.plan_remove([graph.parameter("good.0.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(x)[0], before)
    assert model.good[0].out_features == model.good[1].in_features == 3
