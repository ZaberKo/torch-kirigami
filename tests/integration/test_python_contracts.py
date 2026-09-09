"""Cross-layer contracts that direct operator-rule tests cannot validate."""

import copy
import io
import operator

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize(
    "consumer", ["invalid_shape", "changed_shape", "slice", "branch", "loop", "scalar"]
)
def test_attribute_consumers_outside_the_pruned_branch_are_checked(consumer, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            y = self.b(self.a(x))
            if consumer in ("invalid_shape", "changed_shape"):
                other = x.reshape(self.a.out_features, -1)
            elif consumer == "slice":
                other = x[:, self.a.out_features - 1]
            elif consumer == "branch":
                other = x + 1 if self.a.out_features == 4 else x - 1
            elif consumer == "loop":
                other = x
                for _ in range(self.a.out_features):
                    other = other + 1
            else:
                other = self.a.out_features
            return y, other

    model = Model()
    x = torch.randn(3 if consumer == "changed_shape" else 2, 4)
    before = copy.deepcopy(model.state_dict())
    expected = model(x)
    graph = DependencyGraph.build(model, args=(x,))
    with pytest.raises(PlanningError, match=r"attribute|structure"):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    assert model.a.out_features == 4
    torch.testing.assert_close(model.state_dict(), before)
    torch.testing.assert_close(model(x), expected)
    graph.validate()


@pytest.mark.parametrize(
    "operation",
    [
        operator.and_,
        operator.or_,
        operator.xor,
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
        torch.logical_and,
        torch.logical_or,
        torch.logical_xor,
        pytest.param(lambda a, b: a.bitwise_and(b), id="method-bitwise-and"),
        pytest.param(lambda a, b: a.bitwise_or(b), id="method-bitwise-or"),
        pytest.param(lambda a, b: a.bitwise_xor(b), id="method-bitwise-xor"),
        pytest.param(lambda a, b: a.logical_and(b), id="method-logical-and"),
        pytest.param(lambda a, b: a.logical_or(b), id="method-logical-or"),
        pytest.param(lambda a, b: a.logical_xor(b), id="method-logical-xor"),
    ],
)
@pytest.mark.parametrize("source", ["activation", "parameter"])
def test_python_boolean_operators_run_through_capture_plan_and_backward(
    operation, source, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6, bias=False)

        def forward(self, x):
            if source == "parameter":
                w = self.fc.weight
                return F.linear(x, w * operation(w > 0, w < 1))
            y = self.fc(x)
            return torch.where(operation(y > 0, y < 1), y, 0.0)

    model = Model()
    x = torch.randn(2, 4)
    w = model.fc.weight.detach()[[0, 2, 3, 4, 5]].clone().requires_grad_()
    if source == "parameter":
        expected = F.linear(x, w * operation(w > 0, w < 1))
    else:
        y = F.linear(x, w)
        expected = torch.where(operation(y > 0, y < 1), y, 0.0)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("fc.weight").axis(0).select([1])], preserve_io=False
    )
    actual = model(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.fc.weight.grad, w.grad)


@pytest.mark.parametrize("container", [list, tuple])
def test_unflatten_configuration_survives_plan_and_checkpoint(container, execution_device):
    def make():
        return nn.Sequential(
            nn.Linear(4, 6), nn.Unflatten(1, container((2, 3))), nn.Flatten(1), nn.Linear(6, 2)
        )

    model = make()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4)
    kept = [3, 4, 5]
    expected = F.linear(
        F.linear(x, model[0].weight[kept], model[0].bias[kept]),
        model[3].weight[:, kept],
        model[3].bias,
    )
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("0.weight").axis(0).select([0, 1, 2])]
    )
    plan = PruningPlan.from_dict(plan.to_dict())
    assert model[1].unflattened_size == container((2, 3))
    Pruner(model).apply(plan)
    Pruner(original).apply(plan)
    assert type(model[1].unflattened_size) is container
    assert model[1].unflattened_size == container((1, 3))
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(original(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = make()
    load_checkpoint(restored, stream)
    assert type(restored[1].unflattened_size) is container
    torch.testing.assert_close(restored(x), expected)


def test_attribute_barrier_preserves_independent_pruning_and_readonly_planning(
    monkeypatch, execution_device
):
    class Model(nn.Module):
        def __deepcopy__(self, memo):
            raise AssertionError("Configuration verification must not call custom model copying")

        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)
            self.safe = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.Linear(6, 2))

        def forward(self, x):
            return self.b(self.a(x)), x[:, self.a.out_features - 1], self.safe(x)

    model = Model().eval()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    state = copy.deepcopy(model.state_dict())
    parameters = tuple(model.parameters())
    buffer = model.safe[1].running_mean
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if execution_device == "cuda" else None

    def forbid_parameter_copy(self, memo):
        raise AssertionError("Planning must not copy model weights")

    monkeypatch.setattr(nn.Parameter, "__deepcopy__", forbid_parameter_copy)
    with pytest.raises(PlanningError, match="structure"):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("safe.0.weight").axis(0).select([1])]
    )
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert model.safe[1].running_mean is buffer
    torch.testing.assert_close(model.state_dict(), state)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
    untouched = model(x)[:2]
    Pruner(model).apply(plan)
    torch.testing.assert_close(model(x)[:2], untouched)
    assert model.safe[0].out_features == 5
