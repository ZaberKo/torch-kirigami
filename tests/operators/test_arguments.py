"""operators / arguments contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)

SPELLINGS = [
    (lambda y: torch.cat((y, y), dim=1), lambda y: torch.cat((y, y), axis=1)),
    (lambda y: torch.concat((y, y), dim=1), lambda y: torch.concat((y, y), axis=1)),
    (lambda y: torch.stack((y, y), dim=1), lambda y: torch.stack((y, y), axis=1)),
    (lambda y: y.chunk(2, dim=1), lambda y: y.chunk(2, axis=1)),
    (lambda y: y.unbind(dim=0), lambda y: y.unbind(axis=0)),
    (lambda y: y.narrow(dim=1, start=0, length=4), lambda y: y.narrow(axis=1, start=0, length=4)),
    (lambda y: y.softmax(dim=1), lambda y: y.softmax(axis=1)),
    (lambda y: y.log_softmax(dim=1), lambda y: y.log_softmax(axis=1)),
    (lambda y: y.repeat_interleave(2, dim=1), lambda y: y.repeat_interleave(2, axis=1)),
    (lambda y: y.split(4, dim=1), lambda y: y.split(split_size=4, dim=1)),
    (lambda y: y.repeat(2, 1), lambda y: y.repeat(repeats=(2, 1))),
    (lambda y: y.view(y.size(0), -1), lambda y: y.view(size=(y.size(axis=0), -1))),
    (lambda y: y.reshape(y.size(0), -1), lambda y: y.reshape(shape=(y.size(axis=0), -1))),
    (lambda y: y.permute(1, 0), lambda y: y.permute(dims=(1, 0))),
    (lambda y: y.unsqueeze(dim=1), lambda y: y.unsqueeze(axis=1)),
    (lambda y: y.squeeze(dim=0), lambda y: y.squeeze(axis=0)),
]


@pytest.mark.parametrize("canonical,alias", SPELLINGS)
def test_native_keyword_spellings_have_same_coordinates(canonical, alias, execution_device):
    signatures, plans = [], []
    for operation in (canonical, alias):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 6)

            def forward(self, x):
                return operation(self.fc(x))  # noqa: B023 - used before the loop advances.

        model = Model()
        graph = DependencyGraph.build(model, args=(torch.randn(3, 4),))
        remove = [graph.parameter("fc.weight").axis(0).select([5])]
        impact = graph.propagate(remove=remove)
        assert not any(not d.complete for d in impact.diagnostics)
        signatures.append(
            (
                impact.status,
                tuple(
                    sorted(
                        (s.tensor.id.split(":", 1)[1], s.regions)
                        for s in impact.selections.values()
                    )
                ),
            )
        )
        try:
            plan = Pruner(model, graph=graph).plan(remove=remove, preserve_io=False)
        except PlanningError:
            plans.append(None)
        else:
            plans.append(tuple((r.tensor.paths, r.segments) for r in plan.recipes))
    assert signatures[0] == signatures[1]
    assert plans[0] == plans[1]


@pytest.mark.parametrize("kind", ["linear", "conv", "batch", "layer", "group"])
def test_functional_forms_with_keyword_arguments(kind, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "linear":
                self.weight = nn.Parameter(torch.randn(6, 4))
            elif kind == "conv":
                self.weight = nn.Parameter(torch.randn(6, 4, 1))
            else:
                self.weight = nn.Parameter(torch.randn(4))

        def forward(self, x):
            if kind == "linear":
                return F.linear(input=x, weight=self.weight)
            if kind == "conv":
                return F.conv1d(input=x, weight=self.weight)
            if kind == "batch":
                return F.batch_norm(
                    input=x, running_mean=None, running_var=None, weight=self.weight, training=True
                )
            if kind == "layer":
                return F.layer_norm(input=x, normalized_shape=(4,), weight=self.weight)
            return F.group_norm(input=x, num_groups=2, weight=self.weight)

    x = torch.randn(3, 4, 5) if kind == "conv" else torch.randn(3, 4)
    graph = DependencyGraph.build(Model(), args=(x,))
    impact = graph.propagate(remove=[graph.parameter("weight").axis(0).select([0, 2])])
    assert impact.status == "resolved"


@pytest.mark.parametrize("mode", ["swapaxes", "inplace", "narrow"])
def test_named_arguments_and_negative_narrow(mode, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)
            self.relu = nn.ReLU(inplace=True)
            self.b = nn.Linear(6, 2)

        def forward(self, x):
            y = self.a(x)
            if mode == "swapaxes":
                y = torch.swapaxes(input=y, axis0=0, axis1=1)
                y = y.swapaxes(axis0=0, axis1=1)
            elif mode == "inplace":
                y = self.relu(input=y)
            else:
                y = torch.narrow(y, dim=0, start=-1, length=1)
            return self.b(y)

    model = Model()
    x = torch.randn(2, 4)
    keep = [0, 2, 3, 4, 5]
    y = model.a(x)[:, keep]
    if mode == "inplace":
        y = y.relu()
    elif mode == "narrow":
        y = y[-1:]
    expected = F.linear(y, model.b.weight[:, keep], model.b.bias)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(remove=[graph.parameter("a.weight").axis(0).select([1])])
    torch.testing.assert_close(model(x), expected)
