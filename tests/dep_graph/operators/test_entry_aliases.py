"""Independent spelling coverage, including calls absent from the registry."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan

# This list is deliberately independent of default_rules and registry contents.
ALIASES = {
    "absolute": torch.abs,
    "negative": torch.neg,
    "multiply": torch.mul,
    "subtract": torch.sub,
    "divide": torch.div,
    "clip": torch.clamp,
    "greater": torch.gt,
    "greater_equal": torch.ge,
    "less": torch.lt,
    "less_equal": torch.le,
    "not_equal": torch.ne,
    "detach": torch.detach,
    "masked_fill": torch.masked_fill,
    "relu_": torch.relu_,
}


def invoke(call, name, x):
    if name == "clip":
        return call(x, min=-0.5, max=0.5)
    if name == "masked_fill":
        return call(x, x > 0, 0.0)
    if name in {
        "multiply",
        "subtract",
        "divide",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "not_equal",
    }:
        return call(x, 0.5)
    return call(x)


@pytest.mark.parametrize("name", list(ALIASES))
@pytest.mark.parametrize("method", [False, True])
def test_alias_spelling_compacts_with_independent_base_reference(name, method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 6)
            self.head = nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            if method:
                # The bound receiver replaces the first argument of a function call.
                y = invoke(lambda value, *a, **kw: getattr(value, name)(*a, **kw), name, y)
            else:
                y = invoke(getattr(torch, name), name, y)
            return self.head(y.float())

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    expected = F.linear(
        invoke(ALIASES[name], name, hidden).float(), dense.head.weight[:, keep], dense.head.bias
    )
    torch.testing.assert_close(model(sample), expected)
    model(sample).sum().backward()
    assert model.head.weight.grad is not None


@pytest.mark.parametrize("method", [False, True])
def test_broadcast_to_entries_follow_dynamic_feature_width(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            shape = (2, y.size(-1))
            y = y.broadcast_to(size=shape) if method else torch.broadcast_to(y, size=shape)
            return self.head(y)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(1, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]).expand(2, -1)
    torch.testing.assert_close(
        model(sample), F.linear(hidden, dense.head.weight[:, keep], dense.head.bias)
    )


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("guarded", [False, True])
def test_numel_entries_keep_shape_provenance(method, guarded, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            count = y.numel() if method else torch.numel(input=y)
            y = y + count if guarded else y.reshape(y.size(0), count // y.size(0))
            return self.head(y)

    model = Model().eval()
    sample = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    remove = [graph.parameter("pre.weight").axis(0).select([1])]
    if guarded:
        with pytest.raises(PlanningError, match="argument"):
            pruner.plan_remove(remove)
        assert model.pre.out_features == 6
    else:
        dense = copy.deepcopy(model)
        keep = [0, 2, 3, 4, 5]
        expected = F.linear(
            F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]),
            dense.head.weight[:, keep],
            dense.head.bias,
        )
        pruner.apply(pruner.plan_remove(remove))
        torch.testing.assert_close(model(sample), expected)


@pytest.mark.parametrize("kind", ["layer", "rms", "prelu"])
@pytest.mark.parametrize("alternative", [False, True])
def test_normalization_entry_forms_preserve_parameter_roles(kind, alternative, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)
            self.weight = nn.Parameter(torch.linspace(0.3, 1.4, 6))
            self.bias = nn.Parameter(torch.linspace(-0.4, 0.7, 6)) if kind == "layer" else None

        def forward(self, x):
            y = self.pre(x)
            if kind == "prelu":
                y = y.prelu(self.weight) if alternative else F.prelu(y, self.weight)
            elif kind == "rms":
                call = torch.rms_norm if alternative else F.rms_norm
                y = call(y, (y.size(-1),), self.weight, 1e-4)
            else:
                call = torch.layer_norm if alternative else F.layer_norm
                y = call(y, (y.size(-1),), self.weight, self.bias, 1e-4)
            return self.head(y)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3)
    try:
        graph = DependencyGraph.build(model, args=(sample,))
    except CaptureError as error:
        # Older native normalization argument parsers reject Proxy-valued sizes.
        # The separately tested F forms remain valid alternatives. Do not invent
        # a capture fallback or treat failed capture as successful compaction.
        assert alternative and kind in ("layer", "rms")
        assert "normalized_shape" in str(error) and "Proxy" in str(error)
        torch.testing.assert_close(model.pre.weight, dense.pre.weight)
        return
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    y = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    if kind == "prelu":
        y = torch.where(y >= 0, y, y * dense.weight[keep])
    else:
        variance = (
            y.square().mean(-1, keepdim=True)
            if kind == "rms"
            else y.var(-1, unbiased=False, keepdim=True)
        )
        y = y if kind == "rms" else y - y.mean(-1, keepdim=True)
        y = y / (variance + 1e-4).sqrt() * dense.weight[keep]
        if kind == "layer":
            y = y + dense.bias[keep]
    torch.testing.assert_close(
        model(sample), F.linear(y, dense.head.weight[:, keep], dense.head.bias)
    )
    model(sample).sum().backward()


def test_function_relu_inplace_requires_safe_producer(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            z = torch.relu_(y.detach())
            return self.head(z)

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    with pytest.raises(PlanningError, match="alias/consumer"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("pre.weight").axis(0).select([1])])
    assert model.pre.out_features == 6
