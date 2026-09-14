"""operators / normalization contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.graph_helpers import removed
from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_normalize_default_out_none_is_readonly(explicit_none, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            y = self.fc(x)
            return F.normalize(y, 2.0, 1, 1e-12, None) if explicit_none else F.normalize(y, dim=1)

    model = Model()
    x = torch.randn(2, 4)
    y = F.linear(x, model.fc.weight[[0, 2, 3, 4, 5]], model.fc.bias[[0, 2, 3, 4, 5]])
    expected = y / y.square().sum(1, keepdim=True).sqrt().clamp_min(1e-12)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("fc.weight").axis(0).select([1])]
        )
    )
    torch.testing.assert_close(model(x), expected)


def test_rms_norm_compact_domain_reference(execution_device):
    model = nn.RMSNorm(6, eps=1e-5)
    x = torch.randn(2, 6)
    old = model.weight.detach().clone()
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("weight").axis(0).select([1, 4])]
        )
    )
    kept = x[:, [0, 2, 3, 5]]
    expected = kept * torch.rsqrt(kept.square().mean(-1, keepdim=True) + 1e-5) * old[[0, 2, 3, 5]]
    torch.testing.assert_close(model(kept), expected)


@pytest.mark.parametrize(
    "module,shape,axis",
    [
        (nn.BatchNorm1d(6), (3, 6), 1),
        (nn.BatchNorm2d(6), (3, 6, 4, 4), 1),
        (nn.BatchNorm3d(6), (3, 6, 2, 3, 4), 1),
        (nn.LayerNorm(6), (2, 3, 6), 2),
        (nn.LayerNorm((3, 6)), (2, 3, 6), 2),
        (nn.GroupNorm(2, 6), (2, 6, 3, 3), 1),
    ],
)
def test_normalization_rules(module, shape, axis, execution_device):
    module = module.to(execution_device)
    graph = DependencyGraph.build(module, args=(torch.randn(shape),))
    selection = [0, 3] if isinstance(module, nn.GroupNorm) else [1, 4]
    impact = graph.propagate(remove=[graph.calls("")[0].input().axis(axis).select(selection)])
    assert impact.status == "resolved"
    assert removed(impact, graph.parameter("weight"), len(module.weight.shape) - 1) == set(
        selection
    )
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        assert removed(impact, graph.buffer("running_mean"), 0) == set(selection)
    assert impact.requirements


def test_groupnorm_compact_reference_recomputes_normalization(execution_device):
    torch.manual_seed(9)
    model = nn.GroupNorm(2, 6)
    x = torch.randn(2, 6, 3, 3)
    graph = DependencyGraph.build(model, args=(x,))
    impact = graph.propagate(remove=[graph.parameter("weight").axis(0).select([0, 4])])
    assert impact.status == "resolved"
    keep = [1, 2, 3, 5]
    compact = nn.GroupNorm(2, 4)
    with torch.no_grad():
        compact.weight.copy_(model.weight[keep])
        compact.bias.copy_(model.bias[keep])
    retained = x[:, keep].reshape(2, 2, 2, 3, 3)
    mean = retained.mean((2, 3, 4), keepdim=True)
    variance = retained.var((2, 3, 4), unbiased=False, keepdim=True)
    normalized = ((retained - mean) / (variance + model.eps).sqrt()).reshape(2, 4, 3, 3)
    reference = (
        normalized * model.weight[keep][None, :, None, None] + model.bias[keep][None, :, None, None]
    )
    torch.testing.assert_close(compact(x[:, keep]), reference)


@pytest.mark.parametrize("kind", ["batch", "layer", "group", "softmax"])
def test_normalization_compact_domain_reference(kind, execution_device):
    norm = {
        "batch": lambda: nn.BatchNorm1d(6),
        "layer": lambda: nn.LayerNorm(6),
        "group": lambda: nn.GroupNorm(2, 6),
        "softmax": lambda: nn.Softmax(dim=1),
    }[kind]()
    model = nn.Sequential(nn.Linear(4, 6), norm, nn.Linear(6, 2)).eval()
    x = torch.randn(3, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1, 4])])
    pruner.apply(plan)
    keep = [0, 2, 3, 5]
    h = F.linear(x, old[0].weight[keep], old[0].bias[keep])
    if kind == "batch":
        h = F.batch_norm(
            h,
            old[1].running_mean[keep],
            old[1].running_var[keep],
            old[1].weight[keep],
            old[1].bias[keep],
            training=False,
            eps=old[1].eps,
        )
    elif kind == "layer":
        h = F.layer_norm(h, (4,), old[1].weight[keep], old[1].bias[keep], old[1].eps)
        assert model[1].normalized_shape == (4,)
    elif kind == "group":
        h = F.group_norm(h, 2, old[1].weight[keep], old[1].bias[keep], old[1].eps)
        assert model[1].num_channels == 4
    else:
        h = h.softmax(dim=1)
    reference = F.linear(h, old[2].weight[:, keep], old[2].bias)
    torch.testing.assert_close(model(x), reference)


@pytest.mark.parametrize("dynamic", [True, False])
def test_functional_layernorm_argument_proof(dynamic, execution_device):
    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.scale = nn.Parameter(torch.randn(6))

        def forward(self, x):
            y = self.fc(x)
            return F.layer_norm(y, (y.size(1) if dynamic else 6,), self.scale)

    model = Functional()
    x = torch.randn(2, 4)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    request = [graph.parameter("fc.weight").axis(0).select([1, 4])]
    if dynamic:
        plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(request)
        pruner.apply(plan)
        keep = [0, 2, 3, 5]
        y = F.linear(x, old.fc.weight[keep], old.fc.bias[keep])
        torch.testing.assert_close(model(x), F.layer_norm(y, (4,), old.scale[keep]))
    else:
        with pytest.raises(PlanningError, match="functional argument"):
            Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(request)
