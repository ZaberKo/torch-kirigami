"""Joint structural constraints checked against original-coordinate arithmetic."""

import copy
import itertools
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.numerics import _assert_value_and_input_gradient, _initialize
from tests.support.pruning import build
from torch_kirigami.pruning import PruningPlan


class SharedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 6, dtype=torch.float64)
        self.alias = self.left
        self.right = nn.Linear(4, 6, dtype=torch.float64)
        self.right.weight = self.left.weight
        self.norm = nn.LayerNorm(6, dtype=torch.float64)
        self.out = nn.Linear(6, 2, dtype=torch.float64)

    def forward(self, x):
        return self.out(self.norm(self.left(x) + 0.3 * self.alias(x) + self.right(x)))


def test_request_permutations_and_duplicates_preserve_shared_closure_and_values(execution_device):
    model = _initialize(SharedResidual())
    original = copy.deepcopy(model)
    x = torch.linspace(-0.4, 0.7, 12, dtype=torch.float64).reshape(3, 4)
    graph, pruner = build(model, x)
    requests = [
        graph.parameter("left.weight").axis(0).select([1]),
        graph.parameter("right.bias").axis(0).select([4]),
        graph.parameter("out.weight").axis(1).select([1]),
    ]
    expected = graph.propagate(remove=requests)
    assert expected.status == "resolved"
    plan = pruner.plan(remove=requests)
    for permutation in itertools.permutations(requests):
        impact = graph.propagate(remove=[*permutation, *permutation])
        assert impact.status == "resolved"
        assert all(impact.selection(ref) == expected.selection(ref) for ref in graph.values())
        other = pruner.plan(remove=permutation)
        assert other.recipes == plan.recipes and other.attributes == plan.attributes
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    assert model.left is model.alias and model.left.weight is model.right.weight
    keep = [0, 2, 3, 5]
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    hidden = 1.3 * F.linear(b, original.left.weight[keep], original.left.bias[keep])
    hidden = hidden + F.linear(b, original.right.weight[keep], original.right.bias[keep])
    centered = hidden - hidden.mean(-1, keepdim=True)
    normalized = centered / (centered.square().mean(-1, keepdim=True) + original.norm.eps).sqrt()
    normalized = normalized * original.norm.weight[keep] + original.norm.bias[keep]
    reference = F.linear(normalized, original.out.weight[:, keep], original.out.bias)
    _assert_value_and_input_gradient(model(a), reference, a, b)


class GroupedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.grouped = nn.Conv2d(4, 6, 1, groups=2, dtype=torch.float64)
        self.norm = nn.GroupNorm(2, 6, dtype=torch.float64)
        self.depthwise = nn.Conv2d(6, 12, 1, groups=6, dtype=torch.float64)
        self.bn = nn.BatchNorm2d(12, dtype=torch.float64)
        self.out = nn.Conv2d(12, 2, 1, dtype=torch.float64)

    def forward(self, x):
        return self.out(self.bn(self.depthwise(self.norm(self.grouped(x)))))


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_groupnorm_depthwise_batchnorm_joint_compaction(training, channels_last, execution_device):
    model = _initialize(GroupedNetwork()).train(training)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    original = copy.deepcopy(model)
    x = torch.linspace(-0.7, 1.1, 96, dtype=torch.float64).reshape(2, 4, 3, 4)
    graph, pruner = build(model, x)
    plan = pruner.plan(remove=[graph.parameter("grouped.weight").axis(0).select([1, 4])])
    pruner.apply(plan)
    keep, expanded = [0, 2, 3, 5], [0, 1, 4, 5, 6, 7, 10, 11]
    assert model.grouped.out_channels == model.norm.num_channels == model.depthwise.groups == 4
    assert model.depthwise.out_channels == model.bn.num_features == model.out.in_channels == 8
    for path in ("grouped", "depthwise", "out"):
        if channels_last:
            assert model.get_submodule(path).weight.is_contiguous(memory_format=torch.channels_last)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    hidden = F.conv2d(b, original.grouped.weight[keep], original.grouped.bias[keep], groups=2)
    groups = hidden.reshape(2, 2, 2, 3, 4)
    centered = groups - groups.mean((2, 3, 4), keepdim=True)
    normalized = (
        centered / (centered.square().mean((2, 3, 4), keepdim=True) + original.norm.eps).sqrt()
    )
    hidden = normalized.reshape(2, 4, 3, 4) * original.norm.weight[keep][None, :, None, None]
    hidden = hidden + original.norm.bias[keep][None, :, None, None]
    hidden = F.conv2d(
        hidden, original.depthwise.weight[expanded], original.depthwise.bias[expanded], groups=4
    )
    mean = hidden.mean((0, 2, 3)) if training else original.bn.running_mean[expanded]
    variance = (
        hidden.var((0, 2, 3), unbiased=False) if training else original.bn.running_var[expanded]
    )
    hidden = (hidden - mean[None, :, None, None]) / (variance + original.bn.eps).sqrt()[
        None, :, None, None
    ]
    hidden = (
        hidden * original.bn.weight[expanded][None, :, None, None]
        + original.bn.bias[expanded][None, :, None, None]
    )
    reference = F.conv2d(hidden, original.out.weight[:, expanded], original.out.bias)
    _assert_value_and_input_gradient(model(a), reference, a, b)


class ProjectedAttention(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.q = nn.Linear(4, 6, dtype=torch.float64)
        self.k = nn.Linear(4, 6, dtype=torch.float64)
        self.v = nn.Linear(4, 6, dtype=torch.float64)
        if shared:
            self.k.weight = self.q.weight
        self.out = nn.Linear(6, 2, dtype=torch.float64)
        self.register_buffer(
            "mask", torch.tensor([[True, False, True], [True, True, False], [False, True, True]])
        )

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = q.reshape(x.size(0), x.size(1), 2, q.size(-1) // 2).transpose(1, 2)
        k = k.reshape(x.size(0), x.size(1), 2, k.size(-1) // 2).transpose(1, 2)
        v = v.reshape(x.size(0), x.size(1), 2, v.size(-1) // 2).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask, dropout_p=0.0)
        return self.out(y.transpose(1, 2).reshape(x.size(0), x.size(1), -1))


@pytest.mark.parametrize("shared", [False, True])
def test_projection_dynamic_shape_mask_sdpa_compact_values_and_gradients(shared, execution_device):
    model = _initialize(ProjectedAttention(shared))
    original = copy.deepcopy(model)
    x = torch.linspace(-1, 0.8, 24, dtype=torch.float64).reshape(2, 3, 4)
    graph, pruner = build(model, x)
    plan = pruner.plan(
        remove=[graph.parameter(f"{name}.weight").axis(0).select([1, 4]) for name in ("q", "v")]
    )
    pruner.apply(plan)
    assert (model.q.weight is model.k.weight) == shared
    assert (
        model.q.out_features
        == model.k.out_features
        == model.v.out_features
        == model.out.in_features
        == 4
    )
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    keep = [0, 2, 3, 5]
    q, k, v = [
        F.linear(b, layer.weight[keep], layer.bias[keep]).reshape(2, 3, 2, 2).transpose(1, 2)
        for layer in (original.q, original.k, original.v)
    ]
    logits = q @ k.transpose(-1, -2) / math.sqrt(2)
    weights = logits.masked_fill(~original.mask, -math.inf).softmax(-1)
    compact = (weights @ v).transpose(1, 2).reshape(2, 3, 4)
    reference = F.linear(compact, original.out.weight[:, keep], original.out.bias)
    _assert_value_and_input_gradient(model(a), reference, a, b)
