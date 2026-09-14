"""operators / convolution contracts."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.graph_helpers import indices
from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
    Magnitude,
    Pruner,
)


def test_parameter_kernel_axis_is_explicitly_unsupported():
    graph = DependencyGraph.build(nn.Conv2d(4, 6, 3), args=(torch.randn(2, 4, 7, 7),))
    impact = graph.propagate(remove=[graph.parameter("weight").axis(2).select([1])])
    assert impact.status == "unresolved"
    assert any(d.code == "unsupported_axis" for d in impact.diagnostics)


def test_depthwise_partial_outputs_do_not_choose_input_removal():
    model = nn.Conv1d(4, 8, 1, groups=4)
    x = torch.randn(2, 4, 5)
    graph = DependencyGraph.build(model, args=(x,))
    root = graph.parameter("weight").axis(0)
    input_ = graph.calls("")[0].input()
    partial = graph.propagate(remove=[root.select([2])])
    assert partial.status == "unresolved"
    assert not partial.selection(input_)
    assert set(partial.selection(root.tensor).fully_selected_indices(0)) == {2}
    balanced = graph.propagate(remove=[root.select([0, 2, 4, 6])])
    assert balanced.status == "resolved"
    assert not balanced.selection(input_)
    compact = nn.Conv1d(4, 4, 1, groups=4)
    keep = [1, 3, 5, 7]
    with torch.no_grad():
        compact.weight.copy_(model.weight[keep])
        compact.bias.copy_(model.bias[keep])
    torch.testing.assert_close(compact(x), model(x)[:, keep])
    whole = graph.propagate(remove=[root.select([2, 3])])
    assert whole.status == "resolved"
    assert set(whole.selection(input_).fully_selected_indices(1)) == {1}


@pytest.mark.parametrize(
    "cls,shape",
    [
        (nn.ConvTranspose1d, (2, 6, 5)),
        (nn.ConvTranspose2d, (2, 6, 4, 5)),
        (nn.ConvTranspose3d, (2, 6, 3, 4, 5)),
    ],
)
def test_grouped_transpose_independent_input_and_output_positions(cls, shape, execution_device):
    model = cls(6, 8, 1, groups=2)
    old = copy.deepcopy(model)
    x = torch.randn(shape)
    graph = DependencyGraph.build(model, args=(x,))
    call = graph.calls("")[0]
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [call.input().axis(1).select([0, 4]), call.output().axis(1).select([1, 6])]
        )
    )
    weight = torch.cat((old.weight[[1, 2]][:, [0, 2, 3]], old.weight[[3, 5]][:, [0, 1, 3]]))
    keep_out = [0, 2, 3, 4, 5, 7]
    func = {
        nn.ConvTranspose1d: F.conv_transpose1d,
        nn.ConvTranspose2d: F.conv_transpose2d,
        nn.ConvTranspose3d: F.conv_transpose3d,
    }[cls]
    reference = func(x[:, [1, 2, 3, 5]], weight, old.bias[keep_out], groups=2)
    torch.testing.assert_close(model(x[:, [1, 2, 3, 5]]), reference)
    assert model.in_channels == 4 and model.out_channels == 6


def test_transpose_output_protection_uses_logical_domain():
    model = nn.Sequential(nn.Conv1d(3, 6, 1), nn.ConvTranspose1d(6, 8, 1, groups=2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3, 4),))
    model, result = Pruner(model, graph=graph).prune(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.34),
        strategy=Greedy(Magnitude()),
    )
    assert result.plan.selection_report.widths == (6,)
    assert model[0].out_channels == 4 and model[1].out_channels == 8


@pytest.mark.parametrize(
    "conv,shape",
    [
        (nn.Conv1d, (2, 6, 5)),
        (nn.Conv2d, (2, 6, 4, 5)),
        (nn.Conv3d, (2, 6, 3, 4, 5)),
    ],
)
def test_partitioned_group_convolution_and_numerical_oracle(conv, shape, execution_device):
    torch.manual_seed(1)
    model = conv(6, 4, 1, groups=2)
    x = torch.randn(shape)
    graph = DependencyGraph.build(model, args=(x,))
    impact = graph.propagate(remove=[graph.calls("")[0].input().axis(1).select([0, 4])])
    assert impact.status == "resolved"
    regions = impact.selection(graph.parameter("weight")).regions
    assert len(regions) == 2
    assert tuple(set(r.axes[0]) for r in regions) == ({0, 1}, {2, 3})
    assert tuple(set(r.axes[1]) for r in regions) == ({0}, {1})
    compact = conv(4, 4, 1, groups=2)
    with torch.no_grad():
        compact.weight.copy_(torch.cat((model.weight[:2, [1, 2]], model.weight[2:, [0, 2]])))
        compact.bias.copy_(model.bias)
    reference_input = x.clone()
    reference_input[:, [0, 4]] = 0
    torch.testing.assert_close(compact(x[:, [1, 2, 3, 5]]), model(reference_input))
    assert any(r.kind == "partitioned_compaction" for r in impact.requirements)


def test_depthwise_input_removal_forces_whole_output_blocks(execution_device):
    graph = DependencyGraph.build(nn.Conv2d(4, 8, 1, groups=4), args=(torch.randn(2, 4, 3, 3),))
    impact = graph.propagate(remove=[graph.calls("")[0].input().axis(1).select([1])])
    assert impact.status == "resolved"
    assert indices(impact, graph.parameter("weight"), 0) == {2, 3}
    assert indices(impact, graph.calls("")[0].input(), 1) == {1}
    assert any(r.target == "groups" for r in impact.requirements)


@pytest.mark.parametrize(
    "conv,shape", [(nn.Conv1d, (2, 6, 5)), (nn.Conv2d, (2, 6, 4, 5)), (nn.Conv3d, (2, 6, 3, 4, 5))]
)
def test_grouped_rows_and_different_local_columns(conv, shape, execution_device):
    model = conv(6, 6, 1, groups=2)
    x = torch.randn(shape)
    original = copy.deepcopy(model)
    graph, pruner = build(model, x)
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(
        [
            graph.calls("")[0].input().axis(1).select([0, 4]),
            graph.parameter("weight").axis(0).select([1, 5]),
        ]
    )
    weight_recipe = next(
        r for r in plan.recipes if r.tensor.paths == graph.parameter("weight").paths
    )
    assert len(weight_recipe.segments) == 2
    pruner.apply(plan)
    reference = conv(4, 4, 1, groups=2)
    with torch.no_grad():
        reference.weight.copy_(
            torch.cat([original.weight[[0, 2]][:, [1, 2]], original.weight[[3, 4]][:, [0, 2]]])
        )
        reference.bias.copy_(original.bias[[0, 2, 3, 4]])
    compact_x = x[:, [1, 2, 3, 5]]
    torch.testing.assert_close(model(compact_x), reference(compact_x))
    assert model.in_channels == model.out_channels == 4


@pytest.mark.parametrize("whole", [True, False])
def test_depthwise_groups_and_multiplier(whole, execution_device):
    model = nn.Conv2d(4, 8, 1, groups=4)
    x = torch.randn(2, 4, 3, 3)
    old = copy.deepcopy(model)
    graph, pruner = build(model, x)
    indices = [2, 3] if whole else [1, 2, 5, 6]
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove(
        [graph.parameter("weight").axis(0).select(indices)]
    )
    pruner.apply(plan)
    assert model.groups == (3 if whole else 4)
    keep = [i for i in range(8) if i not in indices]
    compact_x = x[:, [0, 2, 3]] if whole else x
    reference = F.conv2d(compact_x, old.weight[keep], old.bias[keep], groups=model.groups)
    torch.testing.assert_close(model(compact_x), reference)


def test_automatic_group_balance_selects_different_local_positions():
    model = nn.Conv1d(6, 6, 1, groups=2)
    graph, pruner = build(model, torch.randn(2, 6, 4))

    def metric(ctx, batch):
        ranking = {0: 0, 4: 1, 1: 2, 2: 3, 3: 4, 5: 5}
        return [ranking[int(c.key.rsplit(":", 1)[1])] for c in batch]

    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan(
        Pruner(pruner.model, graph=pruner.graph, preserve_io=False).discover_candidates(),
        budget=ChannelRatio(0.34),
        strategy=Greedy(metric),
    )
    assert set(plan.analysis.selection(graph.parameter("weight")).fully_selected_indices(0)) == {
        0,
        4,
    }
    assert plan.selection_report.removed == (2,)
