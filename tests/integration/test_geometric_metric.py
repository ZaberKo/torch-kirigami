"""Executable workflow coverage for static geometric filter scores."""

import copy
import math
import sys

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, IndexSet, Region, Selection
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    DynamicGreedy,
    Granularity,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    WeightTaylor,
    load_checkpoint,
    save_checkpoint,
)

pytest.importorskip("torchvision", reason="Install workflow example dependencies")
pytest.importorskip("datasets", reason="Install workflow example dependencies")

import prune_finetune
from torchvision.models.vision_transformer import VisionTransformer


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def metric_context(device):
    model = nn.Sequential(nn.Linear(3, 5, bias=False), nn.Linear(5, 2)).to(device).eval()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[-2, 0, 1], [2, 0, -1], [0, 0, 0], [1, 2, 0], [2, 1, 0]])
        )
    graph = DependencyGraph.build(model, args=(torch.zeros(2, 3, device=device),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    context = PlanningContext(
        graph,
        graph.operations(),
        space.candidates,
        ChannelRatio(0.4),
        space.channel_axes,
        pruner.constraints,
    )
    return model, pruner, space, context


@pytest.mark.parametrize("block_size", [1, 2, 256])
def test_geometric_scores_match_signed_distance_reference(
    block_size, execution_device, monkeypatch
):
    model, _pruner, space, context = metric_context(execution_device)
    axis = space.channel_axes[0]
    rows = model[0].weight.detach().double().cpu().tolist()
    # Python scalar arithmetic is independent of the tiled torch.cdist implementation.
    expected = [
        sum(
            math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right, strict=True))) for right in rows
        )
        for left in rows
    ]
    distance_tiles = []
    original_cdist = torch.cdist

    def tracked_cdist(left, right, **kwargs):
        distance_tiles.append((left.shape, right.shape, left.device.type))
        return original_cdist(left, right, **kwargs)

    monkeypatch.setattr(torch, "cdist", tracked_cdist)
    metric = prune_finetune.GeometricMedian(
        context.graph, space.channel_axes, block_size=block_size
    )
    assert distance_tiles
    assert all(
        left[0] <= block_size and right[0] <= block_size for left, right, _ in distance_tiles
    )
    assert {device for _, _, device in distance_tiles} == {execution_device}
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(5))
    assert context.score(metric, candidates) == pytest.approx(expected)
    assert context.score(metric, tuple(reversed(candidates))) == pytest.approx(expected[::-1])
    assert [context.score(metric, (candidate,))[0] for candidate in candidates] == pytest.approx(
        expected
    )
    block = Candidate("overlapping block", (axis.select([0, 2]), axis.select([2, 4])), axis)
    assert context.score(metric, (block,)) == pytest.approx(
        [expected[0] + expected[2] + expected[4]]
    )
    assert len(distance_tiles) == math.ceil(5 / block_size) ** 2  # Scoring reuses the snapshot.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e30])
def test_geometric_rejects_nonfinite_weights_and_distances(value, execution_device):
    model, _pruner, space, context = metric_context(execution_device)
    with torch.no_grad():
        model[0].weight[0, 0] = value
    with pytest.raises(PlanningError, match="finite"):
        prune_finetune.GeometricMedian(context.graph, space.channel_axes)


def test_geometric_requires_complete_output_rows_and_static_selection(execution_device):
    _model, _pruner, space, context = metric_context(execution_device)
    axis = space.channel_axes[0]
    metric = prune_finetune.GeometricMedian(context.graph, space.channel_axes)
    partial = Selection(axis.tensor, (Region((IndexSet.of([0]), IndexSet.of([0]))),))
    with pytest.raises(PlanningError, match="complete output-filter"):
        metric.score(
            context, (Candidate("partial", (partial,), axis),), selected=context.impact(())
        )
    with pytest.raises(PlanningError, match="scored output-weight"):
        metric.score(
            context,
            (Candidate("undeclared", (axis.select([0]),)),),
            selected=context.impact(()),
        )
    with pytest.raises(PlanningError, match="static"):
        context.score(metric, (space.candidates[0],), selected=context.impact((axis.select([2]),)))
    with pytest.raises(ValueError, match="output-weight"):
        prune_finetune.GeometricMedian(context.graph, (axis.tensor.axis(1),))
    with pytest.raises(ValueError, match="positive integer"):
        prune_finetune.GeometricMedian(context.graph, (axis,), block_size=0)


@pytest.mark.parametrize("family", ["cnn", "vit"])
def test_geometric_public_plan_apply_and_checkpoint(family, execution_device, tmp_path):
    torch.manual_seed(54)
    if family == "cnn":
        model = nn.Sequential(
            nn.Conv2d(3, 6, 1), nn.ReLU(), nn.Conv2d(6, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        pairs = (("0", "2"),)
    else:
        model = VisionTransformer(
            image_size=8,
            patch_size=4,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            mlp_dim=16,
            num_classes=3,
        )
        nn.init.normal_(model.heads.head.weight)
        pairs = tuple(
            (f"encoder.layers.encoder_layer_{i}.mlp.0", f"encoder.layers.encoder_layer_{i}.mlp.3")
            for i in range(2)
        )
    model = model.to(execution_device).eval()
    dense = copy.deepcopy(model)
    reference = copy.deepcopy(model)
    inputs = torch.randn(2, 3, 8, 8, device=execution_device)
    graph = DependencyGraph.build(model, args=(inputs,))
    targets = tuple(first for first, _second in pairs)
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path=dict.fromkeys(targets, 2)))
    space = pruner.discover_candidates(targets=targets)
    budget = ParameterBudget.from_ratio(model, 0.05)
    plan = prune_finetune.make_plan(pruner, space, budget, "geometric_median")
    assert plan.selection_report.target_met
    for (name, tensor), (dense_name, old) in zip(
        model.state_dict().items(), dense.state_dict().items(), strict=True
    ):
        assert name == dense_name
        torch.testing.assert_close(tensor, old)  # Planning is read-only.
    for first_path, second_path in pairs:
        first = reference.get_submodule(first_path)
        second = reference.get_submodule(second_path)
        ref = graph.parameter(f"{first_path}.weight")
        removed = set(plan.analysis.selection(ref).fully_selected_indices(0))
        retained = [i for i in range(ref.shape[0]) if i not in removed]
        first.weight = nn.Parameter(first.weight.detach()[retained].clone())
        first.bias = nn.Parameter(first.bias.detach()[retained].clone())
        second.weight = nn.Parameter(second.weight.detach()[:, retained].clone())
        if isinstance(first, nn.Linear):
            first.out_features = second.in_features = len(retained)
        else:
            first.out_channels = second.in_channels = len(retained)
    pruned, _result = pruner.apply(plan)
    with torch.no_grad():
        torch.testing.assert_close(pruned(inputs), reference(inputs))
    checkpoint = tmp_path / f"{family}.pt"
    save_checkpoint(pruned, checkpoint)
    restored = load_checkpoint(dense, checkpoint, map_location=execution_device).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), reference(inputs))
    pruned(inputs).sum().backward()
    assert all(parameter.grad is not None for parameter in pruned.parameters())


@pytest.mark.parametrize("metric", ["magnitude", "taylor"])
@pytest.mark.parametrize("selection", ["static", "dynamic"])
def test_workflow_selects_requested_strategy(metric, selection, execution_device, monkeypatch):
    _model, pruner, space, _context = metric_context(execution_device)

    def inspect_plan(candidate_space, *, budget, strategy):
        assert candidate_space is space
        assert isinstance(strategy, Greedy if selection == "static" else DynamicGreedy)
        assert isinstance(
            strategy.metric, GroupMagnitude if metric == "magnitude" else WeightTaylor
        )
        return "verified"

    monkeypatch.setattr(pruner, "plan", inspect_plan)
    assert (
        prune_finetune.make_plan(pruner, space, ParameterBudget(20), metric, selection)
        == "verified"
    )
    with pytest.raises(ValueError, match="requires static"):
        prune_finetune.make_plan(pruner, space, ParameterBudget(20), "geometric_median", "dynamic")


@pytest.mark.parametrize(
    "metric,selection",
    [("geometric_median", "static"), ("magnitude", "dynamic"), ("taylor", "dynamic")],
)
def test_workflow_cli_explicit_metric_selection(metric, selection, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prune_finetune", "--device", "cpu", "--metric", metric, "--selection", selection],
    )
    options = prune_finetune.parse_args()
    assert options.metric == metric and options.selection == selection


def test_workflow_cli_rejects_dynamic_geometric_scores(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prune_finetune",
            "--device",
            "cpu",
            "--metric",
            "geometric_median",
            "--selection",
            "dynamic",
        ],
    )
    with pytest.raises(SystemExit) as error:
        prune_finetune.parse_args()
    assert error.value.code == 2
    assert "requires --selection static" in capsys.readouterr().err
