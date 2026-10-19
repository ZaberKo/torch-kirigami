"""Workflow metric and strategy selection uses the public pruning API."""

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    GroupMagnitude,
    ParameterBudget,
    PlanningContext,
    Pruner,
    WeightTaylor,
)

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")

import importlib


@pytest.mark.parametrize(
    ("entry", "metric"),
    [
        ("prune_finetune", "magnitude"),
        ("prune_finetune", "taylor"),
        ("iterative_pruning", "magnitude"),
        ("group_sparsity", "magnitude"),
        ("soft_pruning", "magnitude"),
        ("stability_pruning", "magnitude"),
    ],
)
def test_workflows_use_library_dependency_scores(entry, metric, execution_device, monkeypatch):
    model = nn.Sequential(nn.Linear(3, 6, bias=False), nn.Linear(6, 2)).to(execution_device)
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(-9, 9, device=execution_device).reshape(6, 3))
        model[1].weight.copy_(torch.arange(1, 13, device=execution_device).reshape(2, 6) / 2.5)
    model[0].weight.grad = torch.linspace(-1, 2, 18, device=execution_device).reshape(6, 3)
    model[1].weight.grad = torch.linspace(-2, 1, 12, device=execution_device).reshape(2, 6)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3, device=execution_device),))
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("0.weight").axis(0)
    space = CandidateSpace(
        [
            Candidate("single", [axis.select([1])], axis),
            Candidate("pair", [axis.select([0, 3])], axis),
        ],
        [axis],
    )

    def inspect_scores(space, *, budget, strategy):
        expected_metric = WeightTaylor if metric == "taylor" else GroupMagnitude
        assert isinstance(strategy.metric, expected_metric)
        context = PlanningContext(
            graph,
            space.candidates,
            budget,
            space.channel_axes,
            pruner.constraints,
        )
        return context.score(strategy.metric, space.candidates)

    # Check the actual library score against both affected weight matrices.
    # Other tests below run each complete plan/apply/train/save/load workflow.
    monkeypatch.setattr(pruner, "plan", inspect_scores)
    module = importlib.import_module(entry)
    args = ("taylor" if metric == "taylor" else "magnitude",) if entry == "prune_finetune" else ()
    budget = ParameterBudget(20)
    scores = module.make_plan(pruner, space, budget, *args)
    first, last = model[0].weight.detach().double().cpu(), model[1].weight.detach().double().cpu()
    if metric == "taylor":
        first_grad, last_grad = (
            model[0].weight.grad.double().cpu(),
            model[1].weight.grad.double().cpu(),
        )
        rows = (first * first_grad).abs().sum(1) + (last * last_grad).abs().sum(0)
    else:
        rows = first.square().sum(1) + last.square().sum(0)
        rows /= rows.mean()
    expected = [float(rows[1]), float(rows[0]) + float(rows[3])]
    assert scores == pytest.approx(expected)
