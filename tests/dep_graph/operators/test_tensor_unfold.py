"""Tensor window views preserve nonwindow coordinates through physical pruning."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    DynamicGreedy,
    Greedy,
    Magnitude,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


class PatchModel(nn.Module):
    def __init__(self, dim=1, size=3, step=1, keywords=False, inplace=False):
        super().__init__()
        self.pre = nn.Linear(3, 6)
        self.head = nn.Linear(6, 2)
        self.dim, self.size, self.step = dim, size, step
        self.keywords, self.inplace = keywords, inplace

    def forward(self, x):
        z = self.pre(x)
        patches = (
            z.unfold(dimension=self.dim, size=self.size, step=self.step)
            if self.keywords
            else z.unfold(self.dim, self.size, self.step)
        )
        if self.inplace:
            patches.relu_()
        return self.head(patches.mean(-1))


def reference(model, sample, keep):
    """Average explicit slices rather than using unfold or library index maps."""
    dim = model.dim % sample.ndim
    projected = F.linear(sample, model.pre.weight[keep], model.pre.bias[keep])
    pooled = torch.stack(
        [
            projected.narrow(dim, start, model.size).mean(dim)
            for start in range(0, projected.shape[dim] - model.size + 1, model.step)
        ],
        dim=dim,
    )
    return F.linear(pooled, model.head.weight[:, keep], model.head.bias)


@pytest.mark.parametrize(
    "shape,dim", [((8, 8, 3), 0), ((2, 8, 3), 1), ((2, 8, 3), -2), ((2, 4, 8, 3), 2)]
)
@pytest.mark.parametrize("size,step", [(3, 1), (2, 4)])
@pytest.mark.parametrize("keywords", [False, True])
def test_tensor_unfold_preserves_feature_pruning(
    shape, dim, size, step, keywords, execution_device
):
    model = PatchModel(dim, size, step, keywords).eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(shape)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    call = next(op for op in graph.operations() if op.node.target == "unfold")
    assert not call.effects.outputs_are_fresh
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])])
    assert model.pre.out_features == model.head.in_features == 6
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    expected = reference(dense, sample, [0, 2, 3, 5])
    actual = model(sample)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_tensor_unfold_output_selection_maps_back_and_checkpoint_restores(execution_device):
    model = PatchModel().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 8, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    call = next(op for op in graph.operations() if op.node.target == "unfold")
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([call.outputs[0].axis(2).select([2])]))
    expected = reference(dense, sample, [0, 1, 3, 4, 5])
    torch.testing.assert_close(model(sample), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(PatchModel(), stream)
    assert restored.pre.out_features == restored.head.in_features == 5
    torch.testing.assert_close(restored(sample), expected)
    restored(sample).sum().backward()


@pytest.mark.parametrize("port", ["input", "windows", "window_elements"])
def test_tensor_unfold_rejects_window_coordinate_changes_atomically(port, execution_device):
    model = PatchModel().eval()
    sample = torch.randn(2, 8, 3)
    original = copy.deepcopy(model.state_dict())
    identities = tuple(id(p) for p in model.parameters())
    graph = DependencyGraph.build(model, args=(sample,))
    call = next(op for op in graph.operations() if op.node.target == "unfold")
    ref, dim = {
        "input": (call.inputs[0], 1),
        "windows": (call.outputs[0], 1),
        "window_elements": (call.outputs[0], -1),
    }[port]
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="unsupported_axis"):
        pruner.plan_remove(
            [graph.parameter("pre.weight").axis(0).select([1]), ref.axis(dim).select([0])]
        )
    assert tuple(id(p) for p in model.parameters()) == identities
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


def test_tensor_unfold_does_not_authorize_inplace_view_writes(execution_device):
    model = PatchModel(inplace=True).eval()
    sample = torch.randn(2, 8, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    with pytest.raises(PlanningError, match="alias/consumer safety"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("pre.weight").axis(0).select([1])])
    assert model.pre.out_features == 6


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_tensor_unfold_candidates_work_with_both_strategies(strategy_type, execution_device):
    model = PatchModel().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 8, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    plan = pruner.plan(
        space,
        budget=ParameterBudget.from_ratio(model, 0.15),
        strategy=strategy_type(Magnitude()),
    )
    by_key = {candidate.key: candidate for candidate in space.candidates}
    impact = pruner.impact(tuple(by_key[key] for key in plan.selected))
    assert impact.complete and impact.status == "resolved"
    removed = impact.selection(graph.parameter("pre.weight")).fully_selected_indices(0)
    keep = [i for i in range(6) if i not in removed]
    assert len(keep) < 6
    pruner.apply(plan)
    torch.testing.assert_close(model(sample), reference(dense, sample, keep))


def test_tensor_unfold_shape_derived_step_remains_guarded(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            z = self.pre(x)
            return self.head(z.unfold(1, 2, z.size(-1)).mean(-1))

    model = Model().eval()
    sample = torch.randn(2, 12, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    with pytest.raises(PlanningError, match="argument"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("pre.weight").axis(0).select([0])])
    assert model.pre.out_features == 4
