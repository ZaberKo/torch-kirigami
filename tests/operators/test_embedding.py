"""operators / embedding contracts."""

import copy
import io
import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningContext,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def test_embedding_uses_feature_axis_budget(execution_device):
    model = nn.Sequential(nn.Embedding(12, 6), nn.Linear(6, 2))
    x = torch.tensor([[1, 3, 6], [2, 4, 5]])
    graph = DependencyGraph.build(model, args=(x,))
    _, result = Pruner(model, graph=graph).prune(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.3),
        strategy=Greedy(Magnitude()),
    )
    assert result.plan.selection_report.widths == (6,) and result.plan.selection_report.removed == (
        2,
    )
    assert model[0].embedding_dim == 4 and model[0].num_embeddings == 12
    model(x).sum().backward()


def renormalized(weight, ids, limit=1.0):
    result = weight.detach().clone()
    rows = ids.unique()
    values = result[rows]
    norms = values.norm(dim=1, keepdim=True)
    result[rows] = values * (limit / (norms + 1e-7)).clamp(max=1)
    return result


@pytest.mark.parametrize("form", ["module", "function", "function_kwargs"])
@pytest.mark.parametrize("tied", [False, True])
def test_max_norm_build_plan_apply_preserves_native_updates(form, tied, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 6, max_norm=1.0)
            self.projection = nn.Linear(6, 7, bias=False)
            if tied:
                self.projection.weight = self.embedding.weight

        def forward(self, ids):
            if form == "module":
                x = self.embedding(ids)
            elif form == "function":
                x = F.embedding(ids, self.embedding.weight, None, 1.0)
            else:
                x = F.embedding(input=ids, weight=self.embedding.weight, max_norm=1.0)
            return self.projection(x)

    model = Model().eval()
    with torch.no_grad():
        model.embedding.weight.copy_(torch.arange(1, 43).reshape(7, 6))
    ids = torch.tensor([[1, 3, 1], [2, 3, 2]])
    original = model.embedding.weight.detach().clone()
    identity = model.embedding.weight
    identity.grad = torch.ones_like(identity)
    original_grad = identity.grad
    expected_weight = renormalized(original, ids)
    with pytest.warns(UserWarning, match="parameter value updates") as emitted:
        graph = DependencyGraph.build(model, args=(ids,))
    assert len(emitted) == 1
    assert model.embedding.weight is identity
    assert identity.grad is original_grad
    torch.testing.assert_close(identity.grad, torch.ones_like(identity))
    torch.testing.assert_close(model.embedding.weight, expected_weight)
    # Rows outside the sample are unchanged. The warning does not mean that the
    # whole vocabulary was calibrated, and build does not replace .grad.
    torch.testing.assert_close(model.embedding.weight[[0, 4, 5, 6]], original[[0, 4, 5, 6]])
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("embedding.weight").axis(1)
    candidate = Candidate("feature:0", (axis.select([0]),), axis)
    context = PlanningContext(graph, (candidate,), ChannelRatio(0.3), (axis,), ())
    scores = context.score(Magnitude(p=1), (candidate,), accepted_impact=context.impact(()))
    # The removed column affects both lookups and projection inputs. Shared
    # weights count once. The reference does not reuse propagated regions.
    expected_score = float(expected_weight[:, 0].abs().sum())
    if not tied:
        expected_score += float(model.projection.weight.detach()[:, 0].abs().sum())
    assert scores[0] == pytest.approx(expected_score)
    version = identity._version
    reference = copy.deepcopy(model)
    with warnings.catch_warnings(record=True) as emitted:
        plan = pruner.plan_remove([graph.parameter("embedding.weight").axis(1).select([1, 4])])
    assert not emitted
    assert identity._version == version
    torch.testing.assert_close(identity, expected_weight)
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3, 5]
    compact_weight = renormalized(reference.embedding.weight[:, keep], ids)
    expected = F.linear(compact_weight[ids], reference.projection.weight[:, keep])
    torch.testing.assert_close(model(ids), expected)
    assert (model.projection.weight is model.embedding.weight) == tied
    model(ids).sum().backward()
    assert model.embedding.weight.grad is not None
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(Model().eval(), stream)
    assert (restored.projection.weight is restored.embedding.weight) == tied
    torch.testing.assert_close(restored(ids), expected)


def test_max_norm_root_and_failure_warning(execution_device):
    ids = torch.tensor([1, 2, 1])
    root = nn.Embedding(7, 6, max_norm=1.0).eval()
    with pytest.warns(UserWarning, match="not rolled back"):
        graph = DependencyGraph.build(root, args=(ids,))
    pruner = Pruner(root, graph=graph, preserve_io=False)
    pruner.apply(pruner.plan_remove([graph.parameter("weight").axis(1).select([1])]))
    assert root(ids).shape == (3, 5)

    bad = nn.Sequential(nn.Embedding(7, 6, max_norm=1.0), nn.BatchNorm1d(6), nn.Linear(5, 2))
    original = bad[0].weight.detach().clone()
    buffers = {name: value.clone() for name, value in bad.named_buffers()}
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if execution_device == "cuda" else None
    with pytest.warns(UserWarning, match="not rolled back"), pytest.raises(CaptureError):
        DependencyGraph.build(bad, args=(ids,))
    torch.testing.assert_close(bad[0].weight, renormalized(original, ids))
    for name, value in bad.named_buffers():
        torch.testing.assert_close(value, buffers[name])
    assert bad.training and bad[1].training
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())


def test_max_norm_warning_can_prevent_execution(execution_device):
    model = nn.Embedding(7, 6, max_norm=1.0)
    original = model.weight.detach().clone()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Metadata execution permits parameter value updates"
        )
        with pytest.raises(CaptureError, match="parameter value updates"):
            DependencyGraph.build(model, args=(torch.tensor([1, 2]),))
    torch.testing.assert_close(model.weight, original)


def test_max_norm_does_not_permit_other_parameter_writes(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 6, max_norm=1.0)

        def forward(self, ids):
            self.embedding.weight.detach().zero_()
            return self.embedding(ids)

    model = Model()
    original = model.embedding.weight.detach().clone()
    with pytest.raises(CaptureError, match="write"):
        DependencyGraph.build(model, args=(torch.tensor([1, 2]),))
    torch.testing.assert_close(model.embedding.weight, original)


def test_max_norm_does_not_permit_unproven_parameter_views(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(7, 6))

        def forward(self, ids):
            return F.embedding(ids, self.weight.detach(), max_norm=1.0)

    model = Model()
    original = model.weight.detach().clone()
    with pytest.raises(CaptureError, match="Parameter/alias write"):
        DependencyGraph.build(model, args=(torch.tensor([1, 2]),))
    torch.testing.assert_close(model.weight, original)


@pytest.mark.parametrize("temporary", [False, True])
def test_max_norm_buffer_and_temporary_weight_preserve_state(temporary, execution_device):
    """Buffer updates are isolated; fresh temporary weights need no permission."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            weight = torch.arange(1.0, 43.0).reshape(7, 6)
            if temporary:
                self.weight = nn.Parameter(weight)
            else:
                self.register_buffer("weight", weight)
            self.projection = nn.Linear(6, 2)

        def forward(self, ids):
            weight = self.weight.clone() if temporary else self.weight
            return self.projection(F.embedding(ids, weight, max_norm=1.0))

    model = Model()
    ids = torch.tensor([1, 3, 1])
    original = model.weight.clone()
    original_projection = model.projection.weight.detach().clone()
    identity = model.weight
    with warnings.catch_warnings(record=True) as emitted:
        graph = DependencyGraph.build(model, args=(ids,))
        pruner = Pruner(model, graph=graph)
        plan = pruner.plan_remove([graph.parameter("projection.weight").axis(1).select([1, 4])])
    assert not emitted
    assert model.weight is identity
    torch.testing.assert_close(model.weight, original)
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3, 5]
    torch.testing.assert_close(model.weight, original[:, keep])
    compact_weight = renormalized(original[:, keep], ids)
    expected = F.linear(compact_weight[ids], original_projection[:, keep], model.projection.bias)
    actual = model(ids)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.weight, original[:, keep] if temporary else compact_weight)
    actual.sum().backward()
    assert model.projection.weight.grad is not None
