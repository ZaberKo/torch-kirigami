"""integration / persistence contracts."""

import io

import torch
from torch.nn import functional as F

from tests.support.models import CachedWeight, WithBuffer
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_final_structure_after_two_rounds(execution_device):
    model = WithBuffer()
    x = torch.randn(2, 4)
    for _ in range(2):
        graph = DependencyGraph.build(model, args=(x,))
        Pruner(model, graph=graph).prune(remove=[graph.parameter("fc.weight").axis(0).select([1])])
        with torch.no_grad():
            model.fc.weight.add_(0.5)
    model.eval()
    model.last.train()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    with torch.inference_mode():
        restored = load_checkpoint(WithBuffer(), stream)
    assert restored.fc is restored.alias
    assert restored.fc.out_features == 4 and restored.offset.shape == (4,)
    assert not restored.training and restored.last.training
    assert not restored.fc.weight.is_inference()
    torch.testing.assert_close(restored(x), model(x))
    restored(x).sum().backward()
    assert restored.fc.weight.grad is not None
    again = io.BytesIO()
    save_checkpoint(restored, again)


def test_single_fused_definition_supports_pruning_and_checkpoint(execution_device):
    import runpy

    namespace = runpy.run_path("examples/fused_attention.py")
    namespace["main"]()


def test_cached_parameter_plan_apply_and_checkpoint(execution_device):
    model = CachedWeight()
    x = torch.randn(2, 4)
    keep = [0, 2, 3]
    expected = F.linear(F.linear(x, model.weight[keep]), model.out.weight[:, keep], model.out.bias)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(remove=[graph.parameter("weight").axis(0).select([1])])
    portable = PruningPlan.from_dict(plan.to_dict())
    Pruner(model).apply(portable)
    assert model.cached is model.alias
    assert model.cached[0]["weight"] is model.weight
    torch.testing.assert_close(model(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(CachedWeight(), stream)
    assert restored.cached is restored.alias and restored.cached[0]["weight"] is restored.weight
    torch.testing.assert_close(restored(x), expected)
