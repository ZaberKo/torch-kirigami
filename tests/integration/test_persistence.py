"""integration / persistence contracts."""

import io
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tests.support.models import CachedWeight, WithBuffer
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    AttributeRecipe,
    ExecutionError,
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
        Pruner(model, graph=graph).apply(
            Pruner(model, graph=graph).plan_remove(
                [graph.parameter("fc.weight").axis(0).select([1])]
            )
        )
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
    plan = Pruner(model, graph=graph).plan_remove([graph.parameter("weight").axis(0).select([1])])
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


@pytest.mark.parametrize("new_widths", [(6, 5), (5, 6), (5, 4), (5, 5)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("boundary", ["apply", "load"])
def test_shared_attribute_recipes_require_one_final_value(
    new_widths, reverse, boundary, execution_device
):
    """An alias cannot declare a no-op while another alias changes the same field."""
    model = WithBuffer()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("fc.weight").axis(0).select([1])]
    )
    edits = (
        AttributeRecipe("fc.out_features", 6, new_widths[0]),
        AttributeRecipe("alias.out_features", 6, new_widths[1]),
    )
    if reverse:
        edits = tuple(reversed(edits))
    plan = replace(
        plan,
        attributes=tuple(a for a in plan.attributes if a.path != "fc.out_features") + edits,
    )
    original_tensors = (*model.parameters(), *model.buffers())
    original_values = tuple(t.detach().clone() for t in original_tensors)
    original_output = model(x).detach().clone()
    modes = tuple(m.training for m in model.modules())
    if new_widths[0] != new_widths[1]:
        error_type = ValueError if boundary == "load" else ExecutionError
        with pytest.raises(error_type, match="Shared attribute edits disagree"):
            if boundary == "load":
                PruningPlan.from_dict(plan.to_dict())
            else:
                Pruner(model).apply(plan)
        assert model.fc is model.alias and model.fc.out_features == 6
        assert tuple(m.training for m in model.modules()) == modes
        assert not hasattr(model, "_kirigami_structure")
        assert all(
            current is old
            for current, old in zip(
                (*model.parameters(), *model.buffers()), original_tensors, strict=True
            )
        )
        for tensor, value in zip(original_tensors, original_values, strict=True):
            torch.testing.assert_close(tensor, value)
        torch.testing.assert_close(model(x), original_output)
        graph.validate(model)
        return
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(x, model.fc.weight[keep], model.fc.bias[keep]) + model.offset[keep]
    expected = F.linear(hidden.relu(), model.last.weight[:, keep], model.last.bias)
    if boundary == "load":
        plan = PruningPlan.from_dict(plan.to_dict())
    Pruner(model).apply(plan)
    assert model.fc is model.alias and model.fc.out_features == 5
    torch.testing.assert_close(model(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(WithBuffer(), stream)
    assert restored.fc is restored.alias and restored.fc.out_features == 5
    torch.testing.assert_close(restored(x), expected)
