"""Local torchvision stochastic-depth semantics used by the VBP example."""

import copy

import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import variance_pruning as workflow
from torchvision.models.convnext import CNBlockConfig, ConvNeXt
from torchvision.ops import StochasticDepth, stochastic_depth

from torch_kirigami import CaptureError, DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import Pruner, load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("mode", ["row", "batch"])
def test_eval_stochastic_depth_preserves_every_coordinate(mode, execution_device):
    model = nn.Sequential(nn.Conv2d(3, 6, 1), StochasticDepth(0.7, mode), nn.Conv2d(6, 2, 1)).eval()
    inputs = torch.randn(2, 3, 4, 4, device=execution_device)
    graph = DependencyGraph.build(model, args=(inputs,), operators=workflow.vbp_operators())
    operation = next(op for op in graph.operations() if op.node.target is stochastic_depth)
    for dim in range(4):
        seed = operation.inputs[0].axis(dim).select([0])
        impact = graph.propagate(remove=(seed,))
        assert impact.selection(operation.outputs[0]) == operation.outputs[0].axis(dim).select([0])
    original = copy.deepcopy(model)
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([0, 4])])
    pruner.apply(plan)
    with torch.no_grad():
        retained = [1, 2, 3, 5]
        hidden = original[0](inputs)[:, retained]
        expected = nn.functional.conv2d(hidden, original[2].weight[:, retained], original[2].bias)
        torch.testing.assert_close(model(inputs), expected)
    assert type(model[1]) is StochasticDepth and model[1].p == 0.7 and model[1].mode == mode
    assert stochastic_depth not in OperatorRegistry.default().functions


@pytest.mark.parametrize("mode", ["row", "batch"])
@pytest.mark.parametrize("probability", [0.0, 0.7])
def test_stochastic_depth_training_capture_is_explicitly_rejected(
    mode, probability, execution_device
):
    model = nn.Sequential(nn.Conv2d(3, 6, 1), StochasticDepth(probability, mode)).train()
    inputs = torch.randn(2, 3, 4, 4, device=execution_device)
    rng = torch.get_rng_state()
    with pytest.raises(CaptureError, match=r"model.eval\(\).*training=False"):
        DependencyGraph.build(model, args=(inputs,), operators=workflow.vbp_operators())
    assert all(module.training for module in model.modules())
    torch.testing.assert_close(torch.get_rng_state(), rng)


def test_convnext_hidden_pruning_with_local_rule_and_checkpoint(execution_device, tmp_path):
    model = ConvNeXt(
        [
            CNBlockConfig(4, 8, 1),
            CNBlockConfig(8, 12, 1),
            CNBlockConfig(12, 16, 1),
            CNBlockConfig(16, None, 1),
        ],
        stochastic_depth_prob=0.2,
        num_classes=3,
    ).eval()
    original = copy.deepcopy(model)
    reference = copy.deepcopy(model)
    inputs = torch.randn(2, 3, 32, 32, device=execution_device)
    graph = DependencyGraph.build(model, args=(inputs,), operators=workflow.vbp_operators())
    pairs, _ = workflow.discover_mlps(graph)
    first, second = list(pairs.items())[1]  # A nonzero stochastic-depth probability.
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter(f"{first}.weight").axis(0).select([1, 3])])
    assert plan.analysis.status == "resolved"
    pruner.apply(plan)
    producer, consumer = reference.get_submodule(first), reference.get_submodule(second)
    retained = [i for i in range(producer.out_features) if i not in (1, 3)]
    producer.weight = nn.Parameter(producer.weight.detach()[retained].clone())
    producer.bias = nn.Parameter(producer.bias.detach()[retained].clone())
    consumer.weight = nn.Parameter(consumer.weight.detach()[:, retained].clone())
    producer.out_features = consumer.in_features = len(retained)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), reference(inputs))
    checkpoint = tmp_path / "convnext.pt"
    save_checkpoint(model, checkpoint)
    restored = load_checkpoint(original, checkpoint, map_location=execution_device).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), reference(inputs))
    # Applying an eval-captured plan leaves the original training computation.
    model.train()
    model(inputs).sum().backward()
    assert model.get_submodule(first).weight.grad is not None
