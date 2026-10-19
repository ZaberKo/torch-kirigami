"""Torchvision model adapters, official weight selection, and compact references."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import (
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import ChannelGate, register_gate_operators

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")


import imagenet_models
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.vision_transformer import VisionTransformer

from tests.support.torchvision_models import workflow_resnet


@pytest.mark.parametrize("builder", [resnet18, resnet34, resnet50])
def test_torchvision_resnet_compaction_matches_masked_reference_and_checkpoint(
    tmp_path, execution_device, builder
):
    torch.set_num_threads(1)
    torch.manual_seed(11)
    # Compare structural mathematics in float64: changing convolution widths can
    # change cuDNN's TF32 algorithm, which is not an exact FP32 numerical reference.
    model = builder(weights=None).to(device=execution_device, dtype=torch.float64).eval()
    reference = copy.deepcopy(model)
    x = torch.randn(2, 3, 64, 64, device=execution_device, dtype=torch.float64)
    removed = [0, 2, 7]
    # BN output is exactly zero for these channels, including its offset. ReLU
    # preserves zero, so removing corresponding conv2 inputs has this reference.
    with torch.no_grad():
        reference.layer1[0].bn1.weight[removed] = 0
        reference.layer1[0].bn1.bias[removed] = 0
        expected = reference(x)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("layer1.0.conv1",))
    pruner.apply(pruner.plan_remove((space.channel_axes[0].select(removed),)))
    torch.testing.assert_close(model(x), expected, atol=1e-9, rtol=1e-9)
    assert model.layer1[0].conv1.out_channels == 61
    assert model.layer1[0].conv2.in_channels == 61
    assert model.fc.out_features == 1000
    model(x).square().mean().backward()
    assert model.layer1[0].conv1.weight.grad is not None
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(
        builder(weights=None), tmp_path / "model.pt", map_location=execution_device
    ).eval()
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("patch_size", [16, 32])
def test_vit_adapter_gate_identity_and_ffn_compaction(
    gated, tmp_path, execution_device, patch_size
):
    torch.set_num_threads(1)
    # torchvision architecture with smaller dimensions keeps numerical regression
    # independent and cheap; full pretrained ViT is also checked manually.
    official = (
        VisionTransformer(
            image_size=32,
            patch_size=patch_size,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            mlp_dim=32,
            num_classes=1000,
        )
        .to(device=execution_device, dtype=torch.float64)
        .eval()
    )
    with torch.no_grad():
        official.heads.head.weight.normal_(std=0.02)
    model = imagenet_models.TraceableViT(copy.deepcopy(official)).eval()
    x = torch.randn(2, 3, 32, 32, device=execution_device, dtype=torch.float64)
    layers = ("encoder.layers.encoder_layer_0",)
    if gated:
        model.encoder.layers[0].mlp[1] = nn.Sequential(
            model.encoder.layers[0].mlp[1],
            ChannelGate(32, -1).to(device=execution_device, dtype=torch.float64),
        )
    skeleton = copy.deepcopy(model)
    with torch.no_grad():
        torch.testing.assert_close(model(x), official(x), rtol=1e-10, atol=1e-10)
        official.encoder.layers[0].mlp[0].weight[[0, 2]] = 0
        official.encoder.layers[0].mlp[0].bias[[0, 2]] = 0
        expected = official(x)
    operators = (
        register_gate_operators(OperatorRegistry.default()) if gated else OperatorRegistry.default()
    )
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    pruner = Pruner(model, graph=graph)
    pruner.apply(
        pruner.plan_remove((graph.parameter(f"{layers[0]}.mlp.0.weight").axis(0).select([0, 2]),))
    )
    torch.testing.assert_close(model(x), expected, rtol=1e-9, atol=1e-9)
    assert model.encoder.layers[0].mlp[0].out_features == 30
    model(x).square().mean().backward()
    assert model.encoder.layers[0].mlp[0].weight.grad is not None
    if gated:
        assert model.encoder.layers[0].mlp[1][1].size == 30
    save_checkpoint(model, tmp_path / "vit.pt")
    restored = load_checkpoint(skeleton, tmp_path / "vit.pt", map_location=execution_device)
    torch.testing.assert_close(restored(x), model(x))


@pytest.mark.parametrize("name", tuple(imagenet_models.MODELS))
def test_supported_pretrained_models_use_explicit_official_weights(monkeypatch, name):
    _, weights = imagenet_models.MODELS[name]
    seen = []

    def build(*, weights):
        seen.append(weights)
        if name.startswith("resnet"):
            return workflow_resnet()
        return VisionTransformer(
            image_size=224,
            patch_size=32 if name.endswith("32") else 16,
            num_layers=2,
            num_heads=2,
            hidden_dim=16,
            mlp_dim=32,
        )

    monkeypatch.setitem(imagenet_models.MODELS, name, (build, weights))
    model = imagenet_models.make_model(name).eval()
    skeleton = imagenet_models.make_model(name, pretrained=False).eval()
    assert seen == [weights, None]
    assert weights.transforms().crop_size == [224]
    assert isinstance(model, imagenet_models.TraceableViT) == name.startswith("vit_")
    assert type(model) is type(skeleton)
