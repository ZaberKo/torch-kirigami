"""Official pretrained ImageNet models and explicit internal pruning positions."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16

from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import Candidate, CandidateSpace
from torch_kirigami.sparsity import ChannelGate, register_gate_operators

MODELS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
}


class TraceableViT(nn.Module):
    """Reuse torchvision ViT weights and computation without FX-incompatible assertions.

    Only FFN intermediate dimensions are candidates. Image resolution, patch
    embedding, attention heads, residual width and classifier remain unchanged.
    """

    def __init__(self, official):
        super().__init__()
        self.conv_proj = official.conv_proj
        self.class_token = official.class_token
        self.encoder = official.encoder
        self.heads = official.heads

    def forward(self, images):
        tokens = self.conv_proj(images).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = self.encoder.dropout(tokens + self.encoder.pos_embedding)
        for block in self.encoder.layers:
            normalized = block.ln_1(tokens)
            attention, _ = block.self_attention(
                normalized, normalized, normalized, need_weights=False
            )
            tokens = tokens + block.dropout(attention)
            tokens = tokens + block.mlp(block.ln_2(tokens))
        return self.heads(self.encoder.ln(tokens)[:, 0])


def layers_for(name):
    if name == "resnet18":
        return tuple(f"layer{stage}.{block}" for stage in range(1, 5) for block in range(2))
    return tuple(f"encoder.layers.encoder_layer_{index}" for index in range(12))


def positions(name, layers, *, gated=False):
    """Return (producer parameter, optional gate path, optional BN scale path)."""
    if name == "resnet18":
        return tuple(
            (
                f"{layer}.conv1.weight",
                f"{layer}.bn1.1" if gated else None,
                f"{layer}.bn1.0.weight" if gated else f"{layer}.bn1.weight",
            )
            for layer in layers
        )
    return tuple(
        (f"{layer}.mlp.0.weight", f"{layer}.mlp.1.1" if gated else None, None) for layer in layers
    )


def make_model(name, layers, *, pretrained=True, gated=False):
    builder, weights = MODELS[name]
    official = builder(weights=weights if pretrained else None)
    model = TraceableViT(official) if name == "vit_b_16" else official
    if gated:
        # Insert identity-initialized gates at explicit, known positions after
        # loading the original weights. No hooks or automatic graph rewriting.
        for layer in layers:
            block = model.get_submodule(layer)
            if name == "resnet18":
                block.bn1 = nn.Sequential(block.bn1, ChannelGate(block.conv1.out_channels, 1))
            else:
                block.mlp[1] = nn.Sequential(
                    block.mlp[1], ChannelGate(block.mlp[0].out_features, -1)
                )
    return model


def build_space(model, example, name, layers, *, gated=False, group_size=8):
    operators = register_gate_operators(OperatorRegistry.default())
    graph = DependencyGraph.build(model, args=(example,), operators=operators)
    axes = tuple(
        graph.parameter(path).axis(0) for path, _, _ in positions(name, layers, gated=gated)
    )
    candidates = tuple(
        Candidate(
            f"{layer}:{start:012d}",
            (axis.select(range(start, min(start + group_size, axis.tensor.shape[0]))),),
            axis,
        )
        for layer, axis in zip(layers, axes, strict=True)
        for start in range(0, axis.tensor.shape[0], group_size)
    )
    return CandidateSpace(graph, candidates=candidates, axes=axes)
