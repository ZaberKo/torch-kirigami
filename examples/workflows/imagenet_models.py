"""Shared pretrained model loading and the explicit torchvision ViT FX adapter."""

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    resnet18,
    resnet34,
    resnet50,
    vit_b_16,
    vit_b_32,
)

MODELS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
}


class TraceableViT(nn.Module):
    """Expose torchvision ViT data flow to FX while keeping its pretrained weights.

    The examples prune only FFN intermediate widths. Patch layout, hidden
    width, attention heads and the classifier remain fixed.
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


def make_model(name, *, pretrained=True):
    """Load official weights or construct the original checkpoint skeleton.

    Gate insertion, candidate definitions and dependency capture belong to the
    individual examples so their structural pruning choices remain visible.
    """
    builder, weights = MODELS[name]
    official = builder(weights=weights if pretrained else None)
    return TraceableViT(official) if name.startswith("vit_") else official
