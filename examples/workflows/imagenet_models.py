"""Shared pretrained model loading and explicit torchvision ViT FX adapters."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    convnext_tiny,
    resnet18,
    resnet34,
    resnet50,
    vit_b_16,
    vit_b_32,
)
from torchvision.models.vision_transformer import EncoderBlock, VisionTransformer

MODELS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_b_32": (vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
}

# These models expose dense two-layer MLPs with an output bias. Keeping this
# catalog separate avoids advertising ConvNeXt in ResNet-specific workflows.
MLP_MODELS = {
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "vit_b_16": MODELS["vit_b_16"],
    "vit_b_32": MODELS["vit_b_32"],
}


class TraceableViT(nn.Module):
    """Expose torchvision ViT data flow to FX while keeping its pretrained weights.

    The adapter does not choose pruning positions. Each workflow discovers
    candidates or validates the structures required by its algorithm; the
    dependency and execution rules determine which changes are supported.
    """

    def __init__(self, official: VisionTransformer) -> None:
        super().__init__()
        self.conv_proj = official.conv_proj
        self.class_token = official.class_token
        self.encoder = official.encoder
        self.heads = official.heads

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify an image batch using the original pretrained submodules."""
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


class HeadPrunableAttention(nn.Module):
    """Self-attention with independent internal width and fixed per-head width.

    Args:
        source: Torchvision's batch-first, equal-width self-attention module.

    Only the self-attention used by torchvision ViT is supported: no masks,
    cross-attention, extra bias tokens, or returned attention weights. Dropout
    follows the module's training mode. Conversion copies projection parameters;
    construct a new optimizer after conversion and physical pruning.
    """

    def __init__(self, source: nn.MultiheadAttention) -> None:
        super().__init__()
        if (
            type(source) is not nn.MultiheadAttention
            or not source.batch_first
            or source.in_proj_weight is None
            or source.kdim != source.embed_dim
            or source.vdim != source.embed_dim
            or source.bias_k is not None
            or source.bias_v is not None
            or source.add_zero_attn
        ):
            raise ValueError("Expected batch-first equal-width ViT self-attention")
        self.head_dim = source.head_dim
        self.dropout = source.dropout
        options = {"device": source.in_proj_weight.device, "dtype": source.in_proj_weight.dtype}
        self.qkv = nn.Linear(
            source.embed_dim, 3 * source.embed_dim, bias=source.in_proj_bias is not None, **options
        )
        self.proj = nn.Linear(
            source.embed_dim, source.embed_dim, bias=source.out_proj.bias is not None, **options
        )
        with torch.no_grad():
            self.qkv.weight.copy_(source.in_proj_weight)
            self.proj.weight.copy_(source.out_proj.weight)
            if self.qkv.bias is not None:
                self.qkv.bias.copy_(source.in_proj_bias)
            if self.proj.bias is not None:
                self.proj.bias.copy_(source.out_proj.bias)
        self.qkv.weight.requires_grad_(source.in_proj_weight.requires_grad)
        self.proj.weight.requires_grad_(source.out_proj.weight.requires_grad)
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad_(source.in_proj_bias.requires_grad)
        if self.proj.bias is not None:
            self.proj.bias.requires_grad_(source.out_proj.bias.requires_grad)
        self.train(source.training)

    @property
    def num_heads(self) -> int:
        """Return the current head count from the compact projection width."""
        return self.proj.in_features // self.head_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply unmasked self-attention without changing the residual width."""
        batch, length = tokens.shape[0], tokens.shape[1]
        # Infer the head count at execution time; no stale Python size needs
        # rewriting after pruning. The per-head feature dimension stays fixed.
        q, k, v = (
            self.qkv(tokens).reshape(batch, length, 3, -1, self.head_dim).permute(2, 0, 3, 1, 4)
        ).unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(attended.transpose(1, 2).reshape(batch, length, -1))


class HeadPrunableEncoderBlock(nn.Module):
    """Keep a complete callable encoder block around the explicit self-attention.

    Args:
        source: Torchvision encoder block whose normalization, dropout and MLP
            modules are reused. Its attention projections are copied.

    A separate block avoids leaving an EncoderBlock whose original forward
    expects the now-replaced native MultiheadAttention call signature.
    """

    def __init__(self, source: EncoderBlock) -> None:
        super().__init__()
        if type(source) is not EncoderBlock:
            raise ValueError("Expected a torchvision EncoderBlock")
        self.ln_1 = source.ln_1
        self.self_attention = HeadPrunableAttention(source.self_attention)
        self.dropout = source.dropout
        self.ln_2 = source.ln_2
        self.mlp = source.mlp
        self.training = source.training

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply attention and FFN residual updates with fixed external width."""
        tokens = tokens + self.dropout(self.self_attention(self.ln_1(tokens)))
        return tokens + self.mlp(self.ln_2(tokens))


class HeadPrunableViT(TraceableViT):
    """Reuse a torchvision ViT's modules, explicitly replacing its attention.

    The supplied model's encoder is reused and modified. Build the same adapter
    around an unpruned skeleton before loading a structural checkpoint. Model
    classes live in this importable module so CLI and fresh-script checkpoints
    share stable class identities, rather than recording `__main__` classes.
    """

    def __init__(self, official: VisionTransformer) -> None:
        super().__init__(official)
        for name, block in tuple(self.encoder.layers.named_children()):
            self.encoder.layers.add_module(name, HeadPrunableEncoderBlock(block))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify images with independently prunable attention and FFN widths."""
        tokens = self.conv_proj(images).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.class_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = self.encoder.dropout(tokens + self.encoder.pos_embedding)
        for block in self.encoder.layers:
            tokens = block(tokens)
        return self.heads(self.encoder.ln(tokens)[:, 0])


def make_head_prunable_model(name: str, *, pretrained: bool = True) -> HeadPrunableViT:
    """Construct the identical adapted skeleton for pruning and restoration."""
    if name not in ("vit_b_16", "vit_b_32"):
        raise ValueError("This workflow supports vit_b_16 and vit_b_32")
    builder, weights = MODELS[name]
    return HeadPrunableViT(builder(weights=weights if pretrained else None))


def make_model(name: str, *, pretrained: bool = True) -> nn.Module:
    """Load official weights or construct the original checkpoint skeleton.

    Gate insertion, candidate definitions and dependency capture belong to the
    individual examples so their structural pruning choices remain visible.
    """
    builder, weights = MODELS[name]
    official = builder(weights=weights if pretrained else None)
    return TraceableViT(official) if name.startswith("vit_") else official


def make_mlp_model(name: str, *, pretrained: bool = True) -> nn.Module:
    """Load a CNN or ViT with dense MLPs for activation-based pruning."""
    builder, weights = MLP_MODELS[name]
    official = builder(weights=weights if pretrained else None)
    return TraceableViT(official) if name.startswith("vit_") else official
