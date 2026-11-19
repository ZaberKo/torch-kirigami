"""Provide pretrained torchvision models and traceable ViT adapters.

The model catalogs pair supported names with official weight enums.
`TraceableViT` exposes the stock ViT dataflow to FX.
It reuses the original modules and weights.

`HeadPrunableViT` replaces attention with explicit Q/K/V and output projections.
These projections can shrink by whole heads while preserving per-head width.
The adapter supports the specified torchvision ViT self-attention layouts.
It does not support attention masks or cross-attention.

The adapters expose structure for dependency analysis and model mutation.
Each workflow entry chooses its own pruning candidates and selection policy.
"""

from collections import Counter

import torch
from torch import fx, nn
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
from torchvision.ops import stochastic_depth

from torch_kirigami import (
    ArgumentRef,
    AxisRelation,
    CaptureError,
    DependencyGraph,
    Operation,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    TensorEffects,
)

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


class EvaluationStochasticDepthRule(OperatorRule):
    """Describe torchvision stochastic depth while it acts as identity."""

    def validate_capture(self, node: fx.Node, module: nn.Module | None) -> None:
        """Reject capture before training stochastic depth can sample."""
        training = node.kwargs.get("training", node.args[3] if len(node.args) > 3 else True)
        if training is not False:
            raise CaptureError("stochastic_depth capture requires model.eval() and training=False")

    def build_spec(self, operation: Operation) -> OperatorSpec:
        """Preserve every axis in evaluation mode."""
        x, y = operation.inputs[0], operation.outputs[0]
        return OperatorSpec(
            relations=tuple(AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape)))
        )

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Return the input tensor unchanged in evaluation mode."""
        return TensorEffects(
            writes_inputs=(), aliases_input=ArgumentRef("input", 0), returns_input=True
        )


def mlp_operators() -> OperatorRegistry:
    """Register evaluation semantics for dense-MLP vision model capture."""
    return OperatorRegistry.default().register(
        stochastic_depth, EvaluationStochasticDepthRule(evaluate_on_meta=True)
    )


def _coordinatewise(operation: Operation) -> bool:
    """Recognize coordinatewise transforms without writes or active randomness."""
    module = operation.module
    if module is not None:
        if type(module) is nn.Dropout:
            return not module.training and not module.inplace
        return type(module) in (
            nn.GELU,
            nn.ReLU,
            nn.SiLU,
            nn.Tanh,
            nn.Sigmoid,
            nn.Identity,
        ) and not getattr(module, "inplace", False)
    if operation.node.op == "call_method":
        return operation.node.target in ("relu", "tanh", "sigmoid")
    if operation.node.target in (F.relu, F.silu):
        return not operation.argument("inplace", 1, False)
    if operation.node.target in (F.gelu, torch.relu, torch.tanh, torch.sigmoid):
        return True
    return (
        operation.node.target is F.dropout
        and not operation.argument("training", 2, True)
        and not operation.argument("inplace", 3, False)
    )


def discover_mlps(graph: DependencyGraph) -> tuple[dict[str, str], dict[str, str]]:
    """Find dense MLP pairs from captured dataflow.

    The producer output must reach one biased Linear through a single-use
    chain of supported coordinatewise transforms. Its last axis remains the
    hidden coordinate. Consecutive pairs may share a boundary Linear.

    Args:
        graph: Evaluation-mode graph captured before calibration hooks exist.

    Returns:
        Producer-to-consumer module paths and exclusion reasons by producer path.
    """
    graph.validate()
    operations = graph.operations()
    by_node = {operation.node: operation for operation in operations}
    calls = Counter(
        id(operation.module) for operation in operations if operation.module is not None
    )
    uses = Counter(
        ref for operation in operations for ref in {*operation.bindings.values(), *operation.inputs}
    )
    pairs: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for producer in operations:
        if type(producer.module) is not nn.Linear:
            continue
        path = producer.module_path
        assert path is not None
        if calls[id(producer.module)] != 1 or any(
            uses[ref] != 1 for ref in producer.bindings.values()
        ):
            excluded[path] = "Producer is called repeatedly or shares a parameter with another use"
            continue
        current = producer
        while True:
            if len(current.outputs) != 1 or len(current.node.users) != 1:
                excluded[path] = "Hidden value branches or does not have exactly one consumer"
                break
            node = next(iter(current.node.users))
            following = by_node.get(node)
            if following is None or following.inputs != current.outputs:
                excluded[path] = "Hidden value is returned or combined with other inputs"
                break
            if type(following.module) is nn.Linear:
                consumer = following.module
                if (
                    consumer.bias is None
                    or consumer.in_features != producer.module.out_features
                    or calls[id(consumer)] != 1
                    or any(uses[ref] != 1 for ref in following.bindings.values())
                ):
                    excluded[path] = (
                        "Consumer must be biased, width-matched, and used without parameter sharing"
                    )
                    break
                assert following.module_path is not None
                pairs[path] = following.module_path
                break
            if (
                not _coordinatewise(following)
                or len(following.outputs) != 1
                or following.outputs[0].shape != producer.outputs[0].shape
                or following.bindings
            ):
                excluded[path] = f"Unsupported hidden transform: {node.op} {node.target}"
                break
            current = following
    for consumer in pairs.values():
        excluded.pop(consumer, None)
    return pairs, excluded
