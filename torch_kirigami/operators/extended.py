"""Common channel, normalization, embedding, and elementwise operator families."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import AxisBarrier, Requirement
from ..errors import CaptureError, UnsupportedOperation
from ..operation import (
    CandidateAxis,
    OperationContext,
    OperatorRegistrar,
    OperatorRule,
    OperatorSpec,
    OutputContract,
)
from ..relations import ReshapeRelation
from ..selection import TensorRef
from .effects import native_effects
from .native import (
    affine_layouts,
    bind,
    equal,
    layer_norm,
    layout,
    one,
    pointwise,
    reduction,
    requirement,
)


def channel_operator(ctx: OperationContext) -> OperatorSpec:
    """Preserve batch/channel coordinates while protecting spatial transformations."""
    x = one(ctx.argument("input", 0))
    outputs = ctx.outputs
    # Public pooling/padding/interpolation APIs place channels immediately before
    # their spatial axes. Batched and unbatched pooling forms share this descriptor.
    name = (
        type(ctx.module).__name__.lower()
        if ctx.module is not None
        else str(getattr(ctx.node.target, "__name__", ctx.node.target))
    )
    spatial = next((n for n in (1, 2, 3) if f"{n}d" in name), None)
    channel = len(x.shape) - spatial - 1 if spatial is not None else 1
    if channel not in (0, 1) or len(x.shape) < channel + 2:
        raise UnsupportedOperation("Expected a channel/spatial tensor")
    relations, constraints = [], []
    for y in (*outputs, *(t for t in ctx.inputs if t != x)):
        if len(y.shape) != len(x.shape) or y.shape[channel] != x.shape[channel]:
            raise UnsupportedOperation("Only channel-preserving spatial calls are supported")
        for d in range(channel + 1):
            relations.append(equal(x, d, y, d, ctx.node.name))
    for tensor in (*ctx.inputs, *outputs):
        constraints.extend(
            AxisBarrier(
                tensor.axis(d), "Spatial resizing needs a separate coordinate rule", ctx.node.name
            )
            for d in range(channel + 1, len(tensor.shape))
        )
    # Pooling/interpolation can select different layouts by backend and grad
    # path. Meta remains useful for shape checks, not for a compact stride proof.
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def padding(ctx: OperationContext) -> OperatorSpec:
    """Preserve only axes untouched by padding, independently of their names/size."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    pad = ctx.module.padding if ctx.module is not None else ctx.argument("pad", 1)
    if isinstance(pad, int) and ctx.module is not None:
        spatial = next(n for n in (1, 2, 3) if f"{n}d" in type(ctx.module).__name__.lower())
        pad = (pad,) * (2 * spatial)
    if not isinstance(pad, (tuple, list)) or len(pad) % 2 or len(pad) > 2 * len(x.shape):
        raise UnsupportedOperation("Padding requires explicit axis pairs")
    changed = {len(x.shape) - 1 - i // 2 for i, value in enumerate(pad) if value != 0}
    # Equal final sizes do not establish identity: (-1, 1) crops one position
    # and appends another. Protect that axis on both sides, allowing other axes.
    return OperatorSpec(
        tuple(equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)) if d not in changed),
        tuple(
            AxisBarrier(t.axis(d), "Padded/cropped coordinates are fixed", ctx.node.name)
            for t in (x, y)
            for d in sorted(changed)
        ),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def instance_norm(ctx: OperationContext) -> OperatorSpec:
    """Bind affine/statistic channels for batched and unbatched InstanceNorm."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    if ctx.module is None:
        channel = 1
    else:
        spatial = {nn.InstanceNorm1d: 1, nn.InstanceNorm2d: 2, nn.InstanceNorm3d: 3}[
            type(ctx.module)
        ]
        channel = len(x.shape) - spatial - 1
    relations = [ReshapeRelation(x, y, ctx.node.name)]
    for name, position in (("running_mean", 1), ("running_var", 2), ("weight", 3), ("bias", 4)):
        value = bind(ctx, name, position)
        if value is not None:
            relations.append(equal(x, channel, one(value), 0, ctx.node.name))
    requirements = () if ctx.module is None else (requirement(ctx, "num_features", x, channel),)
    return OperatorSpec(tuple(relations), (*affine_layouts(ctx), layout(ctx, y)), requirements)


def prelu(ctx: OperationContext) -> OperatorSpec:
    """Bind per-channel PReLU weights; leave the shared scalar slope unchanged."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    weight = one(bind(ctx, "weight", 1))
    relations = [ReshapeRelation(x, y, ctx.node.name)]
    constraints, requirements = [], []
    if weight.shape == (1,):
        constraints.append(
            AxisBarrier(weight.axis(0), "A scalar PReLU slope is shared", ctx.node.name)
        )
    else:
        channel = 1 if len(x.shape) > 1 else 0
        relations.append(equal(x, channel, weight, 0, ctx.node.name))
        if ctx.module is not None:
            requirements.append(requirement(ctx, "num_parameters", x, channel))
    return OperatorSpec(tuple(relations), tuple(constraints), tuple(requirements))


def embedding_preflight(node: torch.fx.Node, module: nn.Module | None) -> None:
    """Reject Embedding renormalization before it can modify original parameters."""
    max_norm = (
        module.max_norm
        if module is not None
        else node.kwargs.get("max_norm", node.args[3] if len(node.args) > 3 else None)
    )
    if max_norm is not None:
        raise CaptureError(f"{node.name}: Embedding max_norm writes parameters during forward")


def embedding(ctx: OperationContext) -> OperatorSpec:
    """Expose embedding feature width while fixing vocabulary IDs and row positions."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    weight = one(bind(ctx, "weight", 1))
    relations = [equal(weight, 1, y, -1, ctx.node.name)]
    relations.extend(equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)))
    constraints = (
        AxisBarrier(weight.axis(0), "Vocabulary/token-ID remapping is unsupported", ctx.node.name),
    )
    requirements = () if ctx.module is None else (requirement(ctx, "embedding_dim", weight, 1),)
    candidates = (
        () if ctx.module is None else (CandidateAxis(f"{weight.paths[0]}:1", weight.axis(1)),)
    )
    return OperatorSpec(
        tuple(relations),
        constraints,
        requirements,
        candidates=candidates,
        contract=OutputContract(output_layout="contiguous"),
    )


def normalize(ctx: OperationContext) -> OperatorSpec:
    """Preserve coordinates while recomputing the norm over the compact domain."""
    result = pointwise(ctx)
    x = one(ctx.argument("input", 0))
    return replace(
        result,
        requirements=(
            Requirement(
                "reduction_domain",
                ctx.node.name,
                (x,),
                "Recompute normalization on retained positions; zero-mask equivalence is not assumed",
            ),
        ),
    )


def cast(ctx: OperationContext) -> OperatorSpec:
    """Preserve input coordinates; a dtype/device reference contributes no axes."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    copy_output = False
    if ctx.node.target == "to":
        position = (
            3 if len(ctx.args) > 1 and isinstance(ctx.args[1], (torch.dtype, TensorRef)) else 4
        )
        copy_output = ctx.argument("copy", position, False)
        if type(copy_output) is not bool:
            raise UnsupportedOperation("Tensor.to copy must be a static boolean")
    return OperatorSpec(
        (ReshapeRelation(x, y, ctx.node.name),),
        contract=OutputContract(output_layout="cast", copy_output=copy_output),
    )


def register_extended(
    registry: OperatorRegistrar,
    modules: Callable[..., None],
    functions: Callable[..., None],
    methods: Callable[..., None],
) -> None:
    """Register common exact public API spellings through the unified interface."""
    modules([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d], instance_norm, fresh=True)
    functions([F.instance_norm], instance_norm, fresh=True)
    modules([nn.RMSNorm], layer_norm, fresh=True)
    functions([F.rms_norm], layer_norm, fresh=True)
    modules([nn.PReLU], prelu, fresh=True)
    functions([F.prelu], prelu, fresh=True)
    functions([F.normalize], normalize, fresh=True)
    registry.register(
        nn.Embedding,
        OperatorRule(
            embedding,
            preflight=embedding_preflight,
            evaluate_on_meta=True,
            effects=partial(native_effects, fresh_output=True),
        ),
    )
    registry.register(
        F.embedding,
        OperatorRule(
            embedding,
            preflight=embedding_preflight,
            evaluate_on_meta=True,
            effects=partial(native_effects, fresh_output=True),
        ),
        opaque=False,
    )
    modules(
        [
            nn.MaxPool1d,
            nn.MaxPool2d,
            nn.MaxPool3d,
            nn.AvgPool1d,
            nn.AvgPool2d,
            nn.AvgPool3d,
            nn.AdaptiveAvgPool1d,
            nn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool3d,
            nn.AdaptiveMaxPool1d,
            nn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool3d,
            nn.MaxUnpool1d,
            nn.MaxUnpool2d,
            nn.MaxUnpool3d,
            nn.Upsample,
        ],
        channel_operator,
        fresh=True,
    )
    modules(
        [
            nn.ReflectionPad1d,
            nn.ReflectionPad2d,
            nn.ReflectionPad3d,
            nn.ReplicationPad1d,
            nn.ReplicationPad2d,
            nn.ReplicationPad3d,
            nn.ConstantPad1d,
            nn.ConstantPad2d,
            nn.ConstantPad3d,
            nn.ZeroPad2d,
        ],
        padding,
        fresh=True,
    )
    functions(
        [
            F.max_pool1d,
            F.max_pool2d,
            F.max_pool3d,
            F.avg_pool1d,
            F.avg_pool2d,
            F.avg_pool3d,
            F.max_pool1d_with_indices,
            F.max_pool2d_with_indices,
            F.max_pool3d_with_indices,
            F.adaptive_avg_pool1d,
            F.adaptive_avg_pool2d,
            F.adaptive_avg_pool3d,
            F.adaptive_max_pool1d,
            F.adaptive_max_pool2d,
            F.adaptive_max_pool3d,
            F.adaptive_max_pool1d_with_indices,
            F.adaptive_max_pool2d_with_indices,
            F.adaptive_max_pool3d_with_indices,
            F.max_unpool1d,
            F.max_unpool2d,
            F.max_unpool3d,
            F.interpolate,
        ],
        channel_operator,
        fresh=True,
    )
    functions([F.pad], padding, fresh=True)
    unary = [
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "abs",
        "square",
        "reciprocal",
        "floor",
        "ceil",
        "round",
        "expm1",
        "log1p",
        "erf",
        "sign",
        "sinh",
        "cosh",
    ]
    binary = [
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "logical_and",
        "logical_or",
        "logical_xor",
        "pow",
        "maximum",
        "minimum",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "where",
        "clamp",
        "clamp_min",
        "clamp_max",
    ]
    functions([getattr(torch, name) for name in (*unary, *binary)], pointwise, fresh=True)
    functions(
        [
            operator.pow,
            operator.eq,
            operator.ne,
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.abs,
            operator.and_,
            operator.or_,
            operator.xor,
        ],
        pointwise,
        fresh=True,
    )
    functions([operator.pos, F.dropout1d, F.dropout2d, F.dropout3d], pointwise, fresh=False)
    methods(
        [name for name in (*unary, *binary) if hasattr(torch.Tensor, name)], pointwise, fresh=True
    )
    methods(["masked_fill"], pointwise, fresh=True)
    methods(
        [
            "to",
            "type_as",
            "float",
            "double",
            "half",
            "bfloat16",
            "bool",
            "long",
            "int",
            "cpu",
            "cuda",
        ],
        cast,
        fresh=False,
    )
    modules(
        [
            nn.ELU,
            nn.CELU,
            nn.SELU,
            nn.LeakyReLU,
            nn.Mish,
            nn.Hardswish,
            nn.Hardsigmoid,
            nn.Hardtanh,
            nn.Softplus,
            nn.Softsign,
            nn.Tanhshrink,
            nn.Threshold,
        ],
        pointwise,
        fresh=True,
    )
    functions(
        [
            F.elu,
            F.celu,
            F.selu,
            F.leaky_relu,
            F.mish,
            F.hardswish,
            F.hardsigmoid,
            F.hardtanh,
            F.softplus,
            F.softsign,
            F.tanhshrink,
            F.threshold,
        ],
        pointwise,
        fresh=True,
    )
    functions([torch.amax, torch.amin, torch.prod, torch.logsumexp], reduction, fresh=True)
    methods(["amax", "amin", "prod", "logsumexp"], reduction, fresh=True)
