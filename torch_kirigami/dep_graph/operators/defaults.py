"""Explicit built-in registrations; no semantic callback wrapping."""

import builtins
import operator
from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..operation import OperatorRule
from .rules import (
    AddmmRule,
    AttentionRule,
    AttributeReadRule,
    BatchNormRule,
    CastRule,
    ChannelOperatorRule,
    ChannelShuffleRule,
    ChunkRule,
    ConcatenateRule,
    ConvolutionRule,
    EinsumRule,
    EmbeddingRule,
    GluRule,
    GroupNormRule,
    IdentityRule,
    IndexRule,
    IndexSelectRule,
    InstanceNormRule,
    LayerNormRule,
    LinearRule,
    MatmulRule,
    MultiheadAttentionRule,
    NarrowRule,
    NativeRule,
    NormalizeRule,
    PaddingRule,
    PermuteRule,
    PixelShuffleRule,
    PointwiseRule,
    PreluRule,
    ReductionRule,
    RepeatRule,
    ReshapeRule,
    ShapeOnlyRule,
    SoftmaxRule,
    SplitRule,
    StackRule,
    UnfoldFoldRule,
)


def default_rules() -> tuple[tuple[type[nn.Module] | Callable | str, OperatorRule, bool], ...]:
    """Return fresh registration entries for a local registry."""
    entries: list[tuple[type[nn.Module] | Callable | str, OperatorRule, bool]] = []

    def modules(targets: Sequence[type[nn.Module]], rule: type[NativeRule], *, fresh: bool) -> None:
        """Register exact module classes as FX leaves."""
        entries.extend((target, rule(fresh=fresh), True) for target in targets)

    def functions(targets: Sequence[Callable], rule: type[NativeRule], *, fresh: bool) -> None:
        """Register native function targets without changing FX wrapping."""
        entries.extend((target, rule(fresh=fresh), False) for target in targets)

    def methods(targets: Sequence[str], rule: type[NativeRule], *, fresh: bool) -> None:
        """Register Tensor method spellings."""
        entries.extend((target, rule(fresh=fresh), False) for target in targets)

    modules([nn.Softmax, nn.LogSoftmax], SoftmaxRule, fresh=True)
    functions([F.softmax, F.log_softmax, torch.softmax, torch.log_softmax], SoftmaxRule, fresh=True)
    methods(["softmax", "log_softmax"], SoftmaxRule, fresh=True)
    modules([nn.Linear], LinearRule, fresh=True)
    functions([F.linear], LinearRule, fresh=True)
    modules(
        [
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ],
        ConvolutionRule,
        fresh=True,
    )
    functions(
        [F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d],
        ConvolutionRule,
        fresh=True,
    )
    modules([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d], BatchNormRule, fresh=True)
    functions([F.batch_norm], BatchNormRule, fresh=True)
    modules([nn.LayerNorm], LayerNormRule, fresh=True)
    functions([F.layer_norm], LayerNormRule, fresh=True)
    modules([nn.GroupNorm], GroupNormRule, fresh=True)
    functions([F.group_norm], GroupNormRule, fresh=True)
    modules([nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d], PointwiseRule, fresh=False)
    functions([F.dropout], PointwiseRule, fresh=False)
    methods(["detach", "contiguous"], PointwiseRule, fresh=False)
    modules([nn.Identity], IdentityRule, fresh=False)
    modules([nn.Flatten, nn.Unflatten], ReshapeRule, fresh=False)
    modules(
        [
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.Sigmoid,
            nn.Tanh,
        ],
        PointwiseRule,
        fresh=True,
    )
    functions(
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.neg,
            torch.add,
            torch.sub,
            torch.mul,
            torch.div,
            torch.neg,
            torch.relu,
            torch.sigmoid,
            torch.tanh,
            torch.clone,
            F.relu,
            F.relu6,
            F.gelu,
            F.silu,
        ],
        PointwiseRule,
        fresh=True,
    )
    methods(
        [
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "relu",
            "sigmoid",
            "tanh",
            "clone",
            "relu_",
            "add_",
            "mul_",
        ],
        PointwiseRule,
        fresh=True,
    )
    functions([operator.matmul, torch.matmul, torch.mm, torch.bmm], MatmulRule, fresh=True)
    methods(["matmul", "mm", "bmm"], MatmulRule, fresh=True)
    functions(
        [
            torch.permute,
            torch.transpose,
            torch.swapaxes,
            torch.swapdims,
            torch.t,
            torch.movedim,
            torch.moveaxis,
        ],
        PermuteRule,
        fresh=False,
    )
    methods(
        ["permute", "transpose", "swapaxes", "swapdims", "t", "movedim", "moveaxis"],
        PermuteRule,
        fresh=False,
    )
    functions(
        [torch.reshape, torch.flatten, torch.squeeze, torch.unsqueeze, torch.unflatten],
        ReshapeRule,
        fresh=False,
    )
    methods(
        ["reshape", "view", "flatten", "squeeze", "unsqueeze", "unflatten"],
        ReshapeRule,
        fresh=False,
    )
    functions([torch.cat, torch.concat, torch.concatenate], ConcatenateRule, fresh=True)
    functions([torch.split, torch.unbind], SplitRule, fresh=False)
    methods(["split", "unbind"], SplitRule, fresh=False)
    functions([operator.getitem], IndexRule, fresh=False)
    # The wrapper extracts proven scalar size expressions first. Tensor overloads
    # must retain PointwiseRule dependencies instead of silently emitting an empty rule.
    functions(
        [operator.floordiv, operator.mod, torch.floor_divide, torch.remainder],
        PointwiseRule,
        fresh=True,
    )
    methods(["floor_divide", "remainder"], PointwiseRule, fresh=True)
    functions([builtins.getattr], AttributeReadRule, fresh=False)
    functions([torch.sum, torch.mean], ReductionRule, fresh=True)
    methods(["sum", "mean"], ReductionRule, fresh=True)
    methods(["size", "dim", "numel"], ShapeOnlyRule, fresh=False)

    modules([nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d], InstanceNormRule, fresh=True)
    functions([F.instance_norm], InstanceNormRule, fresh=True)
    modules([nn.RMSNorm], LayerNormRule, fresh=True)
    functions([F.rms_norm], LayerNormRule, fresh=True)
    modules([nn.PReLU], PreluRule, fresh=True)
    functions([F.prelu], PreluRule, fresh=True)
    functions([F.normalize], NormalizeRule, fresh=True)
    modules([nn.Embedding], EmbeddingRule, fresh=True)
    functions([F.embedding], EmbeddingRule, fresh=True)
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
        ChannelOperatorRule,
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
        PaddingRule,
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
        ChannelOperatorRule,
        fresh=True,
    )
    functions([F.pad], PaddingRule, fresh=True)
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
    functions([getattr(torch, name) for name in (*unary, *binary)], PointwiseRule, fresh=True)
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
        PointwiseRule,
        fresh=True,
    )
    functions([operator.pos, F.dropout1d, F.dropout2d, F.dropout3d], PointwiseRule, fresh=False)
    methods(
        [name for name in (*unary, *binary) if hasattr(torch.Tensor, name)],
        PointwiseRule,
        fresh=True,
    )
    methods(["masked_fill"], PointwiseRule, fresh=True)
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
        CastRule,
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
        PointwiseRule,
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
        PointwiseRule,
        fresh=True,
    )
    functions([torch.amax, torch.amin, torch.prod, torch.logsumexp], ReductionRule, fresh=True)
    methods(["amax", "amin", "prod", "logsumexp"], ReductionRule, fresh=True)

    functions([torch.stack], StackRule, fresh=True)
    functions([torch.chunk], ChunkRule, fresh=False)
    methods(["chunk"], ChunkRule, fresh=False)
    functions([torch.narrow], NarrowRule, fresh=False)
    methods(["narrow"], NarrowRule, fresh=False)
    functions([torch.index_select], IndexSelectRule, fresh=True)
    methods(["index_select"], IndexSelectRule, fresh=True)
    functions([torch.tile, torch.repeat_interleave], RepeatRule, fresh=True)
    functions([torch.broadcast_to], RepeatRule, fresh=False)
    methods(["repeat", "tile", "repeat_interleave"], RepeatRule, fresh=True)
    methods(["expand", "expand_as"], RepeatRule, fresh=False)
    modules([nn.GLU], GluRule, fresh=True)
    functions([F.glu], GluRule, fresh=True)
    modules([nn.ChannelShuffle], ChannelShuffleRule, fresh=True)
    functions([torch.channel_shuffle], ChannelShuffleRule, fresh=True)
    modules([nn.PixelShuffle, nn.PixelUnshuffle], PixelShuffleRule, fresh=True)
    functions(
        list(
            dict.fromkeys(
                [F.pixel_shuffle, F.pixel_unshuffle, torch.pixel_shuffle, torch.pixel_unshuffle]
            )
        ),
        PixelShuffleRule,
        fresh=True,
    )
    modules([nn.Unfold, nn.Fold], UnfoldFoldRule, fresh=False)
    functions([F.unfold, F.fold], UnfoldFoldRule, fresh=False)

    functions([torch.addmm, torch.baddbmm], AddmmRule, fresh=True)
    methods(["addmm", "baddbmm"], AddmmRule, fresh=True)
    # EinsumRule distinguishes product allocation from possible single-input views.
    functions([torch.einsum], EinsumRule, fresh=False)
    functions([F.scaled_dot_product_attention], AttentionRule, fresh=True)
    modules([nn.MultiheadAttention], MultiheadAttentionRule, fresh=True)

    return tuple(entries)
