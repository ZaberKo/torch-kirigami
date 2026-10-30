"""Shape formulas for nominal real arithmetic, independent of pruning."""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.flop_counter import flop_registry


def is_native_pointwise(name: str) -> bool:
    """Identify ATen packets whose overloads all declare pointwise semantics.

    Pointwise work is outside the dense matrix/convolution MAC convention.
    These tags do not determine FLOPs. Require every overload to carry the tag
    because profiler names identify packets rather than individual overloads.
    """
    if not name.startswith("aten::"):
        return False
    packet = getattr(torch.ops.aten, name.removeprefix("aten::"), None)
    if packet is None or not callable(getattr(packet, "overloads", None)):
        return False
    overloads = packet.overloads()
    return bool(overloads) and all(
        torch.Tag.pointwise in getattr(packet, overload).tags for overload in overloads
    )


def elementwise_flops(*args: Any, out_shape: tuple[int, ...], **kwargs: Any) -> int:
    """Count one arithmetic operation per output element, including broadcasting."""
    return math.prod(out_shape)


def add_flops(*args: Any, out_shape: tuple[int, ...], alpha: float = 1, **kwargs: Any) -> int:
    """Count addition and an explicit nonunit scale of the second operand."""
    return math.prod(out_shape) * (1 + (alpha != 1))


def add_product_flops(
    *args: Any, out_shape: tuple[int, ...], value: float = 1, **kwargs: Any
) -> int:
    """Count addcmul/addcdiv arithmetic and an optional nonunit scalar scale."""
    return math.prod(out_shape) * (2 + (value != 1))


def sum_flops(input: tuple[int, ...], *args: Any, out_shape: tuple[int, ...], **kwargs: Any) -> int:
    """Count one fewer addition than input elements in each reduction group."""
    return max(0, math.prod(input) - math.prod(out_shape))


def mean_flops(
    input: tuple[int, ...], *args: Any, out_shape: tuple[int, ...], **kwargs: Any
) -> int:
    """Count reduction additions and one division per output."""
    return sum_flops(input, out_shape=out_shape) + math.prod(out_shape)


def softmax_flops(input: tuple[int, ...], dim: int, *args: Any, **kwargs: Any) -> int:
    """Count stable softmax arithmetic, excluding maximum comparisons."""
    size = input[dim] if input else 1
    total = math.prod(input)
    return 4 * total - total // size if size else 0


def log_softmax_flops(input: tuple[int, ...], *args: Any, **kwargs: Any) -> int:
    """Count stable log-softmax arithmetic with one unit per exp/log."""
    return 4 * math.prod(input)


def normalization_flops(total: int, groups: int, weight: object, bias: object) -> int:
    """Count centered variance with reused centered values and optional affine work."""
    return 5 * total + 2 * groups + total * ((weight is not None) + (bias is not None))


def layer_norm_flops(
    input: tuple[int, ...],
    normalized_shape: tuple[int, ...],
    weight: object,
    bias: object,
    *args: Any,
    **kwargs: Any,
) -> int | None:
    """Count mean, centered variance, epsilon, rsqrt and affine arithmetic."""
    width = math.prod(normalized_shape)
    return (
        normalization_flops(math.prod(input), math.prod(input) // width, weight, bias)
        if width
        else None
    )


def group_norm_flops(
    input: tuple[int, ...],
    weight: object,
    bias: object,
    N: int,
    C: int,
    HxW: int,
    group: int,
    *args: Any,
    **kwargs: Any,
) -> int:
    """Count one normalization domain per sample and group."""
    return normalization_flops(math.prod(input), N * group, weight, bias)


def batch_norm_eval_flops(
    input: tuple[int, ...],
    weight: object,
    bias: object,
    *args: Any,
    **kwargs: Any,
) -> int:
    """Count inference normalization from stored statistics, without algebraic folding."""
    total = math.prod(input)
    return 2 * input[1] + total * (2 + (weight is not None) + (bias is not None))


def batch_norm_flops(
    input: tuple[int, ...],
    weight: object,
    bias: object,
    running_mean: object = None,
    running_var: object = None,
    training: bool = True,
    *args: Any,
    **kwargs: Any,
) -> int | None:
    """Count stored-statistic inference only; report training updates as uncovered."""
    # The no_stats overload places training/momentum/eps after bias. Those
    # arguments are not running-statistic shapes and are outside this formula.
    if training or not isinstance(running_mean, tuple) or not isinstance(running_var, tuple):
        return None
    return batch_norm_eval_flops(input, weight, bias)


def dot_flops(a: tuple[int, ...], b: tuple[int, ...], **kwargs: Any) -> int:
    """Count a vector contraction under the dense two-FLOPs-per-MAC convention."""
    return 2 * a[0]


def matrix_vector_flops(matrix: tuple[int, ...], vector: tuple[int, ...], **kwargs: Any) -> int:
    """Count matrix-vector work without bias additions or scalar scaling."""
    return 2 * math.prod(matrix)


def biased_matrix_vector_flops(
    bias: tuple[int, ...], matrix: tuple[int, ...], vector: tuple[int, ...], **kwargs: Any
) -> int:
    """Use the same dense matrix-vector formula for the biased form."""
    return matrix_vector_flops(matrix, vector)


def conv_flops(input: tuple[int, ...], weight: tuple[int, ...], output: tuple[int, ...]) -> int:
    """Count ordinary convolution products, including implicit padding positions."""
    return 2 * input[0] * math.prod(weight) * math.prod(output[2:])


def cudnn_convolution_flops(
    input: tuple[int, ...],
    weight: tuple[int, ...],
    padding: object,
    stride: object,
    dilation: object,
    groups: int,
    benchmark: bool,
    deterministic: bool,
    allow_tf32: bool,
    *,
    out_shape: tuple[int, ...],
    **kwargs: Any,
) -> int:
    """Bind cuDNN's distinct signature without interpreting benchmark as transposed."""
    return conv_flops(input, weight, out_shape)


def slow_conv2d_flops(
    input: tuple[int, ...],
    weight: tuple[int, ...],
    kernel_size: object,
    bias: object,
    stride: object,
    padding: object,
    *,
    out_shape: tuple[int, ...],
    **kwargs: Any,
) -> int:
    """Bind the six-argument CPU convolution signature."""
    return conv_flops(input, weight, out_shape)


def efficient_attention_flops(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    *args: Any,
    **kwargs: Any,
) -> int:
    """Count dense BMHD attention; retain the native variable-length formula.

    This raw kernel uses batch, sequence, heads, and width order. The public
    SDPA kernels use batch, heads, sequence, and width order instead.
    """
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        return flop_registry[torch.ops.aten._efficient_attention_forward](
            query, key, value, bias, cu_seqlens_q, cu_seqlens_k, *args, **kwargs
        )
    batch, queries, heads, width = query.shape
    return 2 * batch * heads * queries * key.shape[1] * (width + value.shape[-1])


def arithmetic_formulas() -> dict[object, Callable[..., int | None]]:
    """Return local additions to the counter, without changing its global registry.

    None means the formula does not cover this invocation. Comparisons and
    data movement do not contribute floating-point arithmetic.
    """
    result = {
        getattr(torch.ops.aten, name): elementwise_flops
        for name in (
            "mul",
            "mul_",
            "div",
            "div_",
            "divide",
            "divide_",
            "true_divide",
            "true_divide_",
            "multiply",
            "multiply_",
            "neg",
            "neg_",
            "negative",
            "negative_",
            "abs",
            "abs_",
            "absolute",
            "absolute_",
            "sqrt",
            "sqrt_",
            "rsqrt",
            "rsqrt_",
            "exp",
            "exp_",
            "log",
            "log_",
            "reciprocal",
            "reciprocal_",
            "square",
            "square_",
            "pow",
            "pow_",
        )
    }
    result.update(
        {
            torch.ops.aten.add: add_flops,
            torch.ops.aten.add_: add_flops,
            torch.ops.aten.sub: add_flops,
            torch.ops.aten.sub_: add_flops,
            torch.ops.aten.subtract: add_flops,
            torch.ops.aten.subtract_: add_flops,
            torch.ops.aten.rsub: add_flops,
            torch.ops.aten.addcmul: add_product_flops,
            torch.ops.aten.addcmul_: add_product_flops,
            torch.ops.aten.addcdiv: add_product_flops,
            torch.ops.aten.addcdiv_: add_product_flops,
            torch.ops.aten.sum: sum_flops,
            torch.ops.aten.mean: mean_flops,
            torch.ops.aten._softmax: softmax_flops,
            torch.ops.aten._safe_softmax: softmax_flops,
            torch.ops.aten._log_softmax: log_softmax_flops,
            torch.ops.aten.native_layer_norm: layer_norm_flops,
            torch.ops.aten.native_group_norm: group_norm_flops,
            torch.ops.aten._native_batch_norm_legit_no_training: batch_norm_eval_flops,
            torch.ops.aten.native_batch_norm: batch_norm_flops,
            torch.ops.aten._native_batch_norm_legit: batch_norm_flops,
            torch.ops.aten.cudnn_batch_norm: batch_norm_flops,
        }
    )
    return result


def matrix_extra_flops(op: str, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> int:
    """Count fused bias and scalar scaling separately from the dense MAC formula."""
    if op in (
        "aten::addmm",
        "aten::addmm_",
        "aten::baddbmm",
        "aten::baddbmm_",
        "aten::addmv",
        "aten::addmv_",
        "aten::addbmm",
        "aten::addbmm_",
    ):
        beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
        return output.numel() * ((beta != 0) + (beta not in (0, 1)) + (alpha != 1))
    positions = {
        "aten::convolution": 2,
        "aten::_convolution": 2,
        "aten::convolution_overrideable": 2,
        "aten::_slow_conv2d_forward": 3,
    }
    if op in positions:
        bias = args[positions[op]] if len(args) > positions[op] else kwargs.get("bias")
        return output.numel() if bias is not None else 0
    return 0


# Only data movement, indexing, comparisons and selection are free in the
# arithmetic convention. Normalization and activation are not silently free.
FLOP_FREE_OPS = frozenset(
    f"aten::{name}"
    for name in (
        "t",
        "transpose",
        "permute",
        "view",
        "_unsafe_view",
        "reshape",
        "flatten",
        "unflatten",
        "unsqueeze",
        "squeeze",
        "squeeze_",
        "unsqueeze_",
        "expand",
        "expand_as",
        "repeat",
        "repeat_interleave",
        "slice",
        "select",
        "narrow",
        "split",
        "split_with_sizes",
        "chunk",
        "cat",
        "stack",
        "unbind",
        "as_strided",
        "as_strided_",
        "contiguous",
        "clone",
        "copy_",
        "detach",
        "alias",
        "to",
        "_to_copy",
        "type_as",
        "result_type",
        "resolve_conj",
        "resolve_neg",
        "empty",
        "empty_like",
        "empty_strided",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "full",
        "full_like",
        "fill_",
        "zero_",
        "resize_",
        "arange",
        "scalar_tensor",
        "lift_fresh",
        "embedding",
        "index",
        "index_select",
        "gather",
        "masked_fill",
        "masked_fill_",
        "triu",
        "tril",
        "_nnpack_available",
        "_local_scalar_dense",
        "item",
        "is_nonzero",
        "_reshape_alias",
        "_autocast_to_reduced_precision",
        "_autocast_to_full_precision",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "all",
        "any",
        "logical_not",
        "logical_and",
        "where",
        "isneginf",
        "isinf",
        "isnan",
        "relu",
        "relu_",
        "clamp",
        "clamp_min",
        "clamp_min_",
        "amax",
        "amin",
        "max",
        "min",
        # Forward max pooling selects values and indices. Comparisons are free
        # under this convention; backward scatter/add kernels are not included.
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "max_pool1d_with_indices",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        # Coordinate transforms and padding only copy, select, or fill values.
        "pixel_shuffle",
        "pixel_unshuffle",
        "channel_shuffle",
        "native_channel_shuffle",
        "unfold",
        "im2col",
        "max_unpool2d",
        "max_unpool3d",
        "pad",
        "constant_pad_nd",
        "_pad_circular",
        "reflection_pad1d",
        "reflection_pad2d",
        "reflection_pad3d",
        "replication_pad1d",
        "replication_pad2d",
        "replication_pad3d",
    )
)

FLOP_COMPOSITES = frozenset(
    f"aten::{name}"
    for name in (
        "softmax",
        "_safe_softmax",
        "square",
        "square_",
        "absolute",
        "absolute_",
        "negative",
        "negative_",
        "multiply",
        "multiply_",
        "subtract",
        "subtract_",
        "divide",
        "divide_",
        "true_divide",
        "true_divide_",
        "log_softmax",
        "batch_norm",
        "_batch_norm_impl_index",
        "native_batch_norm",
        "layer_norm",
        "group_norm",
    )
)


# Only these audited forward formulas use two FLOPs per dense MAC. Do not
# convert the counter's total: its registry also accepts arbitrary FLOP formulas.
MAC_FORMULAS = frozenset(
    f"aten::{name}"
    for name in (
        "mm",
        "addmm",
        "addmm_",
        "bmm",
        "baddbmm",
        "baddbmm_",
        "addbmm",
        "addbmm_",
        "dot",
        "vdot",
        "mv",
        "addmv",
        "addmv_",
        "_scaled_mm",
        "convolution",
        "_convolution",
        "cudnn_convolution",
        "_slow_conv2d_forward",
        "convolution_overrideable",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_scaled_dot_product_flash_attention_for_cpu",
        "_flash_attention_forward",
        "_efficient_attention_forward",
    )
)

# These wrappers contribute through their constituent matrix/conv calls, not
# through formulas registered directly on the wrappers.
MAC_COMPOSITES = frozenset(
    f"aten::{name}"
    for name in (
        "linear",
        "matmul",
        "einsum",
        "conv1d",
        "conv2d",
        "conv3d",
        "_convolution_mode",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "scaled_dot_product_attention",
        "_scaled_dot_product_attention_math",
    )
)

# Explicit exclusions under the matrix/conv MAC convention. Unfamiliar kernels
# must appear in uncovered_ops["macs"], not silently become zero-cost operations.
MAC_FREE_OPS = frozenset(
    f"aten::{name}"
    for name in (
        "add",
        "add_",
        "sub",
        "mul",
        "div",
        "div_",
        "neg",
        "pow",
        "sqrt",
        "rsqrt",
        "abs",
        "exp",
        "log",
        "tanh",
        "sigmoid",
        "relu",
        "relu_",
        "gelu",
        "silu",
        "clamp_min",
        "clamp_min_",
        "clamp",
        "where",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "all",
        "any",
        "logical_not",
        "logical_and",
        "softmax",
        "_softmax",
        "_safe_softmax",
        "log_softmax",
        "_log_softmax",
        "batch_norm",
        "_batch_norm_impl_index",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit_no_training",
        "native_batch_norm",
        "cudnn_batch_norm",
        "miopen_batch_norm",
        "layer_norm",
        "native_layer_norm",
        "group_norm",
        "native_group_norm",
        "avg_pool1d",
        "avg_pool2d",
        "avg_pool3d",
        "adaptive_avg_pool1d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
        "_adaptive_avg_pool2d",
        "_adaptive_avg_pool3d",
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "max_pool1d_with_indices",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "mean",
        "sum",
        "amax",
        "amin",
        "max",
        "min",
        "t",
        "transpose",
        "permute",
        "view",
        "_unsafe_view",
        "reshape",
        "flatten",
        "unflatten",
        "unsqueeze",
        "squeeze",
        "squeeze_",
        "unsqueeze_",
        "lgamma",
        "special_gammaln",
        "expand",
        "expand_as",
        "repeat",
        "repeat_interleave",
        "slice",
        "select",
        "narrow",
        "split",
        "split_with_sizes",
        "chunk",
        "cat",
        "stack",
        "unbind",
        "as_strided",
        "as_strided_",
        "contiguous",
        "clone",
        "copy_",
        "detach",
        "alias",
        "to",
        "_to_copy",
        "type_as",
        "resolve_conj",
        "resolve_neg",
        "empty",
        "empty_like",
        "empty_strided",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "full",
        "full_like",
        "fill_",
        "zero_",
        "resize_",
        "arange",
        "scalar_tensor",
        "lift_fresh",
        "dropout",
        "native_dropout",
        "embedding",
        "index",
        "index_select",
        "gather",
        "masked_fill",
        "isneginf",
        "isinf",
        "isnan",
        "masked_fill_",
        "triu",
        "tril",
        "_nnpack_available",
        "_local_scalar_dense",
        "item",
        "is_nonzero",
        "_reshape_alias",
        "_autocast_to_reduced_precision",
        "_autocast_to_full_precision",
        # Fold adds overlapping patches but performs no dense matrix product.
        "col2im",
    )
)
