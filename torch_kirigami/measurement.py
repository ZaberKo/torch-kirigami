"""Inference complexity and CPU/CUDA timing, independent of pruning policy."""

import gc
import math
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from .bindings import ordinary_tensors, storage_key
from .capture import isolated_execution, tensor_leaves

__all__ = [
    "ModelComplexity",
    "calculate_model_complexity",
    "count_parameters",
    "measure_module_latency",
]


def count_parameters(model: nn.Module) -> int:
    """Count elements in unique registered Parameters without executing the model.

    Includes frozen and unused parameters; excludes buffers. Multiple names for
    the same Parameter count once. Distinct Parameters sharing storage remain
    distinct entities. No device transfer, sample input, or FX graph is needed.
    Counts tensor sizes, not nonzero values: masking weights does not lower it.
    """
    return sum(parameter.numel() for parameter in model.parameters())


# Only these audited forward formulas use two FLOPs per dense MAC. Do not
# convert the counter's total: its registry also accepts arbitrary FLOP formulas.
_MAC_FORMULAS = frozenset(
    f"aten::{name}"
    for name in (
        "mm",
        "addmm",
        "bmm",
        "baddbmm",
        "_scaled_mm",
        "convolution",
        "_convolution",
        "cudnn_convolution",
        "_slow_conv2d_forward",
        "convolution_overrideable",
        "_scaled_dot_product_efficient_attention",
        "_scaled_dot_product_flash_attention",
        "_scaled_dot_product_cudnn_attention",
        "_flash_attention_forward",
        "_efficient_attention_forward",
    )
)

# These wrappers contribute through their constituent matrix/conv calls, not
# through formulas registered directly on the wrappers.
_MAC_COMPOSITES = frozenset(
    f"aten::{name}"
    for name in (
        "linear",
        "matmul",
        "einsum",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "scaled_dot_product_attention",
        "_scaled_dot_product_attention_math",
    )
)

# Explicit exclusions under the matrix/conv MAC convention. Unfamiliar kernels
# must appear in unsupported_ops, not silently become zero-cost operations.
_MAC_FREE_OPS = frozenset(
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
    )
)


@dataclass(frozen=True)
class ModelComplexity:
    """Recorded FLOPs, dense MACs and unique registered parameter elements.

    flops is the native counter's total for registered formulas, including any
    additional non-MAC formulas. It is not a complete model FLOP count. Both
    operation counts cover the entire supplied batch and executed path.

    One multiply-accumulate counts as one MAC (two FLOPs). Matrix products and
    convolutions count; bias, normalization, activation and other elementwise
    work do not. unsupported_ops lists operations with unverified MAC coverage,
    not these explicit exclusions; a nonempty list means completeness is unknown.
    This field describes MAC coverage only, not FLOP coverage: an empty tuple
    does not imply that all floating-point operations were counted.
    Parameters include frozen weights, deduplicated by Parameter identity.
    """

    macs: int
    params: int
    flops: int
    unsupported_ops: tuple[str, ...] = ()


def _to_device(value, device, memo):
    """Move one jointly isolated input tree, preserving repeated object identity."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        if value.device != device and value.numel():
            key = ("storage", storage_key(value))
            if key in memo:
                raise ValueError(
                    "Move storage-sharing input views to the measurement device together"
                )
            memo[key] = True
        result = value.to(device)
    elif isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_to_device(v, device, memo) for v in value)
    elif isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((k, _to_device(v, device, memo)) for k, v in value.items())
    elif isinstance(value, tuple):
        values = tuple(_to_device(v, device, memo) for v in value)
        result = type(value)(*values) if hasattr(value, "_fields") else values
    else:
        return value
    memo[id(value)] = result
    return result


@contextmanager
def _evaluation(target, input_args, input_kwargs, device) -> Iterator[tuple]:
    if not isinstance(target, nn.Module):
        raise TypeError("Measurement requires an nn.Module")
    tensors = (*target.parameters(), *target.buffers())
    device = torch.device(device if device is not None else tensors[0].device if tensors else "cpu")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Measurement supports CPU and CUDA only")
    if device.type == "cpu":
        device = torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
    if any(t.device != device for t in tensors):
        raise ValueError("Move all model parameters and buffers to the measurement device first")
    args = tuple(input_args) if isinstance(input_args, (tuple, list)) else (input_args,)
    kwargs = input_kwargs or {}
    constants = tuple(ordinary_tensors(target))
    identities = {id(t) for t in constants}
    storages = {storage_key(t) for t in constants if t.numel()}
    if any(
        t.device != device and (id(t) in identities or (t.numel() and storage_key(t) in storages))
        for t in tensor_leaves((args, kwargs))
    ):
        raise ValueError("Move model-shared inputs and ordinary attributes to the device together")
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context, isolated_execution(target, args, kwargs) as (args, kwargs, _):
        if any(t.device != device for t in tensor_leaves((args, kwargs))):
            args, kwargs = _to_device((args, kwargs), device, {})
        target.eval()
        yield args, kwargs, device


@torch.no_grad()
def calculate_model_complexity(
    target: nn.Module,
    input_args: Any,
    device: torch.device | str | None = None,
    input_kwargs: dict[str, Any] | None = None,
) -> ModelComplexity:
    """Record eager inference FLOPs, MACs and parameters without changing training state.

    Args:
        target: Eager module already on the chosen device; do not pass a compiled
            wrapper. Parameter values must not be mutated by its forward.
        input_args: One input or positional argument tuple/list. Nested tensor
            containers are moved to device and isolated from forward writes.
        device: CPU/CUDA device, inferred from model tensors when omitted.
        input_kwargs: Keyword inputs, including nested tensor containers.

    Returns:
        Counts for the entire supplied batch and explicit unsupported operations.
        FLOPs retain the native counter total; MAC coverage diagnostics do not
        certify FLOP completeness.
        Converts only audited native matrix/conv FLOP formulas to MACs. SDPA uses the math
        backend here so fused CPU attention cannot silently bypass counting.

    Model modes, registered buffer bindings/values, inputs and torch RNG state
    are restored even on failure. This does not isolate arbitrary Python side
    effects or parameter writes. Counting runs outside inference_mode; latency
    uses a separate, uninstrumented forward and the original attention backend.
    """
    with (
        torch.inference_mode(False),
        _evaluation(target, input_args, input_kwargs, device) as (args, kwargs, _),
    ):
        with (
            profile(activities=[ProfilerActivity.CPU]) as trace,
            sdpa_kernel(SDPBackend.MATH),
            FlopCounterMode(display=False) as counter,
        ):
            target(*args, **kwargs)
        counts = {
            str(op).replace("aten.", "aten::", 1): flops
            for op, flops in counter.get_flop_counts().get("Global", {}).items()
        }
        counted = counts.keys() & _MAC_FORMULAS
        macs = 0
        for op in counted:
            flops = counts[op]
            if type(flops) is not int or flops < 0 or flops % 2:
                raise ValueError(f"{op}: expected a nonnegative even FLOP count for dense MACs")
            macs += flops // 2
        # A registered composite formula may suppress decomposition. Its count
        # has no verified MAC meaning, so it cannot be ignored as a wrapper.
        unsupported = counts.keys() - counted - _MAC_FREE_OPS
        ignored = (_MAC_FREE_OPS | _MAC_COMPOSITES) - unsupported
        for event in trace.events():
            if "::" not in event.name or event.name in counted or event.name in ignored:
                continue
            parent = event.cpu_parent
            while parent is not None and parent.name not in counted:
                parent = parent.cpu_parent
            # Backend kernels below an already counted convolution/matmul do
            # not represent additional work or missing formulas.
            if parent is None:
                unsupported.add(event.name)
        return ModelComplexity(
            macs=macs,
            params=count_parameters(target),
            flops=counter.get_total_flops(),
            unsupported_ops=tuple(sorted(unsupported)),
        )


@torch.inference_mode()
def measure_module_latency(
    target: nn.Module,
    input_args: Any,
    device: torch.device | str | None = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    repetitions: int = 20,
    warmup: int = 5,
    compile: bool = False,
    compile_kwargs: dict[str, Any] | None = None,
) -> float:
    """Return median inference milliseconds for the entire supplied batch.

    Args:
        target: Module already on device, with read-only inference behavior.
        input_args: One input or positional argument tuple/list.
        device: CPU/CUDA device, inferred from model tensors when omitted.
        input_kwargs: Optional keyword inputs.
        repetitions: Positive number of timed forwards.
        warmup: Nonnegative untimed forwards after compilation.
        compile: Whether to use torch.compile; failures propagate explicitly.
        compile_kwargs: Arguments for torch.compile, such as backend or mode.

    CPU uses perf_counter; CUDA uses events on the specified device's current
    stream and synchronizes before/after timing. Compilation, warmup, input
    transfer and state isolation are excluded. Modes, buffers, torch RNG and GC
    state are restored on success/failure. Each call creates its own compilation
    wrapper; call again after physical pruning. No NPU or NAS-specific behavior.
    """
    if type(repetitions) is not int or repetitions <= 0 or type(warmup) is not int or warmup < 0:
        raise ValueError("Require positive repetitions and nonnegative warmup")
    if type(compile) is not bool or (compile_kwargs and not compile):
        raise ValueError("compile_kwargs requires compile=True")
    with _evaluation(target, input_args, input_kwargs, device) as (args, kwargs, device):

        def call(*args, **kwargs):
            return target(*args, **kwargs)

        # Compiling a module can attach bookkeeping attributes to that module,
        # invalidating an existing dependency graph. A callable wrapper retains
        # Module.__call__ semantics without permanently annotating the model.
        forward = torch.compile(call, **(compile_kwargs or {})) if compile else target
        if compile:
            forward(*args, **kwargs)
        for _ in range(warmup):
            forward(*args, **kwargs)
        events = []
        if device.type == "cuda":
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(repetitions)
            ]
            # Initialize lazy event resources outside the timed loop.
            for start, end in events:
                start.record()
                end.record()
            torch.cuda.synchronize(device)
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            times = []
            if device.type == "cuda":
                for start, end in events:
                    start.record()
                    forward(*args, **kwargs)
                    end.record()
                torch.cuda.synchronize(device)
                times = [start.elapsed_time(end) for start, end in events]
            else:
                for _ in range(repetitions):
                    start = time.perf_counter()
                    forward(*args, **kwargs)
                    times.append((time.perf_counter() - start) * 1000)
        finally:
            if gc_enabled:
                gc.enable()
        if not all(math.isfinite(t) and t >= 0 for t in times):
            raise ValueError("Invalid latency samples")
        return float(statistics.median(times))
