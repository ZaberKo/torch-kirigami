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

from .capture import isolated_execution

__all__ = ["ModelComplexity", "calculate_model_complexity", "measure_module_latency"]

# Composite wrappers are counted through their constituent matrix/conv calls.
# Other entries have no MAC cost under our matrix/conv-only convention. Keep
# this explicit: unfamiliar kernels must appear in unsupported_ops, not as zero.
_MAC_FREE_OR_COMPOSITE = frozenset(
    [
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
        "add",
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
        "gelu",
        "silu",
        "clamp_min",
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
        "layer_norm",
        "native_layer_norm",
        "group_norm",
        "native_group_norm",
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
    ]
)


@dataclass(frozen=True)
class ModelComplexity:
    """MACs for one supplied batch and all unique registered parameter elements.

    One multiply-accumulate counts as one MAC (two FLOPs). Matrix products and
    convolutions count; bias, normalization, activation and other elementwise
    work do not. unsupported_ops lists observed operations outside the supported
    convention; a nonempty list means macs is only a partial count. Parameters
    include frozen weights, deduplicated by Parameter identity.
    """

    macs: int
    params: int
    unsupported_ops: tuple[str, ...] = ()


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, tuple):
        values = tuple(_to_device(v, device) for v in value)
        return type(value)(*values) if hasattr(value, "_fields") else values
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


@contextmanager
def _evaluation(target, input_args, input_kwargs, device) -> Iterator[tuple]:
    if not isinstance(target, nn.Module):
        raise TypeError("Measurement requires an nn.Module")
    tensors = (*target.parameters(), *target.buffers())
    device = torch.device(device if device is not None else tensors[0].device if tensors else "cpu")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Measurement supports CPU and CUDA only")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
    if any(t.device != device for t in tensors):
        raise ValueError("Move all model parameters and buffers to the measurement device first")
    args = tuple(input_args) if isinstance(input_args, (tuple, list)) else (input_args,)
    args, kwargs = _to_device(args, device), _to_device(input_kwargs or {}, device)
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context, isolated_execution(target, args, kwargs) as (args, kwargs, _):
        target.eval()
        yield args, kwargs, device


@torch.no_grad()
def calculate_model_complexity(
    target: nn.Module,
    input_args: Any,
    device: torch.device | str | None = None,
    input_kwargs: dict[str, Any] | None = None,
) -> ModelComplexity:
    """Count eager inference MACs and parameters without changing training state.

    Args:
        target: Eager module already on the chosen device; do not pass a compiled
            wrapper. Parameter values must not be mutated by its forward.
        input_args: One input or positional argument tuple/list. Nested tensor
            containers are moved to device and isolated from forward writes.
        device: CPU/CUDA device, inferred from model tensors when omitted.
        input_kwargs: Keyword inputs, including nested tensor containers.

    Returns:
        Counts for the entire supplied batch and explicit unsupported operations.
        Uses PyTorch's native FLOP formulas divided by two. SDPA uses the math
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
        counted = {
            str(op).replace("aten.", "aten::", 1)
            for op in counter.get_flop_counts().get("Global", {})
        }
        ignored = {f"aten::{op}" for op in _MAC_FREE_OR_COMPOSITE}
        unsupported = set()
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
            counter.get_total_flops() // 2,
            sum(parameter.numel() for parameter in target.parameters()),
            tuple(sorted(unsupported)),
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
