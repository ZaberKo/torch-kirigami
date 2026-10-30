"""CPU/CUDA inference timing with isolated model execution."""

import gc
import math
import statistics
import time
from typing import Any

import torch
from torch import nn

from .execution import isolated_evaluation


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
    with isolated_evaluation(target, input_args, input_kwargs, device) as (args, kwargs, device):

        def call(*args: object, **kwargs: object) -> object:
            """Invoke the target through a wrapper that avoids module mutation."""
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
