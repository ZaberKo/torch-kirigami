"""Measure model complexity and inference latency for workflow reports.

``measure_model`` calls the public complexity and latency measurement APIs with
the supplied example tensor and device options, then returns comparable input,
device, compilation and warmup metadata alongside parameter/MAC counts and
unsupported-operation diagnostics. These measurements describe the current
model and run configuration; they do not estimate accuracy or guarantee a
pruning speedup.
"""

from argparse import Namespace

import torch
from torch import nn

from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency


def measure_model(
    model: nn.Module, example: torch.Tensor, options: Namespace
) -> dict[str, int | float | str | bool | tuple[str, ...] | tuple[int, ...]]:
    """Measure complexity and latency using the supplied example unchanged."""
    print(f"Counting model complexity on {example.device}; batch={example.shape[0]}", flush=True)
    complexity = calculate_model_complexity(model, example, device=options.device)
    print(
        f"Measuring latency on {example.device}; batch={example.shape[0]}; "
        f"compile={options.compile_latency}. "
        "Initial compilation runs before timing and can keep the CPU busy.",
        flush=True,
    )
    latency = measure_module_latency(
        model,
        example,
        device=options.device,
        compile=options.compile_latency,
        warmup=options.latency_warmup,
        repetitions=options.latency_repetitions,
    )
    return {
        "#Params": complexity.params,
        "#MACs": complexity.macs,
        "latency_ms": latency,
        "unsupported_ops": complexity.unsupported_ops,
        "device": str(example.device),
        "dtype": str(example.dtype),
        "input_shape": tuple(example.shape),
        "compiled": options.compile_latency,
        "warmup": options.latency_warmup,
        "repetitions": options.latency_repetitions,
        # Environment metadata, including for CUDA; not current thread utilization.
        "cpu_threads": torch.get_num_threads(),
    }
