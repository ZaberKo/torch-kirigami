"""Measure model complexity and inference latency for workflow reports.

`measure_model` calls the public Torch Prisma complexity and latency APIs.
It uses the supplied example tensor on its current device.
The result includes parameter counts, MACs, FLOPs, and separate coverage diagnostics.
It also records input, device, compilation, and warmup settings for comparison.

The measurements describe the current model and run configuration.
They do not estimate accuracy or guarantee faster inference after pruning.
The adapter restores module modes. Other forward side effects persist.
"""

from argparse import Namespace

import torch
from torch import nn
from torch_prisma import calculate_model_complexity, measure_module_latency


def measure_model(
    model: nn.Module, example: torch.Tensor, options: Namespace
) -> dict[str, int | float | str | bool | tuple[int, ...] | dict[str, tuple[str, ...]]]:
    """Measure an evaluation forward without moving the model or example.

    Args:
        model: Model prepared on the example's device. Each module's mode is
            restored after success or failure. Other forward side effects persist.
        example: Batch used unchanged for both complexity and latency measurement.
        options: CLI options supplying `compile_latency`, `latency_warmup`, and
            `latency_repetitions`. Timing uses the example's actual device.

    Returns:
        Counts, coverage diagnostics, median batch latency, and execution settings.
    """
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        print(
            f"Counting model complexity on {example.device}; batch={example.shape[0]}", flush=True
        )
        complexity = calculate_model_complexity(model, example)
        print(
            f"Measuring latency on {example.device}; batch={example.shape[0]}; "
            f"compile={options.compile_latency}. "
            "Initial compilation runs before timing and can keep the CPU busy.",
            flush=True,
        )

        # Compile a callable so compilation bookkeeping is not attached to the model.
        def forward(x: torch.Tensor) -> torch.Tensor:
            return model(x)

        latency = measure_module_latency(
            forward,
            example,
            device=example.device,
            compile=options.compile_latency,
            warmup=options.latency_warmup,
            repetitions=options.latency_repetitions,
        )
    finally:
        for module, training in modes:
            module.training = training
    return {
        "#Params": complexity.params,
        "#MACs": complexity.macs,
        "#FLOPs": complexity.flops,
        "latency_ms": latency.median_ms,
        "uncovered_ops": complexity.uncovered_ops,
        "device": str(example.device),
        "dtype": str(example.dtype),
        "input_shape": tuple(example.shape),
        "compiled": options.compile_latency,
        "warmup": options.latency_warmup,
        "repetitions": options.latency_repetitions,
        # Environment metadata, including for CUDA; not current thread utilization.
        "cpu_threads": torch.get_num_threads(),
    }
