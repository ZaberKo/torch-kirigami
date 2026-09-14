"""Comparable model measurements shared by the workflow examples."""

import torch

from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency


def measure_model(model, example, options):
    """Use an identical inference batch for baseline and compact-model measurements."""
    inputs = example[:1].expand(options.benchmark_batch_size, *example.shape[1:]).contiguous()
    complexity = calculate_model_complexity(model, inputs, device=options.device)
    latency = measure_module_latency(
        model,
        inputs,
        device=options.device,
        compile=options.compile,
        warmup=options.warmup,
        repetitions=options.repetitions,
    )
    return {
        "#Params": complexity.params,
        "#MACs": complexity.macs,
        "latency_ms": latency,
        "unsupported_ops": complexity.unsupported_ops,
        "device": str(inputs.device),
        "dtype": str(inputs.dtype),
        "input_shape": tuple(inputs.shape),
        "compiled": options.compile,
        "warmup": options.warmup,
        "repetitions": options.repetitions,
        "cpu_threads": torch.get_num_threads(),
    }
