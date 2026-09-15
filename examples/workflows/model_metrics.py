"""Comparable model measurements shared by the workflow examples."""

import torch

from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency


def measure_model(model, example, options):
    """Use an identical inference batch for baseline and compact-model measurements."""
    inputs = example[:1].expand(options.val_batch_size, *example.shape[1:]).contiguous()
    print(f"Counting model complexity on {inputs.device}", flush=True)
    complexity = calculate_model_complexity(model, inputs, device=options.device)
    print(
        f"Measuring latency on {inputs.device}; compile={options.compile_latency}. "
        "Initial compilation runs before timing and can keep the CPU busy.",
        flush=True,
    )
    latency = measure_module_latency(
        model,
        inputs,
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
        "device": str(inputs.device),
        "dtype": str(inputs.dtype),
        "input_shape": tuple(inputs.shape),
        "compiled": options.compile_latency,
        "warmup": options.latency_warmup,
        "repetitions": options.latency_repetitions,
        # Environment metadata, including for CUDA; not current thread utilization.
        "cpu_threads": torch.get_num_threads(),
    }
