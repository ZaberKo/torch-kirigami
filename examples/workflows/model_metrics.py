"""Shared example reporting; measurement implementations live in the library."""

import torch

from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency


def add_arguments(parser):
    parser.add_argument("--compile", action="store_true", help="Measure torch.compile inference")
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)


def measure_model(model, x, options):
    # Use a fixed inference batch before/after pruning, independent of the
    # ImageNet training batch. Repeating a sample changes no shape-based MAC calculation.
    inputs = x[:1].expand(options.benchmark_batch_size, *x.shape[1:]).contiguous()
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


def report_comparison(before, after):
    print({"measurement": "after", **after})
    print(
        {
            "parameter_reduction": 1 - after["#Params"] / before["#Params"]
            if before["#Params"]
            else None,
            "MAC_reduction": 1 - after["#MACs"] / before["#MACs"]
            if before["#MACs"] and not before["unsupported_ops"] and not after["unsupported_ops"]
            else None,
            "latency_speedup": before["latency_ms"] / after["latency_ms"]
            if after["latency_ms"]
            else None,
        }
    )
