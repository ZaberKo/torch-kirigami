"""Execute representative PyTorch OpInfo samples through public measurement.

This optional audit requires `uv pip install expecttest` and the testing helpers
shipped with the installed PyTorch wheel. These private helpers are not used by
the library or regular tests. Each OpInfo contributes up to three float32 samples.
Passing this probe does not establish numerical correctness for all arguments.

Run `PYTHONPATH=. .venv/bin/python experiments/measurement_opinfo_probe.py
--device cuda --output results.jsonl`. Run CPU and CUDA separately.
"""

import argparse
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_modules import module_db

from torch_kirigami.dep_graph.capture import tensor_leaves
from torch_kirigami.measurement import calculate_model_complexity


class Operation(nn.Module):
    """Wrap one supplied operation without changing its arguments."""

    def __init__(self, operation: Any) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.operation(*args, **kwargs)


def probe(
    info: Any,
    device: str,
    sample_index: int,
    dtype: torch.dtype = torch.float32,
    variant: str = "function",
) -> dict[str, Any]:
    """Distinguish sample failures, ordinary execution failures, and counting failures."""
    row = {
        "name": info.name,
        "variant": info.variant_test_name,
        "device": device,
        "sample_index": sample_index,
        "dtype": str(dtype),
        "variant_form": variant,
    }
    stage = "sample"
    try:
        if dtype not in info.supported_dtypes(device):
            return {**row, "status": "unsupported_dtype"}
        operation = (
            info.op
            if variant == "function"
            else info.method_variant
            if variant == "method"
            else info.inplace_variant
        )
        if operation is None:
            return {**row, "status": "variant_unavailable"}
        samples = info.sample_inputs(device, dtype, requires_grad=False)
        sample = next(islice(samples, sample_index, sample_index + 1), None)
        if sample is None:
            return {**row, "status": "no_more_samples"}
        args = (sample.input, *sample.args)
        if any(t.numel() > 65_536 for t in tensor_leaves((args, sample.kwargs))):
            return {**row, "status": "sample_size_limit"}
        model = Operation(operation)
        stage = "baseline"
        with torch.no_grad():
            model(*args, **sample.kwargs)
        stage = "measurement"
        result = calculate_model_complexity(model, args, device=device, input_kwargs=sample.kwargs)
        return {**row, "status": "measured", **result.__dict__}
    except Exception as error:  # noqa: BLE001 - Preserve every failure in this audit report.
        return {**row, "status": f"{stage}_error", "error": f"{type(error).__name__}: {error}"}


def probe_module(info: Any, device: str, sample_index: int, dtype: torch.dtype) -> dict[str, Any]:
    """Exercise constructors and evaluation forwards from PyTorch ModuleInfo."""
    row = {"name": info.name, "device": device, "sample_index": sample_index, "dtype": str(dtype)}
    stage = "sample"
    try:
        if dtype not in info.supported_dtypes(device):
            return {**row, "status": "unsupported_dtype"}
        samples = info.module_inputs_func(info, device, dtype, requires_grad=False, training=False)
        sample = next(islice(iter(samples), sample_index, sample_index + 1), None)
        if sample is None:
            return {**row, "status": "no_more_samples"}
        if sample.forward_input is None:
            return {**row, "status": "no_forward_sample"}
        args, kwargs = sample.forward_input.args, sample.forward_input.kwargs
        if any(t.numel() > 65_536 for t in tensor_leaves((args, kwargs))):
            return {**row, "status": "sample_size_limit"}
        stage = "constructor"
        model = info.module_cls(*sample.constructor_input.args, **sample.constructor_input.kwargs)
        model.to(device=device, dtype=dtype).eval()
        stage = "baseline"
        with torch.no_grad():
            model(*args, **kwargs)
        stage = "measurement"
        result = calculate_model_complexity(model, args, device=device, input_kwargs=kwargs)
        return {**row, "status": "measured", **result.__dict__}
    except Exception as error:  # noqa: BLE001 - Keep failures as audit evidence.
        return {**row, "status": f"{stage}_error", "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    """Write each result immediately so one failed operation does not lose the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples_per_op", type=int, default=3)
    parser.add_argument("--kind", choices=("operators", "modules"), default="operators")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "float16", "bfloat16", "complex64", "int64"),
        default="float32",
    )
    parser.add_argument("--variant", choices=("function", "method", "inplace"), default="function")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for --device cuda")
    if args.samples_per_op < 1:
        parser.error("--samples_per_op must be positive")
    if args.kind == "modules" and args.variant != "function":
        parser.error("--variant applies to operators only")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    counts: Counter[str] = Counter()
    with args.output.open("w") as output:
        for info in op_db if args.kind == "operators" else module_db:
            for sample_index in range(args.samples_per_op):
                row = (
                    probe(info, args.device, sample_index, getattr(torch, args.dtype), args.variant)
                    if args.kind == "operators"
                    else probe_module(info, args.device, sample_index, getattr(torch, args.dtype))
                )
                if row["status"] == "no_more_samples":
                    break
                counts[row["status"]] += 1
                output.write(json.dumps(row) + "\n")
                output.flush()
                if row["status"] in ("unsupported_dtype", "variant_unavailable", "sample_error"):
                    break
    print(json.dumps({"torch": torch.__version__, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
