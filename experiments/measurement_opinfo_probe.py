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

from torch_kirigami.dep_graph.capture import tensor_leaves
from torch_kirigami.measurement import calculate_model_complexity


class Operation(nn.Module):
    """Wrap one supplied operation without changing its arguments."""

    def __init__(self, operation: Any) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.operation(*args, **kwargs)


def probe(info: Any, device: str, sample_index: int) -> dict[str, Any]:
    """Distinguish sample failures, ordinary execution failures, and counting failures."""
    row = {
        "name": info.name,
        "variant": info.variant_test_name,
        "device": device,
        "sample_index": sample_index,
    }
    stage = "sample"
    try:
        if torch.float32 not in info.supported_dtypes(device):
            return {**row, "status": "no_float32_sample"}
        samples = info.sample_inputs(device, torch.float32, requires_grad=False)
        sample = next(islice(samples, sample_index, sample_index + 1), None)
        if sample is None:
            return {**row, "status": "no_more_samples"}
        args = (sample.input, *sample.args)
        if any(t.numel() > 65_536 for t in tensor_leaves((args, sample.kwargs))):
            return {**row, "status": "sample_size_limit"}
        model = Operation(info.op)
        stage = "baseline"
        with torch.no_grad():
            model(*args, **sample.kwargs)
        stage = "measurement"
        result = calculate_model_complexity(model, args, device=device, input_kwargs=sample.kwargs)
        return {**row, "status": "measured", **result.__dict__}
    except Exception as error:  # noqa: BLE001 - Preserve every failure in this audit report.
        return {**row, "status": f"{stage}_error", "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    """Write each result immediately so one failed operation does not lose the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples_per_op", type=int, default=3)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required for --device cuda")
    if args.samples_per_op < 1:
        parser.error("--samples_per_op must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    counts: Counter[str] = Counter()
    with args.output.open("w") as output:
        for info in op_db:
            for sample_index in range(args.samples_per_op):
                row = probe(info, args.device, sample_index)
                if row["status"] == "no_more_samples":
                    break
                counts[row["status"]] += 1
                output.write(json.dumps(row) + "\n")
                output.flush()
                if row["status"] in ("no_float32_sample", "sample_error"):
                    break
    print(json.dumps({"torch": torch.__version__, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
