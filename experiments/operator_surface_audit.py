"""Inventory visible PyTorch operations without executing arbitrary API calls.

Run `PYTHONPATH=. .venv/bin/python experiments/operator_surface_audit.py
--output result.json`. Direct registration does not prove support for every
overload. Missing registration does not prove failure because FX can expand
Python wrappers. Lifecycle tests establish those separate claims.
"""

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import OperatorRegistry


def inventory() -> dict[str, object]:
    """Inspect every visible callable and module class in the declared namespaces."""
    registry = OperatorRegistry.default()
    namespaces = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "F": F,
        "nn": nn,
        "torch.special": torch.special,
        "torch.linalg": torch.linalg,
        "torch.fft": torch.fft,
    }
    entries = []
    for namespace, owner in namespaces.items():
        for name in sorted(dir(owner)):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if not callable(value):
                continue
            if namespace == "nn":
                if not inspect.isclass(value) or not issubclass(value, nn.Module):
                    continue
                rule = registry.modules.get(value)
            elif namespace == "Tensor":
                rule = registry.methods.get(name)
            else:
                if inspect.isclass(value):
                    continue
                rule = registry.functions.get(value)
            doc = inspect.getdoc(value) or ""
            entries.append(
                {
                    "entry": f"{namespace}.{name}",
                    "rule": type(rule).__name__ if rule else None,
                    "status": "direct_rule" if rule else "no_direct_rule",
                    "signature_or_summary": doc.split("\n\n", 1)[0][:400],
                }
            )
    return {
        "torch": torch.__version__,
        "scope": list(namespaces),
        "tensor_properties": [
            f"Tensor.{name}"
            for name in sorted(dir(torch.Tensor))
            if not name.startswith("_") and inspect.isdatadescriptor(getattr(torch.Tensor, name))
        ],
        "interpretation": "Visible names include infrastructure and backend entry points. No-direct-rule entries may expand through FX or require additional semantics. This inventory is not a claim of complete operator or overload support.",
        "counts": dict(Counter(str(entry["status"]) for entry in entries)),
        "entries": entries,
    }


def main() -> None:
    """Write a reproducible per-entry inventory and print its scope counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
