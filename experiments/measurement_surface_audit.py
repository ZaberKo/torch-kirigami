"""Inventory PyTorch entry points and measurement coverage declarations.

Run with `PYTHONPATH=. .venv/bin/python experiments/measurement_surface_audit.py
--output result.json`. This script does not execute arbitrary public functions.
A formula or wrapper declaration does not prove support for every invocation.
Native schema introspection is audit-only and is not a library dependency.
"""

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import flop_registry

from torch_kirigami.measurement.formulas import (
    FLOP_COMPOSITES,
    FLOP_FREE_OPS,
    MAC_COMPOSITES,
    MAC_FORMULAS,
    MAC_FREE_OPS,
    arithmetic_formulas,
    is_native_pointwise,
)


def inventory() -> dict[str, object]:
    """Enumerate public callables and all registered ATen schema packets."""
    namespaces = {
        "torch": torch,
        "Tensor": torch.Tensor,
        "F": F,
        "nn": nn,
        "torch.special": torch.special,
        "torch.linalg": torch.linalg,
        "torch.fft": torch.fft,
        "torch.sparse": torch.sparse,
        "torch.masked": torch.masked,
        "torch.signal.windows": torch.signal.windows,
    }
    entries = {}
    for namespace, owner in namespaces.items():
        names = []
        for name in sorted(dir(owner)):
            if name.startswith("_"):
                continue
            value = getattr(owner, name)
            if not callable(value):
                continue
            if namespace == "nn":
                if not inspect.isclass(value) or not issubclass(value, nn.Module):
                    continue
            elif inspect.isclass(value):
                continue
            names.append(name)
        entries[namespace] = names

    # dir(torch.ops.aten) omits packets that Python has not accessed yet.
    # The audit uses the installed schema catalogue, not lazy attribute names.
    schemas = torch._C._jit_get_all_schemas()
    packets: dict[str, list[str]] = {}
    for schema in schemas:
        if schema.name.startswith("aten::"):
            packets.setdefault(schema.name, []).append(schema.overload_name or "default")
    native = {str(op).replace("aten.", "aten::", 1) for op in flop_registry}
    arithmetic = {str(op).replace("aten.", "aten::", 1) for op in arithmetic_formulas()}
    mac_free = MAC_FREE_OPS | FLOP_FREE_OPS | arithmetic
    rows = []
    for name, overloads in sorted(packets.items()):
        if name in MAC_FORMULAS:
            flops = "dense_core_formula"
        elif name in arithmetic:
            flops = "nominal_arithmetic_formula"
        elif name in native:
            flops = "native_formula"
        elif name in FLOP_FREE_OPS:
            flops = "zero_by_convention"
        elif name in FLOP_COMPOSITES | MAC_COMPOSITES:
            flops = "composite_requires_execution_check"
        else:
            flops = "no_direct_formula"
        macs = (
            "dense_core_formula"
            if name in MAC_FORMULAS
            else "zero_by_convention"
            if name in mac_free or is_native_pointwise(name)
            else "composite_requires_execution_check"
            if name in MAC_COMPOSITES
            else "no_direct_formula"
        )
        rows.append(
            {"name": name, "overloads": sorted(set(overloads)), "flops": flops, "macs": macs}
        )
    return {
        "torch": torch.__version__,
        "public_entry_counts": {name: len(items) for name, items in entries.items()},
        "public_entries": entries,
        "tensor_properties": [
            name
            for name in sorted(dir(torch.Tensor))
            if not name.startswith("_") and inspect.isdatadescriptor(getattr(torch.Tensor, name))
        ],
        "aten_packet_count": len(rows),
        "aten_schema_count": sum(len(row["overloads"]) for row in rows),
        "flop_declaration_counts": dict(Counter(row["flops"] for row in rows)),
        "mac_declaration_counts": dict(Counter(row["macs"] for row in rows)),
        "aten_packets": rows,
        "interpretation": (
            "Public names include aliases and utilities. Packet declarations do not prove "
            "overload coverage. No-direct-formula operations can decompose into counted "
            "operations. Numerical and per-call diagnostic tests remain necessary."
        ),
    }


def main() -> None:
    """Write a reproducible inventory without invoking arbitrary operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key.endswith("count") or key.endswith("counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
