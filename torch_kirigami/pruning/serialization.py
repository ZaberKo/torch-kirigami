"""Closed, versioned codecs for immutable pruning data; never load Python code."""

from dataclasses import fields, is_dataclass


def _types():
    from ..configuration import FrozenList, FrozenScalar
    from ..selection import AxisRef, IndexSet, Region, Selection, TensorRef
    from .state import ModelStructure, ModuleState, TensorState
    from .types import (
        AnalysisSummary,
        AttributeRecipe,
        BudgetReport,
        CoordinateSegment,
        PruningPlan,
        TensorRecipe,
    )

    return {
        t.__name__: t
        for t in (
            FrozenList,
            FrozenScalar,
            AxisRef,
            IndexSet,
            Region,
            Selection,
            TensorRef,
            ModelStructure,
            ModuleState,
            TensorState,
            AnalysisSummary,
            AttributeRecipe,
            BudgetReport,
            CoordinateSegment,
            PruningPlan,
            TensorRecipe,
        )
    }


def encode(value):
    """Convert approved records to basic data without retaining arbitrary objects."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, tuple):
        return {"tuple": [encode(v) for v in value]}
    if is_dataclass(value) and type(value).__name__ in _types():
        return {
            "type": type(value).__name__,
            "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)},
        }
    raise TypeError(f"Nonportable plan value: {type(value).__name__}")


def decode(value, *, _depth=0):
    """Validate a closed data schema and reconstruct only approved record types."""
    if _depth > 100:
        raise ValueError("Pruning data nesting limit exceeded")
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, dict):
        raise ValueError("Expected basic scalar or tagged pruning record")
    if set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return tuple(decode(v, _depth=_depth + 1) for v in value["tuple"])
    if set(value) != {"type", "fields"} or not isinstance(value["type"], str):
        raise ValueError("Invalid pruning record")
    cls = _types().get(value["type"])
    if cls is None or not isinstance(value["fields"], dict):
        raise ValueError("Unknown pruning record type")
    if set(value["fields"]) != {f.name for f in fields(cls)}:
        raise ValueError(f"Invalid fields for {value['type']}")
    return cls(**{k: decode(v, _depth=_depth + 1) for k, v in value["fields"].items()})
