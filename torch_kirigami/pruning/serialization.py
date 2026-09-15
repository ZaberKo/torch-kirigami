"""Closed data codecs; explicit schemas never import their consumers."""

from dataclasses import fields, is_dataclass
from types import MappingProxyType

from ..configuration import FrozenDict, FrozenList, FrozenScalar
from ..selection import AxisRef, IndexSet, Region, Selection, TensorRef
from .types import (
    AnalysisSummary,
    AttributeRecipe,
    CoordinateSegment,
    ModelStructure,
    ModuleState,
    ParameterReport,
    SelectionReport,
    TensorRecipe,
    TensorState,
)

RECORD_TYPES = MappingProxyType(
    {
        t.__name__: t
        for t in (
            FrozenDict,
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
            SelectionReport,
            ParameterReport,
            CoordinateSegment,
            TensorRecipe,
        )
    }
)


def encode(value: object, *, record_types: MappingProxyType = RECORD_TYPES) -> object:
    """Convert approved records to basic data without retaining arbitrary objects."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, tuple):
        return {"tuple": [encode(v, record_types=record_types) for v in value]}
    if is_dataclass(value) and record_types.get(type(value).__name__) is type(value):
        return {
            "type": type(value).__name__,
            "fields": {
                f.name: encode(getattr(value, f.name), record_types=record_types)
                for f in fields(value)
            },
        }
    raise TypeError(f"Nonportable plan value: {type(value).__name__}")


def decode(
    value: object, *, record_types: MappingProxyType = RECORD_TYPES, _depth: int = 0
) -> object:
    """Validate a closed data schema and reconstruct only approved record types."""
    if _depth > 100:
        raise ValueError("Pruning data nesting limit exceeded")
    if value is None or type(value) in (str, int, float, bool):
        return value
    if not isinstance(value, dict):
        raise ValueError("Expected basic scalar or tagged pruning record")
    if set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return tuple(
            decode(v, record_types=record_types, _depth=_depth + 1) for v in value["tuple"]
        )
    if set(value) != {"type", "fields"} or not isinstance(value["type"], str):
        raise ValueError("Invalid pruning record")
    cls = record_types.get(value["type"])
    if cls is None or not isinstance(value["fields"], dict):
        raise ValueError("Unknown pruning record type")
    if set(value["fields"]) != {f.name for f in fields(cls)}:
        raise ValueError(f"Invalid fields for {value['type']}")
    try:
        return cls(
            **{
                k: decode(v, record_types=record_types, _depth=_depth + 1)
                for k, v in value["fields"].items()
            }
        )
    except (TypeError, ValueError, AttributeError, IndexError) as error:
        raise ValueError(f"Invalid {value['type']} record: {error}") from error
