"""Shared optimization eligibility checks for native structural computations.

Ordinary analysis still accepts extension types. Passing this check permits
consumers to apply their own narrower reuse proofs; it does not prove that any
arbitrary computation on the record is cacheable.
"""

from dataclasses import fields

from .contracts import (
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Divisible,
    Fixed,
    LayoutConstraint,
    NonEmpty,
    ShapeExpr,
)
from .operation import PartitionedLayout
from .operators.shapes import CallArgumentConstraint
from .relations import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
)
from .selection import AxisRef, IndexSet, Region, Selection, TensorRef

_NATIVE_RECORD_TYPES = (
    AxisRef,
    TensorRef,
    IndexSet,
    Region,
    Selection,
    AxisPort,
    BlockMap,
    AxisRelation,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Divisible,
    Fixed,
    LayoutConstraint,
    NonEmpty,
    ShapeExpr,
    PartitionedLayout,
    CallArgumentConstraint,
)


def has_native_semantics(value: object) -> bool:
    """Require native behavior at every depth before bypassing repeated analysis.

    Frozen containers and value equality do not prove native behavior: an integer
    subclass can override arithmetic inside an otherwise built-in mapping. Reject
    such extensions without invoking their methods. They remain valid inputs to
    ordinary analysis; this predicate only controls optimization eligibility.
    """
    if type(value) in (str, int, bool, type(None)):
        return True
    if type(value) is tuple:
        return all(has_native_semantics(item) for item in value)
    if type(value) in _NATIVE_RECORD_TYPES:
        return all(has_native_semantics(getattr(value, field.name)) for field in fields(value))
    return False
