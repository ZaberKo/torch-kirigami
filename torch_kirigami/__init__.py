"""Structural dependency analysis for PyTorch, independent of pruning policy."""

from .capture import TensorFacts
from .contracts import (
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Diagnostic,
    Divisible,
    Fixed,
    Impact,
    Layout,
    NonEmpty,
    Requirement,
    ShapeExpr,
)
from .errors import AnalysisLimitError, CaptureError, KirigamiError, StaleGraphError
from .graph import CallRef, DependencyGraph
from .registry import (
    CallEffects,
    CandidateAxis,
    OperationContext,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
)
from .relations import (
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    Port,
    ReshapeRelation,
    SliceRelation,
)
from .selection import AxisRef, IndexSet, Region, Selection, TensorRef

__all__ = [
    "AnalysisLimitError",
    "AxisBarrier",
    "AxisRef",
    "AxisRelation",
    "Balanced",
    "Barrier",
    "BlockBalance",
    "BlockMap",
    "BroadcastRelation",
    "CallEffects",
    "CallRef",
    "CandidateAxis",
    "CaptureError",
    "DependencyGraph",
    "Diagnostic",
    "Divisible",
    "Fixed",
    "Impact",
    "IndexSet",
    "KirigamiError",
    "Layout",
    "NonEmpty",
    "OperationContext",
    "OperatorRegistry",
    "OperatorRule",
    "OperatorSpec",
    "PermuteRelation",
    "Port",
    "Region",
    "Requirement",
    "ReshapeRelation",
    "Selection",
    "ShapeExpr",
    "SliceRelation",
    "StaleGraphError",
    "TensorFacts",
    "TensorRef",
]
