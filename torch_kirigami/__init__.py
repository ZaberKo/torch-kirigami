"""Structural dependency analysis for PyTorch models, independent of pruning policy.

The top-level package captures a model as a `DependencyGraph` that records which
tensor axes must change together when channels, heads, or other coordinates are
removed. It exports the graph, the `OperatorRegistry` and `OperatorRule` records
used to describe custom operators, the axis relations and constraint contracts
those rules produce, the `TensorRef`, `Selection`, and `Region` coordinate
records, the library exceptions, and the process-wide `config` settings.
Pruning planning and execution live in `torch_kirigami.pruning`, and
sparse-training primitives live in `torch_kirigami.sparsity`.
"""

from .dep_graph.contracts import (
    ArgumentRef,
    AttributeBinding,
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Diagnostic,
    Divisible,
    Fixed,
    Impact,
    LayoutConstraint,
    NonEmpty,
    Requirement,
    ShapeExpr,
)
from .dep_graph.dependency_graph import DependencyGraph
from .dep_graph.operation import (
    CandidateAxis,
    Operation,
    OperatorRule,
    OperatorSpec,
    OutputContract,
    PartitionedLayout,
    TensorEffects,
    TensorFacts,
)
from .dep_graph.registry import OperatorRegistry
from .dep_graph.relations import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
)
from .dep_graph.selection import AxisRef, IndexSet, Region, Selection, TensorRef
from .errors import (
    AnalysisLimitError,
    CaptureError,
    KirigamiError,
    StaleGraphError,
    UnsupportedOperation,
)
from .settings import config

__all__ = [
    "AnalysisLimitError",
    "ArgumentRef",
    "AttributeBinding",
    "AxisBarrier",
    "AxisPort",
    "AxisRef",
    "AxisRelation",
    "Balanced",
    "Barrier",
    "BlockBalance",
    "BlockMap",
    "BroadcastRelation",
    "CandidateAxis",
    "CaptureError",
    "DependencyGraph",
    "Diagnostic",
    "Divisible",
    "Fixed",
    "Impact",
    "IndexSet",
    "KirigamiError",
    "LayoutConstraint",
    "NonEmpty",
    "Operation",
    "OperatorRegistry",
    "OperatorRule",
    "OperatorSpec",
    "OutputContract",
    "PartitionedLayout",
    "PermuteRelation",
    "Region",
    "Requirement",
    "ReshapeRelation",
    "Selection",
    "ShapeExpr",
    "SliceRelation",
    "StaleGraphError",
    "TensorEffects",
    "TensorFacts",
    "TensorRef",
    "UnsupportedOperation",
    "config",
]
