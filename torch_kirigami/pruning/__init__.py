"""Plan and apply structural pruning on the original PyTorch Module."""

from .candidates import CandidateSpace
from .checkpoint import load_checkpoint, save_checkpoint
from .granularity import Granularity
from .groups import ParameterGroup
from .metrics import GroupMagnitude, Magnitude, WeightTaylor
from .plan import PruningPlan, PruningResult
from .planner import PlanningContext
from .pruner import Pruner
from .strategies import DynamicGreedy, Greedy
from .types import (
    AnalysisSummary,
    AttributeRecipe,
    Candidate,
    ChannelCount,
    ChannelRatio,
    CoordinateSegment,
    ExecutionError,
    Metric,
    MetricContext,
    ModelStructure,
    ParameterBudget,
    ParameterReport,
    PlanningError,
    RewriteContext,
    RewriteResult,
    SelectionReport,
    Strategy,
    StrategyResult,
    TensorRecipe,
)

__all__ = [
    "AnalysisSummary",
    "AttributeRecipe",
    "Candidate",
    "CandidateSpace",
    "ChannelCount",
    "ChannelRatio",
    "CoordinateSegment",
    "DynamicGreedy",
    "ExecutionError",
    "Granularity",
    "Greedy",
    "GroupMagnitude",
    "Magnitude",
    "Metric",
    "MetricContext",
    "ModelStructure",
    "ParameterBudget",
    "ParameterGroup",
    "ParameterReport",
    "PlanningContext",
    "PlanningError",
    "Pruner",
    "PruningPlan",
    "PruningResult",
    "RewriteContext",
    "RewriteResult",
    "SelectionReport",
    "Strategy",
    "StrategyResult",
    "TensorRecipe",
    "WeightTaylor",
    "load_checkpoint",
    "save_checkpoint",
]
