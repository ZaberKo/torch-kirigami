"""Plan and apply structural pruning on the original PyTorch Module."""

from ..operator_recipes import AttributeRecipe, OperatorRecipe, TensorRecipe
from .axis_domains import IdentityAxisIndex
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
    SelectionReport,
    Strategy,
    StrategyResult,
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
    "IdentityAxisIndex",
    "Magnitude",
    "Metric",
    "MetricContext",
    "ModelStructure",
    "OperatorRecipe",
    "ParameterBudget",
    "ParameterGroup",
    "ParameterReport",
    "PlanningContext",
    "PlanningError",
    "Pruner",
    "PruningPlan",
    "PruningResult",
    "SelectionReport",
    "Strategy",
    "StrategyResult",
    "TensorRecipe",
    "WeightTaylor",
    "load_checkpoint",
    "save_checkpoint",
]
