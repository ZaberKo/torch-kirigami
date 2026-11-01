"""Plan and apply structural pruning on the original PyTorch Module."""

from ..operator_recipes import AttributeRecipe, OperatorRecipe, TensorRecipe
from .axis_domains import IdentityAxisIndex
from .candidates import CandidateSpace
from .checkpoint import load_checkpoint, save_checkpoint
from .dvbp import (
    ActivationCovariance,
    DenoisedStatistics,
    DenoisedVariance,
    apply_ob2c_updates,
    collect_covariances,
    denoise_covariance,
    mp_median,
    prepare_ob2c_updates,
    solve_ob2c,
)
from .granularity import Granularity
from .groups import ParameterGroup, unique_groups
from .metrics import GroupMagnitude, Magnitude, WeightTaylor, parameter_regions
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
    "ActivationCovariance",
    "AnalysisSummary",
    "AttributeRecipe",
    "Candidate",
    "CandidateSpace",
    "ChannelCount",
    "ChannelRatio",
    "CoordinateSegment",
    "DenoisedStatistics",
    "DenoisedVariance",
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
    "apply_ob2c_updates",
    "collect_covariances",
    "denoise_covariance",
    "load_checkpoint",
    "mp_median",
    "parameter_regions",
    "prepare_ob2c_updates",
    "save_checkpoint",
    "solve_ob2c",
    "unique_groups",
]
