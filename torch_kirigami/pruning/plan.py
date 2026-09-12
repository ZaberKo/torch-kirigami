"""Portable plans and execution reports built on shared records and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import nn

from ..selection import TensorRef, TensorRefMap
from .serialization import RECORD_TYPES, decode, encode
from .state import validate_plan
from .types import (
    AnalysisSummary,
    AttributeRecipe,
    BudgetReport,
    CoordinateSegment,
    ModelStructure,
    PlanningError,
    TensorRecipe,
)


@dataclass(frozen=True)
class PruningPlan:
    """Portable structural decision; contains no model, graph, or live tensor.

    A recipe may be replayed on a compatible original model. Weights may change
    after scoring; applying a saved decision never recalculates its importance.
    """

    analysis: AnalysisSummary
    recipes: tuple[TensorRecipe, ...]
    attributes: tuple[AttributeRecipe, ...]
    selected: tuple[str, ...]
    budget: BudgetReport
    notes: tuple[str, ...]
    before: ModelStructure
    after: ModelStructure

    def __post_init__(self):
        for name, cls in (
            ("recipes", TensorRecipe),
            ("attributes", AttributeRecipe),
            ("selected", str),
            ("notes", str),
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, cls) for value in values):
                raise TypeError(f"Invalid plan {name}")
            object.__setattr__(self, name, values)
        for name, cls in (
            ("analysis", AnalysisSummary),
            ("budget", BudgetReport),
            ("before", ModelStructure),
            ("after", ModelStructure),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"Invalid plan {name}")
        if len(set(self.selected)) != len(self.selected):
            raise ValueError("Duplicate selected candidate keys")

    def to_dict(self) -> dict:
        """Export basic data suitable for weights-only torch loading."""
        return {
            "format": "torch-kirigami.plan",
            "plan": encode(self, record_types=PLAN_TYPES),
        }

    @classmethod
    def from_dict(cls, data) -> PruningPlan:
        """Load and validate a portable plan without importing model classes."""
        if not isinstance(data, dict) or set(data) != {"format", "plan"}:
            raise ValueError("Invalid plan envelope")
        if data["format"] != "torch-kirigami.plan":
            raise ValueError("Unsupported plan format")
        result = decode(data["plan"], record_types=PLAN_TYPES)
        if not isinstance(result, cls):
            raise ValueError("Payload is not a pruning plan")
        try:
            validate_plan(result)
        except (TypeError, ValueError, PlanningError) as error:
            raise ValueError(f"Invalid pruning plan: {error}") from error
        return result

    def explain(self) -> str:
        """Describe the joint decision, budget, and physical modifications."""
        lines = [f"Pruning plan: {self.analysis.status}; {len(self.recipes)} tensor replacements"]
        if self.budget.scope != "manual":
            lines.append(
                f"Channels: target {sum(self.budget.targets)}, removed {sum(self.budget.removed)}, "
                f"shortfall {self.budget.shortfall}; {self.budget.trials} trials"
            )
        lines.extend(f"{r.tensor.paths[0]}: {r.tensor.shape} -> {r.shape}" for r in self.recipes)
        if self.selected:
            lines.append("Selected candidates: " + ", ".join(self.selected))
        lines.extend(f"{a.path}: {a.old} -> {a.new}" for a in self.attributes)
        lines.extend(self.notes)
        lines.extend(f"Excluded {key}: {reason}" for key, reason in self.budget.exclusions)
        if self.budget.limit_reached:
            lines.append("Strategy trial limit reached; no claim of infeasibility")
        return "\n".join(lines)


@dataclass(frozen=True)
class PruningResult:
    """Execution details returned separately from the original model."""

    plan: PruningPlan
    structure: ModelStructure
    parameter_map: Mapping[nn.Parameter, nn.Parameter]
    coordinate_maps: Mapping[TensorRef, tuple[CoordinateSegment, ...]]
    report: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "parameter_map", MappingProxyType(dict(self.parameter_map)))
        object.__setattr__(
            self,
            "coordinate_maps",
            TensorRefMap(tuple(self.coordinate_maps.items())),
        )
        object.__setattr__(self, "report", tuple(self.report))


PLAN_TYPES = MappingProxyType({**RECORD_TYPES, "PruningPlan": PruningPlan})
