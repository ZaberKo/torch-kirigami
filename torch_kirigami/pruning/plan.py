"""Portable plans and execution reports built on shared records and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import nn

from ..dep_graph.selection import TensorRef, TensorRefMap
from ..operator_recipes import AttributeRecipe, TensorRecipe
from .serialization import RECORD_TYPES, decode, encode
from .state import validate_plan
from .types import (
    AnalysisSummary,
    CoordinateSegment,
    ModelStructure,
    ParameterReport,
    PlanningError,
    SelectionReport,
)


@dataclass(frozen=True)
class PruningPlan:
    """Portable structural decision; contains no model, graph, or live tensor.

    A recipe may be replayed on a compatible original model. Weights may change
    after scoring; applying a saved decision never recalculates its importance.

    Attributes:
        analysis: Frozen dependency analysis for the requested removal.
        recipes: Tensor replacement recipes for physical compaction.
        attributes: Module attribute edits required by the compact structure.
        selected: Candidate keys selected by an automatic strategy, if any.
        selection_report: Measured budget outcome or manual-selection details.
        notes: Explanatory notes attached to the plan.
        before: Structural preconditions for applying the plan.
        after: Expected structure after applying the plan.
    """

    analysis: AnalysisSummary
    recipes: tuple[TensorRecipe, ...]
    attributes: tuple[AttributeRecipe, ...]
    selected: tuple[str, ...]
    selection_report: SelectionReport | ParameterReport
    notes: tuple[str, ...]
    before: ModelStructure
    after: ModelStructure

    def __post_init__(self) -> None:
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
        for name, expected_record in (
            ("analysis", AnalysisSummary),
            ("selection_report", (SelectionReport, ParameterReport)),
            ("before", ModelStructure),
            ("after", ModelStructure),
        ):
            if not isinstance(getattr(self, name), expected_record):
                raise TypeError(f"Invalid plan {name}")
        if len(set(self.selected)) != len(self.selected):
            raise ValueError("Duplicate selected candidate keys")

    def to_dict(self) -> dict[str, object]:
        """Export basic data suitable for weights-only torch loading.

        Returns:
            A dictionary containing the plan's closed record schema.
        """
        return {
            "format": "torch-kirigami.plan",
            "plan": encode(self, record_types=PLAN_TYPES),
        }

    @classmethod
    def from_dict(cls, data: object) -> PruningPlan:
        """Load and validate a portable plan without importing model classes.

        Args:
            data: Dictionary returned by `to_dict()`.

        Returns:
            A validated portable plan.

        Raises:
            ValueError: The envelope, schema, or plan records are invalid.
        """
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
        """Describe the joint decision, budget, and physical modifications.

        Returns:
            Human-readable plan details.
        """
        lines = [f"Pruning plan: {self.analysis.status}; {len(self.recipes)} tensor replacements"]
        if isinstance(self.selection_report, ParameterReport):
            report = self.selection_report
            lines.append(
                f"Parameters: {report.before_params} -> {report.after_params}, "
                f"maximum {report.max_params}; {report.trials} trials"
            )
        elif self.selection_report.scope != "manual":
            lines.append(
                f"Channels: remaining {self.selection_report.remaining}, limits {self.selection_report.targets}, "
                f"shortfall {self.selection_report.shortfall}; {self.selection_report.trials} trials"
            )
        lines.extend(f"{r.tensor.paths[0]}: {r.tensor.shape} -> {r.shape}" for r in self.recipes)
        if self.selected:
            lines.append("Selected candidates: " + ", ".join(self.selected))
        lines.extend(f"{a.path}: {a.old} -> {a.new}" for a in self.attributes)
        lines.extend(self.notes)
        lines.extend(
            f"Excluded {key}: {reason}" for key, reason in self.selection_report.exclusions
        )
        if self.selection_report.limit_reached:
            lines.append("Strategy trial limit reached; no claim of infeasibility")
        return "\n".join(lines)


@dataclass(frozen=True)
class PruningResult:
    """Execution details returned separately from the original model.

    Attributes:
        plan: Portable plan that produced this result.
        structure: Model structure after application.
        parameter_map: Mapping from replaced parameters to their compact versions.
        coordinate_maps: Original-to-compact region mappings for replaced tensors.
        report: Execution notes, including optimizer and graph rebuild requirements.
    """

    plan: PruningPlan
    structure: ModelStructure
    parameter_map: Mapping[nn.Parameter, nn.Parameter]
    coordinate_maps: Mapping[TensorRef, tuple[CoordinateSegment, ...]]
    report: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_map", MappingProxyType(dict(self.parameter_map)))
        object.__setattr__(
            self,
            "coordinate_maps",
            TensorRefMap(tuple(self.coordinate_maps.items())),
        )
        object.__setattr__(self, "report", tuple(self.report))


PLAN_TYPES = MappingProxyType({**RECORD_TYPES, "PruningPlan": PruningPlan})
