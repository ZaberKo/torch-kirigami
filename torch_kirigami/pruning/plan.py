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
    """A static, checked structural pruning decision.

    A plan stores structural decisions, not weights, and holds no model, graph,
    or live tensor. Only `Pruner.apply()` executes it. A plan can be applied to
    any model whose structure matches `before`; replay gathers the receiving
    model's current weights and never rescores candidates. Construction checks
    field types only; `from_dict()` and `Pruner.apply()` also check that the
    recipes, structures, and report agree.

    Attributes:
        analysis: Frozen dependency analysis for the requested removal.
        recipes: Tensor replacement recipes for physical compaction.
        attributes: Module attribute edits required by the compact structure.
        selected: Candidate keys selected by an automatic strategy, in acceptance
            order. Empty for manual plans.
        selection_report: Measured budget outcome, or an empty manual report.
        notes: Explanatory notes attached to the plan.
        before: Structural preconditions for applying the plan.
        after: Expected structure after applying the plan.

    Raises:
        TypeError: A field has the wrong record type.
        ValueError: `selected` contains duplicate keys.
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
        """Encode the plan as basic Python data.

        The result contains only dictionaries, lists, and scalars, so it can be
        stored with `torch.save` and read back with `torch.load(weights_only=True)`.

        Returns:
            A dictionary with a `format` tag and the encoded `plan` records.
        """
        return {
            "format": "torch-kirigami.plan",
            "plan": encode(self, record_types=PLAN_TYPES),
        }

    @classmethod
    def from_dict(cls, data: object) -> PruningPlan:
        """Decode and validate a plan produced by `to_dict()`.

        Only the closed set of plan record types is reconstructed; no model
        classes are imported.

        Args:
            data: Dictionary returned by `to_dict()`.

        Returns:
            A validated plan.

        Raises:
            ValueError: The envelope, schema, or plan records are invalid, or the
                recipes, structures, and report disagree.
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
        """Describe the decision, the budget outcome, and the physical modifications.

        Returns:
            Human-readable, multi-line plan details.
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
    """Details of one `Pruner.apply()` call, returned alongside the pruned model.

    Attributes:
        plan: The plan that was applied.
        structure: Model structure after application.
        parameter_map: Read-only mapping from each replaced `nn.Parameter` object
            to its new compact `nn.Parameter`. Use it to rebuild optimizers;
            optimizer state is not migrated.
        coordinate_maps: Original-to-compact region mappings for every replaced
            tensor, keyed by tensor reference.
        report: Execution notes, including the optimizer and graph rebuild
            reminders.
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
