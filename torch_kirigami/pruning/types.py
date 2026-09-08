"""Small immutable records shared by planning and physical execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from ..errors import KirigamiError
from ..selection import AxisRef, Region, Selection, TensorRef


class PlanningError(KirigamiError):
    """The request has no verified executable recipe under the current policy."""


class ExecutionError(KirigamiError):
    """A structural precondition failed or modifications could not be committed."""


@dataclass(frozen=True)
class Candidate:
    """A named batch of original-coordinate seeds, not a global block constraint."""

    key: str
    remove: tuple[Selection, ...]
    axis: AxisRef | None = None

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Candidate key must be a nonempty string")
        object.__setattr__(self, "remove", tuple(self.remove))
        if not self.remove or not all(isinstance(s, Selection) and s for s in self.remove):
            raise ValueError("Candidate requires nonempty selections")


@dataclass(frozen=True)
class ChannelRatio:
    """Upper bound on removed positions relative to this snapshot's axis widths.

    Args:
        ratio: Fraction in [0, 1). Integer targets round down.
        scope: Local per-axis caps or a global cap without hidden local caps.
        axes: Explicit logical axes; required with caller-supplied candidates.
    """

    ratio: float
    scope: str = "local"
    axes: tuple[AxisRef, ...] | None = None

    def __post_init__(self):
        if not math.isfinite(self.ratio) or not 0 <= self.ratio < 1:
            raise ValueError("ratio must be finite and in [0, 1)")
        if self.scope not in ("local", "global"):
            raise ValueError("scope must be local or global")
        if self.axes is not None:
            object.__setattr__(self, "axes", tuple(dict.fromkeys(self.axes)))


class Metric(Protocol):
    """Score an aligned candidate batch; lower scores are selected first."""

    def __call__(self, context, candidate_batch):
        """Return a finite one-dimensional score per candidate."""
        ...


class Strategy(Protocol):
    """Select registered candidate keys using a shared PlanningContext."""

    def __call__(self, context):
        """Return registered keys; the framework verifies the combined result."""
        ...


@dataclass(frozen=True)
class TensorRecipe:
    """Gather Cartesian retained segments and concatenate in original group order.

    Segments are disjoint regions in the original tensor. Concatenation defines
    the old-to-new coordinate mapping without allocating an elementwise map.
    """

    tensor: TensorRef
    segments: tuple[Region, ...]
    concat_dim: int = 0
    memory_format: str = "contiguous"

    def __post_init__(self):
        object.__setattr__(self, "segments", tuple(Region(tuple(r.axes)) for r in self.segments))

    @property
    def shape(self):
        """Return the resulting tensor shape without allocating tensor data."""
        shapes = [tuple(len(a) for a in r.axes) for r in self.segments]
        if len(shapes) == 1:
            return shapes[0]
        result = list(shapes[0])
        result[self.concat_dim] = sum(s[self.concat_dim] for s in shapes)
        return tuple(result)


@dataclass(frozen=True)
class AttributeRecipe:
    """Assign a verified module attribute after checking its old value."""

    path: str
    old: Any
    new: Any

    def __post_init__(self):
        def frozen(value):
            if isinstance(value, tuple):
                return all(frozen(v) for v in value)
            return value is None or type(value) in (int, float, bool, str)

        if not self.path or not frozen(self.old) or not frozen(self.new):
            raise ValueError("Attribute recipes require a path and immutable scalar/tuple values")


@dataclass(frozen=True)
class CoordinateSegment:
    """Map an old Cartesian region into a compact Cartesian region in order."""

    source: Region
    destination: Region


@dataclass(frozen=True)
class BudgetReport:
    """Frozen denominator, target, and measured joint removals for this round."""

    axes: tuple[AxisRef, ...] = ()
    widths: tuple[int, ...] = ()
    removed: tuple[int, ...] = ()
    targets: tuple[int, ...] = ()
    scope: str = "manual"
    trials: int = 0
    limit_reached: bool = False
    exclusions: tuple[tuple[str, str], ...] = ()

    @property
    def shortfall(self):
        """Return the unfilled channel target, counting coupled axes separately."""
        return max(0, sum(self.targets) - sum(self.removed))


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
    before: Any
    after: Any

    def to_dict(self):
        """Export versioned basic data suitable for weights-only torch loading."""
        from .serialization import encode

        return {"format": "torch-kirigami.plan", "version": 1, "plan": encode(self)}

    @classmethod
    def from_dict(cls, data):
        """Load and validate a portable plan without importing model classes."""
        from .serialization import decode
        from .state import validate_plan

        if not isinstance(data, dict) or set(data) != {"format", "version", "plan"}:
            raise ValueError("Invalid plan envelope")
        if data["format"] != "torch-kirigami.plan" or data["version"] != 1:
            raise ValueError("Unsupported plan format/version")
        result = decode(data["plan"])
        if not isinstance(result, cls):
            raise ValueError("Payload is not a pruning plan")
        validate_plan(result)
        return result

    def explain(self):
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
    structure: Any
    parameter_map: Any
    coordinate_maps: Any
    report: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisSummary:
    """Frozen original-coordinate analysis facts, independent of a live graph."""

    status: str
    requested: tuple[Selection, ...]
    selections: tuple[Selection, ...]
    reasons: tuple[str, ...]

    def selection(self, tensor):
        """Find a frozen selection by stable tensor identity or registered path."""
        for selected in self.selections:
            ref = selected.tensor
            if ref.id == tensor.id or (ref.paths and set(ref.paths) & set(tensor.paths)):
                return selected
        return Selection(tensor)
