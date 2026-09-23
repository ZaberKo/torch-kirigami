"""Small immutable records shared by planning and physical execution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Protocol, TypeVar

import torch

from ..configuration import FrozenDict, FrozenList, FrozenScalar, freeze, thaw
from ..contracts import Impact, Requirement
from ..errors import KirigamiError
from ..graph import DependencyGraph
from ..measurement import count_parameters
from ..operation import OperationContext, OperatorSpec
from ..regions import concatenated_shape
from ..selection import AxisRef, Region, Selection, TensorRef, resolve_reference


def _static(value: object, depth: int = 0) -> object:
    """Freeze static record trees without retaining arbitrary mutable objects."""
    if depth > 100:
        raise ValueError("Structural configuration nesting limit exceeded")
    if isinstance(value, (tuple, list)):
        return tuple(_static(item, depth + 1) for item in value)
    if isinstance(value, FrozenDict):
        return FrozenDict(
            tuple((_static(k, depth + 1), _static(v, depth + 1)) for k, v in value.items)
        )
    if isinstance(value, FrozenList):
        return FrozenList(tuple(_static(item, depth + 1) for item in value.items))
    if isinstance(value, FrozenScalar):
        return FrozenScalar(value.kind, _static(value.value, depth + 1))
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("Structural configuration must contain static data")


def _paths(values: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize a nonempty set of unique registered paths in declared order."""
    values = tuple(values)
    if not values or any(not isinstance(p, str) for p in values) or len(set(values)) != len(values):
        raise ValueError("Expected unique string paths")
    return values


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

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Candidate key must be a nonempty string")
        object.__setattr__(self, "remove", tuple(self.remove))
        if not self.remove or not all(isinstance(s, Selection) and s for s in self.remove):
            raise ValueError("Candidate requires nonempty selections")
        if self.axis is not None and not isinstance(self.axis, AxisRef):
            raise TypeError("Candidate axis must be an AxisRef")


@dataclass(frozen=True)
class ParameterBudget:
    """Upper bound on the final whole-model parameter count.

    All unique Parameter objects count, including frozen and protected tensors;
    buffers do not. A strategy must reach this absolute target before a plan can
    be returned. Structural granularity can make the result smaller than the cap.
    """

    max_params: int

    def __post_init__(self) -> None:
        if type(self.max_params) is not int or self.max_params < 0:
            raise ValueError("max_params must be a nonnegative integer")

    @classmethod
    def from_ratio(cls, model: torch.nn.Module, pruning_ratio: float) -> ParameterBudget:
        """Convert a whole-model parameter reduction fraction to an absolute cap.

        Args:
            model: Original model supplying the unique-parameter baseline.
            pruning_ratio: Finite fraction in [0, 1); this is not a channel ratio.

        Returns:
            Budget with floor(original_count * (1 - pruning_ratio)) parameters.
            The baseline is read once; later model changes do not alter the cap.
            Granularity may require a greater reduction. No model is retained.
        """
        if type(pruning_ratio) not in (int, float) or not 0 <= pruning_ratio < 1:
            raise ValueError("pruning_ratio must be finite and in [0, 1)")
        # Interpret the supplied decimal value exactly: binary subtraction such
        # as 1 - 0.9 must not turn a mathematically integral cap into one less.
        ratio = Fraction(str(pruning_ratio))
        count = count_parameters(model)
        return cls(count * (ratio.denominator - ratio.numerator) // ratio.denominator)


@dataclass(frozen=True)
class ChannelRatio:
    """Upper bound on removed positions relative to this snapshot's axis widths.

    Args:
        ratio: Fraction in [0, 1). Integer targets round down.
        scope: Local per-axis caps or a global cap without hidden local caps.

    The candidate space supplies the logical channel axes.
    """

    ratio: float
    scope: str = "local"

    def __post_init__(self) -> None:
        if not math.isfinite(self.ratio) or not 0 <= self.ratio < 1:
            raise ValueError("ratio must be finite and in [0, 1)")
        if self.scope not in ("local", "global"):
            raise ValueError("scope must be local or global")


@dataclass(frozen=True)
class ChannelCount:
    """Integer removal caps over explicit logical channel axes.

    Args:
        counts: Local tuple aligned with channel_axes, or one integer for global scope.
        channel_axes: Explicit unique logical axes, matching the candidate space in order.
        scope: Local per-axis caps or a global joint cap.
    """

    counts: int | tuple[int, ...]
    channel_axes: tuple[AxisRef, ...]
    scope: str = "local"

    def __post_init__(self) -> None:
        channel_axes = tuple(self.channel_axes)
        if any(not isinstance(a, AxisRef) for a in channel_axes) or len(set(channel_axes)) != len(
            channel_axes
        ):
            raise ValueError("ChannelCount requires unique explicit channel axes")
        if self.scope not in ("local", "global"):
            raise ValueError("scope must be local or global")
        counts = (self.counts,) if self.scope == "global" else tuple(self.counts)
        if len(counts) != (1 if self.scope == "global" else len(channel_axes)) or any(
            type(c) is not int or c < 0 for c in counts
        ):
            raise ValueError("Invalid integer channel caps")
        widths = tuple(a.tensor.shape[a.dim] for a in channel_axes)
        if (self.scope == "global" and counts[0] > sum(widths)) or (
            self.scope == "local" and any(c > w for c, w in zip(counts, widths, strict=True))
        ):
            raise ValueError("Channel cap exceeds current width")
        object.__setattr__(self, "channel_axes", channel_axes)
        if self.scope == "local":
            object.__setattr__(self, "counts", counts)


def channel_targets(
    budget: ChannelRatio | ChannelCount, widths: tuple[int, ...]
) -> tuple[int, ...]:
    """Resolve ratio or integer caps using one planner-independent definition."""
    if isinstance(budget, ChannelRatio):
        return (
            tuple(math.floor(budget.ratio * w) for w in widths)
            if budget.scope == "local"
            else (math.floor(budget.ratio * sum(widths)),)
        )
    if isinstance(budget, ChannelCount):
        if tuple(a.tensor.shape[a.dim] for a in budget.channel_axes) != tuple(widths):
            raise ValueError("Integer budget channel axes do not match planning widths")
        return (budget.counts,) if budget.scope == "global" else budget.counts
    raise TypeError("Expected ChannelRatio or ChannelCount")


class Metric(Protocol):
    """Score additional removals given a complete committed dependency closure.

    Lower finite scores are preferred. For the same model, statistics and
    `selected`, a candidate's score must not depend on batch size, order or other
    batch members. A batch is a computation convenience, not a joint removal;
    use a multi-selection Candidate to request a joint score. Implementations
    own calibration statistics and must not modify the model or run training.
    """

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        selected: Impact,
    ) -> Sequence[float] | torch.Tensor:
        """Evaluate additions to `selected`, using original graph coordinates.

        `selected` may violate repairable constraints when the empty request is
        initially infeasible. Scoring requires complete influence, not an
        executable intermediate model. Conditional scores are metric-specific;
        subtracting two whole-set scores is not a general implementation.
        """
        ...


@dataclass(frozen=True)
class StrategyResult:
    """Candidate keys and diagnostics returned by a selection strategy.

    The planner independently verifies keys, joint dependencies, execution and
    budgets. Resource measurements and query counts are not strategy assertions.

    Args:
        keys: Distinct registered keys, in accepted order.
        stop_reason: Target reached, no further accepted progress, or trial limit.
        exclusions: Candidate key and reason pairs explaining rejected choices.
    """

    keys: tuple[str, ...]
    stop_reason: Literal["target_reached", "exhausted", "trial_limit"] = "exhausted"
    exclusions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.keys, (str, bytes)):
            raise TypeError("Strategy keys require a sequence, not a bare string")
        keys = tuple(self.keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Strategy keys must be nonempty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("Strategy keys must be unique")
        if self.stop_reason not in ("target_reached", "exhausted", "trial_limit"):
            raise ValueError("Unknown strategy stop reason")
        if isinstance(self.exclusions, (str, bytes)):
            raise TypeError("Strategy exclusions require key/reason pairs, not a bare string")
        entries = tuple(self.exclusions)
        if any(isinstance(item, (str, bytes)) for item in entries):
            raise TypeError("Each strategy exclusion must be a key/reason pair, not a string")
        exclusions = tuple(tuple(item) for item in entries)
        if any(
            len(item) != 2 or any(not isinstance(s, str) or not s for s in item)
            for item in exclusions
        ):
            raise ValueError("Strategy exclusions require nonempty string key/reason pairs")
        excluded = {key for key, _ in exclusions}
        if len(excluded) != len(exclusions):
            raise ValueError("Strategy exclusion keys must be unique")
        if excluded.intersection(keys):
            raise ValueError("Strategy cannot both select and exclude the same key")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "exclusions", exclusions)


_StrategyContext = TypeVar("_StrategyContext", contravariant=True)


class Strategy(Protocol[_StrategyContext]):
    """Select registered candidate keys using a shared PlanningContext."""

    def select(self, context: _StrategyContext) -> StrategyResult:
        """Return registered choices and diagnostics without modifying the model."""
        ...


class MetricContext(Protocol):
    """Minimal planner contract required by scoring metrics."""

    def impact(self, remove: tuple[Selection, ...]) -> Impact:
        """Return dependency impact for candidate selections."""
        ...

    def require_complete(self, impact: Impact) -> None:
        """Reject incomplete dependency influence."""
        ...

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """Return the original candidate universe, independent of scoring batches."""
        ...

    @property
    def graph(self) -> DependencyGraph:
        """Return the dependency graph used for scoring."""
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

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments or not isinstance(self.tensor, TensorRef):
            raise ValueError("Tensor recipe requires a tensor and nonempty segments")
        if type(self.concat_dim) is not int or not 0 <= self.concat_dim < max(
            1, len(self.tensor.shape)
        ):
            raise ValueError("Invalid recipe concatenation dimension")
        if self.memory_format not in ("contiguous", "channels_last", "channels_last_3d"):
            raise ValueError("Unsupported recipe memory format")
        rank = len(self.tensor.shape)
        if self.memory_format != "contiguous" and rank != (
            4 if self.memory_format == "channels_last" else 5
        ):
            raise ValueError("Memory format is incompatible with tensor rank")
        for region in segments:
            if not isinstance(region, Region) or region.empty:
                raise ValueError("Recipe segments must be nonempty regions")
            Selection(self.tensor, (region,))
        try:
            concatenated_shape(segments, self.concat_dim)
        except ValueError as error:
            raise ValueError("Recipe segments cannot concatenate") from error
        object.__setattr__(self, "segments", segments)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the resulting tensor shape without allocating tensor data."""
        return concatenated_shape(self.segments, self.concat_dim)


@dataclass(frozen=True)
class AttributeRecipe:
    """Assign a verified attribute; lists use detached FrozenList configuration."""

    path: str
    old: Any
    new: Any

    def __post_init__(self) -> None:
        def frozen(value: object) -> object:
            """Freeze supported configuration values for portable storage."""
            if isinstance(value, (dict, FrozenDict, list, FrozenList, FrozenScalar, torch.Size)):
                return freeze(thaw(value))
            if isinstance(value, tuple):
                return tuple(frozen(v) for v in value)
            if value is None or type(value) in (int, float, bool, str):
                return value
            raise ValueError("Attribute recipes require scalar/list/tuple configuration values")

        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Attribute recipes require a path")
        object.__setattr__(self, "old", frozen(self.old))
        object.__setattr__(self, "new", frozen(self.new))


@dataclass(frozen=True)
class CoordinateSegment:
    """Map an old Cartesian region into a compact Cartesian region in order."""

    source: Region
    destination: Region

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, Region)
            or not isinstance(self.destination, Region)
            or tuple(map(len, self.source.axes)) != tuple(map(len, self.destination.axes))
        ):
            raise ValueError("Coordinate segments require matching per-axis cardinalities")


@dataclass(frozen=True)
class ParameterReport:
    """Exact parameter counts and search diagnostics, without live model state."""

    before_params: int
    after_params: int
    max_params: int
    trials: int = 0
    limit_reached: bool = False
    exclusions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            any(
                type(n) is not int or n < 0
                for n in (self.before_params, self.after_params, self.max_params, self.trials)
            )
            or self.after_params > self.before_params
        ):
            raise ValueError("Invalid parameter report counts")
        if type(self.limit_reached) is not bool:
            raise TypeError("Parameter report limit_reached must be boolean")
        object.__setattr__(self, "exclusions", tuple(tuple(item) for item in self.exclusions))
        if any(
            len(item) != 2 or any(not isinstance(v, str) for v in item) for item in self.exclusions
        ):
            raise ValueError("Parameter exclusions require key/reason pairs")

    @property
    def target_met(self) -> bool:
        """Whether the final model meets the requested absolute cap."""
        return self.after_params <= self.max_params


@dataclass(frozen=True)
class SelectionReport:
    """Frozen denominator, target, and measured joint removals for this round."""

    channel_axes: tuple[AxisRef, ...] = ()
    widths: tuple[int, ...] = ()
    removed: tuple[int, ...] = ()
    targets: tuple[int, ...] = ()
    scope: str = "manual"
    trials: int = 0
    limit_reached: bool = False
    exclusions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("channel_axes", "widths", "removed", "targets"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "exclusions", tuple(tuple(item) for item in self.exclusions))
        if self.scope not in ("manual", "local", "global"):
            raise ValueError("Invalid selection report scope")
        if type(self.limit_reached) is not bool:
            raise TypeError("Selection limit_reached must be boolean")
        if len(self.channel_axes) != len(self.widths) or len(self.widths) != len(self.removed):
            raise ValueError("Selection channel axes, widths and removals must align")
        if self.scope != "manual" and len(self.targets) != (
            1 if self.scope == "global" else len(self.channel_axes)
        ):
            raise ValueError("Selection targets do not match the scope")
        if any(
            type(n) is not int or n < 0
            for n in (*self.widths, *self.removed, *self.targets, self.trials)
        ):
            raise ValueError("Selection counts must be nonnegative integers")
        if any(not isinstance(axis, AxisRef) for axis in self.channel_axes) or len(
            set(self.channel_axes)
        ) != len(self.channel_axes):
            raise ValueError("Selection report channel axes must be unique AxisRef instances")
        if any(
            width != axis.tensor.shape[axis.dim] or removed > width
            for axis, width, removed in zip(
                self.channel_axes, self.widths, self.removed, strict=True
            )
        ):
            raise ValueError("Selection report disagrees with original widths")
        if any(
            len(item) != 2 or any(not isinstance(v, str) for v in item) for item in self.exclusions
        ):
            raise ValueError("Selection exclusions require key/reason pairs")

    @property
    def shortfall(self) -> int:
        """Return the unfilled channel target, counting coupled logical axes separately."""
        return max(0, sum(self.targets) - sum(self.removed))


@dataclass(frozen=True)
class AnalysisSummary:
    """Frozen original-coordinate analysis facts, independent of a live graph."""

    status: str
    requested: tuple[Selection, ...]
    selections: tuple[Selection, ...]
    reasons: tuple[str, ...]
    tensors: tuple[TensorRef, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("resolved", "unresolved", "conflict"):
            raise ValueError("Invalid analysis status")
        for name in ("requested", "selections", "reasons", "tensors"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(not isinstance(s, Selection) for s in (*self.requested, *self.selections)):
            raise TypeError("Analysis summary requires Selection records")
        if any(not isinstance(reason, str) for reason in self.reasons):
            raise TypeError("Analysis reasons must be strings")

        catalog = self.tensors or tuple(
            dict.fromkeys(s.tensor for s in (*self.requested, *self.selections))
        )
        if any(not isinstance(ref, TensorRef) for ref in catalog):
            raise TypeError("Analysis catalog requires TensorRef records")
        if len({ref.portable().id for ref in catalog}) != len(catalog):
            raise ValueError("Duplicate analysis tensor labels")
        object.__setattr__(self, "tensors", catalog)
        for selection in (*self.requested, *self.selections):
            try:
                resolve_reference(selection.tensor, catalog)
            except KeyError as error:
                raise ValueError("Analysis selection is absent from the tensor catalog") from error

    def selection(self, tensor: TensorRef) -> Selection:
        """Query compatible labels; unknown references raise instead of appearing empty."""
        ref = resolve_reference(tensor, self.tensors)
        return next((s for s in self.selections if s.tensor == ref), Selection(ref))


@dataclass(frozen=True)
class ModuleState:
    """Module type, aliases, ordinary configuration, and registered slot schema."""

    paths: tuple[str, ...]
    type_name: str
    attributes: tuple[tuple[str, object], ...]
    slots: tuple[tuple[str, str, bool], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _paths(self.paths))
        object.__setattr__(self, "attributes", _static(self.attributes))
        slots = tuple(tuple(slot) for slot in self.slots)
        if any(
            len(s) != 3
            or s[0] not in ("parameter", "buffer")
            or not isinstance(s[1], str)
            or type(s[2]) is not bool
            for s in slots
        ):
            raise ValueError("Invalid registered slot schema")
        if len({s[1] for s in slots}) != len(slots):
            raise ValueError("Duplicate registered slots")
        if any(len(item) != 2 or not isinstance(item[0], str) for item in self.attributes):
            raise ValueError("Invalid module attribute schema")
        if len({key for key, _ in self.attributes}) != len(self.attributes):
            raise ValueError("Duplicate module attributes")
        if not isinstance(self.type_name, str) or not self.type_name:
            raise ValueError("Module type name must be a nonempty string")
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True)
class TensorState:
    """Tensor binding facts without storage or process-local object identities."""

    paths: tuple[str, ...]
    kind: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    persistent: tuple[bool, ...]
    storage_aliases: tuple[str, ...]
    type_name: str
    values: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _paths(self.paths))
        for name in ("shape", "stride", "persistent", "storage_aliases"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.kind not in ("parameter", "buffer"):
            raise ValueError("Invalid registered tensor kind")
        if len(self.shape) != len(self.stride) or len(self.paths) != len(self.persistent):
            raise ValueError("Tensor state dimensions or persistence do not align")
        if any(type(n) is not int or n < 0 for n in (*self.shape, *self.stride)):
            raise ValueError("Tensor shape and stride must be nonnegative integers")
        if any(type(v) is not bool for v in (*self.persistent, self.requires_grad)):
            raise TypeError("Persistence and requires_grad must be boolean")
        for name in ("dtype", "device", "type_name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Tensor {name} must be a nonempty string")
        _paths(self.storage_aliases)
        if self.values is not None:
            values = tuple(self.values)
            if len(values) != math.prod(self.shape) or any(
                type(v) not in (int, bool) for v in values
            ):
                raise ValueError("Guard values must match the integer tensor shape")
            object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ModelStructure:
    """Immutable structural state usable without an FX graph or model reference."""

    modules: tuple[ModuleState, ...]
    tensors: tuple[TensorState, ...]
    references: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "references", _static(self.references))
        for values, cls in ((self.modules, ModuleState), (self.tensors, TensorState)):
            if any(not isinstance(value, cls) for value in values):
                raise TypeError("Invalid model structure records")
            paths = [path for value in values for path in value.paths]
            if len(paths) != len(set(paths)):
                raise ValueError("Duplicate structure paths")


def require_compact_shape(impact: Impact, ref: TensorRef) -> tuple[int, ...]:
    """Return the rectangular compact shape or reject a partitioned layout."""
    result = impact.selection(ref).compact_shape()
    if result is None:
        raise PlanningError(f"No rectangular tensor layout for {ref.id}")
    return result


@dataclass(frozen=True)
class RewriteContext:
    """One captured call, its affected requirements, and the combined Impact."""

    graph: DependencyGraph
    operation: OperationContext
    impact: Impact
    requirements: tuple[Requirement, ...]

    def __post_init__(self) -> None:
        requirements = tuple(self.requirements)
        if any(not isinstance(item, Requirement) for item in requirements):
            raise TypeError("Rewrite requirements must be Requirement records")
        object.__setattr__(self, "requirements", requirements)

    @property
    def spec(self) -> OperatorSpec:
        """Return the operator's single shared structural description."""
        return self.graph.operator_spec(self.operation)

    def compact_shape(self, ref: TensorRef) -> tuple[int, ...]:
        """Return the ordinary compact shape, rejecting partitioned layouts."""
        return require_compact_shape(self.impact, ref)


@dataclass(frozen=True)
class RewriteResult:
    """Pure extension result; every supplied requirement must be acknowledged.

    Custom rules are responsible for proving original-forward compatibility.
    Output strides can be supplied when proved, allowing downstream view checks.
    """

    tensors: tuple[TensorRecipe, ...] = ()
    attributes: tuple[AttributeRecipe, ...] = ()
    handled: tuple = ()
    notes: tuple[str, ...] = ()
    output_strides: tuple[tuple[TensorRef, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        for name, cls in (
            ("tensors", TensorRecipe),
            ("attributes", AttributeRecipe),
            ("handled", Requirement),
            ("notes", str),
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, cls) for value in values):
                raise TypeError(f"Invalid rewrite {name}")
            object.__setattr__(self, name, values)
        strides = tuple((ref, tuple(stride)) for ref, stride in self.output_strides)
        if any(
            not isinstance(ref, TensorRef)
            or len(stride) != len(ref.shape)
            or any(type(n) is not int or n < 0 for n in stride)
            for ref, stride in strides
        ):
            raise ValueError("Output strides must match tensor ranks")
        if len({ref for ref, _ in strides}) != len(strides):
            raise ValueError("Duplicate output stride declarations")
        object.__setattr__(self, "output_strides", strides)
