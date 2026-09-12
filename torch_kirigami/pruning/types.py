"""Small immutable records shared by planning and physical execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..configuration import FrozenList, FrozenScalar, freeze, thaw
from ..contracts import Impact, Requirement
from ..errors import KirigamiError
from ..graph import DependencyGraph
from ..operation import OperationContext, OperatorSpec
from ..selection import AxisRef, Region, Selection, TensorRef, resolve_reference


def _static(value, depth=0):
    """Freeze static record trees without retaining arbitrary mutable objects."""
    if depth > 100:
        raise ValueError("Structural configuration nesting limit exceeded")
    if isinstance(value, (tuple, list)):
        return tuple(_static(item, depth + 1) for item in value)
    if isinstance(value, FrozenList):
        return FrozenList(tuple(_static(item, depth + 1) for item in value.items))
    if isinstance(value, FrozenScalar):
        return FrozenScalar(value.kind, _static(value.value, depth + 1))
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError("Structural configuration must contain static data")


def _paths(values):
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

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Candidate key must be a nonempty string")
        object.__setattr__(self, "remove", tuple(self.remove))
        if not self.remove or not all(isinstance(s, Selection) and s for s in self.remove):
            raise ValueError("Candidate requires nonempty selections")
        if self.axis is not None and not isinstance(self.axis, AxisRef):
            raise TypeError("Candidate axis must be an AxisRef")


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
            axes = tuple(self.axes)
            if any(not isinstance(axis, AxisRef) for axis in axes):
                raise TypeError("Budget axes must be AxisRef instances")
            object.__setattr__(self, "axes", tuple(dict.fromkeys(axes)))


@dataclass(frozen=True)
class ChannelCount:
    """Integer removal caps over explicit logical axes.

    Args:
        counts: Local tuple aligned with axes, or one integer for global scope.
        axes: Explicit unique logical axes; no protected-domain filtering occurs.
        scope: Local per-axis caps or a global joint cap.
    """

    counts: int | tuple[int, ...]
    axes: tuple[AxisRef, ...]
    scope: str = "local"

    def __post_init__(self):
        axes = tuple(self.axes)
        if any(not isinstance(a, AxisRef) for a in axes) or len(set(axes)) != len(axes):
            raise ValueError("ChannelCount requires unique explicit axes")
        if self.scope not in ("local", "global"):
            raise ValueError("scope must be local or global")
        counts = (self.counts,) if self.scope == "global" else tuple(self.counts)
        if len(counts) != (1 if self.scope == "global" else len(axes)) or any(
            type(c) is not int or c < 0 for c in counts
        ):
            raise ValueError("Invalid integer channel caps")
        widths = tuple(a.tensor.shape[a.dim] for a in axes)
        if (self.scope == "global" and counts[0] > sum(widths)) or (
            self.scope == "local" and any(c > w for c, w in zip(counts, widths, strict=True))
        ):
            raise ValueError("Channel cap exceeds current width")
        object.__setattr__(self, "axes", axes)
        if self.scope == "local":
            object.__setattr__(self, "counts", counts)


def channel_targets(budget, widths):
    """Resolve ratio or integer caps using one planner-independent definition."""
    if isinstance(budget, ChannelRatio):
        return (
            tuple(math.floor(budget.ratio * w) for w in widths)
            if budget.scope == "local"
            else (math.floor(budget.ratio * sum(widths)),)
        )
    if isinstance(budget, ChannelCount):
        if tuple(a.tensor.shape[a.dim] for a in budget.axes) != tuple(widths):
            raise ValueError("Integer budget axes do not match planning widths")
        return (budget.counts,) if budget.scope == "global" else budget.counts
    raise TypeError("Expected ChannelRatio or ChannelCount")


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
        shapes = [tuple(map(len, r.axes)) for r in segments]
        if any(
            a != b
            for size in shapes[1:]
            for dim, (a, b) in enumerate(zip(shapes[0], size, strict=True))
            if dim != self.concat_dim
        ):
            raise ValueError("Recipe segments cannot concatenate")
        if rank == 0 and len(segments) != 1:
            raise ValueError("Scalar recipe must have exactly one segment")
        object.__setattr__(self, "segments", segments)

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
    """Assign a verified attribute; lists use detached FrozenList configuration."""

    path: str
    old: Any
    new: Any

    def __post_init__(self):
        def frozen(value):
            if isinstance(value, (list, FrozenList, FrozenScalar, torch.Size)):
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

    def __post_init__(self):
        if (
            not isinstance(self.source, Region)
            or not isinstance(self.destination, Region)
            or tuple(map(len, self.source.axes)) != tuple(map(len, self.destination.axes))
        ):
            raise ValueError("Coordinate segments require matching per-axis cardinalities")


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

    def __post_init__(self):
        for name in ("axes", "widths", "removed", "targets"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "exclusions", tuple(tuple(item) for item in self.exclusions))
        if self.scope not in ("manual", "local", "global"):
            raise ValueError("Invalid budget report scope")
        if type(self.limit_reached) is not bool:
            raise TypeError("Budget limit_reached must be boolean")
        if len(self.axes) != len(self.widths) or len(self.widths) != len(self.removed):
            raise ValueError("Budget axes, widths and removals must align")
        if self.scope != "manual" and len(self.targets) != (
            1 if self.scope == "global" else len(self.axes)
        ):
            raise ValueError("Budget targets do not match the scope")
        if any(
            type(n) is not int or n < 0
            for n in (*self.widths, *self.removed, *self.targets, self.trials)
        ):
            raise ValueError("Budget counts must be nonnegative integers")
        if any(not isinstance(axis, AxisRef) for axis in self.axes) or len(set(self.axes)) != len(
            self.axes
        ):
            raise ValueError("Budget report axes must be unique AxisRef instances")
        if any(
            width != axis.tensor.shape[axis.dim] or removed > width
            for axis, width, removed in zip(self.axes, self.widths, self.removed, strict=True)
        ):
            raise ValueError("Budget report disagrees with original widths")
        if any(
            len(item) != 2 or any(not isinstance(v, str) for v in item) for item in self.exclusions
        ):
            raise ValueError("Budget exclusions require key/reason pairs")

    @property
    def shortfall(self):
        """Return the unfilled channel target, counting coupled axes separately."""
        return max(0, sum(self.targets) - sum(self.removed))


@dataclass(frozen=True)
class AnalysisSummary:
    """Frozen original-coordinate analysis facts, independent of a live graph."""

    status: str
    requested: tuple[Selection, ...]
    selections: tuple[Selection, ...]
    reasons: tuple[str, ...]
    tensors: tuple[TensorRef, ...] = ()

    def __post_init__(self):
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

    def selection(self, tensor):
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

    def __post_init__(self):
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

    def __post_init__(self):
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

    def __post_init__(self):
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

    def __post_init__(self):
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

    def __post_init__(self):
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
