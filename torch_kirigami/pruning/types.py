"""Small immutable records shared by planning and physical execution."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol, TypeVar

import torch
from torch_prisma import count_parameters

from ..configuration import FrozenDict, FrozenList, FrozenScalar
from ..dep_graph.contracts import Impact
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import (
    AxisRef,
    Region,
    Selection,
    TensorRef,
    resolve_reference,
)
from ..errors import KirigamiError
from ..settings import config


def _static(value: object, depth: int = 0) -> object:
    """Freeze static record trees without retaining arbitrary mutable objects."""
    if depth > config.max_record_depth:
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


def _decimal_ratio(value: float, *, name: str = "ratio") -> Fraction:
    """Validate a ratio and preserve its supplied decimal value for integer caps."""
    if type(value) not in (int, float) or not 0 <= value < 1:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    # Binary multiplication/subtraction can move an integral target below its
    # boundary. Do not add an epsilon: a truly smaller ratio must still floor down.
    return Fraction(str(value))


class PlanningError(KirigamiError):
    """Planning could not produce a verified plan for the request.

    Raised when analysis is incomplete or conflicting, no execution recipe can be
    verified, or automatic selection misses its budget. A failed bounded search
    is not a proof that no valid plan exists.
    """


class ExecutionError(KirigamiError):
    """A structural precondition failed or modifications could not be committed."""


@dataclass(frozen=True)
class Candidate:
    """A named removal request in original coordinates.

    The selections are seeds: analysis adds their dependency closure. A candidate
    is one request that a strategy may select; it does not constrain how other
    candidates are grouped.

    Attributes:
        key: Label that is unique within a candidate space.
        remove: Nonempty selections analyzed together as one request.
        axis: Optional logical channel axis that this candidate removes positions
            from. `GroupMagnitude` requires it.

    Raises:
        ValueError: `key` is empty or `remove` has no nonempty selections.
        TypeError: `axis` is not an `AxisRef`.
    """

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
    """Cap on the final whole-model parameter count.

    Every element of each unique Parameter counts, including frozen and protected
    Parameters; a Parameter shared by several modules counts once, and buffers
    do not count. Unlike `ChannelRatio` and `ChannelCount`, this budget does not
    limit individual channel widths. Automatic planning must meet the cap or it
    raises `PlanningError`. Structural granularity can push the final count
    below the cap.

    Attributes:
        max_params: Maximum number of parameter elements after pruning.

    Raises:
        ValueError: `max_params` is not a nonnegative integer.
    """

    max_params: int

    def __post_init__(self) -> None:
        if type(self.max_params) is not int or self.max_params < 0:
            raise ValueError("max_params must be a nonnegative integer")

    @classmethod
    def from_ratio(cls, model: torch.nn.Module, pruning_ratio: float) -> ParameterBudget:
        """Convert a whole-model parameter reduction fraction to an absolute cap.

        The model's unique parameter count is read once and the model is not
        retained, so later model changes do not alter the cap.

        Args:
            model: Model that supplies the original unique parameter count.
            pruning_ratio: Fraction of parameters to remove, in `[0, 1)`. This is
                a parameter fraction, not a channel ratio.

        Returns:
            A budget of `floor(original_count * (1 - pruning_ratio))` parameters,
            computed with the exact decimal value of `pruning_ratio`.

        Raises:
            ValueError: `pruning_ratio` is not a number in `[0, 1)`.
        """
        ratio = _decimal_ratio(pruning_ratio, name="pruning_ratio")
        count = count_parameters(model)
        return cls(count * (ratio.denominator - ratio.numerator) // ratio.denominator)


@dataclass(frozen=True)
class ChannelRatio:
    """Caps on final logical channel widths, given as a reduction fraction.

    The candidate space supplies the logical channel axes. Each cap is
    `floor(width * (1 - ratio))`, computed with the exact decimal value of
    `ratio`. These caps limit axis widths, not the parameter count; use
    `ParameterBudget` for that and `ChannelCount` for absolute width caps.

    Attributes:
        ratio: Fraction of the original width to remove, in `[0, 1)`. It applies
            to each axis for local scope and to the summed width for global scope.
        scope: `"local"` for one cap per axis or `"global"` for one cap on the
            summed width.

    Raises:
        ValueError: `ratio` is not a number in `[0, 1)` or `scope` is unknown.
    """

    ratio: float
    scope: str = "local"

    def __post_init__(self) -> None:
        _decimal_ratio(self.ratio)
        if self.scope not in ("local", "global"):
            raise ValueError("scope must be local or global")


@dataclass(frozen=True)
class ChannelCount:
    """Absolute caps on final logical channel widths.

    These caps limit axis widths, not the parameter count. A cap may exceed the
    current width, in which case that axis needs no reduction. A zero cap is
    accepted but may be structurally unattainable.

    Attributes:
        max_channels: One final-width cap per axis for local scope, or one cap on
            the summed width for global scope.
        channel_axes: Logical axes covered by the caps. Include the active axes
            in candidate-space order. Excluded axes from the same space can
            also be counted. Local caps follow this field's order.
        scope: `"local"` for per-axis caps or `"global"` for one summed cap.

    Raises:
        ValueError: The axes are not unique `AxisRef` objects, `scope` is
            unknown, or the caps are not nonnegative integers matching the scope.
    """

    max_channels: int | tuple[int, ...]
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
        counts = (self.max_channels,) if self.scope == "global" else tuple(self.max_channels)
        if len(counts) != (1 if self.scope == "global" else len(channel_axes)) or any(
            type(c) is not int or c < 0 for c in counts
        ):
            raise ValueError("Invalid final channel limits")
        object.__setattr__(self, "channel_axes", channel_axes)
        if self.scope == "local":
            object.__setattr__(self, "max_channels", counts)


class Metric(Protocol):
    """Scoring protocol for candidate removals.

    A metric computes scores; a `Strategy` decides what to select. Lower finite
    scores mean a candidate is preferred for removal. For the same model,
    statistics, and `accepted_impact`, a candidate's score must not depend on the
    batch size, the batch order, or the other batch members. A batch is a
    computational convenience, not a joint removal; use a candidate with several
    selections to score a joint removal. Implementations own any calibration
    statistics and must not modify the model or run training. Scores are
    heuristics, not accuracy guarantees.
    """

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> Sequence[float] | torch.Tensor:
        """Score each candidate as an addition to the accepted removals.

        All coordinates are original graph coordinates. `accepted_impact` can
        violate repairable constraints, for example when the empty request is
        initially infeasible. Scoring requires complete dependency influence,
        not an executable intermediate model. How a score is conditioned on
        accepted removals is metric-specific; subtracting two whole-set scores
        is not a general implementation.

        Args:
            context: Dependency queries and original model bindings.
            candidates: Requests to score separately; batching does not combine them.
            accepted_impact: Joint dependency closure of the already accepted
                removal requests, including their coupled parameter regions. The
                model has not been pruned; accepted removals exist only in
                this analysis.

        Returns:
            One finite real score per candidate, in the supplied order.
        """
        ...


@dataclass(frozen=True)
class StrategyResult:
    """Candidate keys and diagnostics returned by a selection strategy.

    The planner independently verifies the keys, the joint dependency analysis,
    execution support, and the budget. It measures resource sizes and counts
    trials itself instead of trusting the strategy.

    Attributes:
        keys: Unique selected candidate keys, in acceptance order.
        stop_reason: Why the strategy stopped: `"target_reached"`,
            `"exhausted"`, or `"trial_limit"`.
        exclusions: Rejected candidate keys and their reasons, as unique
            `(key, reason)` pairs disjoint from `keys`.

    Raises:
        TypeError: `keys` or `exclusions` is a bare string, or an exclusion is a
            string instead of a pair.
        ValueError: Keys are empty or duplicated, `stop_reason` is unknown, or
            exclusions are malformed, duplicated, or overlap `keys`.
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
    """Selection protocol that chooses candidate keys from a planning context.

    A strategy usually owns a `Metric` and uses it to rank candidates. It must
    not modify the model.
    """

    def select(self, context: _StrategyContext) -> StrategyResult:
        """Select candidates without modifying the model.

        Args:
            context: Planning services and the filtered candidates, usually a
                `PlanningContext`.

        Returns:
            Selected candidate keys and search diagnostics. The planner verifies
            the result before it produces a plan.
        """
        ...


class MetricContext(Protocol):
    """Minimal planner interface that scoring metrics rely on."""

    def impact(self, remove: tuple[Selection, ...]) -> Impact:
        """Analyze selections jointly in original tensor coordinates.

        Args:
            remove: Hypothetical selections to propagate jointly.

        Returns:
            Their combined impact, including constraints and diagnostics.
        """
        ...

    def require_complete(self, impact: Impact) -> None:
        """Reject incomplete dependency influence.

        Args:
            impact: Dependency analysis to check for completeness.

        Raises:
            torch_kirigami.pruning.PlanningError: The influence is incomplete.
        """
        ...

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """All candidates of this planning round, independent of scoring batches."""
        ...

    @property
    def graph(self) -> DependencyGraph:
        """Dependency graph used for scoring."""
        ...


@dataclass(frozen=True)
class CoordinateSegment:
    """Correspondence between an original Cartesian region and its compact region.

    Positions map in ascending order along each axis, so both regions have the
    same number of positions per axis.

    Attributes:
        source: Region in the original tensor coordinates.
        destination: Corresponding region in the compact tensor coordinates.

    Raises:
        ValueError: The regions are not `Region` objects or their per-axis sizes
            differ.
    """

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
    """Measured parameter counts and search diagnostics for a `ParameterBudget` plan.

    The report holds no live model state.

    Attributes:
        before_params: Unique parameter elements before pruning.
        after_params: Unique parameter elements after pruning.
        max_params: Requested final parameter cap.
        trials: Joint selection attempts made through the planning context.
        limit_reached: Whether the strategy stopped at its trial limit.
        exclusions: Rejected candidate keys and their reasons.

    Raises:
        ValueError: A count is not a nonnegative integer, `after_params` exceeds
            `before_params`, or an exclusion is not a pair of strings.
        TypeError: `limit_reached` is not a `bool`.
    """

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
    """Channel widths, final-width caps, and measured removals for one round.

    Channel-budget plans fill every field. Manual plans use the default empty
    report with `"manual"` scope, which has no budget.

    Attributes:
        channel_axes: Logical axes counted by the report.
        widths: Original widths, aligned with `channel_axes`.
        removed: Measured removed positions, aligned with `channel_axes`.
        targets: Final-width caps: one per axis for local scope, one summed cap
            for global scope, and empty for manual reports.
        scope: `"manual"`, `"local"`, or `"global"`.
        trials: Joint selection attempts made through the planning context.
        limit_reached: Whether the strategy stopped at its trial limit.
        exclusions: Rejected candidate keys and their reasons.

    Raises:
        ValueError: The fields do not align, counts are invalid, widths disagree
            with the axes, or exclusions are not pairs of strings.
        TypeError: `limit_reached` is not a `bool`.
    """

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
        """Final channels above the caps; zero for manual reports.

        For local scope, the excess of each axis is summed, so one axis below its
        cap cannot offset another axis above its cap.
        """
        if self.scope == "manual":
            return 0
        if self.scope == "global":
            return max(0, sum(self.remaining) - self.targets[0])
        return sum(max(0, n - cap) for n, cap in zip(self.remaining, self.targets, strict=True))

    @property
    def remaining(self) -> tuple[int, ...]:
        """Final widths, aligned with `channel_axes`."""
        return tuple(w - n for w, n in zip(self.widths, self.removed, strict=True))

    @property
    def target_met(self) -> bool:
        """Whether every final-width cap is met; always true for manual reports."""
        return self.shortfall == 0


@dataclass(frozen=True)
class AnalysisSummary:
    """Frozen dependency analysis in original coordinates, independent of a live graph.

    Attributes:
        status: Analysis status: `"resolved"`, `"unresolved"`, or `"conflict"`.
            Only resolved analyses can be executed.
        requested: Removal requests submitted for analysis.
        selections: Propagated removal closure, including partial results when
            the analysis is unresolved or conflicts with constraints.
        reasons: Unique provenance reasons recorded by the analysis.
        tensors: Catalog of tensor references used to resolve selections.
            Defaults to the tensors referenced by `requested` and `selections`.

    Raises:
        ValueError: `status` is unknown, catalog labels repeat, or a selection
            refers to a tensor outside the catalog.
        TypeError: A record has the wrong type.
    """

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
        """Return the removed region of one tensor.

        Args:
            tensor: Live or portable reference that matches a catalog entry by
                portable ID or by a shared registered path.

        Returns:
            The tensor's removed selection, or an empty selection when the
            analysis removes nothing from it.

        Raises:
            KeyError: The tensor is not in the catalog. Unknown tensors raise
                instead of returning an empty selection.
            ValueError: A catalog entry matches but its shape, kind, or paths
                differ.
            TypeError: `tensor` is not a `TensorRef`.
        """
        ref = resolve_reference(tensor, self.tensors)
        return next((s for s in self.selections if s.tensor == ref), Selection(ref))


@dataclass(frozen=True)
class ModuleState:
    """Module type, aliases, ordinary configuration, and registered slot schema.

    Attributes:
        paths: All registered paths that refer to the module.
        type_name: Fully qualified Python type name.
        attributes: Frozen ordinary attribute names and values.
        slots: Registered parameter/buffer names, kinds, and persistence flags.
    """

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
    """Tensor binding facts without storage or process-local object identities.

    Attributes:
        paths: Registered names that refer to the tensor.
        kind: `parameter` or `buffer`.
        shape: Tensor dimensions.
        stride: Tensor strides.
        dtype: Serialized dtype name.
        device: Serialized device name.
        requires_grad: Whether the tensor requires gradients.
        persistent: Registration persistence flags aligned with `paths`.
        storage_aliases: Registered tensor paths sharing this storage.
        type_name: Tensor subclass type name.
        values: Optional integer values used to guard structural tensors.
    """

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
    """Immutable structural state of a model, without an FX graph or model reference.

    Weight values are not recorded, except the values of guarded structural
    tensors, so ordinary training steps do not change the structure.

    Attributes:
        modules: Module type, configuration, alias, and slot records.
        tensors: Registered tensor metadata and structural guards.
        references: Frozen ordinary tensor references and alias relationships.

    Raises:
        TypeError: A record has the wrong type or contains non-static data.
        ValueError: Paths repeat or the references are nested too deeply.
    """

    modules: tuple[ModuleState, ...]
    tensors: tuple[TensorState, ...]
    references: tuple[object, ...] = ()

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
