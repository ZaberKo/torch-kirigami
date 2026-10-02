"""Region-union weight scores with explicit gradient ownership."""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import Impact
from ..regions import gather_region
from ..selection import AxisRef, IndexSet, Selection, TensorRef
from .types import Candidate, MetricContext, PlanningError

_TAYLOR_CACHE_ENTRIES = 128
_TAYLOR_CACHE_POSITIONS = 262_144
_TAYLOR_STATISTIC_ELEMENTS = 1_048_576


class Magnitude:
    """Compute L1 or L2 over the union of all affected parameter regions.

    Args:
        p: One or two; L2 includes the final square root.
        parameter_filter: Optional predicate accepting (TensorRef, Parameter).
            Bias and normalization parameters are included by default.
    """

    def __init__(
        self,
        p: int = 2,
        *,
        parameter_filter: Callable[[TensorRef, nn.Parameter], bool] | None = None,
    ) -> None:
        if p not in (1, 2):
            raise ValueError("Magnitude supports p=1 or p=2")
        self.p = p
        self.parameter_filter = parameter_filter

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score newly removed regions after the complete accepted selection."""
        return _scores(self, context, candidates, accepted_impact=accepted_impact)


class WeightTaylor:
    """Score real weights using caller-provided current, unscaled gradients.

    The caller owns loss reduction, gradient accumulation, and AMP unscaling.
    This is not a per-example Fisher estimator and never invokes backward.
    Elementwise absolute Taylor reuses complete axis slices with at most 128
    scalar vectors and 262,144 total positions. Signed joint Taylor preserves
    original-region reductions, including cancellation and overflow behavior.
    Irregular conditional regions and inference tensors are not cached.

    Args:
        mode: elementwise_abs sums absolute products; joint_abs takes one
            absolute value after summing all signed products in the candidate.
        parameter_filter: Optional (TensorRef, Parameter) predicate.
    """

    def __init__(
        self,
        mode: str = "elementwise_abs",
        *,
        parameter_filter: Callable[[TensorRef, nn.Parameter], bool] | None = None,
    ) -> None:
        if mode not in ("elementwise_abs", "joint_abs"):
            raise ValueError("Unknown Taylor mode")
        self.mode = mode
        self.parameter_filter = parameter_filter
        # Scalar vectors only; weak bindings prevent a recycled object identity
        # from reusing another tensor's statistics.
        self._axis_statistics: OrderedDict[
            tuple[object, ...],
            tuple[
                weakref.ReferenceType[torch.Tensor],
                weakref.ReferenceType[torch.Tensor],
                tuple[float, ...],
            ],
        ] = OrderedDict()
        self._statistic_positions = 0

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Return aligned scores; missing or nonfinite statistics are errors."""
        return _scores(self, context, candidates, accepted_impact=accepted_impact)

    def _axis_sums(self, weight: torch.Tensor, dim: int) -> tuple[float, ...] | None:
        """Reduce absolute axis statistics when their reuse is supported."""
        gradient = weight.grad
        assert gradient is not None
        if self.mode != "elementwise_abs":
            return None
        width = weight.shape[dim]
        elements_per_position = weight.numel() // width
        if width > _TAYLOR_CACHE_POSITIONS or elements_per_position > _TAYLOR_STATISTIC_ELEMENTS:
            return None
        try:
            key = (id(weight), weight._version, id(gradient), gradient._version, dim, self.mode)
        except RuntimeError:
            return None  # Inference tensors use ordinary selected-region reductions.
        entry = self._axis_statistics.get(key)
        if entry is not None:
            previous_weight, previous_gradient, values = entry
            if previous_weight() is weight and previous_gradient() is gradient:
                self._axis_statistics.move_to_end(key)
                return values
            self._statistic_positions -= len(values)
            del self._axis_statistics[key]
        weights, gradients = weight.detach().movedim(dim, 0), gradient.detach().movedim(dim, 0)
        chunk_size = max(1, _TAYLOR_STATISTIC_ELEMENTS // max(elements_per_position, 1))
        sums = []
        for start in range(0, width, chunk_size):
            products = weights[start : start + chunk_size].to(torch.float64) * gradients[
                start : start + chunk_size
            ].to(torch.float64)
            products = products.abs()
            # Near float64 overflow, even unsigned reductions can differ between
            # batched slices and the original region layout. Keep a wide safety
            # margin for every possible subset of this parameter's products.
            if (
                weight.dtype == torch.float64
                and (products > torch.finfo(torch.float64).max / (2 * max(weight.numel(), 1))).any()
            ):
                return None
            sums.extend(products.reshape(products.shape[0], -1).sum(1).cpu().tolist())
        values = tuple(sums)
        # Unselected positions can be nonfinite. The planner checks only the
        # candidate's resulting score, matching the ordinary region path.
        self._axis_statistics[key] = (weakref.ref(weight), weakref.ref(gradient), values)
        self._statistic_positions += len(values)
        while (
            len(self._axis_statistics) > _TAYLOR_CACHE_ENTRIES
            or self._statistic_positions > _TAYLOR_CACHE_POSITIONS
        ):
            _old_key, (_weight, _gradient, old_values) = self._axis_statistics.popitem(last=False)
            self._statistic_positions -= len(old_values)
        return values


_TAYLOR_SCORE = WeightTaylor.score


_WEIGHT_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
)
_WEIGHT_FUNCTIONS = {
    F.linear: 1,
    F.conv1d: 1,
    F.conv2d: 1,
    F.conv3d: 1,
    F.batch_norm: 3,
    F.layer_norm: 2,
}


class GroupMagnitude:
    """Normalize joint weight energy against its logical channel domain.

    For each candidate, sum `abs(weight) ** p` over its newly removed regions.
    Divide by the mean energy for deleting one surviving position of its declared
    axis. This matches mean-then-mean group magnitude for fixed, one-to-one
    members. Parameter aliases and overlapping roles are counted once, unlike
    implementations that count a dependency member more than once.

    Only exact Linear, Conv1d/2d/3d, BatchNorm1d/2d/3d and LayerNorm weight
    bindings, and corresponding functional weight arguments, participate. Bias
    and buffers do not. Unknown module semantics require an explicit custom
    metric; names such as `weight` are not sufficient to identify a binding.

    A candidate must select positions solely along its declared axis. The
    normalization includes all surviving original positions, independently of
    candidate availability, wrapping or scoring batch. Dynamic scores exclude
    accepted deletions and recompute the surviving-domain normalization. No
    model execution or activation/gradient recalibration occurs.

    At most 32 normalization entries retain one Python float per original axis
    position, about 32 bytes per position including tuple references on CPython.
    Singleton candidates reuse these exact scores. Blocks still require joint
    propagation; singleton scores are not assumed additive.

    Args:
        p: One or two; unlike `Magnitude`, squared energy is not square-rooted.
        parameter_filter: Optional further restriction of recognized weights.
    """

    def __init__(
        self,
        p: int = 2,
        *,
        parameter_filter: Callable[[TensorRef, nn.Parameter], bool] | None = None,
    ) -> None:
        if p not in (1, 2):
            raise ValueError("GroupMagnitude supports p=1 or p=2")
        self.p = p
        self.parameter_filter = parameter_filter
        # Scalars and immutable coordinate metadata only; no context, model,
        # activation, Parameter, or full Impact survives a score call here.
        self._normalizers: OrderedDict[
            tuple[object, ...], tuple[float, float, tuple[float, ...]]
        ] = OrderedDict()
        self._context: weakref.ReferenceType[MetricContext] | None = None

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Return normalized conditional energies, independent of batching."""
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        domains = tuple(_candidate_domain(candidate) for candidate in candidates)
        if not domains:
            return []
        weights = _weight_bindings(context)
        if not weights:
            raise PlanningError("GroupMagnitude found no recognized operator weight bindings")
        bindings = dict(context.graph.tensor_bindings())
        included = weights
        if self.parameter_filter is not None:
            included = set()
            for ref in weights:
                if self.parameter_filter(ref, cast(nn.Parameter, bindings[ref])):
                    included.add(ref)
                context.graph.validate()
        if not included:
            raise PlanningError("GroupMagnitude parameter_filter excluded all recognized weights")

        # A second planning context may add constraints affecting completeness.
        # Never reuse its predecessor's proofs, even for the same graph/weights.
        if self._context is None or self._context() is not context:
            self._normalizers.clear()
            try:
                self._context = weakref.ref(context)
            except TypeError:
                self._context = None  # Custom contexts need not support weak references.

        # Version counters also invalidate direct metric calls after ordinary
        # weight updates. Inference tensors cannot provide a safe cache key.
        try:
            versions = tuple(sorted((id(bindings[r]), bindings[r]._version) for r in included))
        except RuntimeError:
            versions = None
        accepted_regions = tuple((s.tensor, s.regions) for s in accepted_impact.selections.values())
        normalizers = {}
        for axis in dict.fromkeys(axis for axis, _indices in domains):
            key = (context.graph.id, self.p, axis, accepted_regions, versions, frozenset(included))
            cacheable = versions is not None and self._context is not None
            if cacheable and key in self._normalizers:
                normalizers[axis] = self._normalizers[key]
                self._normalizers.move_to_end(key)
                continue
            surviving = IndexSet.span(0, axis.tensor.shape[axis.dim]).subtract(
                accepted_impact.selection(axis.tensor).fully_selected_indices(axis.dim)
            )
            population = tuple(
                Candidate(f"normalization:{position}", (axis.select([position]),), axis)
                for position in surviving
            )
            norms = _scores(
                self, context, population, accepted_impact=accepted_impact, include=included
            )
            scale, mean = _normalization(norms, self.p)
            # A singleton's normalization query is exactly its conditional score
            # query. Store scalar results instead of propagating/evaluating them
            # again when a later score batch visits the same axis positions.
            position_norms = [0.0] * axis.tensor.shape[axis.dim]
            for position, norm in zip(surviving, norms, strict=True):
                position_norms[position] = norm
            entry = (scale, mean, tuple(position_norms))
            normalizers[axis] = entry
            if cacheable:
                self._normalizers[key] = entry
                if len(self._normalizers) > 32:
                    self._normalizers.popitem(last=False)
        norms = [0.0] * len(candidates)
        joint_indices = []
        for index, (axis, positions) in enumerate(domains):
            if len(positions) == 1:
                norms[index] = normalizers[axis][2][next(iter(positions))]
            else:
                joint_indices.append(index)
        if joint_indices:
            joint_norms = _scores(
                self,
                context,
                tuple(candidates[index] for index in joint_indices),
                accepted_impact=accepted_impact,
                include=included,
            )
            for index, norm in zip(joint_indices, joint_norms, strict=True):
                norms[index] = norm
        scores = []
        for value, (axis, _positions) in zip(norms, domains, strict=True):
            scale, mean, _position_norms = normalizers[axis]
            if scale == 0:
                if value != 0:
                    raise PlanningError(
                        "GroupMagnitude has zero singleton normalization but a nonzero joint "
                        "score; use a custom metric for this jointly activated structure"
                    )
                scores.append(0.0)
            else:
                scores.append((value / scale) ** self.p / mean)
        return scores


def _candidate_domain(candidate: Candidate) -> tuple[AxisRef, IndexSet]:
    """Reject ambiguous normalization domains rather than guessing an axis."""
    axis = candidate.axis
    if axis is None or any(selection.tensor != axis.tensor for selection in candidate.remove):
        raise PlanningError("GroupMagnitude requires a candidate selecting its declared axis")
    merged = Selection(axis.tensor)
    for selection in candidate.remove:
        merged = merged.union(selection)
    indices = merged.fully_selected_indices(axis.dim)
    if not indices or merged != axis.select(indices):
        raise PlanningError("GroupMagnitude requires complete slices of its declared axis")
    return axis, indices


def _weight_bindings(context: MetricContext) -> set[TensorRef]:
    """Identify parameter roles from captured operations, never alias spelling."""
    refs = set()
    for operation in context.graph.operations():
        if type(operation.module) in _WEIGHT_MODULES:
            ref = operation.binding("weight")
        elif operation.module is None and operation.node.target in _WEIGHT_FUNCTIONS:
            ref = operation.argument("weight", _WEIGHT_FUNCTIONS[operation.node.target])
        else:
            continue
        if isinstance(ref, TensorRef) and ref.kind == "parameter":
            refs.add(ref)
    return refs


def _normalization(norms: Sequence[float], p: int) -> tuple[float, float]:
    """Scale the mean energy without overflowing at extreme finite magnitudes."""
    if not all(math.isfinite(value) for value in norms):
        raise PlanningError("GroupMagnitude normalization contains nonfinite weights")
    scale = max(norms, default=0.0)
    mean = math.fsum((value / scale) ** p for value in norms) / len(norms) if scale else 0.0
    return scale, mean


def _magnitude_norm(values: torch.Tensor, p: int, *, batched: bool = False) -> torch.Tensor:
    """Reduce regions or independent rows with the same overflow-safe formula."""
    if p == 2:
        dtype = (
            torch.float64 if values.dtype in (torch.float64, torch.complex128) else torch.float32
        )
        values = torch.view_as_real(values.resolve_conj()) if values.is_complex() else values
        values = values.to(dtype).abs()
    else:
        values = values.to(torch.complex128 if values.is_complex() else torch.float64).abs()
    values = values.reshape(values.shape[0], -1) if batched else values.reshape(-1)
    if p == 1:
        return values.sum(-1, dtype=torch.float64)
    scale = values.amax(-1)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / divisor.unsqueeze(-1)).square().sum(-1, dtype=torch.float64).sqrt() * scale.to(
        torch.float64
    )


def _scores(
    metric: Magnitude | WeightTaylor | GroupMagnitude,
    context: MetricContext,
    candidates: tuple[Candidate, ...],
    *,
    accepted_impact: Impact,
    include: set[TensorRef] | None = None,
) -> list[float]:
    """Compute marginal norms or Taylor sums using complete joint propagation."""
    context.graph.validate_impact(accepted_impact)
    context.require_complete(accepted_impact)
    result = [0.0] * len(candidates)
    device_scores: dict[torch.device, list[tuple[int, torch.Tensor | float]]] = {}
    l2 = isinstance(metric, (Magnitude, GroupMagnitude)) and metric.p == 2
    bindings = dict(context.graph.tensor_bindings())
    native_taylor = (
        type(metric) is WeightTaylor
        and "score" not in vars(metric)
        and type(metric).score is _TAYLOR_SCORE
    )
    with torch.no_grad():
        for index, candidate in enumerate(candidates):
            impact = context.impact((*accepted_impact.requested, *candidate.remove))
            context.require_complete(impact)
            totals: dict[torch.device, torch.Tensor | float] = {}
            for selection in impact.parameters:
                if include is not None and selection.tensor not in include:
                    continue
                selection = selection.subtract(accepted_impact.selection(selection.tensor))
                if not selection:
                    continue
                weight = bindings[selection.tensor]
                if include is None and metric.parameter_filter is not None:
                    accepted = metric.parameter_filter(selection.tensor, cast(nn.Parameter, weight))
                    context.graph.validate()  # A user callback cannot invalidate cached bindings.
                    if not accepted:
                        continue
                taylor = isinstance(metric, WeightTaylor)
                if taylor and (weight.is_complex() or weight.grad is None or weight.grad.is_sparse):
                    raise PlanningError(
                        "WeightTaylor requires real parameters and dense current gradients"
                    )
                if native_taylor and len(selection.regions) == 1:
                    region = selection.regions[0]
                    partial = tuple(
                        dim
                        for dim, indices in enumerate(region.axes)
                        if indices != IndexSet.span(0, weight.shape[dim])
                    )
                    if len(partial) == 1:
                        dim = partial[0]
                        values = cast(WeightTaylor, metric)._axis_sums(weight, dim)
                        if values is not None:
                            value = sum(values[position] for position in region.axes[dim])
                            # Preserve ordinary region reduction and error
                            # behavior when cached statistics are nonfinite.
                            if math.isfinite(value):
                                previous = totals.get(weight.device)
                                totals[weight.device] = (
                                    value if previous is None else previous + value
                                )
                                continue
                for region in selection.regions:  # Selection normalizes to a disjoint union.
                    values = gather_region(weight.detach(), region)
                    if taylor:
                        # Promote before multiplication: a float64 reduction cannot
                        # recover products already underflowed/overflowed in float32.
                        values = values.to(torch.float64) * gather_region(
                            weight.grad.detach(), region
                        ).to(torch.float64)
                        if cast(WeightTaylor, metric).mode == "elementwise_abs":
                            values = values.abs()
                        value = values.sum(dtype=torch.float64)
                    else:
                        value = _magnitude_norm(values, metric.p)
                    previous = totals.get(weight.device)
                    totals[weight.device] = (
                        value
                        if previous is None
                        else torch.hypot(previous, value)
                        if l2
                        else previous + value
                    )
            for device, value in totals.items():
                device_scores.setdefault(device, []).append((index, value))
        # One transfer per device/batch, rather than one synchronization per
        # affected parameter region. Different parameter devices can still contribute.
        for entries in device_scores.values():
            tensors = [value for _, value in entries if isinstance(value, torch.Tensor)]
            values = iter(torch.stack(tensors).cpu().tolist() if tensors else ())
            for index, value in entries:
                value = next(values) if isinstance(value, torch.Tensor) else value
                result[index] = math.hypot(result[index], value) if l2 else result[index] + value
    if isinstance(metric, WeightTaylor) and metric.mode == "joint_abs":
        result = [abs(value) for value in result]
    return result
