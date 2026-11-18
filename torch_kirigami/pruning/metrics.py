"""Region-union weight scores with explicit gradient ownership."""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

import torch
from torch import nn

from ..dep_graph.contracts import Impact
from ..dep_graph.selection import (
    AxisRef,
    IntervalBox,
    Selection,
    TensorRef,
    canonical_boxes,
    gather_intervals,
    region_boxes,
)
from ..settings import config
from .types import Candidate, MetricContext, PlanningError


def parameter_regions(
    context: MetricContext,
    candidate: Candidate,
    *,
    accepted_impact: Impact,
) -> tuple[Selection, ...]:
    """Return the parameter regions that one candidate would remove in addition.

    The accepted and candidate requests are analyzed together, and the regions
    already removed by `accepted_impact` are subtracted. Shared parameters and
    overlapping regions count once. A joint request can trigger effects that
    neither request produces alone. Constraint violations do not prevent this
    query, and its result does not prove that the removal can be executed. It
    does not read parameter values or choose a score formula.

    Args:
        context: Dependency queries and original model bindings.
        candidate: One request, which can contain several selections.
        accepted_impact: Complete dependency closure of previously accepted
            removal requests from the same graph. Use `context.impact(())`
            to query a candidate without prior removals.

    Returns:
        Nonempty selections in original coordinates, one per affected
        `nn.Parameter`. Buffers are excluded. An empty tuple is valid and means
        the candidate removes no further parameters. Use
        `context.graph.tensor_bindings()` to resolve parameter values.

    Raises:
        torch_kirigami.pruning.PlanningError: Dependency influence is incomplete.
        ValueError: A reference or `accepted_impact` belongs to another graph.
        torch_kirigami.AnalysisLimitError: An additional selection, or an
            intermediate piece of `Selection.subtract`, exceeds the region
            limit. The built-in metrics score it without this representation.
        torch_kirigami.StaleGraphError: The model changed after capture.
    """
    regions = []
    for selection, accepted in _joint_parameters(context, candidate, accepted_impact):
        remaining = selection.subtract(accepted)
        if remaining:
            regions.append(remaining)
    return tuple(regions)


def _joint_parameters(
    context: MetricContext, candidate: Candidate, accepted_impact: Impact
) -> tuple[tuple[Selection, Selection], ...]:
    """Pair each joint parameter selection of `parameter_regions` with its accepted selection."""
    context.graph.validate_impact(accepted_impact)
    context.require_complete(accepted_impact)
    impact = context.impact((*accepted_impact.requested, *candidate.remove))
    context.require_complete(impact)
    return tuple((s, accepted_impact.selection(s.tensor)) for s in impact.parameters)


def _additional_boxes(
    context: MetricContext, candidate: Candidate, accepted_impact: Impact
) -> tuple[tuple[TensorRef, tuple[IntervalBox, ...]], ...]:
    """Return `canonical_boxes` of the positions that `parameter_regions` selects.

    Parameters with no additional positions are omitted. Unlike
    `parameter_regions`, this never reaches the region limit.
    """
    parts = []
    for selection, accepted in _joint_parameters(context, candidate, accepted_impact):
        boxes = canonical_boxes(region_boxes(selection.regions), region_boxes(accepted.regions))
        if boxes:
            parts.append((selection.tensor, boxes))
    return tuple(parts)


class Magnitude:
    """L1 or L2 norm of the union of all affected parameter regions.

    All selected values form one vector. For L2, squared magnitudes are summed
    across all affected parameters before one square root is taken; the score is
    not a sum of per-module norms. Biases and custom operator parameters are
    included; buffers are not. With accepted removals, only the additional
    regions returned by `torch_kirigami.pruning.parameter_regions` are scored.

    A request without parameter effects raises `PlanningError`. Zero-valued
    parameters, and effects already covered by accepted removals, can score
    zero. Scores are heuristics, not accuracy guarantees; use a custom `Metric`
    for a different scoring scope.

    Args:
        p: Norm exponent, either 1 or 2. Default: 2. L2 includes the square root.

    Raises:
        ValueError: `p` is not 1 or 2.
    """

    def __init__(self, p: int = 2) -> None:
        if p not in (1, 2):
            raise ValueError("Magnitude supports p=1 or p=2")
        self.p = p

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score the parameter regions each candidate removes beyond the accepted ones.

        Args:
            context: Planner access to candidates, graph bindings, and impacts.
            candidates: Requests to score independently.
            accepted_impact: Complete dependency closure of prior removals.

        Returns:
            One score per candidate, in input order. The planning context checks
            that scores are finite when it calls this metric.

        Raises:
            torch_kirigami.pruning.PlanningError: Influence is incomplete, or a
                request that is not already covered has no parameter effects.
        """
        return _parameter_scores(self, context, candidates, accepted_impact=accepted_impact)


class WeightTaylor:
    """First-order Taylor score from real weights and the gradients already in `.grad`.

    The library never calls backward and never unscales AMP gradients: the
    caller computes the gradients and owns loss reduction, gradient
    accumulation, and unscaling. This is not a per-example Fisher estimator.
    All affected Parameters contribute, including biases and custom operator
    parameters; buffers do not. A request without parameter effects raises
    `PlanningError`. Zero contributions, and effects already covered by
    accepted removals, can score zero. Scores are heuristics, not accuracy
    guarantees.

    In `"elementwise_abs"` mode, per-position sums along one axis can be cached
    and reused, within the configured cache limits. The cache is keyed by tensor
    versions, so weight or gradient updates invalidate it. `"joint_abs"` mode
    always reduces the selected regions directly, so signed cancellation and
    overflow behave exactly as for an uncached computation.

    Args:
        mode: `"elementwise_abs"` sums `abs(weight * gradient)`. `"joint_abs"`
            takes one absolute value after summing signed products across all
            newly affected parameter regions. Default: `"elementwise_abs"`.

    Raises:
        ValueError: `mode` is unknown.
    """

    def __init__(self, mode: str = "elementwise_abs") -> None:
        if mode not in ("elementwise_abs", "joint_abs"):
            raise ValueError("Unknown Taylor mode")
        self.mode = mode
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
        self._cache_settings: tuple[int, ...] | None = None

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score the parameter regions each candidate removes, using current gradients.

        Args:
            context: Planner access to candidates, graph bindings, and impacts.
            candidates: Requests to score independently.
            accepted_impact: Complete dependency closure of prior removals.

        Returns:
            One Taylor score per candidate, in input order. The planning context
            checks that scores are finite when it calls this metric.

        Raises:
            torch_kirigami.pruning.PlanningError: Influence is incomplete, a
                request that is not already covered has no parameter effects, or
                an affected parameter is complex or has no dense gradient.
        """
        self._sync_cache_settings()
        return _parameter_scores(self, context, candidates, accepted_impact=accepted_impact)

    def _sync_cache_settings(self) -> None:
        """Release old statistics before scoring under changed resource settings."""
        settings = (
            config.taylor_cache_size,
            config.taylor_cache_positions,
            config.taylor_chunk_elements,
        )
        if settings != self._cache_settings:
            self._axis_statistics.clear()
            self._statistic_positions = 0
            self._cache_settings = settings

    def _axis_sums(self, weight: torch.Tensor, dim: int) -> tuple[float, ...] | None:
        """Reduce absolute axis statistics when their reuse is supported."""
        self._sync_cache_settings()
        gradient = weight.grad
        assert gradient is not None
        if self.mode != "elementwise_abs" or not config.taylor_cache_size:
            return None
        width = weight.shape[dim]
        elements_per_position = weight.numel() // width
        if (
            width > config.taylor_cache_positions
            or not config.taylor_chunk_elements
            or elements_per_position > config.taylor_chunk_elements
        ):
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
        chunk_size = max(1, config.taylor_chunk_elements // max(elements_per_position, 1))
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
            len(self._axis_statistics) > config.taylor_cache_size
            or self._statistic_positions > config.taylor_cache_positions
        ):
            _old_key, (_weight, _gradient, old_values) = self._axis_statistics.popitem(last=False)
            self._statistic_positions -= len(old_values)
        return values


_TAYLOR_SCORE = WeightTaylor.score


class GroupMagnitude:
    """Affected parameter energy normalized by the mean single-position energy on an axis.

    The numerator is the energy `sum(abs(value) ** p)` over the union of the
    candidate's affected parameter regions. The denominator is the mean energy
    of single-position requests on `candidate.axis`. All affected Parameters
    contribute, including biases and custom operator parameters; buffers do not.
    Shared parameters and overlapping regions count once. Operator types and
    parameter names do not affect which parameters are included.

    A candidate must select complete slices along its declared axis only. With
    no accepted removals, the denominator averages over all positions of the
    axis, even when the candidates cover only some positions or contain several
    positions. A block candidate uses the energy of its joint region union as
    the numerator, without dividing by the block size.

    With accepted removals, the numerator excludes parameter regions already in
    `accepted_impact`, and the denominator averages over all axis positions not
    yet removed, including the positions being scored. Scoring does not run the
    model or recalibrate activations or gradients. A request without parameter
    effects raises `PlanningError`. Zero-valued parameters, and requests that
    are already covered, can score zero.

    This heuristic is inspired by DepGraph (CVPR 2023, Section 3.3,
    https://arxiv.org/abs/2301.12900) and Torch-Pruning's
    `GroupMagnitudeImportance`, but it is not a full reproduction of either. It
    uses mean normalization and region deduplication, and it does not implement
    the paper's sparse training or Top-N normalization. It does not estimate
    task loss, and its scores are not accuracy guarantees. Use a custom `Metric`
    for another scoring policy.

    Args:
        p: Energy exponent, either 1 or 2. Default: 2. Unlike `Magnitude(p=2)`,
            no final square root is applied.

    Raises:
        ValueError: `p` is not 1 or 2.
    """

    def __init__(self, p: int = 2) -> None:
        if p not in (1, 2):
            raise ValueError("GroupMagnitude supports p=1 or p=2")
        self.p = p
        # Scalars and immutable coordinate metadata only; no context, model,
        # activation, Parameter, or full Impact survives a score call here.
        self._normalizers: OrderedDict[
            tuple[object, ...], tuple[float, float, tuple[float, ...]]
        ] = OrderedDict()
        self._context: weakref.ReferenceType[MetricContext] | None = None
        self._cache_size = config.normalization_cache_size

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score each candidate's normalized energy beyond the accepted removals.

        Scores do not depend on how candidates are batched.

        Args:
            context: Planner access to candidates, graph bindings, and impacts.
            candidates: Requests to score independently.
            accepted_impact: Complete dependency closure of prior removals.

        Returns:
            One normalized score per candidate, in input order. The planning
            context checks that scores are finite when it calls this metric.

        Raises:
            torch_kirigami.pruning.PlanningError: A candidate does not select complete slices of its
                declared axis, influence is incomplete, or scoring has no
                parameter basis. Also raised for nonfinite normalization values
                or nonzero joint energy with a zero single-position baseline.
        """
        if self._cache_size != config.normalization_cache_size:
            self._normalizers.clear()
            self._cache_size = config.normalization_cache_size
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        domains = tuple(_candidate_domain(candidate) for candidate in candidates)
        if not domains:
            return []
        parameters = _parameter_bindings(context)

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
            versions = tuple(
                sorted((id(tensor), tensor._version) for tensor in parameters.values())
            )
        except RuntimeError:
            versions = None
        # Each population request extends the accepted request in this order.
        # Equal closures of other requests can differ in which of these joint
        # requests reach an analysis limit.
        accepted = tuple((s.tensor, s.regions) for s in accepted_impact.requested)
        normalizers = {}
        for axis in dict.fromkeys(axis for axis, _count, _first in domains):
            key = (context.graph.id, self.p, axis, accepted, versions)
            cacheable = (
                versions is not None
                and self._context is not None
                and config.normalization_cache_size > 0
            )
            if cacheable and key in self._normalizers:
                normalizers[axis] = self._normalizers[key]
                self._normalizers.move_to_end(key)
                continue
            removed = frozenset(
                accepted_impact.selection(axis.tensor).fully_selected_indices(axis.dim)
            )
            surviving = [i for i in range(axis.tensor.shape[axis.dim]) if i not in removed]
            population = tuple(
                Candidate(f"normalization:{position}", (axis.select([position]),), axis)
                for position in surviving
            )
            norms = _parameter_scores(self, context, population, accepted_impact=accepted_impact)
            scale, scaled_mean_energy = _normalization(norms, self.p)
            # A singleton's normalization query is exactly its conditional score
            # query. Store scalar results instead of propagating/evaluating them
            # again when a later score batch visits the same axis positions.
            position_norms = [0.0] * axis.tensor.shape[axis.dim]
            for position, norm in zip(surviving, norms, strict=True):
                position_norms[position] = norm
            entry = (scale, scaled_mean_energy, tuple(position_norms))
            normalizers[axis] = entry
            if cacheable:
                self._normalizers[key] = entry
                if len(self._normalizers) > config.normalization_cache_size:
                    self._normalizers.popitem(last=False)
        norms = [0.0] * len(candidates)
        joint_indices = []
        for index, (axis, count, first) in enumerate(domains):
            if count == 1:
                norms[index] = normalizers[axis][2][first]
            else:
                joint_indices.append(index)
        if joint_indices:
            joint_norms = _parameter_scores(
                self,
                context,
                tuple(candidates[index] for index in joint_indices),
                accepted_impact=accepted_impact,
            )
            for index, norm in zip(joint_indices, joint_norms, strict=True):
                norms[index] = norm
        scores = []
        for value, (axis, _count, _first) in zip(norms, domains, strict=True):
            scale, scaled_mean_energy, _position_norms = normalizers[axis]
            scores.append(_group_score(value, scale, scaled_mean_energy, self.p))
        return scores


def _candidate_domain(candidate: Candidate) -> tuple[AxisRef, int, int]:
    """Return the declared axis, the number of selected slices and the first one.

    The seeds are combined as boxes, so a valid domain never reaches the region
    limit, whatever the seed count and order. Ambiguous domains are rejected
    rather than guessing an axis.
    """
    axis = candidate.axis
    if axis is None or any(selection.tensor != axis.tensor for selection in candidate.remove):
        raise PlanningError("GroupMagnitude requires a candidate selecting its declared axis")
    boxes = canonical_boxes(
        tuple(box for selection in candidate.remove for box in region_boxes(selection.regions))
    )
    # A union of complete slices is one canonical box that spans every other axis.
    if len(boxes) != 1 or any(
        intervals != ((0, size),)
        for dim, (intervals, size) in enumerate(zip(boxes[0], axis.tensor.shape, strict=True))
        if dim != axis.dim
    ):
        raise PlanningError("GroupMagnitude requires complete slices of its declared axis")
    intervals = boxes[0][axis.dim]
    return axis, sum(stop - start for start, stop in intervals), intervals[0][0]


def _parameter_bindings(context: MetricContext) -> dict[TensorRef, nn.Parameter]:
    """Select registered parameters without operator-specific scoring roles."""
    return {
        ref: cast(nn.Parameter, tensor)
        for ref, tensor in context.graph.tensor_bindings()
        if ref.kind == "parameter"
    }


def _group_score(norm: float, scale: float, scaled_mean_energy: float, p: int) -> float:
    """Normalize one union norm with statistics from `_normalization`.

    Raise when joint effects have nonzero energy but all single-position effects
    have zero energy. Ordinary and accelerated scoring share this calculation.
    """
    if scale:
        return (norm / scale) ** p / scaled_mean_energy
    if norm != 0:
        raise PlanningError(
            "GroupMagnitude has zero singleton normalization but a nonzero joint score. "
            "Use a custom metric for this jointly activated structure."
        )
    return 0.0


def _normalization(norms: Sequence[float], p: int) -> tuple[float, float]:
    """Return the maximum norm and the mean of `(norm / maximum) ** p`.

    Inputs are single-position L1 or L2 norms, not energies. Return `(0, 0)`
    for empty or all-zero inputs. Scaling avoids overflow in the mean energy.
    """
    if not all(math.isfinite(value) for value in norms):
        raise PlanningError("GroupMagnitude normalization contains nonfinite weights")
    scale = max(norms, default=0.0)
    scaled_mean_energy = (
        math.fsum((value / scale) ** p for value in norms) / len(norms) if scale else 0.0
    )
    return scale, scaled_mean_energy


def _magnitude_norm(values: torch.Tensor, p: int) -> torch.Tensor:
    """Reduce one region to an L1 or overflow-safe L2 norm."""
    if p == 2:
        dtype = (
            torch.float64 if values.dtype in (torch.float64, torch.complex128) else torch.float32
        )
        values = torch.view_as_real(values.resolve_conj()) if values.is_complex() else values
        values = values.to(dtype).abs()
    else:
        values = values.to(torch.complex128 if values.is_complex() else torch.float64).abs()
    values = values.reshape(-1)
    if p == 1:
        return values.sum(-1, dtype=torch.float64)
    scale = values.amax(-1)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / divisor.unsqueeze(-1)).square().sum(-1, dtype=torch.float64).sqrt() * scale.to(
        torch.float64
    )


def _parameter_scores(
    metric: Magnitude | WeightTaylor | GroupMagnitude,
    context: MetricContext,
    candidates: tuple[Candidate, ...],
    *,
    accepted_impact: Impact,
) -> list[float]:
    """Compute additional-region norms or Taylor sums with joint propagation.

    Both magnitude classes receive union L1 or L2 norms here. `GroupMagnitude`
    converts these norms to normalized energies after this helper returns.
    """
    context.graph.validate_impact(accepted_impact)
    context.require_complete(accepted_impact)
    bindings = dict(context.graph.tensor_bindings())
    totals = []
    for candidate in candidates:
        parts = _additional_boxes(context, candidate, accepted_impact)
        totals.append(_region_totals(metric, bindings, parts))
        if not parts and not all(
            seed.issubset(accepted_impact.selection(seed.tensor)) for seed in candidate.remove
        ):
            # Empty marginal regions can mean prior deletions already cover
            # the parameter effects. Check the candidate alone only on this
            # uncommon path, without adding queries to ordinary scoring.
            original = context.impact(candidate.remove)
            context.require_complete(original)
            if not original.parameters:
                raise PlanningError(
                    f"{type(metric).__name__} request {candidate.key!r} has no affected parameters. "
                    "Use a custom metric for parameter-free changes."
                )
    return _combine_totals(metric, totals)


def _region_totals(
    metric: Magnitude | WeightTaylor | GroupMagnitude,
    bindings: Mapping[TensorRef, torch.Tensor],
    parts: Iterable[tuple[TensorRef, tuple[IntervalBox, ...]]],
) -> dict[torch.device, torch.Tensor | float]:
    """Reduce one request's parameter boxes from `canonical_boxes` to one value per device.

    Parameters are reduced in tensor-ID order, and each one box at a time.
    The value, and the choice of cached Taylor axis statistics, therefore
    depend only on the selected coordinates, not on the propagation order or
    region decomposition that produced them. Temporary memory follows
    `gather_intervals` for the largest box. Accelerated rankings call this
    reduction to reproduce ordinary scores exactly.
    """
    l2 = isinstance(metric, (Magnitude, GroupMagnitude)) and metric.p == 2
    taylor = isinstance(metric, WeightTaylor)
    native_taylor = (
        type(metric) is WeightTaylor
        and "score" not in vars(metric)
        and type(metric).score is _TAYLOR_SCORE
    )
    totals: dict[torch.device, torch.Tensor | float] = {}
    with torch.no_grad():
        for ref, boxes in sorted(parts, key=lambda part: part[0].id):
            weight = bindings[ref]
            if taylor and (weight.is_complex() or weight.grad is None or weight.grad.is_sparse):
                raise PlanningError(
                    "WeightTaylor requires real parameters and dense current gradients"
                )
            if native_taylor and len(boxes) == 1:
                box = boxes[0]
                partial = tuple(
                    dim
                    for dim, intervals in enumerate(box)
                    if intervals != ((0, weight.shape[dim]),)
                )
                if len(partial) == 1:
                    dim = partial[0]
                    values = cast(WeightTaylor, metric)._axis_sums(weight, dim)
                    if values is not None:
                        value = sum(
                            values[position]
                            for start, stop in box[dim]
                            for position in range(start, stop)
                        )
                        # Preserve ordinary region reduction and error
                        # behavior when cached statistics are nonfinite.
                        if math.isfinite(value):
                            previous = totals.get(weight.device)
                            totals[weight.device] = value if previous is None else previous + value
                            continue
            for box in boxes:
                values = gather_intervals(weight.detach(), box)
                if taylor:
                    # Promote before multiplication: a float64 reduction cannot
                    # recover products already underflowed/overflowed in float32.
                    values = values.to(torch.float64) * gather_intervals(
                        weight.grad.detach(), box
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
    return totals


def _combine_totals(
    metric: Magnitude | WeightTaylor | GroupMagnitude,
    totals: Sequence[Mapping[torch.device, torch.Tensor | float]],
) -> list[float]:
    """Combine `_region_totals` results into one score per request.

    Devices are combined in name order, so a score does not depend on which
    devices other requests in the batch use.
    """
    l2 = isinstance(metric, (Magnitude, GroupMagnitude)) and metric.p == 2
    entries: dict[torch.device, list[tuple[int, torch.Tensor | float]]] = {}
    for index, partial in enumerate(totals):
        for device, value in partial.items():
            entries.setdefault(device, []).append((index, value))
    values: list[dict[str, float]] = [{} for _ in totals]
    # One transfer per device/batch, rather than one synchronization per
    # affected parameter region. Different parameter devices can still contribute.
    for device, device_entries in entries.items():
        tensors = [value for _, value in device_entries if isinstance(value, torch.Tensor)]
        transferred = iter(torch.stack(tensors).cpu().tolist() if tensors else ())
        for index, value in device_entries:
            values[index][str(device)] = (
                next(transferred) if isinstance(value, torch.Tensor) else value
            )
    result = []
    for partial in values:
        total = 0.0
        for _device, value in sorted(partial.items()):
            total = math.hypot(total, value) if l2 else total + value
        result.append(total)
    if isinstance(metric, WeightTaylor) and metric.mode == "joint_abs":
        result = [abs(value) for value in result]
    return result
