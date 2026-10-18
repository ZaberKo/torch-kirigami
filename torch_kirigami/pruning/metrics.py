"""Region-union weight scores with explicit gradient ownership."""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from ..dep_graph.contracts import Impact
from ..dep_graph.selection import AxisRef, IndexSet, Selection, TensorRef, gather_region
from ..settings import config
from .types import Candidate, MetricContext, PlanningError


def parameter_regions(
    context: MetricContext,
    candidate: Candidate,
    *,
    accepted_impact: Impact,
) -> tuple[Selection, ...]:
    """Describe additional parameter removals if one candidate is accepted.

    Analyze accepted and candidate requests together, then subtract accepted
    regions. Shared parameters and overlapping regions count once. Joint requests
    can activate effects that neither request produces separately.
    This query does not change parameter values or choose a score formula.

    Args:
        context: Dependency queries and original model bindings.
        candidate: One request, which can contain multiple selections.
        accepted_impact: Complete dependency closure of previously accepted
            removal requests from the same graph. Use `context.impact(())`
            to query a candidate without prior removals.

    Returns:
        Nonempty selections in original coordinates, one per affected `nn.Parameter`.
        Buffers are excluded. An empty tuple is valid and carries no score.
        Use `context.graph.tensor_bindings()` to resolve parameter values.

    Raises:
        PlanningError: Dependency influence is incomplete. Constraint violations
            do not prevent this query. It does not certify an executable plan.
        ValueError: References or the accepted impact belong to another graph.
    """
    context.graph.validate_impact(accepted_impact)
    context.require_complete(accepted_impact)
    impact = context.impact((*accepted_impact.requested, *candidate.remove))
    context.require_complete(impact)
    regions = []
    for selection in impact.parameters:
        remaining = selection.subtract(accepted_impact.selection(selection.tensor))
        if remaining:
            regions.append(remaining)
    return tuple(regions)


class Magnitude:
    """Compute L1 or L2 over the union of all affected parameter regions.

    Include biases and custom operator parameters, but exclude buffers.
    Requests without parameter effects raise `PlanningError`. Zero-valued
    parameters and effects already covered by accepted removals can score zero.
    Use a custom Metric for a different scoring scope.

    Treat all selected values as one vector. For L2, sum squared magnitudes
    across parameters before taking one square root. Do not sum per-module norms.
    With `accepted_impact`, score only additional parameter regions returned by
    `torch_kirigami.pruning.parameter_regions`.

    Args:
        p: Norm exponent, either 1 or 2. Default: 2. L2 includes the square root.
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
        """Score newly removed regions after the complete accepted selection.

        Args:
            context: Planner access to candidates, graph bindings, and impacts.
            candidates: Requests to score independently in the returned order.
            accepted_impact: Complete dependency closure of prior removals.

        Returns:
            One score per candidate, in input order. The planning context checks
            that scores are finite when it calls this metric.

        Raises:
            PlanningError: Influence is incomplete, or a new request has no
                parameter effects from which to compute a score.
        """
        return _parameter_scores(self, context, candidates, accepted_impact=accepted_impact)


class WeightTaylor:
    """Score real weights using caller-provided current, unscaled gradients.

    The caller owns loss reduction, gradient accumulation, and AMP unscaling.
    This is not a per-example Fisher estimator and never invokes backward.
    Elementwise absolute Taylor reuses complete axis slices within the configured
    scalar-vector and position limits. Signed joint Taylor preserves
    original-region reductions, including cancellation and overflow behavior.
    Irregular conditional regions and inference tensors are not cached.
    All affected Parameters contribute, including biases and custom parameters.
    Requests without parameter effects raise `PlanningError`. Zero contributions
    and effects already covered by accepted removals can score zero.

    Args:
        mode: `"elementwise_abs"` sums `abs(weight * gradient)`. `"joint_abs"`
            takes one absolute value after summing signed products across all
            newly affected parameter regions. Default: `"elementwise_abs"`.
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
        """Score newly removed regions using current parameter gradients.

        Args:
            context: Planner access to candidates, graph bindings, and impacts.
            candidates: Requests to score independently in the returned order.
            accepted_impact: Complete dependency closure of prior removals.

        Returns:
            One Taylor score per candidate, in input order. The planning context
            checks that scores are finite when it calls this metric.

        Raises:
            PlanningError: Influence is incomplete, a new request has no parameter
                effects, or an affected parameter lacks a dense current gradient
                or has complex values.
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
    """Normalize affected parameter energy against a declared candidate axis.

    Sum `abs(parameter) ** p` over the candidate's affected parameter regions.
    Divide by the mean energy of single-position requests on `candidate.axis`.
    All affected Parameters participate, including biases and custom operator
    parameters. Buffers do not participate. Shared parameters and overlapping
    regions count once. Operator types and parameter names do not affect inclusion.

    A candidate must select complete slices solely along its declared axis.
    With no accepted removals, the denominator includes all axis positions,
    even when candidates offer only a subset or contain multiple positions.
    Blocks use their joint region union as the numerator, without division by
    the block size.

    Conditional scores exclude parameter regions already in `accepted_impact`.
    The denominator then uses all remaining axis positions, including the positions
    currently under evaluation. Scoring does not execute the model or recalibrate
    activations or gradients. A request without parameter effects raises
    `PlanningError`. Zero-valued parameters and already covered requests can score zero.

    This heuristic draws on DepGraph (CVPR 2023), Section 3.3, and Torch-Pruning's
    `GroupMagnitudeImportance`. It uses mean normalization and region deduplication.
    It does not implement the paper's sparse training or Top-N normalization.
    It does not estimate task loss. Use a custom Metric for another scoring policy.
    Paper: https://arxiv.org/abs/2301.12900.

    Args:
        p: Energy exponent, either 1 or 2. Default: 2. No final square root
            is applied, unlike `Magnitude(p=2)`.
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
        """Return normalized conditional energies, independent of batching.

        Args:
            context: Planner access to candidates, graph bindings, and impacts.
            candidates: Requests to score independently in the returned order.
            accepted_impact: Complete dependency closure of prior removals.

        Returns:
            One normalized score per candidate, in input order. The planning
            context checks that scores are finite when it calls this metric.

        Raises:
            PlanningError: A candidate does not select complete slices of its
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
        accepted_regions = tuple((s.tensor, s.regions) for s in accepted_impact.selections.values())
        normalizers = {}
        for axis in dict.fromkeys(axis for axis, _indices in domains):
            key = (context.graph.id, self.p, axis, accepted_regions, versions)
            cacheable = (
                versions is not None
                and self._context is not None
                and config.normalization_cache_size > 0
            )
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
        for index, (axis, positions) in enumerate(domains):
            if len(positions) == 1:
                norms[index] = normalizers[axis][2][next(iter(positions))]
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
        for value, (axis, _positions) in zip(norms, domains, strict=True):
            scale, scaled_mean_energy, _position_norms = normalizers[axis]
            scores.append(_group_score(value, scale, scaled_mean_energy, self.p))
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
            selections = parameter_regions(context, candidate, accepted_impact=accepted_impact)
            totals: dict[torch.device, torch.Tensor | float] = {}
            for selection in selections:
                weight = bindings[selection.tensor]
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
            if not selections and not all(
                not seed.subtract(accepted_impact.selection(seed.tensor))
                for seed in candidate.remove
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
