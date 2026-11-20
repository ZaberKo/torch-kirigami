"""Region-union weight scores with explicit gradient ownership."""

from __future__ import annotations

import math
import weakref
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from functools import partial, reduce
from operator import add

import torch

from ..dep_graph.contracts import Impact
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import (
    AxisRef,
    IndexSet,
    IntervalBox,
    Selection,
    TensorRef,
    canonical_boxes,
    gather_intervals,
    region_boxes,
    slice_intervals,
)
from ..settings import config
from .types import Candidate, MetricContext, PlanningError

_ParameterBoxes = tuple[tuple[TensorRef, tuple[IntervalBox, ...]], ...]


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
        limit. Use `parameter_boxes` to score without rebuilding a selection.
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


def parameter_boxes(
    context: MetricContext, candidate: Candidate, *, accepted_impact: Impact
) -> _ParameterBoxes:
    """Return additional parameter coordinates as disjoint interval boxes.

    The coordinate set is the same as in `parameter_regions`. Its decomposition
    depends only on the coordinates, and does not build a bounded `Selection`.

    Args:
        context: Dependency queries and original model bindings.
        candidate: The removal to score.
        accepted_impact: Complete dependency closure of prior removals.

    Returns:
        Pairs of parameter reference and nonempty box sequence, in parameter
        order. Each box has one tuple of half-open intervals per tensor axis.
        Pass each box to `torch_kirigami.dep_graph.selection.gather_intervals`
        to read the selected tensor elements.

    Raises:
        torch_kirigami.pruning.PlanningError: The dependency closure is incomplete.
        ValueError: A selection or impact belongs to another graph.
        torch_kirigami.StaleGraphError: The model changed after capture.
    """
    parts = []
    for selection, accepted in _joint_parameters(context, candidate, accepted_impact):
        boxes = canonical_boxes(region_boxes(selection.regions), region_boxes(accepted.regions))
        if boxes:
            parts.append((selection.tensor, boxes))
    return tuple(parts)


def _candidate_parameter_boxes(
    context: MetricContext, candidates: Iterable[Candidate], accepted_impact: Impact
) -> Iterator[_ParameterBoxes]:
    """Yield disjoint additional boxes for each candidate with a parameter basis."""
    for candidate in candidates:
        boxes = parameter_boxes(context, candidate, accepted_impact=accepted_impact)
        if not boxes and not all(
            seed.issubset(accepted_impact.selection(seed.tensor)) for seed in candidate.remove
        ):
            # A prior removal can cover a candidate's parameters without
            # covering its seeds. Check the candidate alone in that case.
            original = context.impact(candidate.remove)
            context.require_complete(original)
            if not original.parameters:
                raise PlanningError(
                    f"Candidate {candidate.key!r} has no affected parameters. "
                    "Use a custom metric for parameter-free changes."
                )
        yield boxes


class Magnitude:
    """L1 or L2 norm of the union of all affected parameter regions.

    All selected values form one vector. For L2, squared magnitudes are summed
    across all affected parameters before one square root is taken; the score is
    not a sum of per-module norms. Biases and custom operator parameters are
    included; buffers are not. With accepted removals, only the additional
    regions returned by `torch_kirigami.pruning.parameter_boxes` are scored.

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
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        return _magnitude_norms(
            dict(context.graph.tensor_bindings()),
            _candidate_parameter_boxes(context, candidates, accepted_impact),
            p=self.p,
        )

    def _norm_positions(self, requested: IndexSet) -> IndexSet | None:
        """Require norms only at the positions the candidates request."""
        return requested

    def _scores_from_norms(
        self,
        axis: AxisRef,
        positions: Iterable[int],
        removed: Collection[int],
        norms: Mapping[int, float],
    ) -> dict[int, float]:
        """Use the available finite norms without axis normalization."""
        return {i: norms[i] for i in positions if math.isfinite(norms[i])}


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
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        bindings = dict(context.graph.tensor_bindings())
        if self.mode == "joint_abs":
            box_sums = _signed_taylor_sums
            finalize = abs
        else:
            use_axis_cache = (
                type(self) is WeightTaylor
                and "score" not in vars(self)
                and type(self).score is _TAYLOR_SCORE
            )
            box_sums = partial(self._absolute_taylor_sums, use_axis_cache=use_axis_cache)
            finalize = float
        totals = [
            self._region_sums(bindings, boxes, box_sums)
            for boxes in _candidate_parameter_boxes(context, candidates, accepted_impact)
        ]
        return [finalize(reduce(add, values, 0.0)) for values in _device_scalar_values(totals)]

    @staticmethod
    def _region_sums(
        bindings: Mapping[TensorRef, torch.Tensor],
        parts: _ParameterBoxes,
        box_sums: Callable[
            [torch.Tensor, torch.Tensor, tuple[IntervalBox, ...]], Iterable[torch.Tensor | float]
        ],
    ) -> dict[torch.device, torch.Tensor | float]:
        """Sum Taylor products per device in parameter and box order."""
        totals: dict[torch.device, torch.Tensor | float] = {}
        with torch.no_grad():
            for ref, boxes in sorted(parts, key=lambda part: part[0].id):
                weight = bindings[ref]
                gradient = weight.grad
                if weight.is_complex() or gradient is None or gradient.is_sparse:
                    raise PlanningError(
                        "WeightTaylor requires real parameters and dense current gradients"
                    )
                device = weight.device
                for product_sum in box_sums(weight, gradient, boxes):
                    previous = totals.get(device)
                    totals[device] = product_sum if previous is None else previous + product_sum
        return totals

    def _absolute_taylor_sums(
        self,
        weight: torch.Tensor,
        gradient: torch.Tensor,
        boxes: tuple[IntervalBox, ...],
        *,
        use_axis_cache: bool,
    ) -> Iterator[float | torch.Tensor]:
        """Use exact axis coverage for cached absolute products, or read the boxes."""
        if use_axis_cache and len(boxes) == 1:
            box = boxes[0]
            restricted = [
                dim for dim, intervals in enumerate(box) if intervals != ((0, weight.shape[dim]),)
            ]
            if len(restricted) == 1:
                dim = restricted[0]
                values = self._axis_sums(weight, dim)
                if values is not None:
                    cached = sum(values[i] for start, stop in box[dim] for i in range(start, stop))
                    if math.isfinite(cached):
                        yield cached
                        return
        for box in boxes:
            yield _taylor_products(weight, gradient, box).abs().sum(dtype=torch.float64)

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
        gradient = weight.grad
        assert gradient is not None
        if not config.taylor_cache_size:
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
            key = (id(weight), weight._version, id(gradient), gradient._version, dim)
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


def _taylor_products(
    weight: torch.Tensor, gradient: torch.Tensor, box: IntervalBox
) -> torch.Tensor:
    """Read one box, promoting before multiplying to preserve the product's range."""
    return gather_intervals(weight.detach(), box).to(torch.float64) * gather_intervals(
        gradient.detach(), box
    ).to(torch.float64)


def _signed_taylor_sums(
    weight: torch.Tensor, gradient: torch.Tensor, boxes: tuple[IntervalBox, ...]
) -> Iterator[torch.Tensor]:
    """Preserve signed contributions until the metric has combined every box."""
    for box in boxes:
        yield _taylor_products(weight, gradient, box).sum(dtype=torch.float64)


class PrunerZero:
    """Score structural removals with the fixed Pruner-Zero expression.

    For each matrix or kernel, min-max scale the supplied gradient statistics
    over the whole tensor and multiply elementwise by squared weights.
    Constant statistics give zero importance. Each candidate sums the scores
    of the additional coordinates it removes through the dependency graph.
    Parameters of rank below two and buffers do not contribute.

    This adapts the per-weight expression from *Pruner-Zero: Evolving Symbolic
    Pruning Metric From Scratch for Large Language Models* (ICML 2024) to
    structural groups. The caller supplies the statistics and decides how to
    collect them. Recreate this metric after recalibration,
    weight updates, or graph rebuilding. The snapshot stays fixed while
    `DynamicGreedy` evaluates overlapping removals.

    Args:
        graph: Dependency graph for the calibrated model.
        gradient_energy: Nonnegative gradient statistics, keyed by parameter
            reference from `graph`. For example, supply the per-weight L2 norm
            across calibration-batch gradients. Each tensor must
            have the shape and device of its parameter.

    Raises:
        torch_kirigami.pruning.PlanningError: A reference or statistic is invalid,
            or the computed per-weight importance is nonfinite.
    """

    @torch.no_grad()
    def __init__(
        self, graph: DependencyGraph, gradient_energy: Mapping[TensorRef, torch.Tensor]
    ) -> None:
        self._graph_id = graph.id
        self._importance: dict[TensorRef, torch.Tensor] = {}
        bindings = dict(graph.tensor_bindings())
        for ref, energy in gradient_energy.items():
            if ref not in bindings or ref.kind != "parameter" or len(ref.shape) < 2:
                raise PlanningError("Pruner-Zero statistics require a matrix or kernel parameter")
            weight = bindings[ref].detach()
            if (
                not weight.is_floating_point()
                or weight.layout != torch.strided
                or not energy.is_floating_point()
                or energy.layout != torch.strided
                or energy.shape != weight.shape
                or energy.device != weight.device
            ):
                raise PlanningError(f"Pruner-Zero requires gradient statistics matching {ref.id}")
            dtype = torch.promote_types(
                torch.float32, torch.promote_types(weight.dtype, energy.dtype)
            )
            energy = energy.to(dtype)
            low, high = torch.aminmax(energy)
            if not (torch.isfinite(high) & torch.isfinite(low) & (low >= 0)):
                raise PlanningError(
                    f"Pruner-Zero requires finite nonnegative statistics for {ref.id}"
                )
            span = high - low
            scale = torch.where(span > 0, span, torch.ones_like(span))
            # Combine the scales before squaring. Taking the roots separately
            # also keeps a small, positive ratio from rounding to zero first.
            terms = (energy - low).sqrt_().div_(scale.sqrt())
            terms.mul_(weight).square_()
            if not torch.isfinite(terms).all():
                raise PlanningError(f"Pruner-Zero importance is nonfinite for {ref.id}")
            self._importance[ref] = terms

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Sum the fixed importances at each candidate's additional coordinates.

        Args:
            context: Planning context for the graph used to construct the metric.
            candidates: Candidate requests to score separately.
            accepted_impact: Complete structural effect of prior removals.

        Returns:
            One score per candidate in input order.

        Raises:
            torch_kirigami.pruning.PlanningError: The graph changed, influence is
                incomplete, no matrix or kernel is affected, or a needed statistic
                is missing.
        """
        if context.graph.id != self._graph_id:
            raise PlanningError("Pruner-Zero requires calibration on the current dependency graph")
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        totals: list[dict[torch.device, torch.Tensor | float]] = []
        for candidate in candidates:
            partial: dict[torch.device, torch.Tensor | float] = {}
            for ref, boxes in parameter_boxes(context, candidate, accepted_impact=accepted_impact):
                if len(ref.shape) < 2:
                    continue
                importance = self._importance.get(ref)
                if importance is None:
                    raise PlanningError(f"Pruner-Zero calibration has no statistics for {ref.id}")
                for box in boxes:
                    value = gather_intervals(importance, box).sum(dtype=torch.float64)
                    device = importance.device
                    partial[device] = partial[device] + value if device in partial else value
            if not partial:
                original = context.impact(candidate.remove)
                context.require_complete(original)
                if not any(len(selection.tensor.shape) >= 2 for selection in original.parameters):
                    raise PlanningError("Pruner-Zero requires an affected matrix or kernel")
            totals.append(partial)
        return [reduce(add, values, 0.0) for values in _device_scalar_values(totals)]


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
                parameter basis. Also raised for nonfinite normalization values,
                an out-of-range normalized score, or nonzero joint energy with
                a zero single-position baseline.
        """
        if self._cache_size != config.normalization_cache_size:
            self._normalizers.clear()
            self._cache_size = config.normalization_cache_size
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        domains = tuple(_candidate_domain(candidate) for candidate in candidates)
        if not domains:
            return []
        parameters = {
            ref: tensor
            for ref, tensor in context.graph.tensor_bindings()
            if ref.kind == "parameter"
        }

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
            surviving = self._surviving_positions(axis, removed)
            population = tuple(
                Candidate(f"normalization:{position}", (axis.select([position]),), axis)
                for position in surviving
            )
            norms = _magnitude_norms(
                parameters,
                _candidate_parameter_boxes(context, population, accepted_impact),
                p=self.p,
            )
            scale, scaled_mean_energy = self._normalization(norms)
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
            joint_norms = _magnitude_norms(
                parameters,
                _candidate_parameter_boxes(
                    context, (candidates[index] for index in joint_indices), accepted_impact
                ),
                p=self.p,
            )
            for index, norm in zip(joint_indices, joint_norms, strict=True):
                norms[index] = norm
        scores = []
        for value, (axis, _count, _first) in zip(norms, domains, strict=True):
            scale, scaled_mean_energy, _position_norms = normalizers[axis]
            scores.append(self._normalized_score(value, scale, scaled_mean_energy))
        return scores

    def _norm_positions(self, requested: IndexSet) -> IndexSet | None:
        """Require the whole axis to support the same population as ordinary scoring."""
        return None

    def _scores_from_norms(
        self,
        axis: AxisRef,
        positions: Iterable[int],
        removed: Collection[int],
        norms: Mapping[int, float],
    ) -> dict[int, float]:
        """Normalize norms whose entire surviving population has been proved."""
        surviving = self._surviving_positions(axis, removed)
        scale, mean = self._normalization([norms[i] for i in surviving])
        return {i: self._normalized_score(norms[i], scale, mean) for i in positions}

    @staticmethod
    def _surviving_positions(axis: AxisRef, removed: Collection[int]) -> tuple[int, ...]:
        """Enumerate the normalization population in original axis order."""
        return tuple(i for i in range(axis.tensor.shape[axis.dim]) if i not in removed)

    def _normalized_score(self, norm: float, scale: float, mean: float) -> float:
        """Divide one union energy by the average single-position energy."""
        if scale:
            try:
                return (norm / scale) ** self.p / mean
            except OverflowError as error:
                raise PlanningError(
                    "GroupMagnitude normalized score exceeds float range"
                ) from error
        if norm != 0:
            raise PlanningError(
                "GroupMagnitude has zero singleton normalization but a nonzero joint score. "
                "Use a custom metric for this jointly activated structure."
            )
        return 0.0

    def _normalization(self, norms: Sequence[float]) -> tuple[float, float]:
        """Return the scale and the average energy relative to it."""
        if not all(math.isfinite(value) for value in norms):
            raise PlanningError("GroupMagnitude normalization contains nonfinite weights")
        scale = max(norms, default=0.0)
        if not scale:
            return 0.0, 0.0
        return scale, math.fsum((norm / scale) ** self.p for norm in norms) / len(norms)


def _candidate_domain(candidate: Candidate) -> tuple[AxisRef, int, int]:
    """Return the declared axis, the number of selected slices and the first one.

    The seeds are combined as boxes, so a valid domain never reaches the region
    limit, whatever the seed count and order. Ambiguous domains are rejected
    rather than guessing an axis.
    """
    axis = candidate.axis
    if axis is None or any(selection.tensor != axis.tensor for selection in candidate.remove):
        raise PlanningError("GroupMagnitude requires a candidate selecting its declared axis")
    intervals = slice_intervals(axis, candidate.remove)
    if intervals is None:
        raise PlanningError("GroupMagnitude requires complete slices of its declared axis")
    return axis, sum(stop - start for start, stop in intervals), intervals[0][0]


def _l1_norm(values: torch.Tensor) -> torch.Tensor:
    """Sum absolute values of a real or complex box in float64."""
    dtype = torch.complex128 if values.is_complex() else torch.float64
    return values.to(dtype).abs().reshape(-1).sum(-1, dtype=torch.float64)


def _l2_norm(values: torch.Tensor) -> torch.Tensor:
    """Compute a scaled L2 norm without squaring values in their original range."""
    dtype = torch.float64 if values.dtype in (torch.float64, torch.complex128) else torch.float32
    values = torch.view_as_real(values.resolve_conj()) if values.is_complex() else values
    values = values.to(dtype).abs().reshape(-1)
    scale = values.amax(-1)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (values / divisor.unsqueeze(-1)).square().sum(-1, dtype=torch.float64).sqrt() * scale.to(
        torch.float64
    )


def _magnitude_norms(
    bindings: Mapping[TensorRef, torch.Tensor],
    requests: Iterable[_ParameterBoxes],
    *,
    p: int,
) -> list[float]:
    """Compute L1 or L2 norms from canonical boxes, in request order.

    Each request lists disjoint boxes once per parameter. Reduce detached
    values in tensor-ID and box order, one box at a time. Keep partial
    results on their devices until the batch is ready to transfer.
    """
    region_norm, merge_tensors, merge_scalars = (
        (_l2_norm, torch.hypot, math.hypot) if p == 2 else (_l1_norm, add, add)
    )
    totals = []
    for parts in requests:
        by_device: dict[torch.device, torch.Tensor] = {}
        for ref, boxes in sorted(parts, key=lambda part: part[0].id):
            weight = bindings[ref].detach()
            device = weight.device
            for box in boxes:
                norm = region_norm(gather_intervals(weight, box))
                previous = by_device.get(device)
                by_device[device] = norm if previous is None else merge_tensors(previous, norm)
        totals.append(by_device)
    return [reduce(merge_scalars, values, 0.0) for values in _device_scalar_values(totals)]


def _device_scalar_values(
    totals: Sequence[Mapping[torch.device, torch.Tensor | float]],
) -> list[list[float]]:
    """Read all scalar tensors in at most one transfer per device.

    Args:
        totals: One mapping of scalar values by device for each request.

    Returns:
        Values in request order, with each request's values in device-name
        order. Python scalar inputs require no transfer.
    """
    entries: dict[torch.device, list[tuple[int, torch.Tensor | float]]] = {}
    for index, by_device in enumerate(totals):
        for device, value in by_device.items():
            entries.setdefault(device, []).append((index, value))
    values: list[list[float]] = [[] for _ in totals]
    for device in sorted(entries, key=str):
        device_entries = entries[device]
        tensors = [value for _, value in device_entries if isinstance(value, torch.Tensor)]
        transferred = iter(torch.stack(tensors).cpu().tolist() if tensors else ())
        for index, value in device_entries:
            values[index].append(next(transferred) if isinstance(value, torch.Tensor) else value)
    return values
