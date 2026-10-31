"""CPD's static calibrated channel-importance metric."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..dep_graph.contracts import Impact
from ..dep_graph.selection import AxisRef, Selection
from .planner import PlanningContext
from .types import Candidate, MetricContext, PlanningError, StrategyResult


def fisher_score_tensor(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    *,
    dim: int,
    activation_elements: int | float,
) -> torch.Tensor:
    """Return CPD's per-channel, batch-normalized Fisher statistics on-device.

    Shared implementation behind :func:`fisher_scores`. The result stays on the
    gradient's device, so a training interval can accumulate many batches and
    pay one host transfer at its end. Converting per batch would serialize the
    training loop on eight device-to-host copies per step instead.

    Args:
        weight: Parameter values from the backward pass.
        gradient: Dense task-loss gradient for ``weight`` from the same pass.
        dim: Output-channel dimension in ``weight``.
        activation_elements: Total elements in the module output, including
            every currently surviving output channel.
    """
    if weight.shape != gradient.shape:
        raise ValueError("CPD Fisher weight and gradient shapes must match")
    if not isinstance(activation_elements, (int, float)) or not math.isfinite(activation_elements):
        raise ValueError("CPD activation_elements must be finite")
    if activation_elements <= 0:
        raise ValueError("CPD activation_elements must be positive")
    if dim < -weight.ndim or dim >= weight.ndim:
        raise ValueError("CPD Fisher dimension is outside the parameter rank")
    if gradient.is_sparse:
        raise ValueError("CPD Fisher requires a dense gradient")
    products = weight.detach().to(torch.float64) * gradient.detach().to(torch.float64)
    reduced = products.movedim(dim, 0).reshape(products.shape[dim], -1).sum(1)
    return reduced.square().div(activation_elements)


def fisher_scores(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    *,
    dim: int,
    activation_elements: int | float,
) -> tuple[float, ...]:
    """Return CPD's per-channel, batch-normalized Fisher statistics.

    CPD reduces the signed product within an output channel before squaring:
    ``sum(weight * gradient) ** 2 / activation_elements``.  This is distinct
    from summing elementwise squares, because the signed terms can cancel.

    Args:
        weight: Parameter values from the backward pass.
        gradient: Dense task-loss gradient for ``weight`` from the same pass.
        dim: Output-channel dimension in ``weight``.
        activation_elements: Total elements in the module output, including
            every currently surviving output channel.
    """
    values = (
        fisher_score_tensor(weight, gradient, dim=dim, activation_elements=activation_elements)
        .cpu()
        .tolist()
    )
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("CPD Fisher statistics must be finite and nonnegative")
    return tuple(values)


@dataclass(frozen=True)
class CPDMetric:
    """Score independent producer channels using accumulated CPD Fisher statistics.

    Args:
        scores: Per-axis sum of :func:`fisher_scores` over calibration batches.

    Statistics belong to one calibration interval and permit static ranking only.
    The caller aggregates per-parameter statistics into combed-group scores;
    that role information is not encoded in a generic candidate axis.
    Dependency propagation and physical compaction remain the caller's concern.
    """

    scores: dict[AxisRef, tuple[float, ...]]

    def __post_init__(self) -> None:
        """Validate immutable calibration statistics before planning begins."""
        scores = {axis: tuple(values) for axis, values in self.scores.items()}
        for axis, values in scores.items():
            if len(values) != axis.tensor.shape[axis.dim] or not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in values
            ):
                raise ValueError("CPD scores must be finite nonnegative axis statistics")
        object.__setattr__(self, "scores", scores)

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Return static scores for complete producer-channel candidates."""
        context.require_complete(accepted_impact)
        if accepted_impact.selections:
            raise PlanningError("CPDMetric only supports static selection")
        values = []
        for candidate in candidates:
            axis = candidate.axis
            if axis not in self.scores:
                raise PlanningError("CPD statistics are missing a candidate axis")
            selection = Selection(axis.tensor)
            for seed in candidate.remove:
                if seed.tensor == axis.tensor:
                    selection = selection.union(seed)
            indices = selection.fully_selected_indices(axis.dim)
            if not indices or selection != axis.select(indices):
                raise PlanningError("CPDMetric requires complete producer-channel candidates")
            score = math.fsum(self.scores[axis][index] for index in indices)
            if not math.isfinite(score):
                raise PlanningError("CPDMetric produced a nonfinite score")
            values.append(score)
        return values


def _group_size(remaining: int, cap: int) -> int:
    """Return how many channels CPD removes from one group in a single interval.

    The source hook computes ``min(round_pow_2((out_channels - g_mask_zeros) // 16), 4)``,
    where ``round_pow_2(x)`` is the largest power of two not exceeding ``x`` and
    one for non-positive input. This workflow deletes structure instead of
    zeroing masks, so an axis that cannot spare a channel reports zero rather
    than a batch that would empty it.
    """
    if remaining <= 1:
        return 0
    quotient = remaining // 16
    base = 1 if quotient <= 0 else 2 ** math.floor(math.log2(quotient))
    return max(1, min(base, cap, remaining - 1))


@dataclass(frozen=True)
class CPDLeastFirst:
    """Select CPD's least-important coupled group once per pruning interval.

    The caller owns the outer compression target and supplies a fresh metric
    after every training interval, matching CPD's score-resetting schedule.

    Args:
        metric: Accumulated, group-aggregated scores for this interval.
        candidates_per_interval: Cap for the CNN adaptive deletion schedule only.
        ffn_axes: Explicit FFN hidden axes, which use the source FFNPruner's
            fixed count of 32 independently of the CNN cap. At the physical
            boundary remove at most width-1, retaining one live channel.
    """

    metric: CPDMetric
    candidates_per_interval: int = 4
    ffn_axes: tuple[AxisRef, ...] = ()

    def __post_init__(self) -> None:
        """Reject a batch size that is not a positive channel count."""
        if type(self.candidates_per_interval) is not int or self.candidates_per_interval < 1:
            raise ValueError("CPD candidates_per_interval must be a positive integer")
        ffn_axes = tuple(self.ffn_axes)
        if len(set(ffn_axes)) != len(ffn_axes) or any(
            axis not in self.metric.scores for axis in ffn_axes
        ):
            raise ValueError("CPD FFN axes must be unique axes with calibrated scores")
        object.__setattr__(self, "ffn_axes", ffn_axes)

    def channel_count(self, axis: AxisRef) -> int:
        """Return fixed FFN or adaptive CNN count, without emptying an axis."""
        if axis in self.ffn_axes:
            return min(32, max(0, axis.tensor.shape[axis.dim] - 1))
        return _group_size(axis.tensor.shape[axis.dim], self.candidates_per_interval)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Return the cheapest complete group, sized like CPD's source hook.

        One candidate axis stands in for one combed group, because an axis's
        channels are exactly the indices that die together. The strategy ranks
        each axis by the sum of its own cheapest ``k`` channels, takes the
        smallest such aggregate, and requests that axis's ``k`` channels at
        once. Summing a per-axis prefix instead of draining one global order
        reproduces the source hook, which never mixes two groups in one step.

        ``PlanningContext.score`` proves every candidate's influence with a full
        dependency propagation before delegating to the metric. That proof is a
        scoring-time convenience rather than the final guarantee: ``Pruner.plan``
        independently re-propagates, re-compiles and re-checks the budget of
        whatever this strategy returns. CPD's metric reads precomputed per-axis
        statistics and never consults candidate influence, so ranking first and
        proving only the selected group keeps that guarantee without paying one
        propagation per candidate.

        Axes with different ``k`` compare prefix sums, so an axis that can still
        remove four channels can outrank one that removes two. The source hook
        compares group sizes the same way.
        """
        if not context.candidates:
            raise PlanningError("CPD found no candidates remaining to prune")
        accepted_impact = context.impact(())
        scores = self.metric.score(context, context.candidates, accepted_impact=accepted_impact)
        axes: dict[AxisRef, list[tuple[float, Candidate]]] = {}
        for candidate, score in zip(context.candidates, scores, strict=True):
            axis = candidate.axis
            if axis is None:
                raise PlanningError("CPD requires producer-channel candidates")
            indices = set()
            for seed in candidate.remove:
                if seed.tensor == axis.tensor:
                    indices.update(seed.fully_selected_indices(axis.dim))
            if len(indices) != 1:
                raise PlanningError("CPDLeastFirst requires single-channel candidates")
            axes.setdefault(axis, []).append((score, candidate))
        ranked: list[tuple[float, list[tuple[float, Candidate]]]] = []
        for axis, entries in axes.items():
            size = self.channel_count(axis)
            if not size:
                continue
            entries.sort(key=lambda entry: (entry[0], entry[1].key))
            if len(entries) < size:
                continue
            chosen = entries[:size]
            ranked.append((math.fsum(score for score, _ in chosen), chosen))
        ranked.sort(key=lambda entry: (entry[0], entry[1][0][1].key))
        for _aggregate, chosen in ranked:
            try:
                context.require_complete(
                    context.impact(
                        (
                            *accepted_impact.requested,
                            *(seed for _score, candidate in chosen for seed in candidate.remove),
                        )
                    )
                )
            except PlanningError:
                continue
            return StrategyResult(tuple(candidate.key for _score, candidate in chosen))
        raise PlanningError("CPD found no structurally complete group to remove")
