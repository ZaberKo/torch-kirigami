"""Exact group-score reuse for proved independent identity-axis domains.

This is a conservative optimization of built-in GroupMagnitude. It does not
solve constraints or replace joint execution checks. Any unproved relationship,
overlapping domain, extension, or full-axis removal uses ordinary ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..contracts import (
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Divisible,
    Fixed,
    Impact,
    LayoutConstraint,
    NonEmpty,
)
from ..operators.shapes import CallArgumentConstraint
from ..relations import (
    AxisRelation,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
)
from ..selection import AxisRef, IndexSet, TensorRef
from .axis_domains import IdentityAxisIndex
from .metrics import (
    GroupMagnitude,
    _candidate_domain,
    _magnitude_norm,
    _normalization,
    _weight_bindings,
)
from .planner import PlanningContext
from .types import Candidate, Metric, PlanningError

_GROUP_SCORE = GroupMagnitude.score
_RELATIONS = (AxisRelation, BroadcastRelation, PermuteRelation, ReshapeRelation, SliceRelation)
_CONSTRAINTS = (
    NonEmpty,
    Fixed,
    Balanced,
    BlockBalance,
    Divisible,
    Barrier,
    AxisBarrier,
    LayoutConstraint,
    CallArgumentConstraint,
)


def _norms(
    members: tuple[AxisRef, ...],
    weights: set[TensorRef],
    bindings: dict[TensorRef, torch.Tensor],
    p: int,
) -> tuple[float, ...]:
    """Batch independent slices using the same parameter order and norm formula."""
    width = members[0].tensor.shape[members[0].dim]
    totals: dict[torch.device, torch.Tensor] = {}
    with torch.no_grad():
        for axis in members:
            if axis.tensor not in weights:
                continue
            weight = bindings[axis.tensor]
            value = _magnitude_norm(weight.detach().movedim(axis.dim, 0), p, batched=True)
            previous = totals.get(weight.device)
            totals[weight.device] = (
                value
                if previous is None
                else torch.hypot(previous, value)
                if p == 2
                else previous + value
            )
    result = [0.0] * width
    for total in totals.values():
        for index, value in enumerate(total.cpu().tolist()):
            result[index] = math.hypot(result[index], value) if p == 2 else result[index] + value
    return tuple(result)


@dataclass(frozen=True)
class IndependentRanking:
    """Plan-local scalar statistics and immutable proofs, never mutable solver state."""

    context: PlanningContext
    metric: GroupMagnitude
    p: int
    domains: tuple[tuple[AxisRef, ...], ...]
    candidates: tuple[tuple[Candidate, int, IndexSet], ...]
    norms: tuple[tuple[float, ...], ...]
    bindings: tuple[torch.Tensor, ...]
    versions: tuple[int, ...]

    @classmethod
    def build(cls, context: PlanningContext, metric: Metric) -> IndependentRanking | None:
        """Collect proved domains; other candidates retain ordinary joint scoring."""
        if (
            type(metric) is not GroupMagnitude
            or metric.parameter_filter is not None
            or "score" in vars(metric)
            or type(metric).score is not _GROUP_SCORE
        ):
            return None
        graph = context.graph
        if not context.impact(()).complete:
            return None
        if any(type(r) not in _RELATIONS for r in graph.relations):
            return None
        constraints = (*graph.constraints, *context.constraints)
        if any(type(c) not in _CONSTRAINTS for c in constraints):
            return None
        axis_index = IdentityAxisIndex(graph.relations)
        domains, candidates, owners = [], [], {}
        for candidate in context.candidates:
            try:
                axis, indices = _candidate_domain(candidate)
            except PlanningError:
                continue
            if len(indices) != 1:
                continue
            if axis.tensor in owners:
                group = owners[axis.tensor]
                if axis not in domains[group]:
                    continue
            else:
                members = axis_index.component(axis)
                if members is None or any(a.tensor in owners for a in members):
                    continue
                group = len(domains)
                domains.append(members)
                owners.update((a.tensor, group) for a in members)
            candidates.append((candidate, group, indices))
        if not domains:
            return None
        for constraint in constraints:
            if type(constraint) in (Barrier, CallArgumentConstraint):
                if any(ref in owners for ref in constraint.refs):
                    return None
            elif (
                type(constraint) is AxisBarrier
                and constraint.axis.tensor in owners
                and constraint.axis in domains[owners[constraint.axis.tensor]]
            ):
                return None
        bindings = dict(graph.tensor_bindings())
        try:
            versions = tuple(t._version for t in bindings.values())
        except RuntimeError:
            return None  # Inference tensors have no mutation-aware cache key.
        weights = _weight_bindings(context)
        if not weights:
            return None
        norms = tuple(_norms(members, weights, bindings, metric.p) for members in domains)
        graph.validate()
        return cls(
            context,
            metric,
            metric.p,
            tuple(domains),
            tuple(candidates),
            norms,
            tuple(bindings.values()),
            versions,
        )

    def score(
        self,
        accepted_impact: Impact,
        axes: tuple[AxisRef, ...],
    ) -> tuple[dict[str, float], dict[str, dict[AxisRef, IndexSet]]] | None:
        """Score proved candidates; return None when accepted changes invalidate reuse.

        A generic candidate may remove another axis of a cached parameter. Check
        the complete accepted closure, not just its original seeds or counts.
        Any cross-axis or partitioned selection falls back to ordinary scoring.
        """
        self.context.graph.validate()
        self.context.graph.validate_impact(accepted_impact)
        self.context.require_complete(accepted_impact)
        if (
            self.metric.p != self.p
            or self.metric.parameter_filter is not None
            or "score" in vars(self.metric)
            or type(self.metric).score is not _GROUP_SCORE
            or tuple(t._version for t in self.bindings) != self.versions
        ):
            return None
        removed, normalization, counted = [], [], []
        for members, norms in zip(self.domains, self.norms, strict=True):
            root = members[0]
            # IndexSet stores intervals, so repeated membership would enumerate
            # removed positions. Only this bounded logical axis needs a set.
            selected = accepted_impact.selection(root.tensor).fully_selected_indices(root.dim)
            if any(
                accepted_impact.selection(axis.tensor) != axis.select(selected) for axis in members
            ):
                return None
            indices = frozenset(selected)
            surviving = [value for i, value in enumerate(norms) if i not in indices]
            if len(surviving) <= 1:
                return None  # Completing an entire axis can activate other axes.
            removed.append(indices)
            normalization.append(_normalization(surviving, self.p))
            counted.append(tuple(axis for axis in axes if axis in members))
        scores, removals = {}, {}
        for candidate, group, indices in self.candidates:
            position = next(iter(indices))
            if position in removed[group]:
                continue
            scale, mean = normalization[group]
            value = (self.norms[group][position] / scale) ** self.p / mean if scale else 0.0
            scores[candidate.key] = value
            removals[candidate.key] = dict.fromkeys(counted[group], indices)
        return scores, removals
