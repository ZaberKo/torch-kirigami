"""Exact group-score reuse for proved independent identity-axis domains.

This is a conservative optimization of built-in GroupMagnitude. It does not
solve constraints or replace joint execution checks. Any unproved relationship,
overlapping domain, extension, or full-axis removal uses ordinary ranking.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ..dep_graph.contracts import (
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
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.operators.shapes import CallArgumentConstraint
from ..dep_graph.relations import (
    AxisRelation,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
)
from ..dep_graph.selection import AxisRef, IndexSet, TensorRef
from .axis_domains import IdentityAxisIndex
from .metrics import (
    GroupMagnitude,
    _candidate_domain,
    _group_score,
    _magnitude_norm,
    _normalization,
    _parameter_bindings,
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


def _component_norms(
    members: tuple[AxisRef, ...],
    bindings: Mapping[TensorRef, torch.Tensor],
    p: int,
) -> tuple[float, ...]:
    """Return one union L1 or L2 norm per position in an identity-axis component.

    The caller proves that component members select independent, aligned slices.
    Only members present in the parameter bindings contribute. Combine L2 norms
    with `hypot`, so the result equals one norm across all affected parameters,
    rather than a sum of per-parameter norms.
    """
    width = members[0].tensor.shape[members[0].dim]
    totals: dict[torch.device, torch.Tensor] = {}
    with torch.no_grad():
        for axis in members:
            if axis.tensor not in bindings:
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
class IndependentScores:
    """Scores for valid components, fixed summaries, and proved covered seeds.

    A key absent from both `scores` and `covered` needs ordinary joint scoring.
    Summary presence alone does not establish that its component is still valid.

    Attributes:
        scores: Cached scores keyed by candidate key.
        removals: Axis positions removed by each summarized candidate.
        covered: Candidate keys whose positions are already removed by the
            accepted impact and need no independent score.
    """

    scores: Mapping[str, float]
    removals: Mapping[str, Mapping[AxisRef, IndexSet]]
    covered: frozenset[str]


@dataclass(frozen=True)
class IndependentRanking:
    """Plan-local scalar statistics and immutable proofs, never mutable solver state.

    Attributes:
        context: Planning context for the current plan.
        metric: GroupMagnitude instance whose scoring semantics were proved.
        p: Norm exponent used for cached statistics.
        domains: Proved identity-axis components used for normalization.
        candidates: Candidate objects with component indices and axis positions.
        removals: Axis positions removed by each candidate.
        norms: Union L1 or L2 norms indexed by component, then axis position.
        bindings: Tensor objects used to validate statistics freshness.
        versions: Tensor mutation counters captured with the statistics.
    """

    context: PlanningContext
    metric: GroupMagnitude
    p: int
    domains: tuple[tuple[AxisRef, ...], ...]
    candidates: tuple[tuple[Candidate, int, int], ...]
    removals: Mapping[str, Mapping[AxisRef, IndexSet]]
    norms: tuple[tuple[float, ...], ...]
    bindings: tuple[torch.Tensor, ...]
    versions: tuple[int, ...]

    @classmethod
    def build(
        cls, context: PlanningContext, metric: Metric, axes: tuple[AxisRef, ...]
    ) -> IndependentRanking | None:
        """Freeze proved domains and their summaries for the requested counting axes."""
        if (
            type(metric) is not GroupMagnitude
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
        if any(type(c) not in _CONSTRAINTS or not has_native_semantics(c) for c in constraints):
            return None
        axis_index = IdentityAxisIndex(graph.relations)
        domains, candidates, owners, removals = [], [], {}, {}
        counted = []
        for candidate in context.candidates:
            if (
                type(candidate) is not Candidate
                or not has_native_semantics(candidate.axis)
                or not has_native_semantics(candidate.remove)
            ):
                continue
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
                counted.append(tuple(axis for axis in axes if axis in members))
                owners.update((a.tensor, group) for a in members)
            candidates.append((candidate, group, next(iter(indices))))
            removals[candidate.key] = MappingProxyType(dict.fromkeys(counted[group], indices))
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
        parameters = _parameter_bindings(context)
        if any(not any(axis.tensor in parameters for axis in members) for members in domains):
            # Ordinary scoring reports parameter-free requests. Do not turn a
            # missing scoring basis into a cached vector of zero importance.
            return None
        norms = tuple(_component_norms(members, parameters, metric.p) for members in domains)
        graph.validate()
        return cls(
            context,
            metric,
            metric.p,
            tuple(domains),
            tuple(candidates),
            MappingProxyType(removals),
            norms,
            tuple(bindings.values()),
            versions,
        )

    def score(
        self,
        accepted_impact: Impact,
    ) -> IndependentScores | None:
        """Rescore valid components and leave invalid components to ordinary scoring.

        A generic candidate may remove another axis of a cached parameter. Check
        the complete accepted closure, not just its original seeds or counts.
        Cross-axis, partitioned and last-position selections invalidate only their
        component. Model/metric changes invalidate the entire statistics snapshot.
        """
        self.context.graph.validate()
        self.context.graph.validate_impact(accepted_impact)
        self.context.require_complete(accepted_impact)
        if (
            self.metric.p != self.p
            or "score" in vars(self.metric)
            or type(self.metric).score is not _GROUP_SCORE
            or tuple(t._version for t in self.bindings) != self.versions
        ):
            return None
        removed, normalization = {}, {}
        for group, (members, norms) in enumerate(zip(self.domains, self.norms, strict=True)):
            root = members[0]
            # IndexSet stores intervals, so repeated membership would enumerate
            # removed positions. Only this bounded logical axis needs a set.
            selected = accepted_impact.selection(root.tensor).fully_selected_indices(root.dim)
            if any(
                accepted_impact.selection(axis.tensor) != axis.select(selected) for axis in members
            ):
                continue
            indices = frozenset(selected)
            surviving = [value for i, value in enumerate(norms) if i not in indices]
            if len(surviving) <= 1:
                continue  # Completing an entire axis can activate other axes.
            removed[group] = indices
            normalization[group] = _normalization(surviving, self.p)
        if not normalization:
            return None
        scores, covered = {}, set()
        for candidate, group, position in self.candidates:
            if group not in normalization:
                continue
            if position in removed[group]:
                covered.add(candidate.key)
                continue
            scale, scaled_mean_energy = normalization[group]
            value = _group_score(self.norms[group][position], scale, scaled_mean_energy, self.p)
            scores[candidate.key] = value
        return IndependentScores(MappingProxyType(scores), self.removals, frozenset(covered))
