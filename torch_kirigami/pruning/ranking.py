"""Exact score reuse for proved independent identity-axis domains.

This is a conservative optimization of built-in `Magnitude` and `GroupMagnitude`.
It does not solve constraints or replace joint execution checks. Any unproved
relationship, extension, or full-axis removal uses ordinary ranking. When
accepted removals cover another part of a member tensor, such as the columns of
a weight whose rows form the component, the component's norms are reduced again
over the remaining parts of each slice. A score also requires
`CandidateFilter.complete_positions` to prove that the joint closure is
complete, because ordinary ranking reports or excludes incomplete closures.
`GroupMagnitude` requires this for every surviving position of the component,
because its normalization reads all of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

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
from ..dep_graph.selection import (
    AxisRef,
    IndexSet,
    Region,
    Selection,
    TensorRef,
    canonical_boxes,
    region_boxes,
)
from ..errors import AnalysisLimitError
from ..settings import config
from .metrics import (
    GroupMagnitude,
    Magnitude,
    _candidate_domain,
    _combine_totals,
    _group_score,
    _normalization,
    _parameter_bindings,
    _region_totals,
)
from .planner import PlanningContext
from .types import Candidate, Metric, PlanningError

_NATIVE_SCORES = {Magnitude: Magnitude.score, GroupMagnitude: GroupMagnitude.score}
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


def _position_norms(
    metric: Magnitude | GroupMagnitude,
    members: tuple[AxisRef, ...],
    bindings: Mapping[TensorRef, torch.Tensor],
    positions: Iterable[int],
    base: tuple[Selection, ...] | None = None,
) -> Mapping[int, float]:
    """Return the ordinary union L1 or L2 norm of single positions of a component.

    The caller proves that one position adds exactly the member slices to the
    accepted closure, whose member selections are `base`, or nothing when
    `base` is None. The parts of those slices outside `base` are then the
    parameter regions that ordinary scoring reduces. `canonical_boxes` and
    `_region_totals` reduce equal coordinates identically, one position at a
    time, so scores are bitwise equal and temporary memory is bounded by one
    slice per member.
    """
    accepted = (None,) * len(members) if base is None else base
    parameters = tuple(
        (member, () if selection is None else region_boxes(selection.regions))
        for member, selection in zip(members, accepted, strict=True)
        if member.tensor in bindings and all(member.tensor.shape)
    )
    ordered = sorted(positions)
    result = {}
    for start in range(0, len(ordered), config.score_batch_size):
        batch = ordered[start : start + config.score_batch_size]
        totals = []
        for position in batch:
            parts = []
            for member, removed in parameters:
                box = tuple(
                    ((position, position + 1),) if dim == member.dim else ((0, size),)
                    for dim, size in enumerate(member.tensor.shape)
                )
                boxes = canonical_boxes((box,), removed)
                if boxes:
                    parts.append((member.tensor, boxes))
            totals.append(_region_totals(metric, bindings, parts))
        result.update(zip(batch, _combine_totals(metric, totals), strict=True))
    return MappingProxyType(result)


def _has_native_score(metric: Metric) -> bool:
    """Accept only built-in scores whose formulas the cached statistics reproduce."""
    native = _NATIVE_SCORES.get(type(metric))
    return native is not None and "score" not in vars(metric) and type(metric).score is native


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
        metric: `Magnitude` or `GroupMagnitude` instance whose scoring
            semantics were proved.
        p: Norm exponent used for cached statistics.
        domains: Proved identity-axis components used for normalization.
        candidates: Candidate objects with component indices and axis positions.
        positions: Axis positions of the candidates of each component.
        removals: Axis positions removed by each candidate.
        norms: Ordinary single-position L1 or L2 norms without accepted
            removals, indexed by component, then axis position. `Magnitude`
            keeps only candidate positions.
        parameters: Parameter bindings reduced by the norms.
        bindings: Tensor objects used to validate statistics freshness.
        versions: Tensor mutation counters captured with the statistics.
        conditional: Norms of the most recent accepted state of each component
            whose accepted closure also removes other parts of a member tensor,
            keyed by the regions of the accepted member selections. Positions
            are added when a score first needs them.
    """

    context: PlanningContext
    metric: Magnitude | GroupMagnitude
    p: int
    domains: tuple[tuple[AxisRef, ...], ...]
    candidates: tuple[tuple[Candidate, int, int], ...]
    positions: tuple[IndexSet, ...]
    removals: Mapping[str, Mapping[AxisRef, IndexSet]]
    norms: tuple[Mapping[int, float], ...]
    parameters: Mapping[TensorRef, torch.Tensor]
    bindings: tuple[torch.Tensor, ...]
    versions: tuple[int, ...]
    conditional: dict[int, tuple[tuple[tuple[Region, ...], ...], dict[int, float]]] = field(
        default_factory=dict, compare=False, repr=False
    )

    @classmethod
    def build(
        cls, context: PlanningContext, metric: Metric, axes: tuple[AxisRef, ...]
    ) -> IndependentRanking | None:
        """Freeze proved domains and their summaries for the requested counting axes."""
        if not _has_native_score(metric):
            return None
        metric = cast("Magnitude | GroupMagnitude", metric)
        graph = context.graph
        if not context.impact(()).complete:
            return None
        if any(type(r) not in _RELATIONS for r in graph.relations):
            return None
        constraints = (*graph.constraints, *context.constraints)
        if any(type(c) not in _CONSTRAINTS or not has_native_semantics(c) for c in constraints):
            return None
        axis_index = graph.identity_axes
        domains, candidates, groups, removals = [], [], {}, {}
        counted = []
        for candidate in context.candidates:
            if (
                type(candidate) is not Candidate
                or not has_native_semantics(candidate.axis)
                or not has_native_semantics(candidate.remove)
            ):
                continue
            try:
                axis, count, position = _candidate_domain(candidate)
            except PlanningError:
                continue
            if count != 1 or any(
                len(selected) != axis.tensor.shape[dim]
                for seed in candidate.remove
                for region in seed.regions
                for dim, selected in enumerate(region.axes)
                if dim != axis.dim
            ):
                continue  # `complete_positions` proves single-slice requests only.
            group = groups.get(axis)
            if group is None:
                members = axis_index.component(axis)
                if members is None:
                    continue
                # Components may share a tensor on different axes, such as the
                # rows and columns of one weight. Once the accepted closure
                # removes another part of a member tensor, `score` reduces the
                # remaining parts of each slice instead of these statistics.
                group = len(domains)
                domains.append(members)
                counted.append(tuple(axis for axis in axes if axis in members))
                groups.update((member, group) for member in members)
            candidates.append((candidate, group, position))
            removals[candidate.key] = MappingProxyType(
                dict.fromkeys(counted[group], IndexSet.span(position, position + 1))
            )
        if not domains:
            return None
        owners = {axis.tensor for axis in groups}
        for constraint in constraints:
            if type(constraint) in (Barrier, CallArgumentConstraint):
                if any(ref in owners for ref in constraint.refs):
                    return None
            elif type(constraint) is AxisBarrier and constraint.axis in groups:
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
        positions: list[set[int]] = [set() for _ in domains]
        for _, group, position in candidates:
            positions[group].add(position)
        try:
            candidate_positions = tuple(IndexSet.of(group) for group in positions)
        except AnalysisLimitError:
            return None  # Ordinary ranking scores candidates at too many separate positions.
        # GroupMagnitude normalizes over every position; Magnitude needs only
        # the candidates' own positions.
        if type(metric) is GroupMagnitude:
            positions = [set(range(members[0].tensor.shape[members[0].dim])) for members in domains]
        norms = tuple(
            _position_norms(metric, members, parameters, group_positions)
            for members, group_positions in zip(domains, positions, strict=True)
        )
        if type(metric) is Magnitude and not all(
            math.isfinite(norms[group][position]) for _, group, position in candidates
        ):
            return None  # Ordinary scoring reports nonfinite candidate scores.
        graph.validate()
        return cls(
            context,
            metric,
            metric.p,
            tuple(domains),
            tuple(candidates),
            candidate_positions,
            MappingProxyType(removals),
            norms,
            MappingProxyType(parameters),
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
        When it removes other parts of a member tensor, the component's norms
        are reduced again over the parts of each slice that remain. Partitioned
        and last-position selections, and unproved joint closures, leave only
        their component to ordinary scoring. Model/metric changes invalidate the
        entire statistics snapshot.
        """
        self.context.graph.validate()
        self.context.graph.validate_impact(accepted_impact)
        self.context.require_complete(accepted_impact)
        if (
            self.metric.p != self.p
            or not _has_native_score(self.metric)
            or tuple(t._version for t in self.bindings) != self.versions
        ):
            return None
        group_scores = type(self.metric) is GroupMagnitude
        checks = self.context._candidate_filter
        removed, normalization = {}, {}
        for group, members in enumerate(self.domains):
            root = members[0]
            width = root.tensor.shape[root.dim]
            state = checks.component_state(accepted_impact, members)
            if state is None:
                continue
            selected, base = state
            # IndexSet stores intervals, so repeated membership would enumerate
            # removed positions. Only this bounded logical axis needs a set.
            indices = frozenset(selected)
            if width - len(indices) <= 1:
                continue  # Completing an entire axis can activate other axes.
            # Ordinary normalization scores every surviving position, so each
            # of those joint closures must also be proved complete.
            complete = frozenset(
                checks.complete_positions(
                    members, accepted_impact, None if group_scores else self.positions[group]
                )
            )
            if group_scores and any(i not in complete for i in range(width) if i not in indices):
                continue
            norms = self.norms[group]
            if base is not None and complete:
                key = tuple(selection.regions for selection in base)
                entry = self.conditional.get(group)
                if entry is None or entry[0] != key:
                    entry = self.conditional[group] = (key, {})
                # Only proved positions are scored, and normalization reads
                # every surviving position, which are then all proved.
                missing = [i for i in complete if i not in entry[1]]
                if missing:
                    entry[1].update(
                        _position_norms(self.metric, members, self.parameters, missing, base)
                    )
                norms = entry[1]
            if group_scores:
                try:
                    normalization[group] = _normalization(
                        [norms[i] for i in range(width) if i not in indices], self.p
                    )
                except PlanningError:
                    continue  # Ordinary ranking reports or excludes this component.
            removed[group] = (indices, complete, norms)
        if not removed:
            return None
        scores, covered = {}, set()
        for candidate, group, position in self.candidates:
            if group not in removed:
                continue
            indices, complete, norms = removed[group]
            if position in indices:
                covered.add(candidate.key)
                continue
            if position not in complete:
                continue  # Ordinary ranking reports or excludes this candidate.
            value = norms[position]
            if not math.isfinite(value):
                continue  # Ordinary scoring reports nonfinite conditional scores.
            if group_scores:
                scale, scaled_mean_energy = normalization[group]
                try:
                    value = _group_score(value, scale, scaled_mean_energy, self.p)
                except PlanningError:
                    continue  # Ordinary ranking reports or excludes this candidate.
            scores[candidate.key] = value
        return IndependentScores(MappingProxyType(scores), self.removals, frozenset(covered))
