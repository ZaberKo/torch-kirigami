"""Exact group-score reuse for proved independent identity-axis domains.

This is a conservative optimization of built-in GroupMagnitude. It does not
solve constraints or replace joint execution checks. Any unproved relationship,
overlapping domain, extension, or full-axis removal uses ordinary ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
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
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    Relation,
    ReshapeRelation,
    SliceRelation,
)
from ..selection import MAX_PARTS, AxisRef, IndexSet, TensorRef
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


def _component(
    root: AxisRef,
    adjacent: dict[TensorRef, list[Relation]],
) -> tuple[AxisRef, ...] | None:
    """Prove a one-to-one axis component without crossing scoped or nonlinear maps.

    Other axes of a tensor cannot become full cross sections while a position
    remains on this axis. Domains must also be tensor-disjoint from each other;
    together those conditions exclude row/column intersections and joint-only
    block completions. Full-axis removals are deliberately not accelerated.
    """
    width = root.tensor.shape[root.dim]
    if not 1 < width <= MAX_PARTS:
        return None
    members, pending = {}, deque((root,))
    while pending:
        axis = pending.popleft()
        if 0 in axis.tensor.shape:
            return None
        if axis.tensor in members:
            if members[axis.tensor] != axis:
                return None
            continue
        members[axis.tensor] = axis
        for relation in adjacent.get(axis.tensor, ()):
            if type(relation) is ReshapeRelation and relation.left.shape == relation.right.shape:
                # The relation already establishes row-major correspondence.
                # Equal shapes make its index map identity, not a layout proof.
                target = relation.right if axis.tensor == relation.left else relation.left
                pending.append(target.axis(axis.dim))
                continue
            if type(relation) is BroadcastRelation:
                offset = len(relation.big.shape) - len(relation.small.shape)
                forward = axis.tensor == relation.small
                target = relation.big if forward else relation.small
                dim = axis.dim + offset if forward else axis.dim - offset
                if not 0 <= dim < len(target.shape) or target.shape[dim] != width:
                    return None
                # Other dimensions are complete cross sections, so broadcasting
                # them preserves exactly the same positions of this axis.
                pending.append(target.axis(dim))
                continue
            if type(relation) is not AxisRelation:
                return None
            if relation.left.scope is not None or relation.right.scope is not None:
                return None
            for origin, target in (
                (relation.left.axis, relation.right.axis),
                (relation.right.axis, relation.left.axis),
            ):
                if origin != axis:
                    continue
                if target.tensor.shape[target.dim] != width or relation.maps != (
                    BlockMap(0, 0, width),
                ):
                    return None
                pending.append(target)
    return tuple(sorted(members.values(), key=lambda a: a.tensor.id))


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
        """Use the fast path only when the entire candidate universe is proved."""
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
        adjacent = defaultdict(list)
        for relation in graph.relations:
            for ref in dict.fromkeys(relation.refs):
                adjacent[ref].append(relation)
        domains, candidates, owners = [], [], {}
        for candidate in context.candidates:
            try:
                axis, indices = _candidate_domain(candidate)
            except PlanningError:
                return None
            if len(indices) != 1:
                return None
            if axis.tensor in owners:
                group = owners[axis.tensor]
                if axis not in domains[group]:
                    return None
            else:
                members = _component(axis, adjacent)
                if members is None or any(a.tensor in owners for a in members):
                    return None
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

    def rank(
        self,
        selected: Impact,
        axes: tuple[AxisRef, ...],
    ) -> tuple[list[Candidate], dict[str, dict[AxisRef, IndexSet]]] | None:
        """Recompute conditional normalization without repropagating untouched domains."""
        self.context.graph.validate()
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
            indices = frozenset(selected.selection(root.tensor).fully_selected_indices(root.dim))
            surviving = [value for i, value in enumerate(norms) if i not in indices]
            if len(surviving) <= 1:
                return None  # Completing an entire axis can activate other axes.
            removed.append(indices)
            normalization.append(_normalization(surviving, self.p))
            counted.append(tuple(axis for axis in axes if axis in members))
        scores, removals = [], {}
        for candidate, group, indices in self.candidates:
            position = next(iter(indices))
            if position in removed[group]:
                continue
            scale, mean = normalization[group]
            value = (self.norms[group][position] / scale) ** self.p / mean if scale else 0.0
            scores.append((value, candidate.key, candidate))
            removals[candidate.key] = dict.fromkeys(counted[group], indices)
        return [c for _, _, c in sorted(scores, key=lambda item: (item[0], item[1]))], removals
