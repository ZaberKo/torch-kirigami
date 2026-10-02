"""Conservative identity-axis proofs shared by ranking and custom strategies."""

from collections import defaultdict, deque
from collections.abc import Iterable

from ..relations import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    Relation,
    ReshapeRelation,
)
from ..selection import MAX_PARTS, AxisRef, TensorRef


class IdentityAxisIndex:
    """Index immutable relations for full-axis, identity-coordinate components.

    This proves index correspondence only, not constraints or execution support.
    Callers must validate their graph and actual selections. `component()` is
    limited to proper subsets on one axis per tensor, with no unknown or scoped
    relations. `equivalent_axes()` only proves mutually implying seeds; other
    relations still require ordinary propagation. No tensor data or dependency
    impacts are retained.

    Args:
        relations: Relations from one fixed dependency graph snapshot.
    """

    def __init__(self, relations: Iterable[Relation]) -> None:
        self._adjacent: dict[TensorRef, list[Relation]] = defaultdict(list)
        for relation in relations:
            for ref in dict.fromkeys(relation.refs):
                self._adjacent[ref].append(relation)
        self._components: dict[AxisRef, tuple[AxisRef, ...] | None] = {}
        self._equivalent: dict[AxisRef, tuple[AxisRef, ...]] = {}

    def equivalent_axes(self, root: AxisRef) -> tuple[AxisRef, ...]:
        """Find seeds with equal closures through bidirectional identity edges.

        Equal selections on these axes imply one another. Their eventual closure
        is therefore the same even when additional nonidentity relations exist.
        This does not extrapolate between different positions, prove layout, or
        replace the analysis of at least one actual request.
        """
        if type(root) is not AxisRef or type(root.tensor) is not TensorRef:
            return (root,)
        if root in self._equivalent:
            return self._equivalent[root]
        members = set()
        pending = deque((root,))
        while pending:
            axis = pending.popleft()
            if axis in members:
                continue
            members.add(axis)
            for relation in self._adjacent.get(axis.tensor, ()):
                if type(relation) is BroadcastRelation:
                    target = _broadcast_axis(relation, axis)
                    if target is not None:
                        pending.append(target)
                    continue
                if (
                    type(relation) is ReshapeRelation
                    and relation.left.shape == relation.right.shape
                ):
                    target = relation.right if axis.tensor == relation.left else relation.left
                    if type(target) is TensorRef:
                        pending.append(target.axis(axis.dim))
                    continue
                endpoints = _equal_endpoints(relation)
                if endpoints is None or axis not in endpoints:
                    continue
                pending.extend(endpoints)
        result = tuple(sorted(members, key=lambda axis: (axis.tensor.id, axis.dim)))
        for axis in result:
            self._equivalent[axis] = result
        return result

    def component(self, root: AxisRef) -> tuple[AxisRef, ...] | None:
        """Return proved equal-coordinate axes, or None when proof is unavailable.

        Only proper subsets of root positions can use this correspondence: a
        full removal can trigger additional axes. Broadcasts preserve this axis
        only when its width matches. Same-shape reshape correspondence comes
        from the declared relation, never from observed shapes alone.
        """
        if type(root) is not AxisRef or type(root.tensor) is not TensorRef:
            return None
        if root in self._components:
            return self._components[root]
        result = self._component(root)
        self._components[root] = result
        if result is not None:
            for axis in result:
                self._components[axis] = result
        return result

    def _component(self, root: AxisRef) -> tuple[AxisRef, ...] | None:
        """Walk exact built-in relations without extrapolating extension behavior."""
        width = root.tensor.shape[root.dim]
        if not 1 < width <= MAX_PARTS:
            return None
        members: dict[TensorRef, AxisRef] = {}
        pending = deque((root,))
        while pending:
            axis = pending.popleft()
            if type(axis) is not AxisRef or type(axis.tensor) is not TensorRef:
                return None
            if 0 in axis.tensor.shape:
                return None
            if axis.tensor in members:
                if members[axis.tensor] != axis:
                    return None
                continue
            members[axis.tensor] = axis
            for relation in self._adjacent.get(axis.tensor, ()):
                if (
                    type(relation) is ReshapeRelation
                    and relation.left.shape == relation.right.shape
                ):
                    target = relation.right if axis.tensor == relation.left else relation.left
                    pending.append(target.axis(axis.dim))
                    continue
                if type(relation) is BroadcastRelation:
                    target = _broadcast_axis(relation, axis)
                    if target is None:
                        return None
                    pending.append(target)
                    continue
                endpoints = _equal_endpoints(relation)
                if endpoints is None:
                    return None
                if axis in endpoints:
                    pending.extend(endpoints)
        return tuple(sorted(members.values(), key=lambda axis: axis.tensor.id))


def _equal_endpoints(relation: Relation) -> tuple[AxisRef, AxisRef] | None:
    """Recognize exact unscoped bidirectional identity relations once."""
    if type(relation) is not AxisRelation:
        return None
    if type(relation.left) is not AxisPort or type(relation.right) is not AxisPort:
        return None
    if relation.left.scope is not None or relation.right.scope is not None:
        return None
    left, right = relation.left.axis, relation.right.axis
    if any(
        type(axis) is not AxisRef or type(axis.tensor) is not TensorRef for axis in (left, right)
    ):
        return None
    width = left.tensor.shape[left.dim]
    if (
        right.tensor.shape[right.dim] != width
        or any(type(mapping) is not BlockMap for mapping in relation.maps)
        or relation.maps != (BlockMap(0, 0, width),)
    ):
        return None
    return left, right


def _broadcast_axis(relation: BroadcastRelation, axis: AxisRef) -> AxisRef | None:
    """Preserve a full structural axis only when its width is not broadcast."""
    if type(relation.small) is not TensorRef or type(relation.big) is not TensorRef:
        return None
    offset = len(relation.big.shape) - len(relation.small.shape)
    forward = axis.tensor == relation.small
    target = relation.big if forward else relation.small
    dim = axis.dim + offset if forward else axis.dim - offset
    if not 0 <= dim < len(target.shape) or target.shape[dim] != axis.tensor.shape[axis.dim]:
        return None
    return target.axis(dim)
