"""Conservative identity-axis proofs shared by the solver, ranking and custom strategies."""

from collections import ChainMap, defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping

from ..errors import AnalysisLimitError
from ..settings import config
from .contracts import Provenance
from .eligibility import has_native_semantics
from .relations import (
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    Relation,
    ReshapeRelation,
    _shared_reshape_axis_counts,
)
from .selection import AxisRef, IndexSet, Selection, TensorRef

IDENTITY_REASON = "Equivalent axes through native identity relations"


class IdentityAxisIndex:
    """Index of axes connected by identity-coordinate relations.

    The index proves only that positions correspond one-to-one across axes; it
    does not check constraints or execution support, and callers must still
    validate their graph and actual selections. `component()` covers proper
    subsets of one axis per tensor and fails when an unknown or partially
    scoped relation is involved; explicit full-tensor scopes are accepted.
    `equivalent_axes()` proves only that equal seeds on the returned axes imply
    one another; other relations still require ordinary propagation. No tensor
    data or dependency impacts are retained.

    Args:
        relations: Relations from one fixed dependency graph snapshot.
    """

    def __init__(self, relations: Iterable[Relation]) -> None:
        self._max_parts = config.max_parts
        self._adjacent: dict[TensorRef, list[Relation]] = defaultdict(list)
        self._native_relations: set[int] = set()
        for relation in relations:
            if has_native_semantics(relation):
                self._native_relations.add(id(relation))
            for ref in dict.fromkeys(relation.refs):
                self._adjacent[ref].append(relation)
        self._components: dict[AxisRef, tuple[AxisRef, ...] | None] = {}
        self._equivalent: dict[AxisRef, tuple[AxisRef, ...]] = {}
        # One spanning tree per class, shared by explanations from every root.
        self._trees: dict[AxisRef, dict[AxisRef, list[tuple[AxisRef, str]]]] = {}

    def equivalent_axes(self, root: AxisRef) -> tuple[AxisRef, ...]:
        """Find axes whose equal seeds have equal closures through identity edges.

        Equal selections on these axes imply one another, so their closures are
        the same even when other, nonidentity relations exist. This does not
        extrapolate between different positions, prove a layout, or replace the
        analysis of at least one actual request.

        Args:
            root: Axis to start from.

        Returns:
            The equivalent axes sorted by tensor ID and dimension, including
            `root`. Returns `(root,)` when `root` is not a native `AxisRef`.
        """
        if type(root) is not AxisRef or not has_native_semantics(root):
            return (root,)
        if root in self._equivalent:
            return self._equivalent[root]
        tree: dict[AxisRef, list[tuple[AxisRef, str]]] = defaultdict(list)
        members = {root}
        for origin, target, reason in self._identity_edges(root):
            members.add(target)
            tree[origin].append((target, reason))
            tree[target].append((origin, reason))
        result = tuple(sorted(members, key=lambda axis: (axis.tensor.id, axis.dim)))
        self._trees.setdefault(result[0], tree)
        for axis in result:
            self._equivalent[axis] = result
        return result

    def _identity_edges(self, root: AxisRef) -> Iterator[tuple[AxisRef, AxisRef, str]]:
        """Yield a spanning tree rooted at the seed, with each relation's explanation."""
        seen = {root}
        pending = deque((root,))
        while pending:
            axis = pending.popleft()
            for relation in self._adjacent.get(axis.tensor, ()):
                if id(relation) not in self._native_relations:
                    continue
                if type(relation) is BroadcastRelation:
                    target = _broadcast_axis(relation, axis)
                elif type(relation) is ReshapeRelation:
                    prefix, suffix = _shared_reshape_axis_counts(
                        relation.left.shape, relation.right.shape
                    )
                    ref = relation.right if axis.tensor == relation.left else relation.left
                    if axis.dim < prefix:
                        target = ref.axis(axis.dim)
                    elif axis.dim >= len(axis.tensor.shape) - suffix:
                        target = ref.axis(axis.dim - len(axis.tensor.shape))
                    else:
                        target = None
                elif type(relation) is PermuteRelation:
                    target = (
                        relation.right.axis(relation.dims.index(axis.dim))
                        if axis.tensor == relation.left
                        else relation.left.axis(relation.dims[axis.dim])
                    )
                else:
                    endpoints = _equal_endpoints(relation)
                    if endpoints is None or axis not in endpoints:
                        continue
                    left, right = endpoints
                    target = right if axis == left else left
                if target is not None and target not in seen:
                    seen.add(target)
                    pending.append(target)
                    yield axis, target, relation.reason

    def slice_provenance(
        self, seed: Selection, slices: tuple[Selection, ...]
    ) -> tuple[Provenance, ...]:
        """Explain the initial equivalent slices using the declared operator relations."""
        if len(slices) <= 1:
            return ()
        by_tensor = {s.tensor: s for s in slices}
        axis = next(i for i, a in enumerate(seed.regions[0].axes) if len(a) != seed.tensor.shape[i])
        root = seed.tensor.axis(axis)
        tree = self._trees[self._equivalent[root][0]]
        # Previously seeded tensors can lie between two new slices. A path
        # across such a tensor explains a transitive identity implication.
        pending = deque(((root, None, root),))
        steps = []
        while pending:
            origin, parent, source = pending.popleft()
            for target, reason in tree[origin]:
                if target == parent:
                    continue
                ancestor = source
                if target.tensor in by_tensor:
                    steps.append(
                        Provenance(
                            by_tensor[source.tensor],
                            by_tensor[target.tensor],
                            reason if source == origin else IDENTITY_REASON,
                        )
                    )
                    ancestor = target
                pending.append((target, origin, ancestor))
        return tuple(steps)

    def maps_within(self, selection: Selection, closure: Mapping[str, Selection]) -> bool:
        """Check all consequences of one selection against a proposed closure."""
        for relation in self._adjacent.get(selection.tensor, ()):
            for target in relation.propagate(selection):
                known = closure.get(target.tensor.id)
                if target and (known is None or not target.issubset(known)):
                    return False
        return True

    def project_slices(
        self, members: tuple[AxisRef, ...], indices: IndexSet, base: Mapping[str, Selection]
    ) -> dict[str, Selection] | None:
        """Extend a complete closure by proper slices of a proved component.

        Identity edges force the new slices on every member. For overlapping
        selections, checking all adjacent relations proves that these additions
        suffice. Other axes must retain their compactness and complete slices
        for callers that reuse their previous constraint checks.
        """
        try:
            added = {m.tensor.id: m.select(indices) for m in members}
            for member in members:
                old = base.get(member.tensor.id)
                if old:
                    added[member.tensor.id] = old.union(added[member.tensor.id])
            closure = ChainMap(added, base)
            for member in members:
                old = base.get(member.tensor.id, Selection(member.tensor))
                joint = added[member.tensor.id]
                if (joint.compact_shape() is None) != (old.compact_shape() is None):
                    return None
                if any(
                    joint.fully_selected_indices(dim) != old.fully_selected_indices(dim)
                    for dim in range(len(member.tensor.shape))
                    if dim != member.dim
                ):
                    return None
                if not self.maps_within(joint, closure):
                    return None
            return added
        except AnalysisLimitError:
            return None

    def canonical_seeds(self, seeds: tuple[Selection, ...]) -> tuple[Selection, ...]:
        """Replace single-axis slices with the same slices on a canonical equivalent axis.

        A seed is rewritten only when it is one native region that is partial on
        exactly one axis; the canonical axis is the first result of
        `equivalent_axes()`. This proves mutual implication for these exact
        coordinates. It does not merge requests, choose channels, or extrapolate
        a closure to other indices. Other seeds are returned unchanged.

        Args:
            seeds: Selections to canonicalize.

        Returns:
            One selection per seed, in input order.
        """
        result = []
        for seed in seeds:
            root = _slice_axis(seed)
            if root is None:
                result.append(seed)
                continue
            first = self.equivalent_axes(root)[0]
            result.append(seed if first == root else first.select(seed.regions[0].axes[root.dim]))
        return tuple(result)

    def equivalent_slices(
        self, seed: Selection, *, prefix: tuple[Selection, ...] = (), first_only: bool = False
    ) -> tuple[Selection, ...]:
        """Return equivalent proper slices that can be seeded together in stable order.

        Only tensors without a prefix seed can supply a shared starting slice.
        The other members still participate in ordinary propagation.
        """
        root = _slice_axis(seed)
        if root is None:
            return (seed,)
        indices = seed.regions[0].axes[root.dim]
        members = self.equivalent_axes(root)
        # Slices on two axes of one tensor can overlap. Seeding them all would
        # need a different provenance delta for each starting point.
        tensors = {axis.tensor for axis in members}
        if len(tensors) != len(members):
            return (seed,)
        if prefix:
            occupied = {s.tensor for s in prefix}
            if seed.tensor in occupied:
                return (seed,)
            members = tuple(axis for axis in members if axis.tensor not in occupied)
        if first_only:
            members = members[:1]
        return tuple(seed if axis == root else axis.select(indices) for axis in members)

    def component(self, root: AxisRef) -> tuple[AxisRef, ...] | None:
        """Return the axes proven to share coordinates with `root`, if a proof exists.

        Only proper subsets of the root positions can use this correspondence,
        because removing every position can trigger additional axes. A broadcast
        preserves the axis only when its width matches. A same-shape reshape
        correspondence comes from the declared relation, never from observed
        shapes alone.

        Args:
            root: Axis whose component is requested.

        Returns:
            The component's axes sorted by tensor ID, one axis per tensor, or
            `None` when no proof is available.
        """
        if type(root) is not AxisRef or not has_native_semantics(root):
            return None
        if self._max_parts != config.max_parts:
            # This index is also public independently of DependencyGraph. A
            # cached proof/failure must not bypass a changed expansion bound.
            self._components.clear()
            self._max_parts = config.max_parts
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
        if not 1 < width <= self._max_parts:
            return None
        members: dict[TensorRef, AxisRef] = {}
        pending = deque((root,))
        while pending:
            axis = pending.popleft()
            if 0 in axis.tensor.shape:
                return None
            if axis.tensor in members:
                if members[axis.tensor] != axis:
                    return None
                continue
            members[axis.tensor] = axis
            for relation in self._adjacent.get(axis.tensor, ()):
                if id(relation) not in self._native_relations:
                    return None
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


def _slice_axis(seed: Selection) -> AxisRef | None:
    """Recognize one native rectangle partial on exactly one axis."""
    if not has_native_semantics(seed) or len(seed.regions) != 1:
        return None
    partial = [
        dim
        for dim, indices in enumerate(seed.regions[0].axes)
        if len(indices) != seed.tensor.shape[dim]
    ]
    return seed.tensor.axis(partial[0]) if len(partial) == 1 else None


def _equal_endpoints(relation: Relation) -> tuple[AxisRef, AxisRef] | None:
    """Recognize full-axis identity edges from already verified native records."""
    if type(relation) is not AxisRelation:
        return None
    if not relation.left.is_full or not relation.right.is_full:
        return None
    left, right = relation.left.axis, relation.right.axis
    width = left.tensor.shape[left.dim]
    if right.tensor.shape[right.dim] != width or relation.maps != (BlockMap(0, 0, width),):
        return None
    return left, right


def _broadcast_axis(relation: BroadcastRelation, axis: AxisRef) -> AxisRef | None:
    """Preserve an axis of a verified native relation when its width is not broadcast."""
    offset = len(relation.big.shape) - len(relation.small.shape)
    forward = axis.tensor == relation.small
    target = relation.big if forward else relation.small
    dim = axis.dim + offset if forward else axis.dim - offset
    if not 0 <= dim < len(target.shape) or target.shape[dim] != axis.tensor.shape[axis.dim]:
        return None
    return target.axis(dim)
