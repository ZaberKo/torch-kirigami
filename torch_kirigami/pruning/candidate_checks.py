"""Shared proofs for excluding irreversible removal requests before scoring."""

import weakref
from collections import ChainMap, OrderedDict, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TypeVar

from ..dep_graph.contracts import (
    IRREVERSIBLE_CONSTRAINTS,
    AlignmentClass,
    AxisBarrier,
    Balanced,
    Constraint,
    Diagnostic,
    Divisible,
    Fixed,
    Impact,
    NonEmpty,
    alignment_classes,
    alignment_conflicts,
)
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.selection import AxisRef, IndexSet, Region, Selection, TensorRef, slice_intervals
from ..errors import AnalysisLimitError
from ..settings import config
from .types import Candidate

# Built-ins whose `check` reads only complete removals and compactness on one axis.
_AXIS_CONSTRAINTS = (NonEmpty, Fixed, Balanced, Divisible, AxisBarrier)

_T = TypeVar("_T")
# Positions that an accepted closure removes on every member, and the accepted
# member selections when they contain more than those slices.
_ComponentState = tuple[IndexSet, tuple[Selection, ...] | None]
_Domain = tuple[AxisRef, IndexSet, tuple[AxisRef, ...]]
_RequestKey = tuple[tuple[TensorRef, tuple[Region, ...]], ...]


def propagate_request(
    graph: DependencyGraph, remove: tuple[Selection, ...], constraints: tuple[Constraint, ...]
) -> Impact:
    """Propagate one planning request, reporting a seed merge beyond `max_parts` as incomplete.

    `DependencyGraph.propagate` raises when the requested selections cannot be
    merged within `max_parts`. Planning treats such a request like one whose
    relations reach the limit, so a strategy can skip it and try other
    candidates.

    Args:
        graph: Graph whose original coordinates define `remove`.
        remove: Selections analyzed together, in request order.
        constraints: Extra constraints checked against the closure.

    Returns:
        The impact of `remove`. When its seeds cannot be merged, the impact
        has no selections, provenance, interfaces or requirements, one
        incomplete `analysis_limit` diagnostic, and status `unresolved`.

    Raises:
        ValueError: A selection or constraint belongs to another graph.
        torch_kirigami.StaleGraphError: The model changed after capture.
    """
    try:
        return graph.propagate(remove=remove, constraints=constraints)
    except AnalysisLimitError as error:
        for selection in remove:
            graph.metadata(selection.tensor)  # A later foreign seed still raises ValueError.
        empty = graph.propagate(remove=(), constraints=constraints)
        diagnostic = Diagnostic(
            "analysis_limit",
            str(error),
            tensors=tuple(dict.fromkeys(selection.tensor.id for selection in remove)),
            complete=False,
        )
        return replace(
            empty,
            requested=remove,
            selections={},
            diagnostics=(diagnostic,),
            requirements=(),
            provenance=(),
            interfaces=(),
            status="unresolved",
        )


class CandidateFilter:
    """Prove candidate effects and irreversible conflicts without general feasibility.

    Native proper-subset identity components supply exact closure projections,
    also when accepted removals cover other slices of member tensors. Other
    requests use ordinary propagation. No model data or full impacts are
    cached; each cache of projection proofs or rejection results is bounded by
    `capacity`. Callers validate graph freshness at batch and callback
    boundaries.

    Args:
        graph: Graph whose original coordinates define the requests.
        constraints: Extra constraints shared by discovery or this planning round.
        query: Optional cached propagation query using these same constraints.
        candidate_count: Number of candidates that each scan checks, or zero
            when unknown. It raises `capacity`.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        constraints: tuple[Constraint, ...],
        *,
        query: Callable[[tuple[Selection, ...]], Impact] | None = None,
        candidate_count: int = 0,
    ) -> None:
        self.graph = graph
        self.constraints = constraints
        self._query = query
        self._candidate_count = candidate_count
        self.index = graph.identity_axes
        self._fixed: dict[AxisRef, Fixed] = {}
        all_constraints = (*graph.constraints, *constraints)
        self._native = all(has_native_semantics(c) for c in all_constraints)
        self._rejections: OrderedDict[_RequestKey, tuple[Diagnostic, ...]] = OrderedDict()
        self._by_axis: dict[AxisRef, list[Constraint]] = defaultdict(list)
        self._by_tensor: dict[str, list[Constraint]] = defaultdict(list)
        self._alignment_by_axis: dict[AxisRef, AlignmentClass] = {}
        self._empty_diagnostics: list[Diagnostic] = []
        self._cache_settings: tuple[int, int] | None = None
        self._domains: OrderedDict[int, tuple[Candidate, _Domain | None]] = OrderedDict()
        # Proofs that read the most recent accepted impact object, keyed by kind.
        self._accepted_impact: weakref.ReferenceType[Impact] | None = None
        self._accepted_proofs: OrderedDict[Hashable, object] = OrderedDict()
        # Proofs that read only the members of pure-slice states.
        self._member_rejections: OrderedDict[
            tuple[tuple[AxisRef, ...], IndexSet], tuple[Diagnostic, ...] | None
        ] = OrderedDict()
        self._complete_positions: OrderedDict[
            tuple[tuple[AxisRef, ...], IndexSet, IndexSet], IndexSet
        ] = OrderedDict()
        for constraint in all_constraints:
            for ref in constraint.refs:
                graph.metadata(ref)
            if self._native and type(constraint) in _AXIS_CONSTRAINTS:
                self._by_axis[constraint.axis].append(constraint)
            elif self._native:
                for ref in dict.fromkeys(constraint.refs):
                    self._by_tensor[ref.id].append(constraint)
            if type(constraint) is Fixed and has_native_semantics(constraint):
                for axis in self.index.equivalent_axes(constraint.axis):
                    self._fixed.setdefault(axis, constraint)
            if self._native:
                try:
                    diagnostic = constraint.check(MappingProxyType({}))
                except AnalysisLimitError:
                    self._native = False
                    continue
                # A repairable empty-query violation may disappear after removal.
                # Retain only persistent rejection or incomplete-proof information.
                if diagnostic is not None and (diagnostic.irreversible or not diagnostic.complete):
                    self._empty_diagnostics.append(diagnostic)
        if self._native:
            classes = alignment_classes(all_constraints, self.index.equivalent_axes)
            for alignment in classes:
                for axis in alignment.axes:
                    self._alignment_by_axis[axis] = alignment
            self._empty_diagnostics.extend(alignment_conflicts(classes, MappingProxyType({})))
        self.native_semantics = self._native and all(
            has_native_semantics(r) for r in graph.relations
        )
        self.reuse_equivalent = (
            self.native_semantics
            and type(graph) is DependencyGraph
            and not any(callable(value) for value in vars(graph).values())
        )

    @property
    def capacity(self) -> int:
        """Return the bound of each cache, or zero when caching is disabled.

        `DynamicGreedy` checks every candidate again after each accepted change,
        in the same order. A least-recently-used cache smaller than one such
        scan evicts each proof before its reuse, so the bound is at least twice
        `candidate_count`: one projection domain and about one proof per
        candidate, with room for states revisited across scans. A
        `candidate_cache_size` of zero still disables every cache.
        """
        size = config.candidate_cache_size
        return max(size, 2 * self._candidate_count) if size else 0

    def parameter_selections(self, candidate: Candidate) -> tuple[Selection, ...] | None:
        """Project complete native parameter effects, retaining conflicts for raw queries.

        None requires ordinary propagation. This never certifies execution or
        excludes a candidate for a complete constraint violation.
        """
        projected = self._project(candidate, None)
        if projected is None:
            return None
        members, indices, _base = projected
        if not self._members_complete(members, indices, None, None) or any(
            not diagnostic.complete for diagnostic in self._empty_diagnostics
        ):
            return None
        selections = {member.tensor.id: member.select(indices) for member in members}
        return tuple(
            selections[key]
            for key in sorted(selections)
            if selections[key] and selections[key].tensor.kind == "parameter"
        )

    def component_state(
        self, accepted: Impact, members: tuple[AxisRef, ...]
    ) -> tuple[IndexSet, tuple[Selection, ...] | None] | None:
        """Return the accepted removals on one identity component.

        Args:
            accepted: Impact of the accepted removals.
            members: An identity component from `IdentityAxisIndex.component`.

        Returns:
            The positions that `accepted` fully removes on every member, and
            None when it removes exactly those slices. Otherwise the second
            item holds the accepted member selections, in member order. A
            complete closure removes the same positions on every member,
            because identity relations map them onto each other; any other
            state returns None and needs ordinary propagation.
        """
        self.graph.validate_impact(accepted)
        self._sync_cache_settings()
        return self._accepted_state(accepted, members)

    def complete_positions(
        self,
        members: tuple[AxisRef, ...],
        accepted: Impact,
        positions: IndexSet | None = None,
    ) -> IndexSet:
        """Return the positions whose single-slice removal has a proved complete closure.

        The request for a position adds the accepted requests and one region
        that selects the slice at that position on one member. Its closure
        adds exactly that slice on each member when the accepted closure
        removes only member slices. When the accepted closure also removes
        other parts of member tensors, such as the columns of a weight whose
        rows are members, each joint state is proved closed instead; see
        `_joint_selections`. Constraints that read a member tensor are checked
        against the joint state, which keeps the accepted selections elsewhere.

        Args:
            members: An identity component from `IdentityAxisIndex.component`.
            accepted: Complete impact of the accepted removals.
            positions: Positions to prove, or None for every position that
                `accepted` does not remove. The work is proportional to their
                number.

        Returns:
            The positions among `positions` that `accepted` does not remove and
            whose joint closure with `accepted` is proved complete. A removal
            of the last position is never proved. A missing position requires
            ordinary propagation; it does not mean that its closure is
            incomplete.
        """
        self.graph.validate_impact(accepted)
        self._sync_cache_settings()
        if not self._native or not accepted.complete:
            return IndexSet()
        state = self._accepted_state(accepted, members)
        if state is None:
            return IndexSet()
        removed, base = state
        root = members[0]
        width = root.tensor.shape[root.dim]
        if width - len(removed) <= 1:
            return IndexSet()  # Whole-axis deletion can activate other axes and relations.
        try:
            surviving = IndexSet.span(0, width).subtract(removed)
            if positions is not None:
                surviving = surviving.intersect(positions)
        except AnalysisLimitError:
            return IndexSet()  # Positions that this proof cannot represent use propagation.
        if not surviving:
            return IndexSet()
        tensors = {member.tensor for member in members}
        local = base is None and all(
            ref in tensors
            for member in members
            for constraint in self._by_tensor.get(member.tensor.id, ())
            for ref in constraint.refs
        )
        context = None if local else accepted

        def prove() -> IndexSet:
            """Check one joint closure per requested surviving position."""
            return IndexSet.of(
                position
                for position in surviving
                if self._members_complete(
                    members, removed.union(IndexSet.of([position])), context, base
                )
            )

        key = (members, removed, surviving)
        try:
            if local:
                return self._cached(self._complete_positions, key, prove)
            return self._cached(self._proofs(accepted), ("complete", *key), prove)
        except AnalysisLimitError:
            return IndexSet()

    def fixed_rejection(self, axis: AxisRef) -> Diagnostic | None:
        """Prove whole-domain Fixed protection through native identity relations."""
        self._sync_cache_settings()
        constraint = self._fixed.get(axis)
        if constraint is None or not axis.tensor.shape[axis.dim]:
            return None
        return constraint.check({constraint.axis.tensor.id: constraint.axis.select([0])})

    def request_key(self, remove: tuple[Selection, ...]) -> _RequestKey:
        """Keep the ordered prefix and use the solver's representative for a disjoint last slice."""
        if self.reuse_equivalent and remove:
            prefix = remove[:-1]
            tail = self.index.equivalent_slices(remove[-1], prefix=prefix, first_only=True)
            remove = (*prefix, *tail)
        return tuple((s.tensor, s.regions) for s in remove)

    def rejections(
        self, candidate: Candidate, accepted_impact: Impact | None = None
    ) -> tuple[Diagnostic, ...]:
        """Return only irreversible diagnostics for this joint removal request.

        A request whose seeds cannot be merged within `max_parts` has an
        incomplete impact and no irreversible diagnostic.
        """
        self._sync_cache_settings()
        projected = self._projected_rejections(candidate, accepted_impact)
        if projected is not None:
            return projected
        cacheable = (
            self.reuse_equivalent
            and self._query is None
            and accepted_impact is None
            and self.capacity > 0
            and has_native_semantics(candidate.remove)
            and type(self.graph) is DependencyGraph
            and not any(callable(value) for value in vars(self.graph).values())
        )
        if cacheable:
            key = self.request_key(candidate.remove)
            if key in self._rejections:
                self.graph.validate()
                self._rejections.move_to_end(key)
                return self._rejections[key]
        requested = () if accepted_impact is None else accepted_impact.requested
        try:
            seeds = (*requested, *candidate.remove)
            impact = (
                propagate_request(self.graph, seeds, self.constraints)
                if self._query is None
                else self._query(seeds)
            )
            result = tuple(d for d in impact.diagnostics if d.irreversible)
            if cacheable and impact.complete:
                self._remember(self._rejections, key, result)
            return result
        finally:
            self.graph.validate()

    def _project(
        self, candidate: Candidate, accepted: Impact | None
    ) -> tuple[tuple[AxisRef, ...], IndexSet, tuple[Selection, ...] | None] | None:
        """Locate a proper-subset component removal, retaining other fallbacks.

        Returns the identity component, the joint positions removed from each
        of its axes, and the accepted member selections of a conditional
        state, or None for a pure-slice state. A pure-slice closure selects
        exactly those slices on every member and nothing else beyond the
        accepted closure. A conditional closure still needs `_joint_selections`.
        """
        for selection in candidate.remove:
            self.graph.metadata(selection.tensor)
        if accepted is not None:
            self.graph.validate_impact(accepted)
        if candidate.axis is not None:
            self.graph.metadata(candidate.axis.tensor)
        if not self._native:
            return None
        domain = self.native_domain(candidate)
        if domain is None:
            return None
        axis, indices, members = domain
        base = None
        if accepted is not None:
            state = self._accepted_state(accepted, members)
            if state is None:
                return None
            before, base = state
            try:
                indices = indices.union(before)
            except AnalysisLimitError:
                return None
        if len(indices) == axis.tensor.shape[axis.dim]:
            return None  # Whole-tensor deletion can activate other axes and relations.
        return members, indices, base

    def native_domain(self, candidate: Candidate) -> _Domain | None:
        """Return a native candidate's axis, positions and identity component.

        Native candidates are deeply immutable, so the result is reused for the
        same object. Extension candidates are never projected.
        """
        self._sync_cache_settings()
        entry = self._domains.get(id(candidate))
        if entry is not None and entry[0] is candidate:
            self._domains.move_to_end(id(candidate))
            return entry[1]
        axis = candidate.axis
        if (
            type(candidate) is not Candidate
            or axis is None
            or not has_native_semantics(axis)
            or not has_native_semantics(candidate.remove)
        ):
            return None
        domain = None
        if all(seed.tensor == axis.tensor for seed in candidate.remove):
            try:
                selection = candidate.remove[0]
                for seed in candidate.remove[1:]:
                    selection = selection.union(seed)
                intervals = slice_intervals(axis, (selection,))
                if intervals is not None:
                    members = self.index.component(axis)
                    if members is not None:
                        domain = (axis, IndexSet(intervals), members)
            except AnalysisLimitError:
                pass  # Ordinary propagation reports the limit or analyzes the request.
        self._remember(self._domains, id(candidate), (candidate, domain))
        return domain

    def _proofs(self, accepted: Impact) -> OrderedDict[Hashable, object] | None:
        """Return the proof cache of the most recent accepted impact object."""
        if type(accepted) is not Impact:
            return None
        if self._accepted_impact is None or self._accepted_impact() is not accepted:
            self._accepted_impact = weakref.ref(accepted)
            self._accepted_proofs.clear()
        return self._accepted_proofs

    def _accepted_state(
        self, accepted: Impact, members: tuple[AxisRef, ...]
    ) -> _ComponentState | None:
        """Implement `component_state`, reusing results for one accepted impact object."""

        def state() -> _ComponentState | None:
            """Compare each member's accepted selection with the component slices."""
            selections = tuple(accepted.selection(member.tensor) for member in members)
            before = selections[0].fully_selected_indices(members[0].dim)
            if all(
                selection == member.select(before)
                for member, selection in zip(members, selections, strict=True)
            ):
                return before, None
            if accepted.complete and all(
                selection.fully_selected_indices(member.dim) == before
                for member, selection in zip(members, selections, strict=True)
            ):
                return before, selections
            return None

        try:
            return self._cached(self._proofs(accepted), ("state", members), state)
        except AnalysisLimitError:
            return None

    def _joint_selections(
        self,
        members: tuple[AxisRef, ...],
        indices: IndexSet,
        accepted: Impact | None,
        base: tuple[Selection, ...] | None,
    ) -> Mapping[str, Selection] | None:
        """Return the joint closure of accepted removals and component positions.

        Each member selects its accepted selection and the slices at `indices`.
        Unlisted tensors keep their accepted selections. With a pure-slice
        state, the component proof establishes this closure. Otherwise the
        state contains the closure, because the accepted closure and the
        identity relations force it. The closure contains the state when every
        relation adjacent to a member maps it back into the state. Native
        relations are monotone, so the state is then the least fixed point.

        The state must also keep the compactness of each member tensor and the
        complete cross-sections of its other axes. Built-in axis constraints
        and alignment classes read only these properties, so on other axes they
        keep their accepted results, and counted removals beyond the component
        stay those of `accepted`. Returns None when this proof fails or reaches
        an analysis limit.
        """
        if base is None:
            view = {member.tensor.id: member.select(indices) for member in members}
        else:
            assert accepted is not None

            found = self._cached(
                self._proofs(accepted),
                ("closure", members, indices),
                lambda: self.index.project_slices(members, indices, accepted.selections),
            )
            if found is None:
                return None
            view = found
        if accepted is None:
            return MappingProxyType(view)
        return MappingProxyType(ChainMap(view, accepted.selections))

    def _projected_rejections(
        self, candidate: Candidate, accepted: Impact | None
    ) -> tuple[Diagnostic, ...] | None:
        """Prove irreversible conflicts from the projected component closure.

        Each checked constraint and joint `alignment_conflicts` class reads the
        joint closure. Other axes keep the conflicts of the empty or accepted
        request.
        """
        projected = self._project(candidate, accepted)
        if projected is None:
            return None
        members, indices, base = projected

        def check() -> tuple[Diagnostic, ...] | None:
            """Run irreversible constraint types on the member axes."""
            # Pure-slice checks read members alone, so their proofs are shared
            # by every accepted impact.
            context = None if base is None else accepted
            selections = self._joint_selections(members, indices, context, base)
            if selections is None:
                return None
            constraints = (
                constraint
                for member in members
                for constraint in self._by_axis.get(member, ())
                if type(constraint) in IRREVERSIBLE_CONSTRAINTS
            )
            return self._check_members(members, selections, constraints)

        if base is None:
            rejected = self._cached(self._member_rejections, (members, indices), check)
        else:
            assert accepted is not None
            rejected = self._cached(self._proofs(accepted), ("rejections", members, indices), check)
        if rejected is None:
            return None
        diagnostics = (
            *self._empty_diagnostics,
            *(() if accepted is None else accepted.diagnostics),
        )
        return tuple(dict.fromkeys(d for d in (*diagnostics, *rejected) if d.irreversible))

    def _members_complete(
        self,
        members: tuple[AxisRef, ...],
        indices: IndexSet,
        accepted: Impact | None,
        base: tuple[Selection, ...] | None,
    ) -> bool:
        """Check that no constraint able to see a projected removal is incomplete.

        `accepted` may be None only when every constraint on a member tensor
        reads member tensors alone.
        """
        selections = self._joint_selections(members, indices, accepted, base)
        if selections is None:
            return False
        constraints = (
            constraint
            for member in members
            for constraint in (
                *self._by_axis.get(member, ()),
                *self._by_tensor.get(member.tensor.id, ()),
            )
        )
        diagnostics = self._check_members(members, selections, constraints)
        return diagnostics is not None and all(d.complete for d in diagnostics)

    def _check_members(
        self,
        members: tuple[AxisRef, ...],
        selections: Mapping[str, Selection],
        constraints: Iterable[Constraint],
    ) -> tuple[Diagnostic, ...] | None:
        """Check constraints and joint alignment against a projected joint closure.

        Only member axes need axis constraints and alignment classes. A
        pure-slice removal leaves every other axis of a member tensor without
        complete removed positions and compact, so those axes keep their
        empty-request results, which `_empty_diagnostics` holds. A conditional
        state keeps the accepted results on those axes; see `_joint_selections`.

        Returns:
            The diagnostics of `constraints` and of the `alignment_conflicts`
            classes of member axes, or None when a check reaches an analysis
            limit. Propagation records an analysis limit as an incomplete,
            reversible diagnostic, so a limit in a skipped constraint cannot
            change a rejection.
        """
        diagnostics = []
        # A member's equivalence class lies inside its identity component.
        classes = {
            id(alignment): alignment
            for member in members
            if (alignment := self._alignment_by_axis.get(member)) is not None
        }
        try:
            for constraint in {id(c): c for c in constraints}.values():
                diagnostic = constraint.check(selections)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
            diagnostics.extend(alignment_conflicts(classes.values(), selections))
        except AnalysisLimitError:
            return None
        return tuple(diagnostics)

    def _sync_cache_settings(self) -> None:
        """Clear every cache after `max_parts` or the cache bound changes.

        Every public query calls this first, so a new bound, including zero,
        applies to the next query whichever proof it uses.
        """
        settings = (self.capacity, config.max_parts)
        if settings != self._cache_settings:
            self._rejections.clear()
            self._domains.clear()
            self._accepted_proofs.clear()
            self._member_rejections.clear()
            self._complete_positions.clear()
            self._cache_settings = settings

    def _cached(self, cache: OrderedDict | None, key: Hashable, compute: Callable[[], _T]) -> _T:
        """Return a bounded LRU entry, computing and storing it when absent."""
        if cache is not None and key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = compute()
        if cache is not None:
            self._remember(cache, key, value)
        return value

    def _remember(self, cache: OrderedDict, key: Hashable, value: object) -> None:
        """Insert into a bounded LRU cache sized by `capacity`."""
        capacity = self.capacity
        if not capacity:
            return
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
