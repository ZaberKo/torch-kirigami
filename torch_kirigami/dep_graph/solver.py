"""Exact dependency closure, constraint indexes, and bounded relation caches."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from ..errors import AnalysisLimitError
from .contracts import (
    AlignmentClass,
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Constraint,
    Diagnostic,
    Divisible,
    Fixed,
    Impact,
    LayoutConstraint,
    NonEmpty,
    Provenance,
    ShapeExpr,
    alignment_classes,
    alignment_conflicts,
)
from .eligibility import has_native_semantics
from .graph_state import DependencyGraphState, validate_reference
from .identity_axes import IDENTITY_REASON, IdentityAxisIndex
from .relations import AxisRelation, Relation
from .selection import (
    IndexSet,
    Region,
    Selection,
    TensorRef,
    canonical_boxes,
    coalesce_selections,
    region_boxes,
)

# These built-ins inspect only the selections named by `refs`, provided their
# nested records are also immutable built-ins. Extensions at any depth can read
# other state and must run on every query, including untouched references.
_LOCAL_CONSTRAINT_TYPES = (
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Divisible,
    Fixed,
    LayoutConstraint,
    NonEmpty,
)
# Built-ins combined by `alignment_classes` over identity-equivalent axes.
_ALIGNMENT_TYPES = (Balanced, Divisible, NonEmpty)


class DependencySolver:
    """Query one completed analysis using graph-local indexes and caches.

    Args:
        data: Completed structural records. Model freshness is checked by the
            public graph before it delegates a query to this solver.
    """

    def __init__(self, data: DependencyGraphState) -> None:
        self._data = data
        adjacency: defaultdict[str, list[Relation]] = defaultdict(list)
        for relation in data.relations:
            for ref in dict.fromkeys(relation.refs):
                adjacency[ref.id].append(relation)
        self._adjacency = {key: tuple(relations) for key, relations in adjacency.items()}
        self._cacheable_relations = {
            id(relation)
            for relation in data.relations
            if type(relation) is AxisRelation and has_native_semantics(relation)
        }
        self._relation_outputs: dict[
            tuple[int, str], tuple[tuple[Region, ...], tuple[Selection, ...]]
        ] = {}
        (
            self._constraint_index,
            self._unindexed_constraints,
            self._empty_constraint_diagnostics,
        ) = self._index_constraints()
        native = tuple(c for c in data.constraints if has_native_semantics(c))
        self._alignment_constraints = tuple(c for c in native if type(c) in _ALIGNMENT_TYPES)
        self.identity_axes = IdentityAxisIndex(data.relations)
        self._native_projection = all(has_native_semantics(r) for r in data.relations)
        self._alignment = self._alignment_classes(())
        (
            self._tensor_operations,
            self._operation_successors,
            self._dynamic_expression_consumers,
        ) = self._index_operations()

    def _index_operations(
        self,
    ) -> tuple[
        dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], tuple[tuple[str, ShapeExpr], ...]
    ]:
        """Index direct tensor/size consumers and immediate FX successors."""
        direct: defaultdict[str, set[str]] = defaultdict(set)
        successors: defaultdict[str, set[str]] = defaultdict(set)
        dynamic = []

        for name, operation in self._data.operations.items():
            for ref in (*operation.inputs, *operation.outputs, *operation.bindings.values()):
                direct[ref.id].add(name)
            for node in operation.node.all_input_nodes:
                successors[node.name].add(name)
                expression = self._data.expressions.get(node)
                if expression is None:
                    continue
                if not has_native_semantics(expression):
                    dynamic.append((name, expression))
                elif expression:
                    for ref in expression.refs:
                        direct[ref.id].add(name)
            for writer in self._data.write_dependencies.get(operation.node, ()):
                successors[writer.name].add(name)
        tensor_operations = {key: tuple(sorted(names)) for key, names in direct.items()}
        operation_successors = {key: tuple(sorted(names)) for key, names in successors.items()}
        return tensor_operations, operation_successors, tuple(dynamic)

    def _index_constraints(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...], dict[int, Diagnostic]]:
        """Index local built-in constraints by tensor and record their empty-state diagnostics."""
        by_tensor: defaultdict[str, list[int]] = defaultdict(list)
        unindexed = []
        empty_diagnostics = {}
        empty = MappingProxyType({})
        for index, constraint in enumerate(self._data.constraints):
            if type(constraint) not in _LOCAL_CONSTRAINT_TYPES or not has_native_semantics(
                constraint
            ):
                unindexed.append(index)
                continue
            for ref in constraint.refs:
                by_tensor[ref.id].append(index)
            diagnostic = self._check_constraint(constraint, empty)
            if diagnostic is not None:
                empty_diagnostics[index] = diagnostic
        constraint_index = {key: tuple(indices) for key, indices in by_tensor.items()}
        return constraint_index, tuple(unindexed), empty_diagnostics

    def _check_constraint(
        self, constraint: Constraint, selections: Mapping[str, Selection]
    ) -> Diagnostic | None:
        """Check one constraint after validating its references, reporting limits as diagnostics."""
        for ref in constraint.refs:
            validate_reference(self._data.refs, ref)
        try:
            return constraint.check(selections)
        except AnalysisLimitError as error:
            return Diagnostic("analysis_limit", str(error), complete=False)

    def _constraint_diagnostics(
        self, selections: Mapping[str, Selection], extra: tuple[Constraint, ...]
    ) -> tuple[Diagnostic, ...]:
        """Check touched built-ins and every extension, keeping declaration order.

        Joint `alignment_conflicts` follow the individual diagnostics.
        """
        active = set(self._unindexed_constraints)
        for identity in selections:
            active.update(self._constraint_index.get(identity, ()))
        results = dict(self._empty_constraint_diagnostics)
        readonly = MappingProxyType(selections)
        for index in sorted(active):
            diagnostic = self._check_constraint(self._data.constraints[index], readonly)
            if diagnostic is None:
                results.pop(index, None)
            else:
                results[index] = diagnostic
        diagnostics = [results[index] for index in sorted(results)]
        # Caller-supplied constraints may change between queries; do not retain
        # them or their diagnostics in the immutable graph's index.
        for constraint in extra:
            diagnostic = self._check_constraint(constraint, readonly)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        native = tuple(c for c in extra if type(c) in _ALIGNMENT_TYPES and has_native_semantics(c))
        key, classes = self._alignment
        # Native constraints are immutable records, so identity proves reuse.
        if len(native) != len(key) or any(a is not b for a, b in zip(native, key, strict=True)):
            self._alignment = key, classes = self._alignment_classes(native)
        try:
            diagnostics.extend(alignment_conflicts(classes, readonly))
        except AnalysisLimitError as error:
            diagnostics.append(Diagnostic("analysis_limit", str(error), complete=False))
        return tuple(diagnostics)

    def _alignment_classes(
        self, extra: tuple[Constraint, ...]
    ) -> tuple[tuple[Constraint, ...], tuple[AlignmentClass, ...]]:
        """Combine graph and native extra alignment constraints, keyed by the extras."""
        classes = alignment_classes(
            (*self._alignment_constraints, *extra), self.identity_axes.equivalent_axes
        )
        return extra, classes

    def affected_operations(self, selections: Mapping[str, Selection]) -> frozenset[str]:
        """Find direct tensor and shape-expression consumers, then all their FX successors."""
        affected = {
            name for identity in selections for name in self._tensor_operations.get(identity, ())
        }
        # Extension subclasses can compute refs or truthiness from external
        # state. Keep their original query-time behavior, including nested ones.
        for name, expression in self._dynamic_expression_consumers:
            if expression and any(ref.id in selections for ref in expression.refs):
                affected.add(name)
        pending = deque(affected)
        while pending:
            # Include value/layout consumers even when their shape and selected
            # coordinates are unchanged, as with reductions and static slices.
            for name in self._operation_successors.get(pending.popleft(), ()):
                if name not in affected:
                    affected.add(name)
                    pending.append(name)
        return frozenset(affected)

    def _seed_selections(self, requested: tuple[Selection, ...]) -> tuple[Selection, ...]:
        """Coalesce safe seeds while retaining ordered failures for invalid input."""
        if not all(
            type(selection) is Selection
            and type(selection.tensor) is TensorRef
            and self._data.refs.get(selection.tensor.id) == selection.tensor
            for selection in requested
        ):
            return requested
        return coalesce_selections(requested)

    def propagate(
        self, *, remove: Iterable[Selection], constraints: Iterable[Constraint] = ()
    ) -> Impact:
        """Propagate a joint request to a fixed point and check constraints on the closure.

        A native compound request can recover from an intermediate analysis
        limit by proving the full closure. Unproved limits remain diagnostics.
        """
        requested = tuple(remove)
        extra_constraints = tuple(constraints)
        initial = self._seed_selections(requested)
        projected = ()
        native = self._native_projection and has_native_semantics(requested)
        if (
            requested
            and native
            and self._data.refs.get(requested[-1].tensor.id) == requested[-1].tensor
        ):
            tail = requested[-1]
            projected = self.identity_axes.equivalent_slices(tail, prefix=requested[:-1])
            if len(projected) > 1:
                initial = (*self._seed_selections(requested[:-1]), *projected)
        current, provenance, diagnostics = self._walk(initial, native=native)
        if len(projected) > 1:
            provenance[:0] = self.identity_axes.slice_provenance(requested[-1], projected)
        if native and diagnostics and len(requested) > 1:
            seeds = (*requested[:-1], projected[0]) if len(projected) > 1 else requested
            recovered = self._project_request(seeds, current)
            if recovered is not None:
                current, steps = recovered
                provenance.extend(steps)
                diagnostics = []
        all_constraints = (*self._data.constraints, *extra_constraints)
        diagnostics.extend(self._constraint_diagnostics(current, extra_constraints))
        affected = self.affected_operations(current)
        activated = {
            id(req)
            for name in affected
            if name in self._data.specs
            for req in self._data.specs[name].requirements
        }
        requirements = tuple(
            req
            for req in self._data.requirements
            if id(req) in activated or any(ref.id in current for ref in req.refs)
        )
        diagnostics = tuple(dict.fromkeys(diagnostics))
        status = (
            "conflict"
            if any(d.severity == "conflict" for d in diagnostics)
            else ("unresolved" if diagnostics else "resolved")
        )
        return Impact(
            self._data.id,
            requested,
            MappingProxyType(dict(sorted(current.items()))),
            diagnostics,
            requirements,
            tuple(provenance),
            tuple(self._data.refs[k] for k in sorted(self._data.interfaces) if k in current),
            status,
            all_constraints,
            self._data.refs,
        )

    def _walk(
        self, initial: tuple[Selection, ...], *, native: bool
    ) -> tuple[dict[str, Selection], list[Provenance], list[Diagnostic]]:
        """Propagate the accumulated selections until the work queue is empty."""
        current, queue, queued = {}, deque(), set()
        provenance, diagnostics = [], []

        def add(selection: Selection) -> Selection | None:
            """Merge a seed or propagated selection into the work queue."""
            validate_reference(self._data.refs, selection.tensor)
            old = current.get(selection.tensor.id)
            if native and old is not None and selection.regions == old.regions:
                return None  # A return edge often yields precisely the known slice.
            if old is None:
                old = Selection(selection.tensor)
            delta = selection.subtract(old)
            if delta:
                current[selection.tensor.id] = delta if native and not old else old.union(delta)
                if selection.tensor.id not in queued:
                    queue.append(selection.tensor.id)
                    queued.add(selection.tensor.id)
            return delta

        for selection in initial:
            add(selection)
        disabled = set()
        while queue:
            identity = queue.popleft()
            queued.remove(identity)
            # Relations need the accumulated selection: separate arrivals may
            # jointly complete a block or a broadcast fiber on a shared path.
            source = current[identity]
            cacheable_source = type(source) is Selection and all(
                has_native_semantics(region) for region in source.regions
            )
            for relation in self._adjacency.get(identity, ()):
                if id(relation) in disabled:
                    continue
                try:
                    key = (id(relation), identity)
                    cached = self._relation_outputs.get(key)
                    cacheable = cacheable_source and id(relation) in self._cacheable_relations
                    if cacheable and cached is not None and cached[0] == source.regions:
                        targets = cached[1]
                    else:
                        targets = relation.propagate(source)
                        if cacheable:
                            # One last input/output per endpoint; merges, scheduling,
                            # constraints and provenance still use the fresh joint query.
                            self._relation_outputs[key] = (source.regions, targets)
                    for target in targets:
                        delta = add(target)
                        if delta:
                            provenance.append(Provenance(source, delta, relation.reason))
                except AnalysisLimitError as error:
                    disabled.add(id(relation))
                    diagnostics.append(
                        Diagnostic(
                            "analysis_limit",
                            str(error),
                            tensors=tuple(r.id for r in relation.refs),
                            complete=False,
                        )
                    )
        return current, provenance, diagnostics

    def _project_request(
        self, requested: tuple[Selection, ...], partial: dict[str, Selection]
    ) -> tuple[dict[str, Selection], list[Provenance]] | None:
        """Prove a native closure directly after an intermediate traversal limit.

        The partial result contains only forced coordinates. Each identity
        slice below is also forced by an original seed. If all relations map
        the proposed result into itself, it is the least closed superset.
        """
        current = dict(partial)
        steps = []
        try:
            for seed in requested:
                for target in self.identity_axes.equivalent_slices(seed):
                    old = current.get(target.tensor.id, Selection(target.tensor))
                    if target.issubset(old):
                        continue
                    before, extra = region_boxes(old.regions), region_boxes(target.regions)
                    for box in canonical_boxes(extra, before):
                        delta = target.tensor.select(
                            (Region(tuple(IndexSet(axis) for axis in box)),)
                        )
                        steps.append(Provenance(seed, delta, IDENTITY_REASON))
                    union = canonical_boxes((*before, *extra))
                    current[target.tensor.id] = target.tensor.select(
                        Region(tuple(IndexSet(axis) for axis in box)) for box in union
                    )
            if all(self.identity_axes.maps_within(s, current) for s in current.values()):
                return current, steps
        except AnalysisLimitError:
            pass  # The direct proof must also fit the configured representation.
        return None
