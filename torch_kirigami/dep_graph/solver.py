"""Exact dependency closure, constraint indexes, and bounded relation caches."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from ..errors import AnalysisLimitError
from .contracts import (
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
)
from .eligibility import has_native_semantics
from .graph_state import DependencyGraphState, validate_reference
from .relations import AxisRelation, Relation
from .selection import (
    Region,
    Selection,
    TensorRef,
    coalesce_selections,
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
        tensor_operations = {key: tuple(sorted(names)) for key, names in direct.items()}
        operation_successors = {key: tuple(sorted(names)) for key, names in successors.items()}
        return tensor_operations, operation_successors, tuple(dynamic)

    def _index_constraints(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...], dict[int, Diagnostic]]:
        """Index local built-ins and retain their possibly invalid empty states."""
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
        """Validate constraint ownership and preserve analysis-limit diagnostics."""
        for ref in constraint.refs:
            validate_reference(self._data.refs, ref)
        try:
            return constraint.check(selections)
        except AnalysisLimitError as error:
            return Diagnostic("analysis_limit", str(error), complete=False)

    def _constraint_diagnostics(
        self, selections: Mapping[str, Selection], extra: tuple[Constraint, ...]
    ) -> tuple[Diagnostic, ...]:
        """Check touched built-ins and every extension, keeping declaration order."""
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
        return tuple(diagnostics)

    def affected_operations(self, selections: Mapping[str, Selection]) -> frozenset[str]:
        """Find data and dimension consumers using one activation rule."""
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
        """Propagate a joint request and check its complete structural closure."""
        requested = tuple(remove)
        current, queue, queued = {}, deque(), set()
        provenance, diagnostics = [], []

        def add(selection: Selection) -> Selection:
            """Merge a seed or propagated selection into the work queue."""
            validate_reference(self._data.refs, selection.tensor)
            old = current.get(selection.tensor.id, Selection(selection.tensor))
            delta = selection.subtract(old)
            if delta:
                current[selection.tensor.id] = old.union(delta)
                if selection.tensor.id not in queued:
                    queue.append(selection.tensor.id)
                    queued.add(selection.tensor.id)
            return delta

        for selection in self._seed_selections(requested):
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
        extra_constraints = tuple(constraints)
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
