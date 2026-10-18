"""Shared proofs for excluding irreversible removal requests before scoring."""

from collections import defaultdict
from collections.abc import Callable
from types import MappingProxyType

from ..dep_graph.contracts import Constraint, Diagnostic, Fixed, Impact
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.selection import AxisRef, IndexSet, Selection
from ..errors import AnalysisLimitError
from .axis_domains import IdentityAxisIndex
from .types import Candidate


class CandidateFilter:
    """Prove candidate effects and irreversible conflicts without general feasibility.

    Native proper-subset identity components supply exact closure projections.
    Other requests use ordinary propagation. No model data or full impacts are
    cached. Callers validate graph freshness at batch and callback boundaries.

    Args:
        graph: Graph whose original coordinates define the requests.
        constraints: Extra constraints shared by discovery or this planning round.
        query: Optional cached propagation query using these same constraints.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        constraints: tuple[Constraint, ...],
        *,
        query: Callable[[tuple[Selection, ...]], Impact] | None = None,
    ) -> None:
        self.graph = graph
        self.constraints = constraints
        self._query = query
        self.index = IdentityAxisIndex(graph.relations)
        self._fixed: dict[AxisRef, Fixed] = {}
        all_constraints = (*graph.constraints, *constraints)
        self._native = all(has_native_semantics(c) for c in all_constraints)
        self._by_tensor: dict[str, list[Constraint]] = defaultdict(list)
        self._empty_diagnostics: list[Diagnostic] = []
        for constraint in all_constraints:
            for ref in constraint.refs:
                graph.metadata(ref)
                if self._native:
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

    def parameter_selections(self, candidate: Candidate) -> tuple[Selection, ...] | None:
        """Project complete native parameter effects, retaining conflicts for raw queries.

        None requires ordinary propagation. This never certifies execution or
        excludes a candidate for a complete constraint violation.
        """
        projected = self._checked_projection(candidate, None)
        if projected is None:
            return None
        selections, diagnostics = projected
        if any(not diagnostic.complete for diagnostic in diagnostics):
            return None
        return tuple(
            selections[key]
            for key in sorted(selections)
            if selections[key] and selections[key].tensor.kind == "parameter"
        )

    def fixed_rejection(self, axis: AxisRef) -> Diagnostic | None:
        """Prove whole-domain Fixed protection through native identity relations."""
        constraint = self._fixed.get(axis)
        if constraint is None or not axis.tensor.shape[axis.dim]:
            return None
        return constraint.check({constraint.axis.tensor.id: constraint.axis.select([0])})

    def rejections(
        self, candidate: Candidate, accepted_impact: Impact | None = None
    ) -> tuple[Diagnostic, ...]:
        """Return only irreversible diagnostics for this joint removal request."""
        projected = self._checked_projection(candidate, accepted_impact)
        if projected is not None:
            _, diagnostics = projected
            return tuple(dict.fromkeys(d for d in diagnostics if d.irreversible))
        requested = () if accepted_impact is None else accepted_impact.requested
        try:
            seeds = (*requested, *candidate.remove)
            impact = (
                self.graph.propagate(remove=seeds, constraints=self.constraints)
                if self._query is None
                else self._query(seeds)
            )
            return tuple(d for d in impact.diagnostics if d.irreversible)
        finally:
            self.graph.validate()

    def _checked_projection(
        self, candidate: Candidate, accepted: Impact | None
    ) -> tuple[dict[str, Selection], tuple[Diagnostic, ...]] | None:
        """Share ownership checks and native constraints between projection consumers.

        Incomplete influence and complete constraint conflicts remain distinct.
        The caller decides whether it needs complete regions or rejection reasons.
        """
        for selection in candidate.remove:
            self.graph.metadata(selection.tensor)
        if accepted is not None:
            self.graph.validate_impact(accepted)
        if candidate.axis is not None:
            self.graph.metadata(candidate.axis.tensor)
        projected = self._project(candidate, accepted) if self._native else None
        if projected is None:
            return None
        selections, members = projected
        checks = {id(c): c for axis in members for c in self._by_tensor.get(axis.tensor.id, ())}
        diagnostics = list(self._empty_diagnostics)
        if accepted is not None:
            diagnostics.extend(accepted.diagnostics)
        try:
            for constraint in checks.values():
                diagnostic = constraint.check(MappingProxyType(selections))
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
        except AnalysisLimitError:
            return None  # Ordinary propagation records limits without inventing a proof.
        return selections, tuple(diagnostics)

    def _project(
        self, candidate: Candidate, accepted: Impact | None
    ) -> tuple[dict[str, Selection], tuple[AxisRef, ...]] | None:
        """Prove an exact proper-subset closure, retaining cross-axis fallbacks."""
        axis = candidate.axis
        if (
            type(candidate) is not Candidate
            or not has_native_semantics(axis)
            or axis is None
            or not has_native_semantics(candidate.remove)
        ):
            return None
        selection = Selection(axis.tensor)
        for seed in candidate.remove:
            if seed.tensor != axis.tensor:
                return None
            selection = selection.union(seed)
        indices = selection.fully_selected_indices(axis.dim)
        if selection != axis.select(indices):
            return None
        members = self.index.component(axis)
        if members is None:
            return None
        current = {} if accepted is None else dict(accepted.selections)
        before = (
            IndexSet()
            if accepted is None
            else accepted.selection(axis.tensor).fully_selected_indices(axis.dim)
        )
        indices = indices.union(before)
        if len(indices) == axis.tensor.shape[axis.dim]:
            return None  # Whole-tensor deletion can activate other axes and relations.
        for member in members:
            if accepted is not None and accepted.selection(member.tensor) != member.select(before):
                return None
            current[member.tensor.id] = member.select(indices)
        return current, members
