"""One candidate universe for planning, budget accounting and sparse training."""

from collections import defaultdict, deque

from ..contracts import Fixed
from ..relations import AxisRelation, BlockMap, BroadcastRelation, ReshapeRelation
from .groups import ParameterGroup, unique_groups
from .types import Candidate, PlanningError


def interface_constraints(graph, preserve_io):
    """Return the standard protection of every external tensor axis."""
    return (
        tuple(Fixed(ref.axis(d)) for ref in graph.interfaces() for d in range(len(ref.shape)))
        if preserve_io
        else ()
    )


def discover(graph, operations):
    """Collect logical domains before instantiating individual candidate requests."""
    domains, keys = {}, {}
    for op in operations:
        for domain in graph.operator_spec(op).candidates:
            axis, block = domain.axis, domain.block_size
            identity = (
                domain.binding if domain.binding is not None else axis,
                block,
                axis.tensor.shape[axis.dim],
            )
            if domain.key in keys:
                if keys[domain.key] != identity:
                    raise PlanningError(f"Conflicting candidate domain key: {domain.key}")
                continue
            keys[domain.key] = identity
            if axis in domains and domains[axis].block_size != block:
                raise PlanningError("Conflicting block sizes for one candidate axis")
            if axis.tensor.shape[axis.dim] % block:
                raise PlanningError("Candidate block must divide the logical width")
            domains.setdefault(axis, domain)
    return tuple(domains.values())


def _candidates(domain):
    axis, block = domain.axis, domain.block_size
    for start in range(0, axis.tensor.shape[axis.dim], block):
        yield Candidate(
            f"{domain.key}:{start:012d}", (axis.select(range(start, start + block)),), axis
        )


def _protected_equal_axes(graph, defaults):
    """Prove full IO protection through unscoped identity axis relationships.

    Keep axes distinct: removing an entire tensor would accidentally couple its
    otherwise independent row/column domains. Other maps use per-candidate proof.
    """
    protected = {c.axis for c in defaults}
    edges = []
    for relation in graph.relations:
        if isinstance(relation, (ReshapeRelation, BroadcastRelation)):
            left, right = relation.refs
            if left.shape == right.shape:
                edges.extend((left.axis(d), right.axis(d)) for d in range(len(left.shape)))
            continue
        if not isinstance(relation, AxisRelation):
            continue
        left, right = relation.left, relation.right
        width = left.tensor.shape[left.axis.dim]
        if (
            left.scope is None
            and right.scope is None
            and right.tensor.shape[right.axis.dim] == width
            and relation.maps == (BlockMap(0, 0, width),)
        ):
            edges.append((left.axis, right.axis))
    adjacent = defaultdict(set)
    for left, right in edges:
        adjacent[left].add(right)
        adjacent[right].add(left)
    pending = deque(protected)
    while pending:
        for axis in adjacent[pending.popleft()] - protected:
            protected.add(axis)
            pending.append(axis)
    return protected


class CandidateSpace:
    """Discover candidates and expose complete dependency groups without a budget.

    Args:
        graph: Fresh dependency snapshot.
        candidates: Optional explicit candidates; requires explicit axes.
        axes: Logical budget axes. Explicit axes retain protected domains.
        preserve_io: Protect external axes, as in Pruner.plan.
        constraints: Additional structural constraints.

    Discovery only removes domains provably protected by external interfaces.
    protected_axes records those excluded domains for cumulative accounting.
    Unknown paths remain visible; group extraction rejects incomplete influence.
    Groups may overlap and do not imply zero-mask/physical-pruning equivalence.
    """

    def __init__(self, graph, *, candidates=None, axes=None, preserve_io=True, constraints=()):
        graph.validate()
        self.graph = graph
        defaults = interface_constraints(graph, preserve_io)
        self.constraints = (*defaults, *tuple(constraints))
        exclusions = []
        self.protected_axes = ()
        self._candidates = None
        self._domains = ()
        if candidates is None:
            domains = discover(graph, graph.operations())
            if axes is None:
                protected = _protected_equal_axes(graph, defaults) if defaults else set()
                if defaults:
                    for domain in domains:
                        if domain.axis not in protected and all(
                            any(
                                d.code == "fixed_axis"
                                for d in graph.propagate(
                                    remove=c.remove, constraints=defaults
                                ).diagnostics
                            )
                            for c in _candidates(domain)
                        ):
                            protected.add(domain.axis)
                self.protected_axes = tuple(d.axis for d in domains if d.axis in protected)
                exclusions = [
                    (f"domain:{d.key}", "All positions are protected by external interfaces")
                    for d in domains
                    if d.axis in protected
                ]
                domains = tuple(d for d in domains if d.axis not in protected)
                axes = tuple(d.axis for d in domains)
            self._domains = domains
        else:
            if axes is None:
                raise ValueError("Custom candidates require explicit budget axes")
            self._candidates = tuple(candidates)
            if len({c.key for c in self._candidates}) != len(self._candidates):
                raise ValueError("Duplicate candidate keys")
            for candidate in self._candidates:
                for selection in candidate.remove:
                    graph.metadata(selection.tensor)
        self.axes = tuple(dict.fromkeys(axes))
        self.exclusions = tuple(exclusions)
        for axis in self.axes:
            graph.metadata(axis.tensor)

    @property
    def candidates(self):
        """Instantiate the fixed candidate universe only when it is requested."""
        self.graph.validate()
        if self._candidates is None:
            self._candidates = tuple(c for domain in self._domains for c in _candidates(domain))
        return self._candidates

    def impact(self, candidates):
        """Analyze a joint batch, including temporary combined candidates."""
        return self.graph.propagate(
            remove=tuple(s for c in candidates for s in c.remove),
            constraints=self.constraints,
        )

    def parameter_groups(self, candidates=None, *, parameter_filter=None):
        """Return unique complete groups; reject incomplete influence or empty filters.

        The filter accepts (TensorRef, Parameter). Repairable count constraints
        are allowed here; the eventual joint pruning request must still compile.
        """
        result = []
        for candidate in self.candidates if candidates is None else tuple(candidates):
            impact = self.impact((candidate,))
            if not impact.complete:
                raise PlanningError(
                    "Incomplete parameter group: "
                    + "; ".join(d.message for d in impact.diagnostics if not d.complete)
                )
            selections = []
            for selection in impact.parameters:
                if parameter_filter is None or parameter_filter(
                    selection.tensor, self.graph.tensor(selection.tensor)
                ):
                    selections.append(selection)
                self.graph.validate()
            result.append(ParameterGroup(self.graph, tuple(selections), candidate.key))
        return unique_groups(result)
