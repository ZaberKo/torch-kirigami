"""One candidate universe for planning, budget accounting and sparse training."""

from collections import defaultdict, deque
from dataclasses import dataclass
from fnmatch import fnmatchcase

from ..contracts import Fixed
from ..relations import AxisRelation, BlockMap, BroadcastRelation, ReshapeRelation
from ..selection import AxisRef
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


@dataclass(frozen=True)
class CandidateSpace:
    """Explicit candidates and logical axes used to count channels.

    Construction performs no discovery or analysis and retains no model. The
    consuming Pruner validates graph ownership. Discovery metadata records excluded
    protected axes for cumulative accounting; explicit axes are never filtered.
    """

    candidates: tuple[Candidate, ...]
    channel_axes: tuple[AxisRef, ...]
    protected_channel_axes: tuple[AxisRef, ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        candidates = tuple(self.candidates)
        if any(not isinstance(c, Candidate) for c in candidates):
            raise TypeError("CandidateSpace requires Candidate records")
        if len({c.key for c in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate keys")
        object.__setattr__(self, "candidates", candidates)
        for name in ("channel_axes", "protected_channel_axes"):
            axes = tuple(getattr(self, name))
            if any(not isinstance(a, AxisRef) for a in axes):
                raise TypeError("Channel axes must be AxisRef instances")
            object.__setattr__(self, name, tuple(dict.fromkeys(axes)))
        exclusions = tuple(tuple(item) for item in self.exclusions)
        if any(len(item) != 2 or any(not isinstance(s, str) for s in item) for item in exclusions):
            raise ValueError("Exclusions require key/reason pairs")
        object.__setattr__(self, "exclusions", exclusions)


def discover_candidates(graph, defaults, targets):
    """Filter declared domains before freezing widths and creating seeds."""
    graph.validate()
    operations = graph.operations()
    if targets is not None:
        if isinstance(targets, str):
            raise TypeError("Candidate targets require an iterable of module path patterns")
        targets = tuple(targets)
        if any(not isinstance(p, str) for p in targets):
            raise TypeError("Candidate targets must be module path patterns")
        aliases = defaultdict(list)
        for path, module in graph.model.named_modules(remove_duplicate=False):
            aliases[id(module)].append(path)
        matching, selected = set(), []
        for operation in operations:
            hits = {
                p
                for p in targets
                if any(fnmatchcase(path, p) for path in aliases[id(operation.module)])
            }
            if hits and graph.operator_spec(operation).candidates:
                selected.append(operation)
                matching.update(hits)
        if set(targets) - matching:
            raise PlanningError(
                f"Candidate targets declare no channel axes: {sorted(set(targets) - matching)}"
            )
        operations = tuple(selected)
    domains = discover(graph, operations)
    protected = _protected_equal_axes(graph, defaults) if defaults else set()
    if defaults:
        for domain in domains:
            if domain.axis not in protected and all(
                any(
                    d.code == "fixed_axis"
                    for d in graph.propagate(remove=c.remove, constraints=defaults).diagnostics
                )
                for c in _candidates(domain)
            ):
                protected.add(domain.axis)
    active = tuple(d for d in domains if d.axis not in protected)
    return CandidateSpace(
        tuple(c for d in active for c in _candidates(d)),
        tuple(d.axis for d in active),
        tuple(d.axis for d in domains if d.axis in protected),
        tuple(
            (f"domain:{d.key}", "All positions are protected by external interfaces")
            for d in domains
            if d.axis in protected
        ),
    )


def parameter_groups(graph, candidates, constraints, parameter_filter):
    """Extract complete groups without requiring individual feasibility."""
    result = []
    for candidate in candidates:
        impact = graph.propagate(remove=candidate.remove, constraints=constraints)
        if not impact.complete:
            raise PlanningError(
                "Incomplete parameter group: "
                + "; ".join(d.message for d in impact.diagnostics if not d.complete)
            )
        selections = []
        for selection in impact.parameters:
            if parameter_filter is None or parameter_filter(
                selection.tensor, graph.tensor(selection.tensor)
            ):
                selections.append(selection)
            graph.validate()
        result.append(ParameterGroup(graph, tuple(selections), candidate.key))
    return unique_groups(result)
