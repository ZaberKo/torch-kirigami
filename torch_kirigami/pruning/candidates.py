"""Candidate spaces shared by planning, channel-budget accounting, and sparse training."""

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import cast

from torch import nn

from ..dep_graph.contracts import Constraint, Fixed
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.operation import CandidateAxis, Operation
from ..dep_graph.selection import AxisRef, Region, Selection, TensorRef
from ..settings import config
from .candidate_checks import CandidateFilter, propagate_request
from .groups import ParameterGroup
from .types import Candidate, PlanningError

_RequestKey = tuple[tuple[TensorRef, tuple[Region, ...]], ...]


def interface_constraints(graph: DependencyGraph, preserve_io: bool) -> tuple[Fixed, ...]:
    """Return `Fixed` constraints for every external input/output axis when requested."""
    return (
        tuple(Fixed(ref.axis(d)) for ref in graph.interfaces() for d in range(len(ref.shape)))
        if preserve_io
        else ()
    )


def discover(
    graph: DependencyGraph, operations: tuple[Operation, ...]
) -> tuple[CandidateAxis, ...]:
    """Collect logical domains before instantiating individual candidate requests."""
    domains: dict[AxisRef, CandidateAxis] = {}
    keys: dict[str, tuple[TensorRef | AxisRef, int, int]] = {}
    for op in operations:
        spec = graph.operator_spec(op)
        if spec is None:
            continue
        for domain in spec.candidates:
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


def _candidates(domain: CandidateAxis) -> Iterator[Candidate]:
    """Yield stable candidate keys for the domain's original channel blocks."""
    axis, block = domain.axis, domain.block_size
    for start in range(0, axis.tensor.shape[axis.dim], block):
        yield Candidate(
            f"{domain.key}:{start:012d}", (axis.select(range(start, start + block)),), axis
        )


@dataclass(frozen=True)
class CandidateSpace:
    """Candidates for one planning round and the logical axes used by channel budgets.

    A candidate space is not the set of all dependency groups in the model: it
    holds only the removal requests a strategy may choose from, plus the
    logical channel axes whose widths channel budgets count. Construction
    performs no discovery or analysis and retains no model; the consuming
    `Pruner` checks that every reference belongs to its graph. Candidate
    filtering never removes `channel_axes`, so the channel-budget baseline stays
    fixed.

    Attributes:
        candidates: Named removal requests in original coordinates, with unique
            keys.
        channel_axes: Unique logical axes counted by channel budgets.
        excluded_channel_axes: Discovered axes whose candidates all have a proven
            irreversible conflict. They are outside the channel-budget baseline.
        exclusions: Rejected candidate keys, or `domain:` keys for excluded axes,
            with their reasons.

    Raises:
        TypeError: A candidate or axis has the wrong type.
        ValueError: Candidate keys repeat or an exclusion is not a pair of
            strings.
    """

    candidates: tuple[Candidate, ...]
    channel_axes: tuple[AxisRef, ...]
    excluded_channel_axes: tuple[AxisRef, ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if any(not isinstance(c, Candidate) for c in candidates):
            raise TypeError("CandidateSpace requires Candidate records")
        if len({c.key for c in candidates}) != len(candidates):
            raise ValueError("Duplicate candidate keys")
        object.__setattr__(self, "candidates", candidates)
        for name in ("channel_axes", "excluded_channel_axes"):
            axes = tuple(getattr(self, name))
            if any(not isinstance(a, AxisRef) for a in axes):
                raise TypeError("Channel axes must be AxisRef instances")
            object.__setattr__(self, name, tuple(dict.fromkeys(axes)))
        exclusions = tuple(tuple(item) for item in self.exclusions)
        if any(len(item) != 2 or any(not isinstance(s, str) for s in item) for item in exclusions):
            raise ValueError("Exclusions require key/reason pairs")
        object.__setattr__(self, "exclusions", exclusions)


def discover_candidates(
    graph: DependencyGraph, constraints: tuple[Constraint, ...], targets: Iterable[str] | None
) -> CandidateSpace:
    """Build a candidate space from declared domains, excluding irreversible conflicts.

    `targets` patterns use `fnmatchcase`, so `*` also matches across `.`.
    Fully excluded domains are removed before the channel baseline is frozen.
    """
    graph.validate()
    checks = CandidateFilter(graph, constraints)
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
            spec = graph.operator_spec(operation)
            if hits and spec is not None and spec.candidates:
                selected.append(operation)
                matching.update(hits)
        if set(targets) - matching:
            raise PlanningError(
                f"Candidate targets declare no channel axes: {sorted(set(targets) - matching)}"
            )
        operations = tuple(selected)
    domains = discover(graph, operations)
    candidates, axes, excluded_axes, exclusions = [], [], [], []
    for domain in domains:
        fixed = checks.fixed_rejection(domain.axis)
        if fixed is not None:
            excluded_axes.append(domain.axis)
            exclusions.append((f"domain:{domain.key}", str(fixed)))
            continue
        kept, rejected = [], []
        for candidate in _candidates(domain):
            reasons = checks.rejections(candidate)
            if reasons:
                rejected.append((candidate.key, "; ".join(map(str, reasons))))
            else:
                kept.append(candidate)
        if kept:
            axes.append(domain.axis)
            candidates.extend(kept)
            exclusions.extend(rejected)
        else:
            excluded_axes.append(domain.axis)
            reasons = "; ".join(dict.fromkeys(reason for _, reason in rejected))
            exclusions.append((f"domain:{domain.key}", reasons or "No declared candidates"))
    graph.validate()
    return CandidateSpace(tuple(candidates), tuple(axes), tuple(excluded_axes), tuple(exclusions))


def filter_candidates(
    graph: DependencyGraph, constraints: tuple[Constraint, ...], space: CandidateSpace
) -> CandidateSpace:
    """Exclude candidates with irreversible conflicts, keeping every channel axis."""
    graph.validate()
    if not isinstance(space, CandidateSpace):
        raise TypeError("Expected a CandidateSpace")
    for axis in (*space.channel_axes, *space.excluded_channel_axes):
        graph.metadata(axis.tensor)
    checks = CandidateFilter(graph, constraints, candidate_count=len(space.candidates))
    kept, exclusions = [], dict(space.exclusions)
    for candidate in space.candidates:
        reasons = checks.rejections(candidate)
        if reasons:
            exclusions[candidate.key] = "; ".join(map(str, reasons))
        else:
            kept.append(candidate)
            exclusions.pop(candidate.key, None)
    graph.validate()
    return CandidateSpace(
        tuple(kept), space.channel_axes, space.excluded_channel_axes, tuple(exclusions.items())
    )


def parameter_groups(
    graph: DependencyGraph,
    candidates: Iterable[Candidate],
    constraints: tuple[Constraint, ...],
    parameter_filter: Callable[[TensorRef, nn.Parameter], bool] | None,
) -> tuple[ParameterGroup, ...]:
    """Extract one group per candidate, preserving order and repeated groups.

    Complete influence is required before parameters are filtered. Constraint
    failures do not prevent extraction, and no execution recipes are compiled.
    An empty filtered group raises instead of breaking candidate-to-group
    alignment. Each candidate is checked as its own ordered request, so the
    first failing candidate in input order raises, whatever its neighbors.
    Equal native requests are analyzed once. An equivalent request shares a
    complete analysis for single slices with the same canonical starting axis.
    At most `impact_cache_size` such parameter selection tuples are kept.
    """
    candidates = tuple(candidates)
    for candidate in candidates:
        for selection in candidate.remove:
            graph.metadata(selection.tensor)
        if candidate.axis is not None:
            graph.metadata(candidate.axis.tensor)
    native = parameter_filter is None and all(
        type(candidate) is Candidate and has_native_semantics((candidate.axis, candidate.remove))
        for candidate in candidates
    )
    with graph._native_queries(constraints, enabled=native):
        checks = CandidateFilter(graph, constraints, candidate_count=len(candidates))
        reuse = native and checks.reuse_equivalent
        # Parameter selections by exact request; these are part of the result.
        analyzed: dict[_RequestKey, tuple[Selection, ...]] = {}
        shared: OrderedDict[_RequestKey, tuple[Selection, ...]] = OrderedDict()
        groups = []
        for candidate in candidates:
            exact = tuple((s.tensor, s.regions) for s in candidate.remove) if native else None
            parameters = analyzed.get(exact) if exact is not None else None
            if parameters is None:
                parameters = checks.parameter_selections(candidate)
            if parameters is None:
                key = (
                    checks.request_key(candidate.remove)
                    if reuse and len(candidate.remove) == 1 and config.impact_cache_size
                    else None
                )
                entry = shared.get(key) if key is not None else None
                if entry is not None:
                    shared.move_to_end(key)
                    parameters = entry
                else:
                    impact = propagate_request(graph, candidate.remove, constraints)
                    if not impact.complete:
                        raise PlanningError(
                            f"Incomplete parameter group for candidate {candidate.key!r}: "
                            + "; ".join(d.message for d in impact.diagnostics if not d.complete)
                        )
                    parameters = impact.parameters
                    if key is not None:
                        shared[key] = parameters
                        shared.move_to_end(key)
                        if len(shared) > config.impact_cache_size:
                            shared.popitem(last=False)
            if exact is not None:
                analyzed[exact] = parameters
            selections = []
            for selection in parameters:
                if parameter_filter is None:
                    selections.append(selection)
                else:
                    keep = parameter_filter(
                        selection.tensor, cast(nn.Parameter, graph.tensor(selection.tensor))
                    )
                    # Even a callback that rejects this region can alter bindings.
                    graph.validate()
                    if keep:
                        selections.append(selection)
            if not selections:
                raise PlanningError(
                    f"Candidate {candidate.key!r} has no parameter regions after extraction and filtering"
                )
            groups.append(ParameterGroup(graph, tuple(selections), candidate.key))
        return tuple(groups)
