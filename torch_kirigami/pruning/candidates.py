"""One candidate universe for planning, budget accounting and sparse training."""

from ..contracts import Fixed
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
    """Generate declared logical-axis candidates, deduplicating shared domains."""
    result, domains, keys = [], {}, {}
    for op in operations:
        for domain in graph.operator_spec(op).candidates:
            axis, block = domain.axis, domain.block_size
            if domain.key in keys and keys[domain.key] != (axis, block):
                raise PlanningError(f"Conflicting candidate domain key: {domain.key}")
            keys[domain.key] = (axis, block)
            if axis in domains and domains[axis].block_size != block:
                raise PlanningError("Conflicting block sizes for one candidate axis")
            domains.setdefault(axis, domain)
    for axis, domain in domains.items():
        block, width = domain.block_size, axis.tensor.shape[axis.dim]
        if width % block:
            raise PlanningError("Candidate block must divide the logical width")
        for start in range(0, width, block):
            result.append(
                Candidate(
                    f"{domain.key}:{start:012d}",
                    (axis.select(range(start, start + block)),),
                    axis,
                )
            )
    return tuple(result), tuple(domains)


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
        if candidates is None:
            candidates, discovered = discover(graph, graph.operations())
            if axes is None:
                protected = set()
                if defaults:
                    for axis in discovered:
                        domain = [c for c in candidates if c.axis == axis]
                        if domain and all(
                            any(
                                d.code == "fixed_axis"
                                for d in graph.propagate(
                                    remove=c.remove, constraints=defaults
                                ).diagnostics
                            )
                            for c in domain
                        ):
                            protected.add(axis)
                    exclusions = [
                        (
                            f"domain:{a.tensor.paths[0] if a.tensor.paths else a.tensor.id.split(':', 1)[-1]}:{a.dim}",
                            "All positions are protected by external interfaces",
                        )
                        for a in discovered
                        if a in protected
                    ]
                axes = tuple(a for a in discovered if a not in protected)
                self.protected_axes = tuple(a for a in discovered if a in protected)
                candidates = tuple(c for c in candidates if c.axis not in protected)
        elif axes is None:
            raise ValueError("Custom candidates require explicit budget axes")
        self.candidates = tuple(candidates)
        self.axes = tuple(dict.fromkeys(axes))
        self.exclusions = tuple(exclusions)
        if len({c.key for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate candidate keys")
        for axis in self.axes:
            graph.metadata(axis.tensor)
        for candidate in self.candidates:
            for selection in candidate.remove:
                graph.metadata(selection.tensor)

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
