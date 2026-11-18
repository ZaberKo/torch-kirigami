"""Cumulative channel budgets measured against original widths across pruning rounds."""

from __future__ import annotations

from ..dep_graph.dependency_graph import DependencyGraph
from ..pruning.budget import channel_targets
from ..pruning.candidates import CandidateSpace
from ..pruning.plan import PruningResult
from ..pruning.serialization import decode, encode
from ..pruning.state import check_structure, snapshot, validate_plan
from ..pruning.types import ChannelCount, ChannelRatio, ModelStructure


def _domains(graph: DependencyGraph, space: CandidateSpace, axes: tuple | None = None) -> tuple:
    """Describe each channel axis by parameter paths, dimension, and declared candidates."""
    graph.validate()
    axes = space.channel_axes if axes is None else axes
    for axis in (*axes, *space.excluded_channel_axes):
        graph.metadata(axis.tensor)
    declarations = {}
    for op in graph.operations():
        spec = graph.operator_spec(op)
        if spec is None:
            continue
        for domain in spec.candidates:
            declarations.setdefault(domain.axis, set()).add((domain.key, domain.block_size))
    if any(not a.tensor.paths or a.tensor.kind != "parameter" for a in axes):
        raise ValueError("Cumulative budgets require named parameter axes")
    return tuple((a.tensor.paths, a.dim, tuple(sorted(declarations.get(a, ())))) for a in axes)


class CumulativeChannelBudget:
    """Convert a cumulative removal ratio into exact integer width caps for each round.

    The budget records the original width of every channel axis in the initial
    candidate space. `budget()` converts a ratio of those original widths into
    absolute final-width limits for the current model, so repeated pruning
    rounds approach the cumulative target instead of compounding ratios on
    already compact widths. `budget()` does not change any state. After you
    apply a plan, call `update()` with the result and a freshly rebuilt graph
    and candidate space before you request the next budget. Only shape changes
    observed in a successful result count as progress. Use `state_dict()` and
    `load_state_dict()` to restore the budget for an equivalent restored model.

    Args:
        graph: Dependency graph for the current model structure.
        space: Initial candidate space. Its channel axes define the tracked
            domains and their original widths. Each axis must belong to a
            named parameter.
        scope: `"local"` gives every axis its own cap. `"global"` caps the
            total width of all axes and sets no per-axis caps.

    Attributes:
        scope: Accounting scope, `"local"` or `"global"`.

    Raises:
        ValueError: `scope` is invalid, or a channel axis does not belong to a
            named parameter.
    """

    def __init__(
        self, graph: DependencyGraph, space: CandidateSpace, *, scope: str = "local"
    ) -> None:
        if scope not in ("local", "global"):
            raise ValueError("scope must be local or global")
        self.scope = scope
        self._domains = _domains(graph, space)
        self._initial = tuple(a.tensor.shape[a.dim] for a in space.channel_axes)
        self._current = self._initial
        self._structure = snapshot(graph.model, guarded=graph.constant_guards())

    def _validate_space(
        self,
        graph: DependencyGraph,
        space: CandidateSpace,
        structure: ModelStructure,
        widths: tuple[int, ...],
        *,
        domains: tuple | None = None,
    ) -> tuple:
        """Resolve recorded domains in a rebuilt graph and check widths and structure."""
        domains = self._domains if domains is None else domains
        try:
            axes = tuple(graph.parameter(paths[0]).axis(dim) for paths, dim, _ in domains)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Cumulative budget domains changed") from error
        # A last remaining channel/block can become irreversibly blocked. Keep
        # its original denominator, but never accept an arbitrary missing axis.
        if (
            len(set(axes)) != len(axes)
            or not set(space.channel_axes).issubset(axes)
            or not set(axes).issubset((*space.channel_axes, *space.excluded_channel_axes))
            or _domains(graph, space, axes) != domains
        ):
            raise ValueError("Cumulative budget domains changed")
        if tuple(a.tensor.shape[a.dim] for a in axes) != tuple(widths):
            raise ValueError("Unrecorded channel width change")
        check_structure(graph.model, structure)
        return axes

    def budget(
        self, graph: DependencyGraph, space: CandidateSpace, ratio: int | float
    ) -> ChannelCount:
        """Return this round's width caps for a cumulative ratio of the original widths.

        Args:
            graph: Fresh dependency graph for the current model structure.
            space: Candidate space rebuilt for that graph.
            ratio: Cumulative fraction of the original widths to remove, in
                `[0, 1)`. Each limit is `floor(original_width * (1 - ratio))`.

        Returns:
            A `ChannelCount` with absolute final-width limits over the current
            axes: one limit per axis for local scope, or one total for global
            scope. An axis that is already narrower than its limit needs no
            further removal.

        Raises:
            ValueError: `ratio` is outside `[0, 1)`, or the graph, candidate domains,
                structure, or observed widths differ from the recorded state.
        """
        targets = channel_targets(ChannelRatio(ratio, scope=self.scope), self._initial)
        axes = self._validate_space(graph, space, self._structure, self._current)
        # Targets refer to original widths, not the already compact model.
        # A target above a current width is already satisfied; do not request
        # an additional deletion merely because an earlier round overshot it.
        return ChannelCount(targets[0] if self.scope == "global" else targets, axes, self.scope)

    def update(self, result: PruningResult, graph: DependencyGraph, space: CandidateSpace) -> None:
        """Record the widths produced by an applied plan.

        The state changes only after every check passes.

        Args:
            result: Result returned by applying the preceding plan.
            graph: Fresh graph for the compacted model.
            space: Candidate space rebuilt from that graph.

        Raises:
            ValueError: The result does not start from the recorded structure,
                grows an axis, or does not match the rebuilt graph and space.
        """
        validate_plan(result.plan)
        if result.plan.before != self._structure or result.structure != result.plan.after:
            raise ValueError("Pruning result does not follow the recorded structure")
        states = {path: state for state in result.structure.tensors for path in state.paths}
        expected = tuple(states[paths[0]].shape[dim] for paths, dim, _ in self._domains)
        if any(new > old for new, old in zip(expected, self._current, strict=True)):
            raise ValueError("Cumulative pruning cannot grow an axis")
        self._validate_space(graph, space, result.structure, expected)
        self._current, self._structure = expected, result.structure

    def state_dict(self) -> dict[str, object]:
        """Return the budget state as portable plain data.

        Returns:
            A dictionary with the scope, encoded domains, original and current
            widths, and the encoded model structure that the next round requires.
        """
        return {
            "scope": self.scope,
            "domains": encode(self._domains),
            "initial": list(self._initial),
            "current": list(self._current),
            "structure": encode(self._structure),
        }

    def load_state_dict(
        self, state: dict[str, object], graph: DependencyGraph, space: CandidateSpace
    ) -> None:
        """Restore saved state for a restored model.

        All checks run before any state changes. The saved domains must include
        every domain recorded by this budget.

        Args:
            state: Mapping returned by `state_dict()`.
            graph: Fresh graph for the restored model.
            space: Candidate space rebuilt from that graph.

        Raises:
            ValueError: The state has different keys or scope, invalid widths,
                or domains, structure, or widths that do not match the graph
                and space.
        """
        if (
            set(state) != {"scope", "domains", "initial", "current", "structure"}
            or state["scope"] != self.scope
        ):
            raise ValueError("Incompatible cumulative budget state")
        domains = decode(state["domains"])
        initial, current = tuple(state["initial"]), tuple(state["current"])
        structure = decode(state["structure"])
        if not isinstance(structure, ModelStructure) or not set(self._domains).issubset(domains):
            raise ValueError("Incompatible cumulative budget domains or structure")
        if (
            len(initial) != len(domains)
            or len(current) != len(domains)
            or any(
                type(a) is not int or type(b) is not int or not 0 < b <= a
                for a, b in zip(initial, current, strict=True)
            )
        ):
            raise ValueError("Invalid cumulative widths")
        self._validate_space(graph, space, structure, current, domains=domains)
        self._domains = domains
        self._initial, self._current, self._structure = initial, current, structure
