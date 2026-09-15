"""Cumulative channel accounting independent of round selection and training."""

from __future__ import annotations

import math

from ..graph import DependencyGraph
from ..pruning.candidates import CandidateSpace
from ..pruning.plan import PruningResult
from ..pruning.serialization import decode, encode
from ..pruning.state import check_structure, snapshot, validate_plan
from ..pruning.types import ChannelCount, ModelStructure


def _domains(graph: DependencyGraph, space: CandidateSpace, axes: tuple | None = None) -> tuple:
    """Return stable parameter-domain descriptors for cumulative accounting."""
    graph.validate()
    axes = space.channel_axes if axes is None else axes
    for axis in (*axes, *space.protected_channel_axes):
        graph.metadata(axis.tensor)
    declarations = {}
    for op in graph.operations():
        for domain in graph.operator_spec(op).candidates:
            declarations.setdefault(domain.axis, set()).add((domain.key, domain.block_size))
    if any(not a.tensor.paths or a.tensor.kind != "parameter" for a in axes):
        raise ValueError("Cumulative budgets require named parameter axes")
    return tuple((a.tensor.paths, a.dim, tuple(sorted(declarations.get(a, ())))) for a in axes)


class CumulativeChannelBudget:
    """Convert cumulative ratios to exact remaining integer caps.

    Args:
        graph: Dependency snapshot for the current model structure.
        space: Initial CandidateSpace with the desired frozen logical domains.
        scope: Local per-axis accounting or global accounting without local caps.

    budget() is read-only. After an applied plan, call update(result, new_graph, new_space)
    before requesting the next budget. Only observed successful shape changes
    count as progress. State can be rebound to an equivalent restored model.
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
        """Validate that a rebuilt graph preserves recorded domains and widths."""
        domains = self._domains if domains is None else domains
        try:
            axes = tuple(graph.parameter(paths[0]).axis(dim) for paths, dim, _ in domains)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("Cumulative budget domains changed") from error
        # A last remaining channel/block can become wholly IO-protected. Keep
        # its original denominator, but never accept an arbitrary missing axis.
        if (
            len(set(axes)) != len(axes)
            or not set(space.channel_axes).issubset(axes)
            or not set(axes).issubset((*space.channel_axes, *space.protected_channel_axes))
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
        """Return this round's cap for a cumulative ratio of original widths."""
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio < 1:
            raise ValueError("ratio must be finite and in [0, 1)")
        axes = self._validate_space(graph, space, self._structure, self._current)
        if self.scope == "global":
            count = max(
                0,
                math.floor(ratio * sum(self._initial)) - (sum(self._initial) - sum(self._current)),
            )
            return ChannelCount(count, axes, "global")
        counts = tuple(
            max(0, math.floor(ratio * initial) - (initial - current))
            for initial, current in zip(self._initial, self._current, strict=True)
        )
        return ChannelCount(counts, axes)

    def update(self, result: PruningResult, graph: DependencyGraph, space: CandidateSpace) -> None:
        """Accept an applied result and rebuilt space; reject inconsistent transitions."""
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
        """Return portable baseline, observed widths and structural preconditions."""
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
        """Restore against a fresh space, validating before changing any state."""
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
