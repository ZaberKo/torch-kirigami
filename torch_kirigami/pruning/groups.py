"""Live parameter regions shared by structural selection and sparse training."""

from dataclasses import dataclass

from ..graph import DependencyGraph
from ..selection import Selection


@dataclass(frozen=True, eq=False)
class ParameterGroup:
    """A deduplicated parameter-region union bound to one dependency snapshot.

    Args:
        graph: Source graph; rebuild groups after structural changes.
        selections: Parameter selections, including partial or partitioned regions.
        key: Descriptive label, not a claim of disjointness or zero invariance.
    """

    graph: DependencyGraph
    selections: tuple[Selection, ...]
    key: str = ""

    def __post_init__(self):
        self.graph.validate()
        merged = {}
        for selection in self.selections:
            self.graph.metadata(selection.tensor)
            if selection.tensor.kind != "parameter":
                raise ValueError("ParameterGroup accepts only parameters")
            if selection:
                old = merged.get(selection.tensor, Selection(selection.tensor))
                merged[selection.tensor] = old.union(selection)
        if not merged:
            raise ValueError("ParameterGroup requires nonempty parameter regions")
        object.__setattr__(
            self, "selections", tuple(sorted(merged.values(), key=lambda s: s.tensor.paths))
        )

    def equivalent(self, other):
        """Compare coordinates and graph identity, independently of group labels."""
        return (
            isinstance(other, ParameterGroup)
            and self.graph is other.graph
            and self.selections == other.selections
        )

    def bindings(self):
        """Return current (Parameter, Selection) pairs after freshness validation."""
        bindings = dict(self.graph.tensor_bindings())
        return tuple((bindings[s.tensor], s) for s in self.selections)


def unique_groups(groups):
    """Canonicalize equivalent groups without merging partially overlapping groups."""
    result = []
    for group in groups:
        if not isinstance(group, ParameterGroup):
            raise TypeError("Expected ParameterGroup")
        if not any(group.equivalent(old) for old in result):
            result.append(group)
    return tuple(result)
