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


def group_equivalence_classes(groups):
    """Return original indices partitioned by geometric group equality.

    Element counts and axis bounds are decomposition-independent bucket keys,
    not equality proofs. Bounds avoid materializing a potentially fragmented
    projection that could exceed the coordinate limit of otherwise valid regions.
    Exact comparison still resolves bucket collisions.
    """
    groups = tuple(groups)
    buckets, classes = {}, []
    for index, group in enumerate(groups):
        if not isinstance(group, ParameterGroup):
            raise TypeError("Expected ParameterGroup")
        signature = (
            id(group.graph),
            tuple(
                (
                    selection.tensor,
                    selection.count,
                    tuple(
                        (
                            min(r.axes[d].intervals[0][0] for r in selection.regions),
                            max(r.axes[d].intervals[-1][1] for r in selection.regions),
                        )
                        for d in range(len(selection.tensor.shape))
                    ),
                )
                for selection in group.selections
            ),
        )
        bucket = buckets.setdefault(signature, [])
        for number in bucket:
            if group.equivalent(groups[classes[number][0]]):
                classes[number].append(index)
                break
        else:
            bucket.append(len(classes))
            classes.append([index])
    return tuple(tuple(indices) for indices in classes)


def unique_groups(groups):
    """Canonicalize equivalent groups without merging partially overlapping groups."""
    groups = tuple(groups)
    return tuple(groups[indices[0]] for indices in group_equivalence_classes(groups))
