"""Live parameter regions shared by structural selection and sparse training."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from torch import nn

from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import Selection, TensorRef


@dataclass(frozen=True, eq=False)
class ParameterGroup:
    """A union of parameter coordinates bound to one dependency graph.

    Construction merges overlapping selections for each parameter. It stores
    coordinates, not parameter copies or numerical values. `bindings()` reads
    the graph's live parameters, so ordinary weight updates need no new group.
    Rebuild the graph and groups after parameter replacement or structural changes.

    Direct construction does not propagate dependencies or prove that selections
    describe a complete pruning effect. Use `Pruner.parameter_groups` to extract
    a separate group for each candidate. This record contains no constraints
    or execution recipes and does not authorize physical pruning.

    Attributes:
        graph: Source graph that owns the parameter references.
        selections: Nonempty parameter selections in original coordinates, merged
            by parameter and sorted by parameter paths. Buffers are not accepted.
        key: Descriptive label. Extraction uses the candidate key. Direct
            construction defaults to an empty string. `equivalent()` ignores it.
    """

    graph: DependencyGraph
    selections: tuple[Selection, ...]
    key: str = ""

    def __post_init__(self) -> None:
        self.graph.validate()
        merged: dict[TensorRef, Selection] = {}
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

    def equivalent(self, other: object) -> bool:
        """Return whether both groups select the same regions in the same graph.

        Ignore keys and region decomposition. This compares selected coordinates,
        not current values or the full structural effects of candidate removals.
        Python `==` remains object identity because this dataclass uses `eq=False`.
        """
        return (
            isinstance(other, ParameterGroup)
            and self.graph is other.graph
            and self.selections == other.selections
        )

    def bindings(self) -> tuple[tuple[nn.Parameter, Selection], ...]:
        """Return live `(nn.Parameter, Selection)` pairs in selection order.

        Validate the source graph before resolving its bindings. Return the
        original parameter objects without gathering or copying their values.
        """
        bindings = dict(self.graph.tensor_bindings())
        return tuple((cast(nn.Parameter, bindings[s.tensor]), s) for s in self.selections)


def group_equivalence_classes(groups: Iterable[ParameterGroup]) -> tuple[tuple[int, ...], ...]:
    """Return original indices partitioned by geometric group equality.

    Element counts and axis bounds are decomposition-independent bucket keys,
    not equality proofs. Bounds avoid materializing a potentially fragmented
    projection that could exceed the coordinate limit of otherwise valid regions.
    Exact comparison still resolves bucket collisions.
    """
    groups = tuple(groups)
    buckets: dict[tuple[object, ...], list[int]] = {}
    classes: list[list[int]] = []
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


def unique_groups(groups: Iterable[ParameterGroup]) -> tuple[ParameterGroup, ...]:
    """Keep the first occurrence of each equivalent parameter group.

    This public helper is exported from `torch_kirigami.pruning`. Use it before
    constructing a sparse regularizer when equivalent groups should count once.
    Assign regularizer coefficients after deduplication to preserve alignment.
    This function does not combine coefficients or merge partially overlapping
    groups. Equal parameter regions do not prove equal pruning effects.

    Args:
        groups: Groups to compare by graph identity and parameter regions.
            Keys do not participate in equality.

    Returns:
        Original group objects in first-occurrence order. Partial overlaps remain
        separate groups. The retained objects keep their original keys.
        An empty iterable returns an empty tuple.
    """
    groups = tuple(groups)
    return tuple(groups[indices[0]] for indices in group_equivalence_classes(groups))
