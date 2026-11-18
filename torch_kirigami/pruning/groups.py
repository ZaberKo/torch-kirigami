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

    Raises:
        ValueError: A selection refers to a buffer or another graph, or no
            selection is nonempty.
        torch_kirigami.StaleGraphError: The model changed after capture.
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

        Keys and the decomposition into regions are ignored. Only selected
        coordinates are compared, not current values or the full structural
        effects of the candidate removals. Python `==` still compares object
        identity because this dataclass uses `eq=False`.

        Args:
            other: Object to compare with.

        Returns:
            `True` when `other` is a `ParameterGroup` of the same graph object
            with equal selections.
        """
        return (
            isinstance(other, ParameterGroup)
            and self.graph is other.graph
            and self.selections == other.selections
        )

    def bindings(self) -> tuple[tuple[nn.Parameter, Selection], ...]:
        """Return live `(nn.Parameter, Selection)` pairs in selection order.

        The source graph is checked for freshness before its bindings are
        resolved. The original parameter objects are returned; their values are
        not gathered or copied.

        Returns:
            One pair per selection, in the order of `selections`.

        Raises:
            torch_kirigami.StaleGraphError: The model changed after capture.
        """
        bindings = dict(self.graph.tensor_bindings())
        return tuple((cast(nn.Parameter, bindings[s.tensor]), s) for s in self.selections)


def checked_groups(groups: Iterable[ParameterGroup]) -> tuple[ParameterGroup, ...]:
    """Return `groups` as a tuple after checking the type of every item.

    Raises:
        TypeError: An item is not a `ParameterGroup`.
    """
    groups = tuple(groups)
    if any(not isinstance(group, ParameterGroup) for group in groups):
        raise TypeError("Expected ParameterGroup items")
    return groups


def group_equivalence_classes(groups: Iterable[ParameterGroup]) -> tuple[tuple[int, ...], ...]:
    """Return original indices partitioned by geometric group equality.

    Element counts and axis bounds are decomposition-independent bucket keys,
    not equality proofs. Bounds avoid materializing a potentially fragmented
    projection that could exceed the coordinate limit of otherwise valid regions.
    Exact comparison still resolves bucket collisions.

    Raises:
        TypeError: An item is not a `ParameterGroup`.
    """
    groups = checked_groups(groups)
    buckets: dict[tuple[object, ...], list[int]] = {}
    classes: list[list[int]] = []
    for index, group in enumerate(groups):
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

    Use it before building a sparse regularizer when equivalent groups should
    count once, and assign regularizer coefficients after deduplication so they
    stay aligned with the groups. This function does not combine coefficients
    or merge partially overlapping groups. Equal parameter regions do not prove
    equal pruning effects.

    Args:
        groups: Groups compared by graph identity and parameter regions, as in
            `ParameterGroup.equivalent()`. Keys are ignored.

    Returns:
        The original group objects in first-occurrence order, with their
        original keys. Partially overlapping groups remain separate. An empty
        iterable returns an empty tuple.

    Raises:
        TypeError: An item is not a `ParameterGroup`.
    """
    groups = tuple(groups)
    return tuple(groups[indices[0]] for indices in group_equivalence_classes(groups))
