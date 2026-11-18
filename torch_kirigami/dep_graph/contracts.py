"""Read-only analysis results and constraints shared by built-ins and extensions."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from ..settings import config
from .relations import AxisPort
from .selection import (
    AxisRef,
    IndexSet,
    Region,
    Selection,
    TensorRef,
    canonical_boxes,
    full_region,
    region_boxes,
)


@dataclass(frozen=True)
class Diagnostic:
    """A conflict or an unproved part of an analysis.

    Attributes:
        code: Stable diagnostic category, such as `unsupported` or `empty_axis`.
        message: Human-readable explanation.
        severity: `unresolved` when a condition cannot be proved yet, or
            `conflict` when a declared condition is violated.
        node: FX node name, when the issue belongs to a particular operation.
        tensors: IDs of the tensor entities involved in the issue.
        complete: Whether the influence range is still fully known despite this
            diagnostic. This is independent of `severity`.
        irreversible: Whether this conflict is proved to persist under every
            superset of the current removals, with graph assumptions and
            constraints fixed. Only a conflict can be irreversible. False makes
            no claim about whether additional removals can repair the conflict.

    Raises:
        ValueError: `severity` is invalid, or `irreversible` is set on a
            diagnostic that is not a conflict.
        TypeError: `complete` or `irreversible` is not a boolean.
    """

    code: str
    message: str
    severity: Literal["unresolved", "conflict"] = "unresolved"
    node: str | None = None
    tensors: tuple[str, ...] = ()
    complete: bool = True
    irreversible: bool = False

    def __str__(self) -> str:
        """Render the category, the operation location when known, and the message."""
        location = f" at {self.node}" if self.node else ""
        return f"{self.code}{location}: {self.message}"

    def __post_init__(self) -> None:
        if self.severity not in ("unresolved", "conflict"):
            raise ValueError("Invalid diagnostic severity")
        if not isinstance(self.complete, bool):
            raise TypeError("Diagnostic completeness must be boolean")
        if type(self.irreversible) is not bool:
            raise TypeError("Diagnostic irreversibility must be boolean")
        if self.irreversible and self.severity != "conflict":
            raise ValueError("An irreversible diagnostic must describe a conflict")
        object.__setattr__(self, "tensors", tuple(self.tensors))


class Constraint(Protocol):
    """A deterministic structural condition on selected coordinates.

    For fixed configuration and selected coordinates, each check must return
    the same diagnostic. Query caching can change the number of calls.
    """

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        ...

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Check the accumulated selections.

        Args:
            selections: Selections keyed by graph-local tensor ID.

        Returns:
            A diagnostic if the condition is violated or cannot be proved, otherwise
            None. Checking must not mutate selections or choose additional removals.
        """
        ...


def chosen(selections: Mapping[str, Selection], tensor: TensorRef) -> Selection:
    """Return a tensor's accumulated selection, or an empty selection.

    Raises:
        ValueError: The selection stored under the tensor's ID belongs to a
            different reference.
    """
    selection = selections.get(tensor.id)
    if selection is None:
        return Selection(tensor)
    if selection.tensor != tensor:
        raise ValueError("Tensor reference does not match the selection")
    return selection


@dataclass(frozen=True)
class NonEmpty:
    """Require at least one position to remain on the given axis.

    Attributes:
        axis: Tensor axis that must retain at least one position.
    """

    axis: AxisRef

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Report a conflict if every position on the axis would be removed.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An irreversible `empty_axis` conflict, or None.

        Raises:
            ValueError: The stored selection belongs to a different reference.
        """
        selection = chosen(selections, self.axis.tensor)
        indices = selection.fully_selected_indices(self.axis.dim)
        if len(indices) == self.axis.tensor.shape[self.axis.dim]:
            return Diagnostic(
                "empty_axis",
                "The operator requires a nonempty axis",
                "conflict",
                tensors=(self.axis.tensor.id,),
                irreversible=True,
            )
        return None


@dataclass(frozen=True)
class Fixed:
    """Require every position on an axis to be preserved.

    Attributes:
        axis: Tensor axis whose positions must remain unchanged.
    """

    axis: AxisRef

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Check that no complete position on the protected axis is removed.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An irreversible `fixed_axis` conflict if a complete position is
            removed, an unresolved `partitioned_constraint` diagnostic if the
            selection needs partitioned packing and preservation cannot be
            proved on this physical axis, or None.

        Raises:
            ValueError: The stored selection belongs to a different reference.
        """
        selection = chosen(selections, self.axis.tensor)
        if selection.fully_selected_indices(self.axis.dim):
            return Diagnostic(
                "fixed_axis",
                "A protected axis would change",
                "conflict",
                tensors=(self.axis.tensor.id,),
                irreversible=True,
            )
        if selection and selection.compact_shape() is None:
            return Diagnostic(
                "partitioned_constraint",
                "Cannot prove a physical axis is preserved by partitioned packing; "
                "constrain the corresponding logical input/output axis",
                tensors=(self.axis.tensor.id,),
            )
        return None


@dataclass(frozen=True)
class Balanced:
    """Require equal retained counts across fixed partitions.

    Attributes:
        axis: Logical axis whose original positions belong to the partitions.
        partitions: Original-coordinate index sets defining the fixed groups.
            Partitions must be nonempty, disjoint, and bounded; covering the
            entire axis is optional.
        nonempty: Whether removing an entire partition is a conflict.

    Raises:
        ValueError: `partitions` is empty, or a partition is not a nonempty
            `IndexSet`, exceeds the axis, or overlaps another partition.

    Notes:
        Unequal counts remain unresolved; this constraint never chooses channels.
    """

    axis: AxisRef
    partitions: tuple[IndexSet, ...]
    nonempty: bool = True

    def __post_init__(self) -> None:
        partitions = tuple(self.partitions)
        bounds = IndexSet.span(0, self.axis.tensor.shape[self.axis.dim])
        seen = IndexSet()
        if not partitions:
            raise ValueError("Balance requires nonempty partitions")
        for partition in partitions:
            if not isinstance(partition, IndexSet) or not partition:
                raise ValueError("Each balance partition must be a nonempty IndexSet")
            if partition.subtract(bounds) or partition.intersect(seen):
                raise ValueError("Balance partitions must be bounded and disjoint")
            seen = seen.union(partition)
        object.__setattr__(self, "partitions", partitions)

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Check retained partition counts without choosing a balancing completion.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An unresolved `partitioned_constraint` diagnostic if the selection
            needs partitioned packing, an irreversible `empty_partition` conflict
            if `nonempty` is set and a partition loses every position, an
            unresolved `unbalanced_groups` diagnostic if retained counts differ,
            or None.

        Raises:
            ValueError: The stored selection belongs to a different reference.
        """
        selection = chosen(selections, self.axis.tensor)
        if selection and selection.compact_shape() is None:
            return Diagnostic(
                "partitioned_constraint",
                "Balance must be checked on the corresponding logical axis",
                tensors=(self.axis.tensor.id,),
            )
        indices = selection.fully_selected_indices(self.axis.dim)
        remaining = tuple(len(p) - len(p.intersect(indices)) for p in self.partitions)
        if self.nonempty and any(n == 0 for n in remaining):
            return Diagnostic(
                "empty_partition",
                "A fixed group would become empty",
                "conflict",
                tensors=(self.axis.tensor.id,),
                irreversible=True,
            )
        if len(set(remaining)) > 1:
            return Diagnostic(
                "unbalanced_groups",
                f"Retained counts {remaining} need additional choices",
                tensors=(self.axis.tensor.id,),
            )
        return None


@dataclass(frozen=True)
class BlockBalance:
    """Require equal positive member counts in every surviving group.

    Attributes:
        groups: Axis representing logical groups; whole groups may disappear.
        members: Axis containing one contiguous block of members per group, in
            group order.
        block_size: Original number of members in each group.
        node: Name of the FX node responsible for this constraint.

    Raises:
        ValueError: `block_size` is not a positive integer, or the member axis
            length differs from the group count times `block_size`.
    """

    groups: AxisRef
    members: AxisRef
    block_size: int
    node: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool):
            raise ValueError("Block size must be a positive integer")
        if self.block_size <= 0 or (
            self.groups.tensor.shape[self.groups.dim] * self.block_size
            != self.members.tensor.shape[self.members.dim]
        ):
            raise ValueError("Member axis must consist of equally sized group blocks")

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.groups.tensor, self.members.tensor)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Check member counts in surviving groups without completing the request.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An unresolved `unbalanced_blocks` diagnostic if a surviving group has
            no members left or surviving groups have different member counts,
            otherwise None.

        Raises:
            ValueError: A stored selection belongs to a different reference.
        """
        removed_groups = chosen(selections, self.groups.tensor).fully_selected_indices(
            self.groups.dim
        )
        removed_members = chosen(selections, self.members.tensor).fully_selected_indices(
            self.members.dim
        )
        surviving = IndexSet.span(0, self.groups.tensor.shape[self.groups.dim]).subtract(
            removed_groups
        )
        counts = {
            self.block_size
            - len(
                removed_members.intersect(
                    IndexSet.span(group * self.block_size, (group + 1) * self.block_size)
                )
            )
            for group in surviving
        }
        if counts and (0 in counts or len(counts) > 1):
            return Diagnostic(
                "unbalanced_blocks",
                f"Surviving groups have member counts {sorted(counts)}; additional choices needed",
                node=self.node,
                tensors=tuple(ref.id for ref in self.refs),
            )
        return None


@dataclass(frozen=True)
class Divisible:
    """Require the retained axis length to be divisible by a positive factor.

    Attributes:
        axis: Tensor axis whose retained length is checked.
        factor: Positive divisor required by the consumer.

    Raises:
        ValueError: `factor` is not a positive, non-boolean integer.
    """

    axis: AxisRef
    factor: int

    def __post_init__(self) -> None:
        if not isinstance(self.factor, int) or isinstance(self.factor, bool) or self.factor <= 0:
            raise ValueError("Divisibility factor must be a positive integer")

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Check that the retained axis length is divisible by `factor`.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An unresolved `partitioned_constraint` diagnostic if the selection
            needs partitioned packing, an unresolved `indivisible_axis`
            diagnostic if the retained length is not divisible, or None.

        Raises:
            ValueError: The stored selection belongs to a different reference.
        """
        selection = chosen(selections, self.axis.tensor)
        if selection and selection.compact_shape() is None:
            return Diagnostic(
                "partitioned_constraint",
                "Divisibility must be checked on the corresponding logical axis",
                tensors=(self.axis.tensor.id,),
            )
        remaining = self.axis.tensor.shape[self.axis.dim] - len(
            selection.fully_selected_indices(self.axis.dim)
        )
        if remaining % self.factor:
            return Diagnostic(
                "indivisible_axis",
                f"Retained size {remaining} must be divisible by {self.factor}",
                tensors=(self.axis.tensor.id,),
            )
        return None


@dataclass(frozen=True)
class Barrier:
    """Report an unsupported operation only when a selection reaches its tensors.

    Attributes:
        refs: Tensor references whose selections activate this barrier.
        message: Explanation returned in the diagnostic.
        node: Name of the FX node responsible for the barrier, if any.
        code: Diagnostic category, defaulting to `unsupported`.
    """

    refs: tuple[TensorRef, ...]
    message: str
    node: str | None = None
    code: str = "unsupported"

    def __post_init__(self) -> None:
        object.__setattr__(self, "refs", tuple(self.refs))

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Report the barrier if any referenced tensor has a nonempty selection.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An unresolved, incomplete diagnostic with category `code`, or None.

        Raises:
            ValueError: A stored selection belongs to a different reference.
        """
        if any(chosen(selections, ref) for ref in self.refs):
            return Diagnostic(
                self.code,
                self.message,
                node=self.node,
                tensors=tuple(r.id for r in self.refs),
                complete=False,
            )
        return None


@dataclass(frozen=True)
class AxisBarrier:
    """Report unsupported removal of complete positions on one axis.

    Attributes:
        axis: Tensor axis whose complete cross-sections are protected.
        message: Explanation returned in the diagnostic.
        node: Name of the FX node responsible for the barrier, if any.
    """

    axis: AxisRef
    message: str
    node: str | None = None

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Report the barrier if a complete position on this axis is selected.

        Partial cross-sections do not activate the barrier.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An unresolved, incomplete `unsupported_axis` diagnostic, or None.

        Raises:
            ValueError: The stored selection belongs to a different reference.
        """
        if chosen(selections, self.axis.tensor).fully_selected_indices(self.axis.dim):
            return Diagnostic(
                "unsupported_axis",
                self.message,
                node=self.node,
                tensors=(self.axis.tensor.id,),
                complete=False,
            )
        return None


@dataclass(frozen=True)
class LayoutConstraint:
    """Validate the compact layouts accepted by one tensor use.

    Attributes:
        tensor: Tensor whose removed regions are checked.
        ports: Allowed partition scopes, or empty for ordinary axis compaction.
        node: Name of the FX call imposing the layout restriction, if any.

    Raises:
        ValueError: A port is not an `AxisPort` on `tensor`.

    Notes:
        Shared tensors must satisfy every use's layout separately. Combining the
        allowed ports of incompatible uses would lose that restriction.
    """

    tensor: TensorRef
    ports: tuple[AxisPort, ...] = ()
    node: str | None = None

    def __post_init__(self) -> None:
        ports = tuple(self.ports)
        if any(not isinstance(port, AxisPort) or port.tensor != self.tensor for port in ports):
            raise ValueError("LayoutConstraint ports must belong to the layout tensor")
        object.__setattr__(self, "ports", ports)

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensors whose selections this constraint inspects."""
        return (self.tensor,)

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Check that the selection fits this use's supported packing.

        Without ports, the selection must consist of complete axis
        cross-sections. With ports, every selected element must lie in a
        complete scoped cross-section of at least one port.

        Args:
            selections: Accumulated selections keyed by graph-local tensor ID.

        Returns:
            An unresolved `unsupported_layout` diagnostic, or None.

        Raises:
            ValueError: The stored selection belongs to a different reference.
        """
        selection = chosen(selections, self.tensor)
        if not selection:
            return None
        if not self.ports:
            valid = selection.compact_shape() is not None
        elif self._full_port_cross_sections(selection):
            valid = True
        else:
            covered = tuple(box for port in self.ports for box in port.covered_boxes(selection))
            valid = not canonical_boxes(region_boxes(selection.regions), covered)
        if not valid:
            return Diagnostic(
                "unsupported_layout",
                "Selection is not a supported compact/partitioned layout",
                node=self.node,
                tensors=(self.tensor.id,),
            )
        return None

    def _full_port_cross_sections(self, selection: Selection) -> bool:
        """Recognize complete cross-sections on the axis of a whole-tensor port.

        That port then covers every selected element, so the general union of
        port coverage is unnecessary. Only exact native records take this path.
        """
        if not (
            type(selection) is Selection
            and type(self.tensor) is TensorRef
            and len(selection.regions) == 1
            and type(selection.regions[0]) is Region
            and all(type(size) is int for size in self.tensor.shape)
            and all(type(indices) is IndexSet for indices in selection.regions[0].axes)
        ):
            return False
        partial = [
            dim
            for dim, (indices, size) in enumerate(
                zip(selection.regions[0].axes, self.tensor.shape, strict=True)
            )
            if indices.intervals != ((0, size),)
        ]
        if len(partial) != 1:
            return False
        return any(
            type(port) is AxisPort
            and type(port.axis) is AxisRef
            and type(port.axis.dim) is int
            and port.axis.dim == partial[0]
            and (
                port.scope is None
                or (
                    type(port.scope) is Region
                    and all(type(indices) is IndexSet for indices in port.scope.axes)
                    and port.scope == full_region(self.tensor.shape)
                )
            )
            for port in self.ports
        )


# Every other built-in constraint reports only repairable or incomplete states.
IRREVERSIBLE_CONSTRAINTS = (NonEmpty, Fixed, Balanced)
"""Built-in constraint types whose `check` can return an irreversible diagnostic."""


@dataclass(frozen=True)
class AlignmentClass:
    """Retained-size requirements shared by one set of identity-equivalent axes.

    Attributes:
        axes: Axes of the set that carry `NonEmpty`, in stable order.
        factor: Least common multiple of the set's `Divisible` factors.
        balances: Partitions and `nonempty` flags of the set's `Balanced`
            constraints, without duplicates.
    """

    axes: tuple[AxisRef, ...]
    factor: int
    balances: tuple[tuple[tuple[IndexSet, ...], bool], ...] = ()


def alignment_classes(
    constraints: Iterable[Constraint],
    equivalent: Callable[[AxisRef], tuple[AxisRef, ...]],
) -> tuple[AlignmentClass, ...]:
    """Combine retained-size constraints over identity-equivalent axes.

    Identity-equivalent axes remove the same positions in every complete
    closure, so they share one retained length and one set of retained
    positions. Their `Divisible` factors and `Balanced` partitions are
    combined, even when they are declared on different axes, for example on
    the rows of a producer and the columns of its consumer.

    Args:
        constraints: Constraints to combine. Only exact built-in `NonEmpty`,
            `Divisible` and `Balanced` records contribute; callers verify their
            native semantics.
        equivalent: Returns the identity-equivalent axes of an axis, including
            the axis itself, in a stable order.

    Returns:
        One class per set of equivalent axes with a `Divisible` or `Balanced`
        constraint, in first-constraint order. Sets without a `NonEmpty` axis,
        or with factor 1 and no partitions, cannot conflict and are omitted.
    """
    nonempty = set()
    factors: dict[tuple[AxisRef, ...], int] = {}
    balances: dict[tuple[AxisRef, ...], list[tuple[tuple[IndexSet, ...], bool]]] = {}
    for constraint in constraints:
        if type(constraint) is NonEmpty:
            nonempty.add(constraint.axis)
        elif type(constraint) in (Divisible, Balanced):
            axes = equivalent(constraint.axis)
            if type(constraint) is Divisible:
                factors[axes] = math.lcm(factors.get(axes, 1), constraint.factor)
            else:
                factors.setdefault(axes, 1)
                balances.setdefault(axes, []).append((constraint.partitions, constraint.nonempty))
    classes = []
    for axes, factor in factors.items():
        required = tuple(axis for axis in axes if axis in nonempty)
        groups = tuple(dict.fromkeys(balances.get(axes, ())))
        if required and (factor > 1 or groups):
            classes.append(AlignmentClass(required, factor, groups))
    return tuple(classes)


def alignment_conflicts(
    classes: Iterable[AlignmentClass], selections: Mapping[str, Selection]
) -> tuple[Diagnostic, ...]:
    """Prove axes whose retained size can no longer be aligned without emptying them.

    Additional removals only shrink an axis. Suppose an axis that must stay
    nonempty retains `r` positions, with `0 < r < m`, where `m` is the least
    common multiple of its factors. No smaller positive length is a multiple
    of `m`, and length zero violates `NonEmpty`, so the conflict persists under
    every superset. Neither constraint can prove this alone.

    `Balanced` partitions restrict the reachable lengths further. Every
    partition must keep the same count `c`, at most the smallest current
    count, and at least 1 when emptied partitions are forbidden. Positions
    outside the partitions can keep any count up to their current one. If no
    such choice gives a positive multiple of `m`, larger requests cannot
    produce one either, because they only lower these bounds. Each `Balanced`
    constraint is checked separately, which proves fewer conflicts than a
    joint check of several but never a false one.

    Args:
        classes: Results of `alignment_classes` for the checked constraints.
        selections: Accumulated selections keyed by graph-local tensor ID.

    Returns:
        One irreversible `unreachable_alignment` conflict per class, in class
        order. Its tensors are those of the nonempty axes in the conflict range.
        The message omits the retained counts, so the same axes give an equal
        diagnostic for every conflicting state, including states where
        `Balanced` also reports an emptied partition. An emptied axis is left
        to `NonEmpty`.

    Raises:
        ValueError: A stored selection belongs to a different reference.
        torch_kirigami.AnalysisLimitError: Counting removed positions exceeds
            the region limit.
    """
    diagnostics = []
    for alignment in classes:
        conflicting = []
        for axis in alignment.axes:
            width = axis.tensor.shape[axis.dim]
            removed = IndexSet()
            if axis.tensor.id in selections:
                selection = chosen(selections, axis.tensor)
                # Fully selected positions lie inside the regions' extents on
                # this axis. Without partitions, most axes keep at least
                # `factor` positions by this bound alone, so skip their exact
                # cross-section analysis.
                extents = sum(len(region.axes[axis.dim]) for region in selection.regions)
                if not alignment.balances and width - extents >= alignment.factor:
                    continue
                if selection.compact_shape() is None:
                    continue  # `Divisible` reports partitioned selections instead.
                removed = selection.fully_selected_indices(axis.dim)
            if _unreachable_alignment(alignment, width, removed):
                conflicting.append(axis.tensor.id)
        if conflicting:
            factor = alignment.factor
            if not alignment.balances:
                message = (
                    f"A positive retained size below {factor} cannot become a multiple "
                    f"of {factor} without emptying the axis"
                )
            elif factor > 1:
                message = (
                    f"Balanced partitions cannot keep a positive size divisible by {factor} "
                    "without emptying the axis"
                )
            else:
                message = "Balanced partitions cannot keep equal counts without emptying the axis"
            diagnostics.append(
                Diagnostic(
                    "unreachable_alignment",
                    message,
                    "conflict",
                    tensors=tuple(dict.fromkeys(conflicting)),
                    irreversible=True,
                )
            )
    return tuple(diagnostics)


def _unreachable_alignment(alignment: AlignmentClass, width: int, removed: IndexSet) -> bool:
    """Decide whether no subset of the retained positions satisfies one class."""
    remaining = width - len(removed)
    factor = alignment.factor
    if remaining == 0:
        return False
    if remaining < factor:
        return True
    for partitions, nonempty in alignment.balances:
        counts = [len(p) - len(p.intersect(removed)) for p in partitions]
        outside = remaining - sum(counts)
        least = min(counts)
        groups = len(partitions)
        # `groups * count` modulo `factor` repeats with this period, so one
        # period of the largest counts covers every reachable residue.
        period = factor // math.gcd(groups, factor)
        lowest = max(1 if nonempty else 0, least - period + 1)
        if not any(
            -(-max(groups * count, 1) // factor) * factor <= groups * count + outside
            for count in range(least, lowest - 1, -1)
        ):
            return True
    return False


@dataclass(frozen=True)
class ShapeExpr:
    """Limited provenance of a dimension value, without a general computation IR.

    Attributes:
        kind: Expression category. Leaves are `constant` and `infer` (integer
            payload; `infer` is always -1), `unknown` (a description string),
            `dimension` (a tensor and axis pair), and `shape`, `dim`, or
            `numel` (a tensor reference). `tuple` groups any number of
            operands, and `add`, `sub`, `mul`, `floordiv`, and `mod` take
            exactly two operands.
        value: Category-specific payload. A `dimension` payload is normalized
            to a tensor reference and a nonnegative axis index. Composite
            expressions have no payload.
        args: Operand expressions for `tuple` and the binary operations.

    Raises:
        TypeError: An operand is not a `ShapeExpr`, or a payload has the wrong type.
        ValueError: `kind`, the operand count, or a scalar payload is invalid.
        IndexError: A `dimension` axis is outside the tensor rank.
    """

    kind: Literal[
        "constant",
        "infer",
        "unknown",
        "dimension",
        "shape",
        "dim",
        "numel",
        "tuple",
        "add",
        "sub",
        "mul",
        "floordiv",
        "mod",
    ]
    value: Any = None
    args: tuple[ShapeExpr, ...] = ()

    def __post_init__(self) -> None:
        args = tuple(self.args)
        if any(not isinstance(arg, ShapeExpr) for arg in args):
            raise TypeError("Shape operands must be ShapeExpr instances")
        binary = {"add", "sub", "mul", "floordiv", "mod"}
        leaves = {"constant", "infer", "unknown", "dimension", "shape", "dim", "numel"}
        if self.kind not in binary | leaves | {"tuple"}:
            raise ValueError(f"Unknown shape expression kind: {self.kind}")
        if (self.kind in binary and len(args) != 2) or (self.kind in leaves and args):
            raise ValueError("Invalid shape expression arity")
        value = self.value
        if self.kind in ("constant", "infer"):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("Shape constants must be integers")
            if self.kind == "infer" and value != -1:
                raise ValueError("Inferred dimension must be -1")
        elif self.kind == "dimension":
            if (
                not isinstance(value, (tuple, list))
                or len(value) != 2
                or not isinstance(value[0], TensorRef)
            ):
                raise TypeError("Dimension provenance requires a tensor and axis pair")
            ref, dim = value
            value = (ref, ref.axis(dim).dim)
        elif self.kind in ("shape", "dim", "numel"):
            if not isinstance(value, TensorRef):
                raise TypeError("Shape reads require a TensorRef")
        elif self.kind == "unknown":
            if not isinstance(value, str):
                raise TypeError("Unknown provenance requires a description")
        elif value is not None:
            raise ValueError("Composite shape expressions have no scalar payload")
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "value", value)

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Unique tensor references read by this expression and its operands, in order."""
        own: tuple[TensorRef, ...] = ()
        if isinstance(self.value, TensorRef):
            own = (self.value,)
        elif self.kind == "dimension":
            own = (self.value[0],)
        return tuple(dict.fromkeys(own + tuple(ref for arg in self.args for ref in arg.refs)))


def _requirement_value(value: object, depth: int = 0) -> object:
    """Detach nested payload sequences, rejecting mutable opaque state."""
    if depth > config.max_container_depth:
        raise ValueError("Requirement data is cyclic or too deeply nested")
    if isinstance(value, (tuple, list)):
        return tuple(_requirement_value(item, depth + 1) for item in value)
    if value is None or isinstance(
        value, (str, int, float, bool, TensorRef, AxisRef, IndexSet, Region, ShapeExpr)
    ):
        return value
    if isinstance(value, slice) and all(
        item is None or (isinstance(item, int) and not isinstance(item, bool))
        for item in (value.start, value.stop, value.step)
    ):
        return value
    raise TypeError(f"Unsupported requirement payload: {type(value).__name__}")


@dataclass(frozen=True)
class ArgumentRef:
    """Identify a call parameter used by a structural contract.

    Operator requirements and tensor effects use this record to locate call
    arguments without retaining a mutable operation context.

    Attributes:
        name: Canonical keyword name, resolved with the native alias table.
        position: Position in FX arguments, including any Tensor receiver.
        variadic: Whether all remaining positional arguments belong to this parameter.

    Raises:
        ValueError: `name` is empty or not a string, or `position` is not a
            nonnegative integer.
        TypeError: `variadic` is not a boolean.
    """

    name: str
    position: int
    variadic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Argument name must be a nonempty string")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("Argument position must be a nonnegative integer")
        if type(self.variadic) is not bool:
            raise TypeError("Argument variadic must be boolean")


@dataclass(frozen=True)
class AttributeBinding:
    """Bind a qualified module attribute to a structural size expression.

    The expression uses the shape-provenance representation. Planning evaluates
    it against retained dimensions; this record never edits the model.

    Attributes:
        target: Dotted path of the module attribute to update.
        expression: Shape expression that gives the planned value.
        detail: Human-readable description of the required update.

    Raises:
        ValueError: `target` is empty, not a string, or contains an empty path component.
        TypeError: `expression` is not a `ShapeExpr`.
    """

    target: str
    expression: ShapeExpr
    detail: str = "Update the bound structural attribute"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target, str)
            or not self.target
            or any(not part for part in self.target.split("."))
        ):
            raise ValueError("Attribute binding requires a qualified attribute path")
        if not isinstance(self.expression, ShapeExpr):
            raise TypeError("Attribute binding requires a ShapeExpr")

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor references whose dimensions determine this attribute."""
        return self.expression.refs


@dataclass(frozen=True)
class Requirement:
    """Describe a modification that an executor must implement or reject.

    Attributes:
        kind: Extensible obligation category, such as `metadata_layout` or
            `partitioned_compaction`. The kind `attribute` is reserved; declare
            module attributes with `AttributeBinding` instead.
        target: Location to check, such as an FX node name or a registered
            tensor path.
        tensors: Tensor references determining the modification.
        detail: Human-readable explanation and limitations.
        data: Uniquely named values containing scalars, integer slices,
            structural references, or nested sequences. Sequences are frozen as
            tuples; other payload types are rejected.
        arguments: Specific parameters this requirement validates when they change.
            All other dimension-derived semantic parameters must remain unchanged.

    Raises:
        ValueError: `kind` is `attribute`, `kind` or `target` is invalid,
            `data` entries are not uniquely named pairs, or a payload is too
            deeply nested.
        TypeError: A tensor is not a `TensorRef`, an argument is not an
            `ArgumentRef`, or a payload has an unsupported type.

    Notes:
        A requirement is descriptive; it neither mutates the model nor guarantees
        that an executor can perform the change.
    """

    kind: str
    target: str
    tensors: tuple[TensorRef, ...]
    detail: str
    data: tuple[tuple[str, Any], ...] = ()
    arguments: tuple[ArgumentRef, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "attribute":
            raise ValueError("Declare module attributes with AttributeBinding")
        if not isinstance(self.kind, str) or not self.kind or not isinstance(self.target, str):
            raise ValueError("Requirement kind and target must be strings")
        data = tuple(tuple(item) for item in self.data)
        if any(len(item) != 2 or not isinstance(item[0], str) for item in data):
            raise ValueError("Requirement data must contain named pairs")
        if len({key for key, _ in data}) != len(data):
            raise ValueError("Requirement data keys must be unique")
        tensors = tuple(self.tensors)
        if any(not isinstance(ref, TensorRef) for ref in tensors):
            raise TypeError("Requirement tensors must be TensorRef instances")
        object.__setattr__(self, "tensors", tensors)
        arguments = tuple(self.arguments)
        if any(not isinstance(arg, ArgumentRef) for arg in arguments):
            raise TypeError("Requirement arguments must be ArgumentRef records")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(
            self, "data", tuple((key, _requirement_value(value)) for key, value in data)
        )

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor references in `tensors` and embedded in `data`, deduplicated in order.

        Graphs use these references to validate requirement ownership.
        """

        def visit(value: object) -> Iterator[TensorRef]:
            """Yield tensor references embedded in a requirement payload."""
            if isinstance(value, TensorRef):
                yield value
            elif isinstance(value, AxisRef):
                yield value.tensor
            elif isinstance(value, ShapeExpr):
                yield from value.refs
            elif isinstance(value, tuple):
                for item in value:
                    yield from visit(item)

        return tuple(dict.fromkeys((*self.tensors, *visit(self.data))))


@dataclass(frozen=True)
class Provenance:
    """Explain a newly discovered selection through a propagation step.

    Attributes:
        source: Selection that caused the propagation.
        target: Newly added target regions, excluding previously known selections.
        reason: Semantic relationship responsible for the change.
    """

    source: Selection
    target: Selection
    reason: str


@dataclass(frozen=True)
class Impact:
    """The structural consequences, constraints, and diagnostics of one request.

    An impact is an analysis result, not an executable pruning plan. All
    selections use the original (captured) coordinates of their tensors.

    Attributes:
        graph_id: Identity of the graph that produced this result.
        requested: Original removal requests.
        selections: Read-only closure of known selections, keyed by tensor ID.
        diagnostics: Conflicts and reasons that part of the analysis is unresolved.
        requirements: Modifications the execution layer must handle or explicitly reject.
        provenance: Reasons for newly discovered selections.
        interfaces: Affected external input and output tensor references.
        status: `conflict` if any diagnostic is a conflict, `unresolved` if
            other diagnostics remain, otherwise `resolved`. Status is
            independent of `complete`.
        constraints: Conditions checked against the closure, including
            caller-supplied constraints.
        tensors: Read-only map of all known references, including unaffected
            values, used for ownership checks.

    Raises:
        ValueError: `status` is not `resolved`, `unresolved`, or `conflict`.

    Notes:
        A resolved result only proves the declared structural conditions under the
        graph's assumptions. It does not establish numerical equivalence.
    """

    graph_id: str
    requested: tuple[Selection, ...]
    selections: Mapping[str, Selection]
    diagnostics: tuple[Diagnostic, ...]
    requirements: tuple[Requirement | AttributeBinding, ...]
    provenance: tuple[Provenance, ...]
    interfaces: tuple[TensorRef, ...]
    status: str
    constraints: tuple[Constraint, ...] = field(repr=False)
    tensors: Mapping[str, TensorRef] = field(repr=False)

    def __post_init__(self) -> None:
        if self.status not in ("resolved", "unresolved", "conflict"):
            raise ValueError("Invalid impact status")
        for name in (
            "requested",
            "diagnostics",
            "requirements",
            "provenance",
            "interfaces",
            "constraints",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "selections", MappingProxyType(dict(self.selections)))
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))

    @property
    def complete(self) -> bool:
        """Whether the influence range is fully known.

        This is independent of `status`: a complete result can still contain
        conflicts, and an incomplete result is never resolved.
        """
        return all(d.complete for d in self.diagnostics)

    @property
    def irreversible(self) -> bool:
        """Whether some conflict is proved to survive every additional removal.

        This is a stronger proof than `status == "conflict"` and is independent
        of `complete`. False does not prove feasibility; a conflict without this
        proof can still require joint choices to repair.
        """
        return any(d.irreversible for d in self.diagnostics)

    def selection(self, tensor: TensorRef) -> Selection:
        """Return the accumulated selection for a tensor.

        Args:
            tensor: A reference from the graph that produced this impact.

        Returns:
            The tensor's selection, or an empty selection if it is unaffected.

        Raises:
            ValueError: The reference does not belong to this impact's graph or
                has altered metadata.
        """
        if self.tensors.get(tensor.id) != tensor:
            raise ValueError("Tensor reference does not belong to this impact")
        return chosen(self.selections, tensor)

    @property
    def parameters(self) -> tuple[Selection, ...]:
        """Selections on affected parameters, one per parameter object."""
        return tuple(s for s in self.selections.values() if s.tensor.kind == "parameter")

    @property
    def buffers(self) -> tuple[Selection, ...]:
        """Selections on affected registered buffers, one per buffer object."""
        return tuple(s for s in self.selections.values() if s.tensor.kind == "buffer")
