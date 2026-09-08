"""Read-only analysis results and constraints shared by built-ins and extensions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .relations import Port
from .selection import AxisRef, IndexSet, Selection, TensorRef


@dataclass(frozen=True)
class Diagnostic:
    """Explain a conflict or an unproved part of an analysis.

    Attributes:
        code: Stable diagnostic category.
        message: Human-readable explanation.
        severity: Unresolved for missing proof or conflict for a violation.
        node: FX node name, when the issue belongs to a particular operation.
        tensors: IDs of tensor entities involved in the issue.
    """

    code: str
    message: str
    severity: str = "unresolved"
    node: str | None = None
    tensors: tuple[str, ...] = ()
    complete: bool = True


class Constraint(Protocol):
    """Check a structural condition without choosing additional removals."""

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Return the tensors whose selections this constraint inspects."""
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


def chosen(selections, tensor):
    """Return a tensor's accumulated selection, or an empty selection."""
    return selections.get(tensor.id, Selection(tensor))


@dataclass(frozen=True)
class NonEmpty:
    """Require at least one position to remain on the given axis."""

    axis: AxisRef

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections):
        """Report a conflict if the entire axis would disappear."""
        selection = chosen(selections, self.axis.tensor)
        if not selection:
            return None
        indices = selection.project(self.axis.dim)
        if len(indices) == self.axis.tensor.shape[self.axis.dim]:
            return Diagnostic(
                "empty_axis",
                "The operator requires a nonempty axis",
                "conflict",
                tensors=(self.axis.tensor.id,),
            )
        return None


@dataclass(frozen=True)
class Fixed:
    """Consumer constraint: preserve this axis's structural positions."""

    axis: AxisRef

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections):
        """Check that the protected structural axis is preserved."""
        selection = chosen(selections, self.axis.tensor)
        if selection.project(self.axis.dim):
            return Diagnostic(
                "fixed_axis",
                "A protected axis would change",
                "conflict",
                tensors=(self.axis.tensor.id,),
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
        nonempty: Whether removing an entire partition is a conflict.

    Notes:
        Unequal counts remain unresolved; this constraint never chooses channels.
    """

    axis: AxisRef
    partitions: tuple[IndexSet, ...]
    nonempty: bool = True

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections):
        """Check partition counts without choosing a balancing completion."""
        selection = chosen(selections, self.axis.tensor)
        if selection and selection.compact_shape() is None:
            return Diagnostic(
                "partitioned_constraint",
                "Balance must be checked on the corresponding logical axis",
                tensors=(self.axis.tensor.id,),
            )
        indices = selection.project(self.axis.dim)
        remaining = tuple(len(p) - len(p.intersect(indices)) for p in self.partitions)
        if self.nonempty and any(n == 0 for n in remaining):
            return Diagnostic(
                "empty_partition",
                "A fixed group would become empty",
                "conflict",
                tensors=(self.axis.tensor.id,),
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
        members: Axis containing contiguous blocks of group members.
        block_size: Original number of members in each group.
        node: FX node responsible for this constraint.
    """

    groups: AxisRef
    members: AxisRef
    block_size: int
    node: str | None = None

    def __post_init__(self):
        if self.block_size <= 0 or (
            self.groups.tensor.shape[self.groups.dim] * self.block_size
            != self.members.tensor.shape[self.members.dim]
        ):
            raise ValueError("Member axis must consist of equally sized group blocks")

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.groups.tensor, self.members.tensor)

    def check(self, selections):
        """Check member counts in surviving groups without completing the request."""
        removed_groups = chosen(selections, self.groups.tensor).project(self.groups.dim)
        removed_members = chosen(selections, self.members.tensor).project(self.members.dim)
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
    """Require the retained axis length to be divisible by a positive factor."""

    axis: AxisRef
    factor: int

    def __post_init__(self):
        if not isinstance(self.factor, int) or isinstance(self.factor, bool) or self.factor <= 0:
            raise ValueError("Divisibility factor must be a positive integer")

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections):
        """Check divisibility of the retained logical axis length."""
        selection = chosen(selections, self.axis.tensor)
        if selection and selection.compact_shape() is None:
            return Diagnostic(
                "partitioned_constraint",
                "Divisibility must be checked on the corresponding logical axis",
                tensors=(self.axis.tensor.id,),
            )
        remaining = self.axis.tensor.shape[self.axis.dim] - len(selection.project(self.axis.dim))
        if remaining % self.factor:
            return Diagnostic(
                "indivisible_axis",
                f"Retained size {remaining} must be divisible by {self.factor}",
                tensors=(self.axis.tensor.id,),
            )
        return None


@dataclass(frozen=True)
class Barrier:
    """Block completeness only when a selection reaches one of these tensors."""

    refs: tuple[TensorRef, ...]
    message: str
    node: str | None = None
    code: str = "unsupported"

    def check(self, selections):
        """Report the barrier only if a referenced tensor is affected."""
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
    """Report unsupported changes to full structural positions on one axis."""

    axis: AxisRef
    message: str
    node: str | None = None

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.axis.tensor,)

    def check(self, selections):
        """Report the barrier only if full positions on this axis are affected."""
        if chosen(selections, self.axis.tensor).project(self.axis.dim):
            return Diagnostic(
                "unsupported_axis",
                self.message,
                node=self.node,
                tensors=(self.axis.tensor.id,),
                complete=False,
            )
        return None


@dataclass(frozen=True)
class Layout:
    """Validate the compact layouts accepted by one tensor use.

    Attributes:
        tensor: Tensor whose removed regions are checked.
        ports: Allowed partition scopes, or empty for ordinary axis compaction.
        node: FX call imposing the layout restriction.

    Notes:
        Shared tensors must satisfy every use's layout separately. Combining the
        allowed ports of incompatible uses would lose that restriction.
    """

    tensor: TensorRef
    ports: tuple[Port, ...] = ()
    node: str | None = None

    @property
    def refs(self):
        """Return the tensors whose selections this constraint inspects."""
        return (self.tensor,)

    def check(self, selections):
        """Check that all selected regions fit this use's supported packing."""
        selection = chosen(selections, self.tensor)
        if not selection:
            return None
        if not self.ports:
            valid = selection.compact_shape() is not None
        else:
            covered = Selection(self.tensor)
            for port in self.ports:
                covered = covered.union(port.select(port.project(selection)))
            valid = not selection.subtract(covered)
        if not valid:
            return Diagnostic(
                "unsupported_layout",
                "Selection is not a supported compact/partitioned layout",
                node=self.node,
                tensors=(self.tensor.id,),
            )
        return None


@dataclass(frozen=True)
class ShapeExpr:
    """Record limited dimension provenance without a computation IR.

    Attributes:
        kind: Expression category, such as constant, dimension, or integer operation.
        value: Category-specific payload, such as an integer or tensor-axis binding.
        args: Operand expressions for supported shape operations.
    """

    kind: str
    value: Any = None
    args: tuple[ShapeExpr, ...] = ()


@dataclass(frozen=True)
class Requirement:
    """Describe a modification a future executor would need to implement.

    Attributes:
        kind: Modification category, such as attribute, layout, or graph argument.
        target: Module attribute or graph location to change.
        tensors: Tensor entities determining the modification.
        detail: Human-readable explanation and limitations.
        data: Structured bindings and original-coordinate information.

    Notes:
        A requirement is descriptive; it neither mutates the model nor guarantees
        that an executor can perform the change.
    """

    kind: str
    target: str
    tensors: tuple[TensorRef, ...]
    detail: str
    data: tuple[tuple[str, Any], ...] = ()


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
    """Hold dependency analysis results, not an executable pruning plan.

    Attributes:
        graph_id: Identity of the graph that produced this result.
        requested: Original removal requests.
        selections: Closure of known selections, keyed by tensor ID.
        diagnostics: Conflicts and reasons that part of the analysis is unresolved.
        requirements: Modifications needed by a future executor.
        provenance: Reasons for newly discovered selections.
        interfaces: Affected external input and output tensor references.
        status: Resolved, unresolved, or conflict.
        constraints: Conditions checked against the closure.

    Notes:
        A resolved result only proves the declared structural conditions under the
        graph's assumptions. It does not establish numerical equivalence.
    """

    graph_id: str
    requested: tuple[Selection, ...]
    selections: Mapping[str, Selection]
    diagnostics: tuple[Diagnostic, ...]
    requirements: tuple[Requirement, ...]
    provenance: tuple[Provenance, ...]
    interfaces: tuple[TensorRef, ...]
    status: str
    constraints: tuple[Constraint, ...] = field(repr=False)

    @property
    def complete(self):
        """Whether the influence range is known, independently of constraint validity."""
        return all(d.complete for d in self.diagnostics)

    def selection(self, tensor: TensorRef) -> Selection:
        """Return the accumulated selection for a tensor, or an empty selection."""
        return chosen(self.selections, tensor)

    @property
    def parameters(self):
        """Return affected parameter selections, deduplicated by object identity."""
        return tuple(s for s in self.selections.values() if s.tensor.kind == "parameter")

    @property
    def buffers(self):
        """Return affected registered-buffer selections."""
        return tuple(s for s in self.selections.values() if s.tensor.kind == "buffer")
