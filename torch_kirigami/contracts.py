"""Read-only analysis results and constraints shared by built-ins and extensions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from .relations import AxisPort
from .selection import AxisRef, IndexSet, Region, Selection, TensorRef


@dataclass(frozen=True)
class Diagnostic:
    """Explain a conflict or an unproved part of an analysis.

    Attributes:
        code: Stable diagnostic category.
        message: Human-readable explanation.
        severity: Unresolved for missing proof or conflict for a violation.
        node: FX node name, when the issue belongs to a particular operation.
        tensors: IDs of tensor entities involved in the issue.
        complete: Whether the influence range is known despite this diagnostic.
    """

    code: str
    message: str
    severity: Literal["unresolved", "conflict"] = "unresolved"
    node: str | None = None
    tensors: tuple[str, ...] = ()
    complete: bool = True

    def __str__(self):
        """Render the category, available operation location, and actionable message."""
        location = f" at {self.node}" if self.node else ""
        return f"{self.code}{location}: {self.message}"

    def __post_init__(self):
        if self.severity not in ("unresolved", "conflict"):
            raise ValueError("Invalid diagnostic severity")
        if not isinstance(self.complete, bool):
            raise TypeError("Diagnostic completeness must be boolean")
        object.__setattr__(self, "tensors", tuple(self.tensors))


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
    selection = selections.get(tensor.id)
    if selection is None:
        return Selection(tensor)
    if selection.tensor != tensor:
        raise ValueError("Tensor reference does not match the selection")
    return selection


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
        indices = selection.fully_selected_indices(self.axis.dim)
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
        if selection.fully_selected_indices(self.axis.dim):
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
            Partitions must be nonempty, disjoint, and bounded; covering the
            entire axis is optional.
        nonempty: Whether removing an entire partition is a conflict.

    Notes:
        Unequal counts remain unresolved; this constraint never chooses channels.
    """

    axis: AxisRef
    partitions: tuple[IndexSet, ...]
    nonempty: bool = True

    def __post_init__(self):
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
        indices = selection.fully_selected_indices(self.axis.dim)
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
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool):
            raise ValueError("Block size must be a positive integer")
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
    """Block completeness only when a selection reaches one of these tensors."""

    refs: tuple[TensorRef, ...]
    message: str
    node: str | None = None
    code: str = "unsupported"

    def __post_init__(self):
        object.__setattr__(self, "refs", tuple(self.refs))

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
        node: FX call imposing the layout restriction.

    Notes:
        Shared tensors must satisfy every use's layout separately. Combining the
        allowed ports of incompatible uses would lose that restriction.
    """

    tensor: TensorRef
    ports: tuple[AxisPort, ...] = ()
    node: str | None = None

    def __post_init__(self):
        ports = tuple(self.ports)
        if any(not isinstance(port, AxisPort) or port.tensor != self.tensor for port in ports):
            raise ValueError("LayoutConstraint ports must belong to the layout tensor")
        object.__setattr__(self, "ports", ports)

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
                covered = covered.union(port.select(port.fully_selected_indices(selection)))
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

    def __post_init__(self):
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
        """Return unique tensor sources without retaining capture state."""
        own = ()
        if isinstance(self.value, TensorRef):
            own = (self.value,)
        elif self.kind == "dimension":
            own = (self.value[0],)
        return tuple(dict.fromkeys(own + tuple(ref for arg in self.args for ref in arg.refs)))


def _requirement_value(value, depth=0):
    """Detach nested payload sequences, rejecting mutable opaque state."""
    if depth > 50:
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
    """Identify one call parameter whose changes an execution requirement verifies.

    Args:
        name: Canonical keyword name, resolved with the native alias table.
        position: Positional slot in FX call arguments, including the Tensor receiver.
        variadic: Whether all remaining positional slots belong to this parameter.
    """

    name: str
    position: int
    variadic: bool = False

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Argument name must be a nonempty string")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("Argument position must be a nonnegative integer")
        if type(self.variadic) is not bool:
            raise TypeError("Argument variadic must be boolean")


@dataclass(frozen=True)
class Requirement:
    """Describe a modification an executor needs to implement.

    Attributes:
        kind: Modification category, such as attribute, layout, or graph argument.
        target: Module attribute or graph location to change.
        tensors: Tensor entities determining the modification.
        detail: Human-readable explanation and limitations.
        data: Unique named values containing scalar data, slices, structural
            references, or nested sequences. Sequences are frozen as tuples;
            opaque mutable objects are rejected. Kind names remain extensible.
        arguments: Specific parameters this requirement validates when they change.
            All other dimension-derived semantic parameters must remain unchanged.

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

    def __post_init__(self):
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
        """Include sources embedded in payloads for graph ownership validation."""

        def visit(value):
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
    """Hold dependency analysis results, not an executable pruning plan.

    Attributes:
        graph_id: Identity of the graph that produced this result.
        requested: Original removal requests.
        selections: Closure of known selections, keyed by tensor ID.
        diagnostics: Conflicts and reasons that part of the analysis is unresolved.
        requirements: Modifications the execution layer must handle or explicitly reject.
        provenance: Reasons for newly discovered selections.
        interfaces: Affected external input and output tensor references.
        status: Resolved, unresolved, or conflict.
        constraints: Conditions checked against the closure.
        tensors: Known references, including unaffected values, for ownership checks.

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
    tensors: Mapping[str, TensorRef] = field(repr=False)

    def __post_init__(self):
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
    def complete(self):
        """Whether the influence range is known, independently of constraint validity."""
        return all(d.complete for d in self.diagnostics)

    def selection(self, tensor: TensorRef) -> Selection:
        """Return the accumulated selection for a tensor, or an empty selection."""
        if self.tensors.get(tensor.id) != tensor:
            raise ValueError("Tensor reference does not belong to this impact")
        return chosen(self.selections, tensor)

    @property
    def parameters(self):
        """Return affected parameter selections, deduplicated by object identity."""
        return tuple(s for s in self.selections.values() if s.tensor.kind == "parameter")

    @property
    def buffers(self):
        """Return affected registered-buffer selections."""
        return tuple(s for s in self.selections.values() if s.tensor.kind == "buffer")
