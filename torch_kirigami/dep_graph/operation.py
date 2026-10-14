"""Shared operation facts and rule contracts, independent of orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
from torch import fx, nn

from ..operator_recipes import OperatorRecipe
from .contracts import ArgumentRef, AttributeBinding, Constraint, Impact, Requirement, ShapeExpr
from .relations import Relation
from .selection import AxisRef, Region, Selection, TensorRef


@dataclass(frozen=True)
class TensorFacts:
    """Tensor metadata detached from activation storage.

    Attributes:
        shape: Logical dimensions observed during metadata execution.
        stride: Strides measured in tensor elements.
        dtype: PyTorch element type.
        device: Device on which the sample tensor was observed.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


_KEYWORD_ALIASES = {
    "dim": ("axis",),
    "keepdim": ("keepdims",),
    "dim0": ("axis0",),
    "dim1": ("axis1",),
    "split_size_or_sections": ("split_size",),
    "sizes": ("repeats",),
    "shape": ("size",),
}


def argument_locations(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    name: str,
    position: int,
    *,
    target: object = None,
    variadic: bool = False,
) -> tuple[tuple[str, str | int], ...]:
    """Locate one canonical parameter without conflating equal FX expression nodes."""
    namespace = getattr(target, "__module__", "")
    native = isinstance(target, str) or namespace == "torch" or namespace.startswith("torch.")
    names = (name, *(_KEYWORD_ALIASES.get(name, ()) if native else ()))
    supplied = [key for key in names if key in kwargs]
    if len(supplied) > 1:
        raise ValueError(f"Multiple spellings supplied for {name}: {supplied}")
    if supplied:
        return (("kwargs", supplied[0]),)
    return tuple(
        ("args", i)
        for i in range(position, len(args) if variadic else min(position + 1, len(args)))
    )


def argument(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    name: str,
    position: int,
    default: object = None,
    *,
    target: object = None,
    variadic: bool = False,
) -> object:
    """Resolve canonical names, native aliases, and variadic positional arguments.

    FX/metadata execution has already checked the original call's signature.
    Native Tensor methods and torch functions share these accepted spellings;
    module and third-party function keywords retain their own semantics.
    This same resolver serves normalized values, raw FX operands, and compact
    execution checks. A present alias must never silently fall back to a default.
    """
    locations = argument_locations(args, kwargs, name, position, target=target, variadic=variadic)
    if locations and locations[0][0] == "kwargs":
        value = kwargs[cast(str, locations[0][1])]
        return (value,) if variadic and not isinstance(value, (tuple, list, fx.Node)) else value
    if variadic:
        return args[position:]
    return args[position] if position < len(args) else default


def tensors(value: object) -> Iterator[TensorRef]:
    """Yield tensor references from nested tuples, lists, and dictionaries.

    Args:
        value: A normalized argument or result tree.

    Yields:
        TensorRef leaves in deterministic container traversal order.
    """
    if isinstance(value, TensorRef):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


@dataclass(frozen=True)
class Operation:
    """Inputs and capture facts provided to an operation semantics rule.

    Attributes:
        node: Original FX operation node.
        args: Positional arguments with tensor values replaced by references.
        kwargs: Keyword arguments with tensor values replaced by references.
        result: Result tree containing references and scalar metadata.
        module: Called module, or None for functions and methods.
        module_paths: Original aliases; an empty string denotes the root module.
        bindings: Module-local parameter and buffer paths mapped to references.
        expressions: Read-only shape-expression table shared by inspection contexts.
        metadata: Captured tensor facts indexed by reference ID.
        constants: Small captured integer tensors, used only with declared value guards.
        graph_id: Owning snapshot identity, preserved by inspection copies.
        effects: Write/allocation declaration captured once before metadata execution.
    """

    node: fx.Node
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any
    module: nn.Module | None
    module_paths: tuple[str, ...]
    bindings: dict[str, TensorRef]
    expressions: Mapping[fx.Node, ShapeExpr]
    metadata: Mapping[str, TensorFacts]
    constants: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    graph_id: str = ""
    effects: TensorEffects = field(default_factory=lambda: TensorEffects(writes_inputs=()))

    @property
    def name(self) -> str:
        """Return the FX call identity within this graph."""
        return self.node.name

    @property
    def inputs(self) -> tuple[TensorRef, ...]:
        """Return tensor arguments in flattened container traversal order."""
        return tuple(tensors((self.args, self.kwargs)))

    @property
    def outputs(self) -> tuple[TensorRef, ...]:
        """Return tensor results in flattened container traversal order."""
        return tuple(tensors(self.result))

    def argument(
        self, name: str, position: int, default: object = None, *, variadic: bool = False
    ) -> object:
        """Resolve an argument supplied by keyword or position.

        Args:
            name: Canonical keyword name.
            position: Corresponding positional argument index.
            default: Value to use if neither spelling was supplied.
            variadic: Collect remaining positional arguments, such as view sizes.

        Returns:
            The normalized argument, preferring an explicitly supplied keyword.
        """
        return argument(
            self.args,
            self.kwargs,
            name,
            position,
            default,
            target=self.node.target if self.module is None else None,
            variadic=variadic,
        )

    def raw_argument(
        self, name: str, position: int, default: object = None, *, variadic: bool = False
    ) -> object:
        """Resolve the original FX operand using the same public-name aliases."""
        return argument(
            self.node.args,
            self.node.kwargs,
            name,
            position,
            default,
            target=self.node.target if self.module is None else None,
            variadic=variadic,
        )

    @property
    def module_path(self) -> str | None:
        """Return the canonical original module path, retaining all aliases separately."""
        return self.module_paths[0] if self.module_paths else None


@dataclass(frozen=True)
class CandidateAxis:
    """Declare a logical channel axis and its default contiguous removal blocks.

    The stable key names a structural domain, independently of FX call identity.
    Seeds use the logical axis; registered relations map them to parameter regions.
    A binding declares that repeated keys represent the same logical coordinate
    system across calls, even when the seed is a call-specific activation axis.
    An optional alignment_axis declares the corresponding logical width for
    Granularity. It must have the same original width as axis. Use it when the
    seed tensor can require partitioned packing: its physical layout cannot by
    itself establish the retained logical width. Relations must couple both axes.
    """

    key: str
    axis: AxisRef
    block_size: int = 1
    binding: TensorRef | None = None
    alignment_axis: AxisRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or not isinstance(self.axis, AxisRef):
            raise ValueError("Candidate domain requires a stable key and an AxisRef")
        if (
            not isinstance(self.block_size, int)
            or isinstance(self.block_size, bool)
            or self.block_size <= 0
        ):
            raise ValueError("Candidate block size must be a positive integer")
        if self.binding is not None and (
            not isinstance(self.binding, TensorRef)
            or self.binding.kind not in ("parameter", "buffer")
        ):
            raise ValueError("Shared candidate domains require a registered tensor binding")
        if self.alignment_axis is not None and (
            not isinstance(self.alignment_axis, AxisRef)
            or self.alignment_axis.tensor.shape[self.alignment_axis.dim]
            != self.axis.tensor.shape[self.axis.dim]
        ):
            raise ValueError("Alignment axis must be an AxisRef with the same original width")


@dataclass(frozen=True)
class PartitionedLayout:
    """Compact disjoint original partitions, then concatenate in declared order.

    Shared by dependency constraints and physical lowering. Partitions describe
    storage coordinates; retained regions preserve each original axis order.
    """

    tensor: TensorRef
    partitions: tuple[Region, ...]
    concat_dim: int = 0

    def __post_init__(self) -> None:
        partitions = tuple(self.partitions)
        dim = self.tensor.axis(self.concat_dim).dim
        if not partitions:
            raise ValueError("Partitioned layout requires partitions")
        seen = Selection(self.tensor)
        for region in partitions:
            selected = Selection(self.tensor, (region,))
            if not selected or selected.subtract(seen) != selected:
                raise ValueError("Layout partitions must be nonempty and disjoint")
            seen = seen.union(selected)
        object.__setattr__(self, "partitions", partitions)
        object.__setattr__(self, "concat_dim", dim)

    def retained_regions(self, selection: Selection) -> tuple[Region, ...]:
        """Return retained Cartesian regions without allocating tensor data."""
        if selection.tensor != self.tensor:
            raise ValueError("Selection belongs to a different layout tensor")
        result = []
        for region in self.partitions:
            axes = tuple(
                indices.subtract(selection.fully_selected_indices(dim, region))
                for dim, indices in enumerate(region.axes)
            )
            if all(axes):
                result.append(Region(axes))
        return tuple(result)


@dataclass(frozen=True)
class OutputContract:
    """Declare layout facts for original-call validation.

    Native meta execution can check shapes but cannot prove backend strides.
    Shape argument permissions are described separately by requirements.

    Attributes:
        output_layout: `backend_dependent` always leaves output strides unknown;
            `unknown` propagates input uncertainty through meta execution.
            `contiguous` establishes layout independently of input strides;
            `cast` additionally normalizes conversion arguments.
        copy_output: A cast explicitly requests a new tensor even without a dtype
            change. Conversion rules normalize the public overloads into this fact.
    """

    output_layout: Literal[
        "unknown",
        "contiguous",
        "cast",
        "backend_dependent",
    ] = "unknown"
    copy_output: bool = False

    def __post_init__(self) -> None:
        if type(self.copy_output) is not bool:
            raise TypeError("Output copy contract must be boolean")
        if self.output_layout not in (
            "unknown",
            "contiguous",
            "cast",
            "backend_dependent",
        ):
            raise ValueError("Invalid output layout")


@dataclass(frozen=True)
class OperatorSpec:
    """Structural facts emitted by a pure operation rule.

    Attributes:
        relations: Deterministic index correspondences to propagate.
        constraints: Conditions that must hold for a proposed change.
        requirements: Descriptions of later model edits; never surgery callbacks.
        candidates: Logical axes and block sizes for optional default discovery.
        layouts: Shared physical partitions used by generic compaction lowering.
        contract: Optional original-call argument/layout behavior.
        expression: Scalar dimension provenance for a dimension-producing call.
        constants: Integer references whose values must remain unchanged.
    """

    relations: tuple[Relation, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    requirements: tuple[Requirement | AttributeBinding, ...] = ()
    candidates: tuple[CandidateAxis, ...] = ()
    layouts: tuple[PartitionedLayout, ...] = ()
    contract: OutputContract | None = None
    expression: ShapeExpr | None = None
    constants: tuple[TensorRef, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "relations",
            "constraints",
            "requirements",
            "candidates",
            "layouts",
            "constants",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name, expected in (
            ("requirements", (Requirement, AttributeBinding)),
            ("candidates", CandidateAxis),
            ("layouts", PartitionedLayout),
            ("constants", TensorRef),
        ):
            if any(not isinstance(item, expected) for item in getattr(self, name)):
                raise TypeError(f"Invalid operator {name} descriptor")
        if self.contract is not None and not isinstance(self.contract, OutputContract):
            raise TypeError("Operator contract must be OutputContract")
        if self.expression is not None and not isinstance(self.expression, ShapeExpr):
            raise TypeError("Operator expression must be ShapeExpr")
        for records, method in ((self.relations, "propagate"), (self.constraints, "check")):
            if any(
                not callable(getattr(item, method, None)) or not hasattr(item, "refs")
                for item in records
            ):
                raise TypeError(f"Operator records must provide refs and {method}")


@dataclass(frozen=True)
class TensorEffects:
    """Declare writes and allocation guarantees, before metadata execution.

    `writes_inputs=None` means unknown; an empty tuple declares no input writes.
    Argument references use the same positional/keyword resolver as shape guards.
    `outputs_are_fresh` guarantees independent new storage for every Tensor
    result, including independence from model state and other returned tensors.
    False provides no allocation guarantee; it does not assert aliasing.
    Declarations must hold throughout the rule's supported compact shapes.
    """

    writes_inputs: tuple[ArgumentRef, ...] | None = None
    outputs_are_fresh: bool = False

    def __post_init__(self) -> None:
        if self.writes_inputs is not None:
            writes = tuple(self.writes_inputs)
            if any(not isinstance(ref, ArgumentRef) for ref in writes):
                raise TypeError("Input writes require ArgumentRef entries")
            object.__setattr__(self, "writes_inputs", tuple(dict.fromkeys(writes)))
        if type(self.outputs_are_fresh) is not bool:
            raise TypeError("Output allocation guarantee must be boolean")


class OperatorRule(ABC):
    """Define one operator family without retaining per-call or planning state.

    Implement `build_spec`; other hooks are optional. Inheriting `tensor_effects`
    declares that the operator does not modify its inputs. Rules that do modify
    inputs must override it. This is an extension contract, not an automatic
    proof of arbitrary Python behavior. All hooks must leave model state intact.

    Args:
        evaluate_on_meta: Permit compact-call validation on metadata tensors.
        cache_meta_output: Declare metadata evaluation depends only on argument
            metadata and captured configuration. Allocation and mutation checks
            remain mandatory even when this flag is true.
    """

    def __init__(self, *, evaluate_on_meta: bool = False, cache_meta_output: bool = False) -> None:
        self.evaluate_on_meta = evaluate_on_meta
        self.cache_meta_output = cache_meta_output

    @abstractmethod
    def build_spec(self, operation: Operation) -> OperatorSpec:
        """Describe coordinates, constraints and bound attributes for one call."""
        raise NotImplementedError

    def validate_capture(self, node: fx.Node, module: nn.Module | None) -> None:
        """Reject unsupported configurations before sample metadata execution."""
        return None

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare a read-only input contract with no output allocation guarantee."""
        return TensorEffects(writes_inputs=())

    def build_recipe(
        self, operation: Operation, spec: OperatorSpec, impact: Impact
    ) -> OperatorRecipe | None:
        """Supply special edits; omitted edits still use shared compilation.

        Return None to provide no special edits. Raise UnsupportedOperation to
        reject this request instead of falling back. Every edit is checked
        against the joint Impact and structural declarations before execution.
        """
        return None
