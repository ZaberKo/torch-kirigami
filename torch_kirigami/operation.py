"""Shared operation facts and rule contracts, independent of orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import fx, nn

from .contracts import Constraint, Requirement, ShapeExpr
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


def argument_locations(args, kwargs, name, position, *, target=None, variadic=False):
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


def argument(args, kwargs, name, position, default=None, *, target=None, variadic=False):
    """Resolve canonical names, native aliases, and variadic positional arguments.

    FX/metadata execution has already checked the original call's signature.
    Native Tensor methods and torch functions share these accepted spellings;
    module and third-party function keywords retain their own semantics.
    This same resolver serves normalized values, raw FX operands, and compact
    execution checks. A present alias must never silently fall back to a default.
    """
    locations = argument_locations(args, kwargs, name, position, target=target, variadic=variadic)
    if locations and locations[0][0] == "kwargs":
        value = kwargs[locations[0][1]]
        return (value,) if variadic and not isinstance(value, (tuple, list, fx.Node)) else value
    if variadic:
        return args[position:]
    return args[position] if position < len(args) else default


def tensors(value):
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
class OperationContext:
    """Inputs and capture facts provided to an operation semantics rule.

    Attributes:
        node: Original FX operation node.
        args: Positional arguments with tensor values replaced by references.
        kwargs: Keyword arguments with tensor values replaced by references.
        output: Result tree containing references and scalar metadata.
        module: Called module, or None for functions and methods.
        module_path: An original module alias; an empty string denotes the root.
        bindings: Module-local parameter and buffer paths mapped to references.
        expressions: Read-only shape-expression table shared by inspection contexts.
        metadata: Captured tensor facts indexed by reference ID.
        constants: Small captured integer tensors, used only with declared value guards.
        graph_id: Owning snapshot identity, preserved by inspection copies.
    """

    node: fx.Node
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any
    module: nn.Module | None
    module_path: str | None
    bindings: dict[str, TensorRef]
    expressions: Mapping[fx.Node, ShapeExpr]
    metadata: Mapping[str, TensorFacts]
    constants: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    graph_id: str = ""

    @property
    def inputs(self):
        """Return tensor arguments in flattened container traversal order."""
        return tuple(tensors((self.args, self.kwargs)))

    @property
    def outputs(self):
        """Return tensor results in flattened container traversal order."""
        return tuple(tensors(self.output))

    def argument(self, name: str, position: int, default=None, *, variadic=False):
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

    def raw_argument(self, name, position, default=None, *, variadic=False):
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

    def binding(self, name: str) -> TensorRef | None:
        """Return a module-local parameter or buffer binding, or None."""
        return self.bindings.get(name)


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

    def __post_init__(self):
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

    def __post_init__(self):
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
        output_layout: ``backend_dependent`` always leaves output strides unknown;
            ``unknown`` propagates input uncertainty through meta execution.
            ``contiguous`` establishes layout independently of input strides;
            ``cast`` additionally normalizes conversion arguments.
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

    def __post_init__(self):
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
    requirements: tuple[Requirement, ...] = ()
    candidates: tuple[CandidateAxis, ...] = ()
    layouts: tuple[PartitionedLayout, ...] = ()
    contract: OutputContract | None = None
    expression: ShapeExpr | None = None
    constants: tuple[TensorRef, ...] = ()

    def __post_init__(self):
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
            ("requirements", Requirement),
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
class CallEffects:
    """Effects available before capture for isolation and downstream alias safety.

    Native registrations declare allocation alongside their semantic rule.
    Capture and execution consume this same contract; OutputContract only
    describes layout and cannot replace pre-execution mutation checks.
    """

    mutates_input: bool = False
    fresh_output: bool = False


class OperatorRule:
    """One definition for capture checks, structural analysis, and lowering.

    Args:
        analyze: Pure callback producing an OperatorSpec from capture metadata.
        lower: Optional pure callback producing declarative rewrite descriptions.
        preflight: Optional callback accepting an FX node and its called module
            before metadata execution. Raise CaptureError for recognized writes.
        effects: Optional callback with the same arguments, returning CallEffects.
        evaluate_on_meta: Whether the native operation is safe to evaluate on meta tensors.
            Third-party callbacks default to declared output facts instead.
    """

    def __init__(
        self, analyze=None, *, lower=None, preflight=None, effects=None, evaluate_on_meta=False
    ):
        self._analyze = analyze
        self._lower = lower
        self._preflight = preflight
        self._effects = effects
        self.evaluate_on_meta = evaluate_on_meta

    def analyze(self, context: OperationContext) -> OperatorSpec:
        """Produce shared structural facts without modifying model state."""
        if self._analyze is None:
            raise NotImplementedError("Implement OperatorRule.analyze")
        return self._analyze(context)

    def preflight(self, node, module) -> None:
        """Check effects using call arguments and configuration, before ShapeProp."""
        if self._preflight is not None:
            self._preflight(node, module)

    def effects(self, node, module) -> CallEffects:
        """Describe writes/copies without requiring output metadata."""
        return self._effects(node, module) if self._effects is not None else CallEffects()

    def lower(self, context):
        """Return custom recipes, or None for the shared descriptor compiler.

        Default compilation belongs to pruning. Declaring standard structural
        semantics does not require importing an executor into the analysis layer.
        """
        return self._lower(context) if self._lower is not None else None
