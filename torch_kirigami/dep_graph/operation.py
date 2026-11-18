"""Captured operation records and the public operator-rule extension contract.

An `OperatorRule` turns one captured `Operation` into an `OperatorSpec` that
describes the call's structural semantics. `TensorEffects` separately declares
which inputs the call can write, which input its result can alias, and whether
its results use fresh storage.
"""

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
    "split_size_or_sections": ("split_size", "split_sizes"),
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
    """Locate an argument by keyword or positional slot without reading its value.

    Args:
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        name: Canonical argument name.
        position: Zero-based positional slot, including a method's receiver.
        target: Call target. Native keyword aliases such as `axis` for `dim`
            apply only when it is a Tensor method name or an object defined in
            a `torch` module.
        variadic: Include all positional slots from `position` onward.

    Returns:
        Location pairs `("kwargs", name)` or `("args", index)`. A supplied
        keyword yields only its own location. A missing argument yields an
        empty tuple.

    Raises:
        ValueError: More than one spelling of a native keyword was supplied.
    """
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
    """Resolve one call argument from its keyword, native alias, or positional slot.

    Native Tensor methods and torch functions accept NumPy-style keyword
    aliases, such as `axis` for `dim` and `keepdims` for `keepdim`. Keywords
    of modules and third-party functions keep their own meaning. This resolver does not validate the complete call
    signature.

    Args:
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        name: Canonical argument name.
        position: Zero-based positional slot, including a method's receiver.
        default: Result for a missing nonvariadic argument.
        target: Call target. Native keyword aliases apply only when it is a
            Tensor method name or an object defined in a `torch` module.
        variadic: Collect remaining positional arguments instead of one value.

    Returns:
        The supplied value. A keyword takes precedence over positional slots.
        In variadic mode, positional arguments form a possibly empty tuple; a
        keyword tuple, list, or FX node is returned unchanged, and any other
        keyword value is wrapped in a one-item tuple.

    Raises:
        ValueError: More than one spelling of a native keyword was supplied.
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
        `TensorRef` leaves in container order; dictionaries yield their values
        in insertion order.
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
    """One captured call, as presented to `OperatorRule.build_spec`.

    Attributes:
        node: Original FX operation node.
        args: Positional arguments with tensor values replaced by references.
        kwargs: Keyword arguments with tensor values replaced by references.
        result: Result tree containing references and scalar metadata.
        module: Called module, or None for functions and methods.
        module_paths: Every path at which the called module is registered in the
            original model. The empty string denotes the root module. Function
            and method calls have an empty tuple.
        bindings: The called module's parameters and buffers, keyed by
            module-local path. Function and method calls have no bindings.
        expressions: Read-only map from FX nodes to the shape expressions that
            produced their dimension values.
        metadata: Captured tensor facts indexed by reference ID.
        constants: Flattened values of small captured integer and boolean
            tensors, indexed by reference ID. A rule must list a reference in
            `OperatorSpec.constants` to guard its values.
        graph_id: Identity of the owning dependency-graph snapshot.
        effects: Tensor effects that the rule declared before metadata execution.
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
        """FX node name that identifies this call within its graph."""
        return self.node.name

    @property
    def inputs(self) -> tuple[TensorRef, ...]:
        """Tensor references in `args` and then `kwargs`, in container order."""
        return tuple(tensors((self.args, self.kwargs)))

    @property
    def outputs(self) -> tuple[TensorRef, ...]:
        """Tensor references in `result`, in container order."""
        return tuple(tensors(self.result))

    def argument(
        self, name: str, position: int, default: object = None, *, variadic: bool = False
    ) -> object:
        """Resolve a normalized argument supplied by keyword or position.

        Native keyword aliases apply only to function and Tensor method calls.

        Args:
            name: Canonical keyword name.
            position: Zero-based positional slot, including a method's receiver.
            default: Value to use if neither spelling was supplied.
            variadic: Collect remaining positional arguments, such as view sizes.

        Returns:
            The normalized argument, with tensors replaced by references. A
            supplied keyword takes precedence over positional slots.

        Raises:
            ValueError: More than one spelling of a native keyword was supplied.
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
        """Resolve the original FX operand instead of a normalized value.

        Args:
            name: Canonical argument name.
            position: Zero-based positional slot, including a method's receiver.
            default: Result for a missing nonvariadic argument.
            variadic: Collect all remaining positional arguments.

        Returns:
            The original operand or container from `node.args` or `node.kwargs`,
            resolved with the same aliases and keyword precedence as `argument`.

        Raises:
            ValueError: More than one spelling of a native keyword was supplied.
        """
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
        """First entry of `module_paths`, or None for function and method calls."""
        return self.module_paths[0] if self.module_paths else None


@dataclass(frozen=True)
class CandidateAxis:
    """Logical channel axis offered for default candidate discovery.

    Default discovery splits the axis into contiguous blocks. The spec's
    relations map selections on the axis to parameter regions.

    Attributes:
        key: Nonempty structural domain name, independent of FX call identity.
        axis: Logical axis on which candidate selections start.
        block_size: Positive number of consecutive coordinates per default
            candidate.
        binding: Optional parameter or buffer reference that identifies shared
            coordinates across calls. Repeated keys with this binding describe
            the same logical domain.
        alignment_axis: Optional axis used to check retained-width granularity.
            It must have the same original width as `axis`, with relations that
            couple both axes. Use it when partitioned packing hides the logical
            width of the seed tensor.

    Raises:
        ValueError: A field violates one of the requirements above.
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
    """Tensor layout made of disjoint partitions that are compacted separately.

    Compaction shrinks each original partition independently and then
    concatenates the results in declared order. Dependency constraints and
    physical compaction share this description. Retained regions preserve the
    original order on each axis.

    Attributes:
        tensor: Tensor whose storage coordinates define the partitions.
        partitions: Nonempty, disjoint regions in concatenation order.
        concat_dim: Axis along which compact partitions concatenate. A negative
            value is normalized to a nonnegative axis index.

    Raises:
        ValueError: No partitions are given, or a partition is empty or overlaps
            an earlier partition.
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
        """Shrink each partition by the axis indices that a selection fully removes.

        Args:
            selection: Removed original coordinates on this layout's tensor.

        Returns:
            Retained Cartesian regions in partition order. An axis index is
            removed from a partition only when the selection covers its whole
            section within that partition. Partitions that lose every index on
            some axis are omitted. No tensor data is allocated.

        Raises:
            ValueError: The selection refers to another tensor.
        """
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
    """Output-layout facts used when the original call is replayed on meta tensors.

    Native meta execution can check shapes but cannot prove backend strides.
    Shape argument permissions are described separately by requirements.

    Attributes:
        output_layout: How replay treats output strides. `unknown` marks the
            outputs uncertain when any input layout is uncertain. `contiguous`
            establishes a known output layout regardless of input strides.
            `cast` replays the call as a `Tensor.to` conversion to the captured
            output dtype and otherwise behaves like `unknown`.
            `backend_dependent` always marks output strides as uncertain.
        copy_output: Whether a `cast` explicitly requests a new tensor even when
            dtype and device are unchanged. Conversion rules normalize the
            public `Tensor.to` overloads into this flag.

    Raises:
        TypeError: `copy_output` is not a `bool`.
        ValueError: `output_layout` is not one of the listed values.
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
    """Structural semantics of one call, as returned by `OperatorRule.build_spec`.

    Sequence fields are converted to tuples.

    Attributes:
        relations: Deterministic index correspondences to propagate.
        constraints: Conditions that must hold for a proposed change.
        requirements: Declarative descriptions of later model edits, never
            surgery callbacks.
        candidates: Logical axes and block sizes for optional default discovery.
        layouts: Physical partitions used by generic compaction.
        contract: Optional output-layout facts for original-call replay.
        expression: Scalar dimension provenance for a call that produces a
            dimension value.
        constants: Tensor references whose captured values must remain
            unchanged. Graph construction treats the call as unsupported
            unless each reference is a registered parameter or buffer.

    Raises:
        TypeError: A field contains a descriptor of the wrong type, or a
            relation or constraint lacks `refs` or its `propagate` or `check`
            method.
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
    """Write, aliasing, and fresh-storage declarations for one call.

    Rules return these declarations before metadata execution. They must hold
    for every original and compact shape that the rule supports.

    Attributes:
        writes_inputs: Arguments that the call can modify, identified by
            `ArgumentRef`. None means unknown writes. An empty tuple declares
            read-only inputs. Container and variadic arguments include all
            Tensor leaves, using the same argument resolver as shape guards.
        outputs_are_fresh: Whether each Tensor result has independent new storage.
            This guarantee includes independence from inputs, model state, and
            other results. False gives no guarantee and does not assert aliasing.
        mutable_parameters: Paths or argument references that permit value updates
            to registered `nn.Parameter` objects during this call. Module-owned
            parameters use relative paths. Parameter arguments use `ArgumentRef`
            entries also listed in `writes_inputs`.
            Only values may change. Identity, shape, layout, dtype, device, and
            `requires_grad` must stay unchanged. This field does not cover buffers.
        aliases_input: The only argument whose storage a result can share.
            None leaves the possible storage sources unspecified.
        returns_input: Whether the call returns the same Tensor object named by
            `aliases_input`, with unchanged storage, shape, strides, dtype, and
            device. Set this only for a single Tensor result.

    Raises:
        TypeError: A field has the wrong type.
        ValueError: `aliases_input` is variadic or combined with fresh outputs,
            `returns_input` is set without `aliases_input`, a parameter path is
            empty, or a parameter `ArgumentRef` is missing from `writes_inputs`.

    Notes:
        Capture does not copy parameters. It warns before metadata execution
        when a call declares parameter updates, and the updated values remain
        in the original model even if capture later fails. Capture isolates
        registered buffers, so buffer writes need no `mutable_parameters`
        entry; declare writes to buffer arguments in `writes_inputs`.
    """

    writes_inputs: tuple[ArgumentRef, ...] | None = None
    outputs_are_fresh: bool = False
    mutable_parameters: tuple[str | ArgumentRef, ...] = ()
    aliases_input: ArgumentRef | None = None
    returns_input: bool = False

    def __post_init__(self) -> None:
        if self.writes_inputs is not None:
            writes = tuple(self.writes_inputs)
            if any(not isinstance(ref, ArgumentRef) for ref in writes):
                raise TypeError("Input writes require ArgumentRef entries")
            object.__setattr__(self, "writes_inputs", tuple(dict.fromkeys(writes)))
        if type(self.outputs_are_fresh) is not bool:
            raise TypeError("Output allocation guarantee must be boolean")
        if type(self.returns_input) is not bool:
            raise TypeError("Input return guarantee must be boolean")
        if self.aliases_input is not None:
            if not isinstance(self.aliases_input, ArgumentRef):
                raise TypeError("Output alias requires an ArgumentRef")
            if self.outputs_are_fresh or self.aliases_input.variadic:
                raise ValueError("Output alias requires a single argument and non-fresh output")
        elif self.returns_input:
            raise ValueError("Input return guarantee requires an alias argument")
        if isinstance(self.mutable_parameters, str):
            raise TypeError("Parameter updates require a sequence of paths or ArgumentRef entries")
        updates = tuple(self.mutable_parameters)
        for ref in updates:
            if isinstance(ref, str):
                if not ref:
                    raise ValueError("Parameter update paths must not be empty")
            elif isinstance(ref, ArgumentRef):
                if self.writes_inputs is None or ref not in self.writes_inputs:
                    raise ValueError("Parameter update arguments must also declare an input write")
            else:
                raise TypeError("Parameter updates require module paths or ArgumentRef entries")
        object.__setattr__(self, "mutable_parameters", tuple(dict.fromkeys(updates)))


class OperatorRule(ABC):
    """Base class for the semantics of one operator family.

    A rule describes each call's structural semantics through `build_spec` and
    its tensor effects through `tensor_effects`. It must not retain per-call or
    planning state. Subclasses must implement `build_spec`; the other hooks are
    optional. The inherited `tensor_effects` declares read-only inputs, so
    rules for operators that modify inputs must override it. The library
    trusts these declarations; they are an extension contract, not an
    automatic proof of arbitrary Python behavior. No hook may modify model
    state.

    Args:
        evaluate_on_meta: Permit compact-call validation on metadata tensors.
        cache_meta_output: Declare that metadata evaluation depends only on
            argument metadata and captured configuration. Allocation and
            mutation checks still run when this flag is true.
    """

    # Defaults also apply when an extension initializes only its own configuration.
    evaluate_on_meta: bool = False
    cache_meta_output: bool = False

    def __init__(self, *, evaluate_on_meta: bool = False, cache_meta_output: bool = False) -> None:
        self.evaluate_on_meta = evaluate_on_meta
        self.cache_meta_output = cache_meta_output

    @abstractmethod
    def build_spec(self, operation: Operation) -> OperatorSpec:
        """Describe the coordinates, constraints, and bound attributes of one call.

        Args:
            operation: Captured call with normalized values and tensor metadata.

        Returns:
            Structural facts independent of scoring and model mutation.

        Raises:
            torch_kirigami.UnsupportedOperation: The rule cannot analyze this call.
                Graph construction records a diagnostic and places a barrier on
                the call's tensors and data ancestors. Any other exception
                propagates and stops graph construction.
        """
        raise NotImplementedError

    def validate_capture(self, node: fx.Node, module: nn.Module | None) -> None:
        """Check a call before metadata execution on the example inputs.

        The default accepts every call.

        Args:
            node: Captured FX call.
            module: Called module, or None for a function or Tensor method.

        Raises:
            torch_kirigami.CaptureError: The call cannot execute under the capture
                contract. This stops graph construction instead of creating a
                local barrier.
        """
        return None

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare writes and output storage guarantees for one call.

        Capture calls this hook before metadata execution.

        Args:
            node: Captured FX call.
            module: Called module, or None for a function or Tensor method.

        Returns:
            Effect declarations valid for supported original and compact shapes.
            The default declares read-only inputs and no allocation guarantee.
        """
        return TensorEffects(writes_inputs=())

    def build_recipe(
        self, operation: Operation, spec: OperatorSpec, impact: Impact
    ) -> OperatorRecipe | None:
        """Supply operator-specific edits; the shared compiler checks still apply.

        Args:
            operation: Captured call and original bindings.
            spec: Structural facts produced by `build_spec`.
            impact: Joint removal selections and their dependency closure.

        Returns:
            Additional declarative edits, or None to use only shared compilation.
            Returning None does not declare the request executable.

        Raises:
            torch_kirigami.UnsupportedOperation: The rule rejects this request. The compiler
                must not use a generic replacement to bypass this rejection.
        """
        return None
