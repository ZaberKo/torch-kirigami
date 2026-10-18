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
    """Locate an argument by name or positional slot without comparing its value.

    Args:
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        name: Canonical argument name.
        position: Zero-based positional slot, including a method's receiver.
        target: Function or method name used to identify native keyword aliases.
        variadic: Include all positional slots from `position` onward.

    Returns:
        Pairs of argument container name and key or index. Missing arguments
        produce an empty tuple. An explicit keyword takes precedence.

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
    """Resolve canonical names, native aliases, and variadic positional arguments.

    Native Tensor methods and torch functions share accepted keyword aliases.
    Module and third-party function keywords retain their own semantics.
    This resolver does not validate the complete call signature.

    Args:
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        name: Canonical argument name.
        position: Zero-based positional slot, including a method's receiver.
        default: Result for a missing nonvariadic argument.
        target: Function or method name used to identify native keyword aliases.
        variadic: Collect remaining positional arguments instead of one value.

    Returns:
        The supplied value, with an explicit keyword taking precedence. Variadic
        positional arguments form a tuple. Variadic keyword containers remain
        containers, and noncontainer keyword values other than FX nodes are wrapped.

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
        """Resolve an original FX operand instead of a normalized tensor reference.

        Args:
            name: Canonical argument name.
            position: Zero-based positional slot, including a method's receiver.
            default: Result for a missing nonvariadic argument.
            variadic: Collect all remaining positional arguments.

        Returns:
            The original operand or container, using the same aliases and
            keyword precedence as `argument`.
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
        """Return the canonical original module path, retaining all aliases separately."""
        return self.module_paths[0] if self.module_paths else None


@dataclass(frozen=True)
class CandidateAxis:
    """Declare a logical channel axis and its default contiguous removal blocks.

    Registered relations map selections on the logical axis to parameter regions.

    Attributes:
        key: Stable structural domain name, independent of FX call identity.
        axis: Logical axis on which candidate selections start.
        block_size: Number of consecutive coordinates per default candidate.
        binding: Registered tensor that identifies shared coordinates across calls.
            Repeated keys with this binding describe the same logical domain.
        alignment_axis: Optional axis used to check retained-width granularity.
            It must have the same original width as `axis`, with relations that
            couple both axes. Use it when partitioned packing hides the logical
            width of the seed tensor.
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

    Dependency constraints and physical compaction share this description.
    Retained regions preserve the original order on each axis.

    Attributes:
        tensor: Tensor whose storage coordinates define the partitions.
        partitions: Nonempty, disjoint regions in concatenation order.
        concat_dim: Axis along which compact partitions concatenate.
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
        """Return each nonempty partition after complete axis sections are removed.

        Args:
            selection: Removed original coordinates on this layout's tensor.

        Returns:
            Retained Cartesian regions in partition order. Partial axis sections
            do not remove an axis index. No tensor data is allocated.

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

    Declarations must hold throughout the rule's supported compact shapes.

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

    Notes:
        Capture warns before permitted parameter updates. These changes remain
        in the original model, including when capture fails. Capture isolates
        buffers separately. `writes_inputs` describes writes to buffer arguments.
    """

    writes_inputs: tuple[ArgumentRef, ...] | None = None
    outputs_are_fresh: bool = False
    mutable_parameters: tuple[str | ArgumentRef, ...] = ()

    def __post_init__(self) -> None:
        if self.writes_inputs is not None:
            writes = tuple(self.writes_inputs)
            if any(not isinstance(ref, ArgumentRef) for ref in writes):
                raise TypeError("Input writes require ArgumentRef entries")
            object.__setattr__(self, "writes_inputs", tuple(dict.fromkeys(writes)))
        if type(self.outputs_are_fresh) is not bool:
            raise TypeError("Output allocation guarantee must be boolean")
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

    # Defaults also apply when an extension initializes only its own configuration.
    evaluate_on_meta: bool = False
    cache_meta_output: bool = False

    def __init__(self, *, evaluate_on_meta: bool = False, cache_meta_output: bool = False) -> None:
        self.evaluate_on_meta = evaluate_on_meta
        self.cache_meta_output = cache_meta_output

    @abstractmethod
    def build_spec(self, operation: Operation) -> OperatorSpec:
        """Describe coordinates, constraints and bound attributes for one call.

        Args:
            operation: Captured call with normalized values and tensor metadata.

        Returns:
            Structural facts independent of scoring and model mutation.

        Raises:
            UnsupportedOperation: The rule cannot analyze this call. Analysis
                retains a diagnostic and blocks affected dependency requests.
        """
        raise NotImplementedError

    def validate_capture(self, node: fx.Node, module: nn.Module | None) -> None:
        """Check a call before sample metadata execution.

        Args:
            node: Captured FX call.
            module: Called module, or None for a function or Tensor method.

        Raises:
            CaptureError: The call cannot execute under the capture contract.
                This stops graph construction, rather than creating a local barrier.
        """
        return None

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare writes and output allocation guarantees for one call.

        Args:
            node: Captured FX call, before sample metadata execution.
            module: Called module, or None for a function or Tensor method.

        Returns:
            Effect declarations valid for supported original and compact shapes.
            The default declares read-only inputs and no allocation guarantee.
        """
        return TensorEffects(writes_inputs=())

    def build_recipe(
        self, operation: Operation, spec: OperatorSpec, impact: Impact
    ) -> OperatorRecipe | None:
        """Supply special edits while retaining the shared compiler checks.

        Args:
            operation: Captured call and original bindings.
            spec: Structural facts produced by `build_spec`.
            impact: Joint removal selections and their dependency closure.

        Returns:
            Additional declarative edits, or None to use only shared compilation.
            Returning None does not declare the request executable.

        Raises:
            UnsupportedOperation: The rule rejects this request. The compiler
                must not use a generic replacement to bypass this rejection.
        """
        return None
