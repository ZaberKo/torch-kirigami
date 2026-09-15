"""Original-call contracts and conservative metadata/layout validation."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn

from ..configuration import thaw
from ..contracts import Impact
from ..graph import DependencyGraph
from ..operation import OperationContext, argument
from ..operators.coordinates import narrow_index, retained_indices
from ..operators.shapes import evaluate, reevaluate
from ..selection import TensorRef
from .layouts import view_preserves_stride_boundaries
from .recipes import compact_stride
from .types import (
    AttributeRecipe,
    PlanningError,
    RewriteContext,
    TensorRecipe,
    require_compact_shape,
)


def _check_slice_coordinates(ctx: RewriteContext, new_index: object) -> None:
    """Reject slices that select different original coordinates after compaction."""
    op, impact = ctx.operation, ctx.impact
    x, y = op.inputs[0], op.outputs[0]
    expanded = dict(next(r for r in ctx.requirements if r.kind == "slice_arguments").data)["index"]
    new_raw = new_index if isinstance(new_index, tuple) else (new_index,)
    new_expanded = []
    for item in new_raw:
        new_expanded.extend(
            [slice(None)] * (len(x.shape) - len(new_raw) + 1) if item is Ellipsis else [item]
        )
    new_expanded.extend([slice(None)] * (len(x.shape) - len(new_expanded)))
    out_dim = 0
    for dim, (item, new_item) in enumerate(zip(expanded, new_expanded, strict=True)):
        old = list(range(x.shape[dim]))
        kept = list(retained_indices(impact, x.axis(dim)))
        if isinstance(item, int):
            if not -len(kept) <= new_item < len(kept) or kept[new_item] != old[item]:
                raise PlanningError(
                    f"{op.node.name}: integer slice selects a different original coordinate"
                )
        else:
            expected = [old[item][i] for i in retained_indices(impact, y.axis(out_dim))]
            if kept[new_item] != expected:
                raise PlanningError(
                    f"{op.node.name}: slice has incorrect original-coordinate correspondence"
                )
            out_dim += 1


def _meta_module(
    operation: OperationContext,
    attributes: Mapping[str, AttributeRecipe],
    tensor: Callable[[TensorRef], torch.Tensor],
) -> nn.Module:
    """Copy a built-in module structure with compact meta bindings and attributes.

    Parameters and buffers are resolved from analysis metadata, never copied from
    live weights. Registered containers are copied before changing any binding.
    """

    def clone(original: nn.Module, prefix: str = "") -> nn.Module:
        """Copy a module subtree while replacing registered tensor bindings."""
        result = copy.copy(original)
        for field in ("_parameters", "_buffers"):
            bindings = {}
            for name, value in getattr(original, field).items():
                ref = operation.bindings.get(prefix + name)
                bindings[name] = tensor(ref) if ref is not None else value
            setattr(result, field, bindings)
        result._modules = {
            name: clone(child, prefix + name + ".") if child is not None else None
            for name, child in original._modules.items()
        }
        owner_path = ".".join(part for part in (operation.module_path, prefix.rstrip(".")) if part)
        for edit in attributes.values():
            parent, _, name = edit.path.rpartition(".")
            if parent == owner_path:
                object.__setattr__(result, name, thaw(edit.new))
        return result

    return clone(operation.module)


def check_forward(
    graph: DependencyGraph,
    operations: Sequence[OperationContext],
    active: Iterable[tuple[RewriteContext, bool]],
    impact: Impact,
    recipes: Mapping[str, TensorRecipe],
    attributes: Mapping[str, AttributeRecipe],
    strides: Mapping[str, tuple[int, ...]],
) -> None:
    """Verify declared call contracts using compact metadata and original coordinates."""
    active_by_name = {ctx.operation.node.name: (ctx, builtin) for ctx, builtin in active}
    operations_by_name = {op.node.name: op for op in operations}
    values, uncertain = {}, set()

    def shape(ref: TensorRef) -> tuple[int, ...]:
        """Return the compact shape for one tensor reference."""
        return recipes[ref.id].shape if ref.id in recipes else require_compact_shape(impact, ref)

    def tensor(ref: TensorRef) -> torch.Tensor:
        """Materialize metadata-only tensor facts for one reference."""
        if ref.id not in values:
            facts = graph.metadata(ref)
            size = shape(ref)
            if ref.id in recipes:
                values[ref.id] = torch.empty_strided(
                    size,
                    compact_stride(size, recipes[ref.id].memory_format),
                    dtype=facts.dtype,
                    device="meta",
                )
            elif size == ref.shape:
                values[ref.id] = torch.empty_strided(
                    size, facts.stride, dtype=facts.dtype, device="meta"
                )
            else:
                values[ref.id] = torch.empty(size, dtype=facts.dtype, device="meta")
                uncertain.add(ref.id)  # Caller-changed input strides are not specified.
        return values[ref.id]

    def tree(value: object) -> object:
        """Replace tensor references recursively inside call arguments."""
        if isinstance(value, TensorRef):
            return tensor(value)
        if isinstance(value, tuple):
            return tuple(tree(v) for v in value)
        if isinstance(value, list):
            return [tree(v) for v in value]
        if isinstance(value, dict):
            return {k: tree(v) for k, v in value.items()}
        return value

    def record(refs: object, output: object, shape_hint: str) -> None:
        """Record and validate outputs from a compact metadata forward call."""
        if isinstance(refs, TensorRef):
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != shape(refs):
                raise PlanningError(
                    f"Original forward produces the wrong compact shape at {refs.id}{shape_hint}"
                )
            # Meta has no CPU/CUDA autocast dispatch. Keep the captured execution
            # dtype at every port so affected and unaffected branches agree.
            # Tensor.to can compact a strided view: rebuilding metadata preserves
            # the stride proof rather than silently strengthening it during a cast.
            dtype = graph.metadata(refs).dtype
            values[refs.id] = (
                output
                if output.dtype == dtype
                else torch.empty_strided(output.shape, output.stride(), dtype=dtype, device="meta")
            )
        elif isinstance(refs, (tuple, list)):
            if len(refs) != len(output):
                raise PlanningError("Original forward changes output port count")
            for r, v in zip(refs, output, strict=True):
                record(r, v, shape_hint)
        elif isinstance(refs, dict):
            for key, r in refs.items():
                record(r, output[key], shape_hint)

    for op in operations:
        entry = active_by_name.get(op.node.name)
        if entry is None:
            continue
        ctx, builtin = entry
        if ctx.spec.expression is not None:
            continue
        if not builtin:
            for ref in op.outputs:
                if ref.id in strides:
                    values[ref.id] = torch.empty_strided(
                        shape(ref), strides[ref.id], dtype=graph.metadata(ref).dtype, device="meta"
                    )
                else:
                    tensor(ref)
                    uncertain.add(ref.id)
            continue
        normalized_args = reevaluate(op.node.args, op.args, op.expressions, shape)
        normalized_kwargs = reevaluate(op.node.kwargs, op.kwargs, op.expressions, shape)
        for req in ctx.requirements:
            data = dict(req.data)
            if req.kind == "index_arguments":
                kept = list(retained_indices(impact, data["axis"]))
                if any(i >= len(kept) or kept[i] != i for i in data["indices"]):
                    raise PlanningError(
                        f"{op.node.name}: static indices change original coordinates"
                    )
            if req.kind == "slice_arguments":
                if "narrow_dim" in data:
                    dim = argument(
                        normalized_args, normalized_kwargs, "dim", 1, target=op.node.target
                    ) % len(op.inputs[0].shape)
                    if dim != data["narrow_dim"]:
                        raise PlanningError("narrow dimension changed")
                    start = argument(
                        normalized_args, normalized_kwargs, "start", 2, target=op.node.target
                    )
                    length = argument(
                        normalized_args, normalized_kwargs, "length", 3, target=op.node.target
                    )
                    try:
                        index = narrow_index(shape(op.inputs[0]), dim, start, length)
                    except ValueError as error:
                        raise PlanningError(f"{op.node.name}: {error}") from error
                else:
                    index = normalized_args[1]
                _check_slice_coordinates(ctx, tuple(index) if isinstance(index, list) else index)
            if req.kind == "partition_arguments":
                axis = data["axis"]
                # AxisPort identity is unchanged only if the original static split
                # boundaries still match each retained old partition in order.
                kept = list(retained_indices(impact, axis))
                if not data["unbound"]:
                    if "chunks" in data:
                        chunks = data["chunks"]
                        sections = (len(kept) + chunks - 1) // chunks
                    else:
                        binding = req.arguments[0]
                        sections = argument(
                            normalized_args,
                            normalized_kwargs,
                            binding.name,
                            binding.position,
                            target=op.node.target,
                        )
                    sizes = (
                        [min(sections, len(kept) - i) for i in range(0, len(kept), sections)]
                        if isinstance(sections, int)
                        else list(sections)
                    )
                    expected = [shape(r)[axis.dim] for r in op.outputs]
                    if sizes != expected or sum(sizes) != len(kept):
                        raise PlanningError(
                            f"{op.node.name}: static split needs original-forward editing"
                        )
        args, kwargs = tree(normalized_args), tree(normalized_kwargs)
        shape_hint = ""
        for req in ctx.requirements:
            data = dict(req.data)
            if req.kind == "shape_arguments":
                dimensions = evaluate(data["expression"], shape)
                while (
                    isinstance(dimensions, tuple)
                    and len(dimensions) == 1
                    and isinstance(dimensions[0], tuple)
                ):
                    dimensions = dimensions[0]
                expected = shape(op.outputs[0])
                if len(dimensions) != len(expected) or any(
                    requested != -1 and requested != size
                    for requested, size in zip(dimensions, expected, strict=False)
                ):
                    shape_hint = (
                        " If a fixed number was intended to follow a tensor dimension, "
                        "replace it with tensor.size(dim) or a valid -1 inference, then "
                        "rebuild the dependency graph. Keep algorithmic constants fixed."
                    )
                if (
                    data["requires_view"]
                    and any(r.id in uncertain for r in op.inputs)
                    and not view_preserves_stride_boundaries(
                        shape(op.inputs[0]), shape(op.outputs[0])
                    )
                ):
                    raise PlanningError(
                        f"{op.node.name}: input stride cannot be proved for view after pruning. "
                        "If a copy is acceptable, use reshape(...) or contiguous().view(...) "
                        "in forward and rebuild the dependency graph; shape and index "
                        "constraints still apply."
                    )
                if not data["unpack_shape"]:
                    args, kwargs = (tensor(op.inputs[0]), dimensions), {}
                else:
                    args, kwargs = (tensor(op.inputs[0]), *dimensions), {}
        inplace = graph.operator_rule(op).effects(op.node, op.module).mutates_input
        if inplace:
            source = op.raw_argument("input", 0)
            producer = operations_by_name.get(getattr(source, "name", None))
            fresh = (
                producer is not None
                and graph.operator_rule(producer)
                .effects(producer.node, producer.module)
                .fresh_output
            )
            if not fresh or len(source.users) != 1:
                raise PlanningError(
                    f"{op.node.name}: cannot prove in-place alias/consumer safety. "
                    "If no consumer relies on modifying the original tensor, use an "
                    "out-of-place operation and rebuild the dependency graph."
                )
        contract = ctx.spec.contract
        try:
            if contract is not None and contract.output_layout == "cast":
                output = tensor(op.inputs[0]).to(
                    device="meta",
                    dtype=graph.metadata(op.outputs[0]).dtype,
                    memory_format=op.kwargs.get("memory_format", torch.preserve_format),
                    copy=contract.copy_output
                    or graph.metadata(op.inputs[0]).device != graph.metadata(op.outputs[0]).device,
                )
            elif op.module is not None:
                module = _meta_module(op, attributes, tensor)
                output = module.forward(*args, **kwargs)  # Exact built-in types only; bypass hooks.
            elif op.node.op == "call_method":
                output = getattr(args[0], str(op.node.target))(*args[1:], **kwargs)
            else:
                output = op.node.target(*args, **kwargs)
            record(op.output, output, shape_hint)
        except (RuntimeError, ValueError, TypeError, IndexError, NotImplementedError) as error:
            raise PlanningError(
                f"{op.node.name}: original forward is not proved executable: {error}{shape_hint}"
            ) from error
        # Shape metadata alone does not establish actual strides when an input
        # layout was unspecified. Linear and explicit contiguous establish a
        # known output layout; otherwise preserve that uncertainty downstream.
        establishes_layout = contract is not None and contract.output_layout == "contiguous"
        if not establishes_layout and any(r.id in uncertain for r in op.inputs):
            uncertain.update(r.id for r in op.outputs)
        if contract is not None and contract.output_layout == "backend_dependent":
            uncertain.update(r.id for r in op.outputs)
