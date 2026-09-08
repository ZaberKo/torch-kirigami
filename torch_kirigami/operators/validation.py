"""Original-call contracts and conservative metadata/layout validation."""

from __future__ import annotations

import copy

import torch

from ..pruning.types import PlanningError
from ..selection import IndexSet, TensorRef
from .shapes import evaluate as _expression
from .shapes import reevaluate as _reevaluate


def _keep(impact, axis):
    return IndexSet.span(0, axis.tensor.shape[axis.dim]).subtract(
        impact.selection(axis.tensor).project(axis.dim)
    )


def _slice_coordinates(ctx, new_index):
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
        kept = list(_keep(impact, x.axis(dim)))
        if isinstance(item, int):
            if not -len(kept) <= new_item < len(kept) or kept[new_item] != old[item]:
                raise PlanningError(
                    f"{op.node.name}: integer slice selects a different original coordinate"
                )
        else:
            expected = [old[item][i] for i in _keep(impact, y.axis(out_dim))]
            if kept[new_item] != expected:
                raise PlanningError(
                    f"{op.node.name}: slice has incorrect original-coordinate correspondence"
                )
            out_dim += 1


def check_forward(graph, operations, active, impact, recipes, attributes, strides):
    """Verify declared call contracts using compact metadata and original coordinates."""
    from ..pruning.rewrite import RewriteContext

    active_by_name = {ctx.operation.node.name: (ctx, builtin) for ctx, builtin in active}
    values, uncertain = {}, set()

    def shape(ref):
        return (
            recipes[ref.id].shape
            if ref.id in recipes
            else RewriteContext(graph, None, impact, ()).shape(ref)
        )

    def tensor(ref):
        if ref.id not in values:
            facts = graph.metadata(ref)
            size = shape(ref)
            if ref.id in recipes:
                values[ref.id] = torch.empty(size, dtype=facts.dtype, device="meta").contiguous(
                    memory_format=torch.contiguous_format
                    if recipes[ref.id].memory_format == "contiguous"
                    else getattr(torch, recipes[ref.id].memory_format)
                )
            elif size == ref.shape:
                values[ref.id] = torch.empty_strided(
                    size, facts.stride, dtype=facts.dtype, device="meta"
                )
            else:
                values[ref.id] = torch.empty(size, dtype=facts.dtype, device="meta")
                uncertain.add(ref.id)  # Caller-changed input strides are not specified.
        return values[ref.id]

    def tree(value):
        if isinstance(value, TensorRef):
            return tensor(value)
        if isinstance(value, tuple):
            return tuple(tree(v) for v in value)
        if isinstance(value, list):
            return [tree(v) for v in value]
        if isinstance(value, dict):
            return {k: tree(v) for k, v in value.items()}
        return value

    def record(refs, output):
        if isinstance(refs, TensorRef):
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != shape(refs):
                raise PlanningError(
                    f"Original forward produces the wrong compact shape at {refs.id}"
                )
            values[refs.id] = output
        elif isinstance(refs, (tuple, list)):
            if len(refs) != len(output):
                raise PlanningError("Original forward changes output port count")
            for r, v in zip(refs, output, strict=True):
                record(r, v)
        elif isinstance(refs, dict):
            for key, r in refs.items():
                record(r, output[key])

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
        normalized_args = _reevaluate(op.node.args, op.args, op.expressions, shape)
        normalized_kwargs = _reevaluate(op.node.kwargs, op.kwargs, op.expressions, shape)
        for req in ctx.requirements:
            if req.kind == "index_arguments":
                data = dict(req.data)
                kept = list(_keep(impact, data["axis"]))
                if any(i >= len(kept) or kept[i] != i for i in data["indices"]):
                    raise PlanningError(
                        f"{op.node.name}: static indices change original coordinates"
                    )
            if req.kind == "slice_arguments":
                data = dict(req.data)
                if "narrow_dim" in data:
                    dim = normalized_kwargs.get("dim", normalized_args[1]) % len(op.inputs[0].shape)
                    if dim != data["narrow_dim"]:
                        raise PlanningError("narrow dimension changed")
                    start = normalized_kwargs.get("start", normalized_args[2])
                    length = normalized_kwargs.get("length", normalized_args[3])
                    index = [slice(None)] * len(op.inputs[0].shape)
                    index[dim] = slice(start, start + length)
                else:
                    index = normalized_args[1]
                _slice_coordinates(ctx, tuple(index) if isinstance(index, list) else index)
            if req.kind == "partition_arguments":
                axis = dict(req.data)["axis"]
                # Port identity is unchanged only if the original static split
                # boundaries still match each retained old partition in order.
                kept = list(_keep(impact, axis))
                if not dict(req.data)["unbound"]:
                    name, position = dict(req.data)["argument"]
                    sections = normalized_kwargs.get(
                        name,
                        normalized_args[position] if len(normalized_args) > position else None,
                    )
                    if "chunks" in dict(req.data):
                        chunks = dict(req.data)["chunks"]
                        sections = (len(kept) + chunks - 1) // chunks
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
        for req in ctx.requirements:
            if req.kind == "shape_arguments":
                dimensions = _expression(dict(req.data)["expression"], shape)
                while (
                    isinstance(dimensions, tuple)
                    and len(dimensions) == 1
                    and isinstance(dimensions[0], tuple)
                ):
                    dimensions = dimensions[0]
                if dict(req.data)["requires_view"] and any(r.id in uncertain for r in op.inputs):
                    raise PlanningError(f"{op.node.name}: input stride cannot be proved")
                if not dict(req.data)["unpack_shape"]:
                    args, kwargs = (tensor(op.inputs[0]), dimensions), {}
                else:
                    args, kwargs = (tensor(op.inputs[0]), *dimensions), {}
        inplace = graph.operator_rule(op).effects(op.node, op.module).mutates_input
        if inplace:
            source = op.node.args[0] if op.node.args else None
            producer = next(
                (p for p in operations if p.node.name == getattr(source, "name", None)), None
            )
            contract = graph.operator_spec(producer).contract if producer is not None else None
            fresh = contract is not None and contract.fresh_output
            if not fresh or len(source.users) != 1:
                raise PlanningError(f"{op.node.name}: cannot prove in-place alias/consumer safety")
        try:
            if ctx.spec.contract is not None and ctx.spec.contract.output_layout == "cast":
                output = tensor(op.inputs[0]).to(
                    device="meta",
                    dtype=graph.metadata(op.outputs[0]).dtype,
                    memory_format=op.kwargs.get("memory_format", torch.preserve_format),
                )
            elif op.module is not None:

                def shell(original, prefix="", operation=op):
                    result = copy.copy(original)
                    for field in ("_parameters", "_buffers"):
                        setattr(
                            result,
                            field,
                            {
                                n: tree(operation.bindings[prefix + n])
                                if prefix + n in operation.bindings
                                else v
                                for n, v in getattr(original, field).items()
                            },
                        )
                    result._modules = {
                        n: shell(child, prefix + n + ".") if child is not None else None
                        for n, child in original._modules.items()
                    }
                    owner_path = ".".join(
                        p for p in (operation.module_path, prefix.rstrip(".")) if p
                    )
                    for attr in attributes.values():
                        if attr.path.rpartition(".")[0] == owner_path:
                            object.__setattr__(result, attr.path.rpartition(".")[2], attr.new)
                    return result

                module = shell(op.module)
                output = module.forward(*args, **kwargs)  # Exact built-in types only; bypass hooks.
            elif op.node.op == "call_method":
                output = getattr(args[0], str(op.node.target))(*args[1:], **kwargs)
            else:
                output = op.node.target(*args, **kwargs)
            contract = ctx.spec.contract
            if contract is not None and contract.output_layout == "convolution":
                rank = len(op.outputs[0].shape)
                fmt = (
                    torch.channels_last
                    if rank == 4
                    else torch.channels_last_3d
                    if rank == 5
                    else None
                )
                if fmt is not None and any(
                    tensor(r).ndim == rank
                    and tensor(r).is_contiguous(memory_format=fmt)
                    and not tensor(r).is_contiguous()
                    for r in (*op.inputs, *op.bindings.values())
                ):
                    output = output.contiguous(memory_format=fmt)
            record(op.output, output)
        except (RuntimeError, ValueError, TypeError, IndexError, NotImplementedError) as error:
            raise PlanningError(
                f"{op.node.name}: original forward is not proved executable: {error}"
            ) from error
        # Shape metadata alone does not establish actual strides when an input
        # layout was unspecified. Linear and explicit contiguous establish a
        # known output layout; otherwise preserve that uncertainty downstream.
        establishes_layout = (
            ctx.spec.contract is not None and ctx.spec.contract.output_layout == "contiguous"
        )
        if not establishes_layout and any(r.id in uncertain for r in op.inputs):
            uncertain.update(r.id for r in op.outputs)
        if ctx.spec.contract is not None and ctx.spec.contract.output_layout == "backend_dependent":
            uncertain.update(r.id for r in op.outputs)
