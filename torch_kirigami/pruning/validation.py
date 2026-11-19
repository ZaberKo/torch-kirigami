"""Original-call contracts and conservative metadata/layout validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress

import torch
from torch import fx, nn

from ..bindings import attribute_owner, copy_module_tree
from ..configuration import freeze, thaw
from ..dep_graph.contracts import AttributeBinding, Impact, Requirement
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.operation import Operation, OperatorRule, OperatorSpec, TensorFacts, argument
from ..dep_graph.operators.coordinates import retained_indices, slice_index
from ..dep_graph.operators.defaults import is_native_rule
from ..dep_graph.operators.effects import (
    metadata_read,
    mutable_parameter_inputs,
    returned_input,
    written_inputs,
)
from ..dep_graph.operators.shapes import evaluate, reevaluate
from ..dep_graph.selection import TensorRef
from ..operator_recipes import TensorRecipe
from .layouts import view_preserves_stride_boundaries
from .recipes import compact_stride
from .types import PlanningError, require_compact_shape


def _meta_key(value: object) -> object:
    """Freeze actual meta arguments, including strides and scalar container types."""
    if isinstance(value, torch.Tensor):
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.device.type != "meta"
        ):
            raise TypeError("Only strided metadata tensors can key a meta proof")
        return (
            "tensor",
            TensorFacts(tuple(value.shape), tuple(value.stride()), value.dtype, value.device),
            value.storage_offset(),
            value.is_conj(),
            value.is_neg(),
            value.requires_grad,
        )
    if type(value) in (tuple, list):
        return (type(value), tuple(_meta_key(item) for item in value))
    if type(value) is dict:
        return (dict, tuple((_meta_key(k), _meta_key(v)) for k, v in value.items()))
    if type(value) is slice:
        return (slice, _meta_key(value.start), _meta_key(value.stop), _meta_key(value.step))
    return freeze(value)


def _freeze_meta_output(value: object) -> object:
    """Store independent output facts without retaining mutable metadata tensors."""
    if isinstance(value, torch.Tensor):
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.device.type != "meta"
            or value.storage_offset() != 0
            or value.is_conj()
            or value.is_neg()
            or value.requires_grad
        ):
            raise TypeError("This output needs more than ordinary tensor metadata")
        return TensorFacts(tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
    if isinstance(value, (tuple, list)):
        items = [_freeze_meta_output(item) for item in value]
        return items if isinstance(value, list) else tuple(items)
    if type(value) is dict:
        return {key: _freeze_meta_output(item) for key, item in value.items()}
    freeze(value)  # Reject opaque mutable results instead of retaining them.
    return value


def _restore_meta_output(value: object) -> object:
    """Rebuild fresh metadata tensors from a cached output fact tree."""
    if type(value) is TensorFacts:
        return torch.empty_strided(value.shape, value.stride, dtype=value.dtype, device="meta")
    if isinstance(value, (tuple, list)):
        items = [_restore_meta_output(item) for item in value]
        return items if isinstance(value, list) else tuple(items)
    if type(value) is dict:
        return {key: _restore_meta_output(item) for key, item in value.items()}
    freeze(value)
    return value


def _check_slice_coordinates(
    op: Operation,
    impact: Impact,
    original_index: tuple[object, ...],
    compact_index: tuple[object, ...],
) -> None:
    """Reject slices that select different original coordinates after compaction."""
    x, y = op.inputs[0], op.outputs[0]
    if len(compact_index) != len(original_index):
        raise PlanningError(
            f"{op.name}: compact slice changes the indexed axes. "
            f"Expected {len(original_index)} index entries, got {len(compact_index)}."
        )
    dim = out_dim = 0
    for entry, (item, new_item) in enumerate(zip(original_index, compact_index, strict=True)):
        if type(new_item) is not type(item):
            raise PlanningError(
                f"{op.name}: compact slice changes the indexed axes at entry {entry}. "
                f"Expected {type(item).__name__}, got {type(new_item).__name__}."
            )
        if item is None:
            out_dim += 1
            continue
        old = range(x.shape[dim])
        kept = list(retained_indices(impact, x.axis(dim)))
        if isinstance(item, int):
            original = old[item]
            if not -len(kept) <= new_item < len(kept):
                raise PlanningError(
                    f"{op.name}: index {new_item} on axis {dim} is out of bounds for "
                    f"compact width {len(kept)}. It must still select original position {original}."
                )
            if kept[new_item] != original:
                raise PlanningError(
                    f"{op.name}: integer slice selects a different original coordinate on "
                    f"axis {dim}. Expected position {original}, got {kept[new_item]}."
                )
        else:
            if new_item.step is not None and new_item.step <= 0:
                raise PlanningError(
                    f"{op.name}: slice step on axis {dim} becomes {new_item.step} "
                    "after pruning. PyTorch requires a positive step."
                )
            indexed = old[item]
            expected = [indexed[i] for i in retained_indices(impact, y.axis(out_dim))]
            actual = kept[new_item]
            if actual != expected:
                mismatch = next(
                    (
                        (i, before, after)
                        for i, (before, after) in enumerate(zip(expected, actual, strict=False))
                        if before != after
                    ),
                    None,
                )
                detail = f"Expected {len(expected)} positions, got {len(actual)}."
                if mismatch is not None:
                    i, before, after = mismatch
                    detail = (
                        f"At result position {i}, expected original position {before}, got {after}."
                    )
                raise PlanningError(
                    f"{op.name}: slice has incorrect original-coordinate correspondence on "
                    f"axis {dim}. {detail}"
                )
            out_dim += 1
        dim += 1


def _meta_module(
    operation: Operation,
    edits: tuple[tuple[str, object], ...],
    tensor: Callable[[TensorRef], torch.Tensor],
    *,
    rule: OperatorRule,
) -> nn.Module:
    """Copy a module's configuration and replace every captured tensor binding.

    Native leaves read their configuration and use registered tensor slots.
    Extension calls get isolated configuration, including ordinary aliases to
    registered tensors. Neither path invokes the module's copy hooks.
    """
    original = operation.module
    replacements = {}
    for path, value in (*original.named_parameters(), *original.named_buffers()):
        ref = operation.bindings.get(path)
        if ref is None:
            raise PlanningError(f"{operation.name}: no captured tensor binding for {path}")
        replacements[id(value)] = tensor(ref)
    state = vars(original)
    if (
        is_native_rule(operation.node, original, rule)
        and not original._modules
        and not any(callable(value) for value in state.values())
    ):
        # The built-in forward cannot use unrelated Python state attached to the
        # instance, such as a lock or an activation retained by an observer.
        state = state.copy()
        for name in ("_modules", "_parameters", "_buffers"):
            state[name] = {
                key: replacements.get(id(value), value) for key, value in state[name].items()
            }
        result = object.__new__(type(original))
        object.__setattr__(result, "__dict__", state)
    else:
        result = copy_module_tree(original, replacements)
    for path, new in edits:
        owner, name = attribute_owner(result, path)
        object.__setattr__(owner, name, thaw(new))
    return result


def _check_index_coordinates(op: Operation, requirement: Requirement, impact: Impact) -> None:
    """Check a static vector against the original positions on its compact axis."""
    data = dict(requirement.data)
    axis = data["axis"]
    size = axis.tensor.shape[axis.dim]
    kept = list(retained_indices(impact, axis))
    for i in data["indices"]:
        if not -len(kept) <= i < len(kept):
            raise PlanningError(
                f"{op.name}: static index {i} is out of bounds for compact width "
                f"{len(kept)} and cannot preserve original coordinates"
            )
        if kept[i] != i % size:
            raise PlanningError(
                f"{op.name}: static indices change original coordinates. "
                f"Index {i} selected position {i % size} before pruning and "
                f"would select original position {kept[i]} after pruning."
            )


def _replay_arguments(
    op: Operation,
    requirements: tuple[Requirement, ...],
    impact: Impact,
    shape: Callable[[TensorRef], tuple[int, ...]],
    uncertain: set[str],
) -> tuple[tuple[object, ...], dict[str, object], str]:
    """Recompute size expressions and verify original-coordinate argument contracts."""
    try:
        args = reevaluate(op.node.args, op.args, op.expressions, shape)
        kwargs = reevaluate(op.node.kwargs, op.kwargs, op.expressions, shape)
    except (ArithmeticError, TypeError, ValueError, IndexError) as error:
        raise PlanningError(
            f"{op.name}: cannot evaluate compact call arguments: {error}"
        ) from error
    hints = []
    for req in requirements:
        if req.kind == "index_arguments":
            _check_index_coordinates(op, req, impact)
            data = dict(req.data)
            if data.get("mask"):
                axis = data["axis"]
                width = shape(axis.tensor)[axis.dim]
                size = axis.tensor.shape[axis.dim]
                if width != size:
                    raise PlanningError(
                        f"{op.name}: fixed boolean mask needs width {size} on axis {axis.dim}. "
                        f"Pruning would change it to {width}."
                    )
                # Meta cannot evaluate boolean nonzero; replay the guarded True positions.
                index = [slice(None)] * len(axis.tensor.shape)
                index[axis.dim] = list(data["indices"])
                args = (args[0], tuple(index))
        elif req.kind == "slice_arguments":
            try:
                index = slice_index(op.node.target, shape(op.inputs[0]), args, kwargs)
            except ValueError as error:
                raise PlanningError(f"{op.name}: {error}") from error
            _check_slice_coordinates(op, impact, dict(req.data)["index"], index)
        elif req.kind == "partition_arguments":
            if req.arguments:
                binding = req.arguments[0]
                sections = argument(
                    args, kwargs, binding.name, binding.position, target=op.node.target
                )
                if isinstance(sections, int) and sections <= 0:
                    raise PlanningError(
                        f"{op.name}: split size becomes {sections} after pruning; "
                        "the compact split requires a positive split size"
                    )
            # The relations partition the old axis in order. If replay produces
            # the same retained size at each port, the boundaries also agree.
            hints.append(" The compact partitions must preserve the original output ports.")
        elif req.kind == "shape_arguments":
            data = dict(req.data)
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
                hints.append(
                    " If a fixed number was intended to follow a tensor dimension, "
                    "replace it with tensor.size(dim) or a valid -1 inference, then "
                    "rebuild the dependency graph. Keep algorithmic constants fixed."
                )
            source = op.inputs[0]
            if (
                data["requires_view"]
                and any(ref.id in uncertain for ref in op.inputs)
                and not view_preserves_stride_boundaries(shape(source), expected)
            ):
                raise PlanningError(
                    f"{op.name}: input stride cannot be proved for view after pruning. "
                    "If a copy is acceptable, use reshape(...) or contiguous().view(...) "
                    "in forward and rebuild the dependency graph; shape and index "
                    "constraints still apply."
                )
            args = (source, *dimensions) if data["unpack_shape"] else (source, dimensions)
            kwargs = {}
    return args, kwargs, "".join(hints)


def _temporary_write_issue(
    source: fx.Node,
    writer: Operation,
    operations: Mapping[str, Operation],
    positions: Mapping[fx.Node, int],
) -> str | None:
    """Explain why a write cannot be traced to an isolated temporary Tensor."""
    consumer = writer.node
    value_only = returned_input(consumer, writer.effects) is source
    while True:
        for user in source.users:
            if user is consumer:
                continue
            # A value-only write keeps metadata reads valid. Later uses of the
            # same Tensor see the write in both the original and compact calls.
            if value_only and (metadata_read(user) or positions[user] > positions[writer.node]):
                continue
            if user is writer.node:
                return f"'{source.name}' also reaches the writer through another argument."
            if positions[user] < positions[writer.node]:
                earlier = operations.get(user.name)
                if (
                    value_only
                    and not user.users
                    and earlier is not None
                    and not earlier.effects.mutable_parameters
                    and returned_input(user, earlier.effects) is source
                    and written_inputs(user, earlier.effects) == {source}
                ):
                    # Another value-only write in the same sequence. Its
                    # unused return creates no separate alias path.
                    continue
                return f"'{source.name}' also feeds earlier consumer '{user.name}'."
            return f"'{source.name}' also feeds '{user.name}' without a value-only write proof."
        producer = operations.get(source.name)
        if producer is None:
            return f"'{source.name}' comes from a model input or a stored value."
        effects = producer.effects
        if effects.outputs_are_fresh:
            return None
        if not effects.returns_input or not isinstance(producer.result, TensorRef):
            return (
                f"'{source.name}' does not guarantee fresh storage or return an identified "
                "input unchanged in storage and shape."
            )
        origin = returned_input(source, effects)
        if origin is None:
            return f"'{source.name}' has no single captured Tensor source for its result."
        consumer, source = source, origin


def check_forward(
    graph: DependencyGraph,
    operations: Sequence[Operation],
    active: Iterable[
        tuple[Operation, OperatorSpec, tuple[Requirement | AttributeBinding, ...], OperatorRule]
    ],
    impact: Impact,
    recipes: Mapping[str, TensorRecipe],
    call_edits: Mapping[str, tuple[tuple[str, object], ...]],
    strides: Mapping[str, tuple[int, ...]],
    *,
    meta_outputs: dict[str, tuple[object, object]] | None = None,
) -> None:
    """Check original calls against compact metadata and retained coordinates.

    Args:
        graph: Dependency graph for the unmodified model.
        operations: Captured calls in execution order.
        active: Calls to validate, with specs, requirements, and operator rules.
        impact: Joint removal selections and their dependency closure.
        recipes: Planned tensor replacements indexed by tensor reference ID.
        call_edits: Frozen attribute assignments indexed by call, with module-local paths.
        strides: Proven compact strides indexed by tensor reference ID.
        meta_outputs: Optional cache of reusable metadata outputs, updated in place.

    Raises:
        torch_kirigami.pruning.PlanningError: A call cannot preserve its declared compact shape,
            coordinates, layout, or mutation contract.

    Notes:
        Model tensors are not modified. Only calls whose rules permit metadata
        evaluation execute, using meta tensors and copied module configuration.
    """
    active_by_name = {
        op.name: (spec, requirements, rule) for op, spec, requirements, rule in active
    }
    operations_by_name = {op.node.name: op for op in operations}
    positions = None
    values, uncertain = {}, set()
    shapes: dict[TensorRef, tuple[int, ...]] = {}

    def shape(ref: TensorRef) -> tuple[int, ...]:
        """Return the compact shape for one tensor reference."""
        if ref not in shapes:
            # Recipes can pack partitioned selections that have no ordinary
            # compact shape. Both recipes and the Impact stay fixed in this call.
            shapes[ref] = (
                recipes[ref.id].shape if ref.id in recipes else require_compact_shape(impact, ref)
            )
        return shapes[ref]

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
            container = tuple if isinstance(refs, tuple) else list
            if not isinstance(output, container) or len(refs) != len(output):
                raise PlanningError(f"Original forward changes output ports{shape_hint}")
            for r, v in zip(refs, output, strict=True):
                record(r, v, shape_hint)
        elif isinstance(refs, dict):
            if not isinstance(output, dict) or refs.keys() != output.keys():
                raise PlanningError(f"Original forward changes output keys{shape_hint}")
            for key, r in refs.items():
                record(r, output[key], shape_hint)

    for op in operations:
        entry = active_by_name.get(op.node.name)
        if entry is None:
            continue
        spec, requirements, rule = entry
        requirements = tuple(r for r in requirements if isinstance(r, Requirement))
        requires_replay = any(req.check == "replay" for req in requirements)
        if spec.expression is not None and not requires_replay:
            continue
        effects = op.effects
        inplace = effects.writes_inputs is None or bool(effects.writes_inputs)
        if inplace:
            updates = mutable_parameter_inputs(op.node, effects)
            parameter_paths = {
                path for ref in op.inputs if ref.kind == "parameter" for path in ref.paths
            }
            buffer_paths = {path for ref in op.inputs if ref.kind == "buffer" for path in ref.paths}
            for source in written_inputs(op.node, effects):
                # Capture isolates registered buffers. Parameter updates need an
                # explicit permission. Neither permits writes through views.
                if source.op == "get_attr" and (
                    (source in updates and str(source.target) in parameter_paths)
                    or (effects.writes_inputs is not None and str(source.target) in buffer_paths)
                ):
                    continue
                if positions is None:
                    positions = {node: i for i, node in enumerate(op.node.graph.nodes)}
                issue = _temporary_write_issue(source, op, operations_by_name, positions)
                if issue is not None:
                    raise PlanningError(
                        f"{op.name}: cannot prove in-place alias/consumer safety. {issue} "
                        "If no consumer relies on modifying the original tensor, use an "
                        "out-of-place operation and rebuild the dependency graph."
                    )
        if not rule.evaluate_on_meta:
            if requires_replay:
                raise PlanningError(
                    f"{op.name}: declared original-call checks require evaluate_on_meta=True; "
                    "they cannot be acknowledged by a custom recipe"
                )
            for ref in op.outputs:
                if ref.id in strides:
                    values[ref.id] = torch.empty_strided(
                        shape(ref), strides[ref.id], dtype=graph.metadata(ref).dtype, device="meta"
                    )
                else:
                    tensor(ref)
                    uncertain.add(ref.id)
            continue
        normalized_args, normalized_kwargs, shape_hint = _replay_arguments(
            op, requirements, impact, shape, uncertain
        )
        args, kwargs = tree(normalized_args), tree(normalized_kwargs)
        contract = spec.contract
        simulate_cast = (
            contract is not None and contract.output_layout == "cast" and not requires_replay
        )
        edits = call_edits.get(op.name, ())
        key = None
        if (
            meta_outputs is not None
            and rule.cache_meta_output
            and effects.outputs_are_fresh
            and not inplace
        ):
            # Opaque arguments keep the original per-call evaluation.
            with suppress(TypeError):
                key = (
                    _meta_key(args),
                    _meta_key(kwargs),
                    tuple((name, _meta_key(tensor(ref))) for name, ref in op.bindings.items()),
                    edits,
                    simulate_cast,
                )
        try:
            cached = meta_outputs.get(op.node.name) if key is not None else None
            if cached is not None and cached[0] == key:
                output = _restore_meta_output(cached[1])
            elif simulate_cast:
                output = tensor(op.inputs[0]).to(
                    device="meta",
                    dtype=graph.metadata(op.outputs[0]).dtype,
                    memory_format=op.kwargs.get("memory_format", torch.preserve_format),
                    copy=contract.copy_output
                    or graph.metadata(op.inputs[0]).device != graph.metadata(op.outputs[0]).device,
                )
            elif op.module is not None:
                module = _meta_module(op, edits, tensor, rule=rule)
                output = module.forward(*args, **kwargs)  # Metadata replay bypasses module hooks.
            elif op.node.op == "call_method":
                output = getattr(args[0], str(op.node.target))(*args[1:], **kwargs)
            else:
                output = op.node.target(*args, **kwargs)
            if spec.expression is not None:
                dimensions = tuple(output) if isinstance(output, (torch.Size, list)) else output
                if freeze(dimensions) != freeze(evaluate(spec.expression, shape)):
                    raise PlanningError(
                        f"{op.name}: compact scalar result disagrees with its declared expression"
                    )
            record(op.result, output, shape_hint)
            if key is not None and (cached is None or cached[0] != key):
                try:
                    facts = _freeze_meta_output(output)
                except TypeError:
                    pass
                else:
                    meta_outputs[op.node.name] = (key, facts)
        except (RuntimeError, ValueError, TypeError, LookupError, ArithmeticError) as error:
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
