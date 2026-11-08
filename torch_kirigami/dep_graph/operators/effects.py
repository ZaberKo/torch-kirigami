"""Capture-time call effects, also used by compact-forward validation."""

from __future__ import annotations

import builtins
import inspect
import operator

import torch
from torch import fx, nn

from ...errors import CaptureError
from ..contracts import ArgumentRef
from ..operation import TensorEffects, argument, argument_locations

AUGMENTED_BINARY = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.itruediv: operator.truediv,
    operator.ifloordiv: operator.floordiv,
    operator.imod: operator.mod,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
    operator.ilshift: operator.lshift,
    operator.irshift: operator.rshift,
    operator.imatmul: operator.matmul,
}
_MUTATORS = (*AUGMENTED_BINARY, operator.setitem, operator.delitem)
_MUTATOR_DUNDERS = frozenset(f"__{target.__name__}__" for target in _MUTATORS)
_INDEXED_WRITES = frozenset(
    (
        "index_add_",
        "index_copy_",
        "index_fill_",
        "index_put_",
        "index_reduce_",
        "masked_scatter_",
        "put_",
        "scatter_",
        "scatter_add_",
        "scatter_reduce_",
    )
)
INPUT = ArgumentRef("input", 0)


def inplace_name(name: str) -> bool:
    """Recognize mutating method names."""
    return name in _MUTATOR_DUNDERS or (
        name not in ("and_", "or_", "not_") and name.endswith("_") and not name.endswith("__")
    )


def named_inplace(node: fx.Node) -> bool:
    """Recognize mutating call targets without confusing Python keyword escapes."""
    return node.op in ("call_function", "call_method") and (
        node.target in _MUTATORS or inplace_name(getattr(node.target, "__name__", str(node.target)))
    )


def metadata_read(node: fx.Node) -> bool:
    """Identify native reads of Tensor metadata independent of element values."""
    if node.op == "call_method":
        return node.target in ("size", "dim", "numel", "ndimension", "nelement")
    if node.op != "call_function":
        return False
    if node.target is builtins.getattr:
        return node.args[1] in ("shape", "ndim", "dtype", "device")
    return node.target is torch.numel


def indexed_write(node: fx.Node) -> bool:
    """Identify native indexed writes whose destination is the first argument."""
    if node.op == "call_method":
        return node.target in _INDEXED_WRITES
    if node.op != "call_function":
        return False
    name = getattr(node.target, "__name__", "")
    return node.target is operator.setitem or (
        name in _INDEXED_WRITES and node.target is getattr(torch, name, None)
    )


def native_effects(
    node: fx.Node,
    module: nn.Module | None,
    *,
    fresh_output: bool,
    returns_written_input: bool = False,
) -> TensorEffects:
    """Declare public in-place arguments and guaranteed allocations in one place."""
    target = node.target
    inplace = named_inplace(node) or bool(getattr(module, "inplace", False))
    if node.op == "call_function":
        try:
            bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
            value = bound.arguments.get("inplace", False)
            inplace = inplace or value
        except (TypeError, ValueError):
            inplace = inplace or node.kwargs.get("inplace", False)
    mutates = inplace is not False and inplace is not None
    return TensorEffects(
        writes_inputs=(INPUT,) if mutates else (),
        outputs_are_fresh=fresh_output and not mutates,
        aliases_input=INPUT if mutates and returns_written_input else None,
        returns_input=inplace is True and returns_written_input,
    )


def to_copy(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    """Read Tensor.to's copy argument across its public overloads.

    A boolean in the second slot after the receiver selects the dtype or
    Tensor overload. The device overload takes a dtype or None in that slot.
    This also distinguishes overloads before FX arguments have been evaluated.
    """
    position = 3 if len(args) > 2 and type(args[2]) is bool else 4
    return argument(args, kwargs, "copy", position, False)


def normalization_effects(
    node: fx.Node, module: nn.Module | None, *, instance: bool
) -> TensorEffects:
    """Declare native running-statistic updates without treating inputs as writable.

    Registered buffers remain isolated by capture. Directly bound parameters
    used as statistics receive the same explicit permission and warning as
    other declared native parameter updates. A view does not receive permission.
    """
    names = ("running_mean", "running_var")
    if module is not None:
        updates = module.training or (instance and not module.track_running_stats)
        if not instance:
            updates = module.training and module.track_running_stats
            names += ("num_batches_tracked",)
        paths = tuple(
            name for name in names if updates and isinstance(getattr(module, name), nn.Parameter)
        )
        return TensorEffects(writes_inputs=(), outputs_are_fresh=True, mutable_parameters=paths)
    flag = "use_input_stats" if instance else "training"
    updates = argument(node.args, node.kwargs, flag, 5, instance, target=node.target)
    if updates is False:
        return TensorEffects(writes_inputs=(), outputs_are_fresh=True)
    native = node.target in (torch.batch_norm, torch.instance_norm)
    statistics = tuple(
        ArgumentRef(name, position)
        for name, position in zip(names, (3, 4) if native else (1, 2), strict=True)
        if argument(node.args, node.kwargs, name, position, target=node.target) is not None
    )
    return TensorEffects(
        writes_inputs=statistics,
        outputs_are_fresh=True,
        mutable_parameters=statistics,
    )


def written_inputs(node: fx.Node, effects: TensorEffects) -> set[fx.Node]:
    """Resolve declared writes using the existing canonical argument locator.

    Container arguments conservatively include every FX-valued leaf. Unknown
    writes include all inputs; they are never silently treated as read-only.
    """
    if effects.writes_inputs is None:
        return set(node.all_input_nodes)
    return argument_nodes(node, effects.writes_inputs)


def returned_input(node: fx.Node, effects: TensorEffects) -> fx.Node | None:
    """Return the FX argument that the call returns as the same Tensor."""
    ref = effects.aliases_input
    if not effects.returns_input or ref is None:
        return None
    source = argument(node.args, node.kwargs, ref.name, ref.position, target=node.target)
    return source if isinstance(source, fx.Node) else None


def argument_nodes(
    node: fx.Node, refs: tuple[ArgumentRef, ...], *, required: bool = False
) -> set[fx.Node]:
    """Return the FX values in the specified argument locations."""
    result = set()
    for ref in refs:
        locations = argument_locations(
            node.args,
            node.kwargs,
            ref.name,
            ref.position,
            target=node.target,
            variadic=ref.variadic,
        )
        if required and not locations:
            raise CaptureError(f"Missing storage source argument {ref.name} at {node.name}")
        for kind, key in locations:
            value = node.args[key] if kind == "args" else node.kwargs[key]
            fx.map_arg(value, lambda item: result.add(item) or item)
    return result


def mutable_parameter_inputs(node: fx.Node, effects: TensorEffects) -> set[fx.Node]:
    """Resolve arguments permitted to update direct registered parameters.

    Callers must check that each source is a direct parameter binding. The
    declaration does not permit mutation through an unproven alias or view.
    """
    if not effects.mutable_parameters:
        return set()
    return written_inputs(
        node,
        TensorEffects(
            writes_inputs=tuple(
                ref for ref in effects.mutable_parameters if isinstance(ref, ArgumentRef)
            )
        ),
    )
