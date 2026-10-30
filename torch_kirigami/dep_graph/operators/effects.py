"""Capture-time call effects, also used by compact-forward validation."""

from __future__ import annotations

import inspect
import operator

import torch
from torch import fx, nn

from ..contracts import ArgumentRef
from ..operation import TensorEffects, argument


def named_inplace(node: fx.Node) -> bool:
    """Recognize mutating API names without confusing Python keyword escapes."""
    if node.op not in ("call_function", "call_method"):
        return False
    if node.target in (operator.and_, operator.or_, operator.not_):
        return False
    mutators = (
        operator.iadd,
        operator.isub,
        operator.imul,
        operator.imatmul,
        operator.itruediv,
        operator.ifloordiv,
        operator.imod,
        operator.ipow,
        operator.iand,
        operator.ior,
        operator.ixor,
        operator.ilshift,
        operator.irshift,
        operator.setitem,
        operator.delitem,
    )
    if node.target in mutators:
        return True
    name = getattr(node.target, "__name__", str(node.target))
    if name in {f"__{target.__name__}__" for target in mutators}:
        return True
    return name.endswith("_") and not name.endswith("__")


def native_effects(node: fx.Node, module: nn.Module | None, *, fresh_output: bool) -> TensorEffects:
    """Declare public in-place arguments and guaranteed allocations in one place."""
    target = node.target
    mutates = named_inplace(node)
    mutates = mutates or bool(getattr(module, "inplace", False))
    if node.op == "call_function":
        try:
            bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
            value = bound.arguments.get("inplace", False)
            mutates = mutates or (value is not False and value is not None)
        except (TypeError, ValueError):
            mutates = mutates or node.kwargs.get("inplace") is True
    return TensorEffects(
        writes_inputs=(ArgumentRef("input", 0),) if mutates else (),
        outputs_are_fresh=fresh_output and not mutates,
    )


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
    result = set()
    for ref in effects.writes_inputs:
        locations = argument(
            node.args,
            node.kwargs,
            ref.name,
            ref.position,
            target=node.target,
            variadic=ref.variadic,
        )
        fx.map_arg(locations, lambda item: result.add(item) or item)
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
