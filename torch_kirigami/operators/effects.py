"""Capture-time call effects, also used by compact-forward validation."""

import inspect
import operator

from ..operation import CallEffects


def named_inplace(node):
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


def native_effects(node, module, *, fresh_output):
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
    return CallEffects(mutates, fresh_output and not mutates)
