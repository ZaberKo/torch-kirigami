"""Capture-time call effects, also used by compact-forward validation."""

import inspect
import operator

import torch

from ..registry import CallEffects


def native_effects(node, module):
    """Recognize public in-place arguments and a small set of guaranteed copies."""
    target = node.target
    name = getattr(target, "__name__", str(target))
    mutates = name.endswith("_") and node.op in ("call_function", "call_method")
    mutates = mutates or bool(getattr(module, "inplace", False))
    if node.op == "call_function":
        try:
            bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
            value = bound.arguments.get("inplace", False)
            mutates = mutates or (value is not False and value is not None)
        except (TypeError, ValueError):
            mutates = mutates or node.kwargs.get("inplace") is True
    fresh = target in (torch.clone, "clone", operator.add, torch.add)
    return CallEffects(mutates, fresh)
