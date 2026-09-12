"""Shared immutable guards for recognizable Python module configuration."""

from dataclasses import dataclass

import torch
from torch.nn.modules import module as module_runtime


@dataclass(frozen=True)
class FrozenScalar:
    """An exact scalar type/value guard; floating NaNs have stable equality."""

    kind: str
    value: object


@dataclass(frozen=True)
class FrozenList:
    """Preserve the list container type without retaining mutable configuration."""

    items: tuple


def freeze(value, *, _depth=0):
    """Freeze scalars and nested list/tuple configuration, rejecting other objects."""
    if _depth > 50:
        raise TypeError("Configuration is cyclic or too deeply nested")
    if value is None:
        return value
    if type(value) in (str, int, float, bool):
        return FrozenScalar(type(value).__name__, value.hex() if type(value) is float else value)
    if type(value) is torch.Size:
        return FrozenScalar("size", tuple(value))
    if type(value) in (tuple, list):
        items = tuple(freeze(v, _depth=_depth + 1) for v in value)
        return FrozenList(items) if type(value) is list else items
    raise TypeError("Unrecognized configuration value")


def thaw(value):
    """Restore independently owned list/tuple configuration from a frozen guard."""
    if type(value) is torch.Size:
        return value
    if isinstance(value, FrozenScalar):
        if value.kind == "size":
            return torch.Size(value.value)
        return float.fromhex(value.value) if value.kind == "float" else value.value
    if isinstance(value, FrozenList):
        return [thaw(v) for v in value.items]
    if isinstance(value, tuple):
        return tuple(thaw(v) for v in value)
    return value


def attributes(module):
    """Collect the same immutable configuration for graph and portable guards."""
    result = []
    for name, value in sorted(vars(module).items()):
        if name == "_kirigami_structure":
            continue
        try:
            result.append((name, freeze(value)))
        except TypeError:
            continue
    return tuple(result)


def forward_hook_paths(model):
    """Find unmodeled module call hooks, including expanded root/parent boundaries."""
    # PyTorch has public global-hook registration but no public inspection API.
    # Keep its stable registry access localized here and test both supported versions.
    global_hooks = (
        ("<global>",)
        if (module_runtime._global_forward_hooks or module_runtime._global_forward_pre_hooks)
        else ()
    )
    return global_hooks + tuple(
        path or "<root>"
        for path, module in model.named_modules()
        if module._forward_hooks or module._forward_pre_hooks
    )


def has_registration_hooks():
    """Detect registration callbacks whose substitutions cannot preserve a transaction.

    PyTorch exposes registration but no public inspection API; keep access to
    these process-wide registries alongside the forward-hook compatibility check.
    """
    return bool(
        module_runtime._global_parameter_registration_hooks
        or module_runtime._global_buffer_registration_hooks
        or module_runtime._global_module_registration_hooks
    )
