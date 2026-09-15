"""Shared immutable guards for recognizable Python module configuration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MemberDescriptorType
from typing import cast

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

    items: tuple[object, ...]


@dataclass(frozen=True)
class FrozenDict:
    """Preserve insertion order and exact key/value types in static configuration."""

    items: tuple[tuple[object, object], ...]


def object_attributes(value: object) -> dict[str, object]:
    """Read instance dictionary and initialized Python slots without properties."""
    result = dict(vars(value))
    for cls in reversed(type(value).__mro__):
        for name, descriptor in vars(cls).items():
            if isinstance(descriptor, MemberDescriptorType) and hasattr(value, name):
                result[name] = getattr(value, name)
    return result


def freeze(value: object, *, _depth: int = 0) -> object:
    """Freeze scalars and nested list/tuple/dict configuration, rejecting other objects."""
    if _depth > 50:
        raise TypeError("Configuration is cyclic or too deeply nested")
    if value is None:
        return value
    if type(value) in (str, int, float, bool):
        return FrozenScalar(type(value).__name__, value.hex() if type(value) is float else value)
    if type(value) is torch.Size:
        return FrozenScalar("size", tuple(cast(torch.Size, value)))
    if type(value) is torch.device:
        return FrozenScalar("device", str(value))
    if type(value) in (torch.dtype, torch.layout, torch.memory_format):
        return FrozenScalar(type(value).__name__, str(value).removeprefix("torch."))
    if type(value) is dict:
        return FrozenDict(
            tuple(
                (freeze(k, _depth=_depth + 1), freeze(v, _depth=_depth + 1))
                for k, v in value.items()
            )
        )
    if type(value) in (tuple, list):
        items = tuple(
            freeze(v, _depth=_depth + 1) for v in cast(tuple[object, ...] | list[object], value)
        )
        return FrozenList(items) if type(value) is list else items
    raise TypeError("Unrecognized configuration value")


def thaw(value: object) -> object:
    """Restore independently owned list/tuple/dict configuration from a frozen guard."""
    if type(value) is torch.Size:
        return value
    if isinstance(value, FrozenScalar):
        if value.kind == "size":
            return torch.Size(cast(tuple[int, ...], value.value))
        if value.kind == "device":
            return torch.device(cast(str, value.value))
        if value.kind in ("dtype", "layout", "memory_format"):
            result = getattr(torch, cast(str, value.value), None)
            if type(result).__name__ != value.kind:
                raise ValueError("Invalid PyTorch configuration constant")
            return result
        return float.fromhex(cast(str, value.value)) if value.kind == "float" else value.value
    if isinstance(value, FrozenDict):
        return {thaw(k): thaw(v) for k, v in value.items}
    if isinstance(value, FrozenList):
        return [thaw(v) for v in value.items]
    if isinstance(value, tuple):
        return tuple(thaw(v) for v in value)
    return value


def attributes(module: torch.nn.Module) -> tuple[tuple[str, object], ...]:
    """Collect the same immutable configuration for graph and portable guards."""
    result = []
    for name, value in sorted(object_attributes(module).items()):
        if name == "_kirigami_structure":
            continue
        try:
            result.append((name, freeze(value)))
        except TypeError:
            continue
    return tuple(result)


def forward_hook_paths(model: torch.nn.Module) -> tuple[str, ...]:
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


def has_registration_hooks() -> bool:
    """Detect registration callbacks whose substitutions cannot preserve a transaction.

    PyTorch exposes registration but no public inspection API; keep access to
    these process-wide registries alongside the forward-hook compatibility check.
    """
    return bool(
        module_runtime._global_parameter_registration_hooks
        or module_runtime._global_buffer_registration_hooks
        or module_runtime._global_module_registration_hooks
    )
