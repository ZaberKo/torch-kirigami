"""Immutable snapshots of recognizable module configuration, used as structural guards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MemberDescriptorType
from typing import cast

import torch
from torch.nn.modules import module as module_runtime

from .settings import config

STRUCTURE_ATTRIBUTE = "_kirigami_structure"


@dataclass(frozen=True)
class FrozenScalar:
    """A frozen scalar that compares equal only to the same exact type and value.

    Floats are stored as hexadecimal strings, so NaN guards compare equal and
    `0.0` differs from `-0.0`.

    Attributes:
        kind: Exact type name, such as `"int"`, `"float"`, `"size"`, `"device"`,
            or `"dtype"`.
        value: Captured value. Floats are hexadecimal strings, sizes are tuples,
            and devices and PyTorch constants are their string names.
    """

    kind: str
    value: object


@dataclass(frozen=True)
class FrozenList:
    """A frozen list that stays distinct from an equal tuple.

    Attributes:
        items: Frozen values in list order.
    """

    items: tuple[object, ...]


@dataclass(frozen=True)
class FrozenDict:
    """A frozen dictionary that keeps insertion order and exact key and value types.

    Attributes:
        items: Frozen key/value pairs in insertion order.
    """

    items: tuple[tuple[object, object], ...]


def slot_names(cls: type) -> tuple[str, ...]:
    """Return the names of slot descriptors in `cls`, from the root base to `cls`."""
    return tuple(
        name
        for base in reversed(cls.__mro__)
        for name, descriptor in vars(base).items()
        if isinstance(descriptor, MemberDescriptorType)
    )


def object_attributes(value: object, *, slots: tuple[str, ...] | None = None) -> dict[str, object]:
    """Return the instance dictionary merged with every initialized slot.

    Args:
        value: Object to inspect.
        slots: Precomputed `slot_names(type(value))`. Callers may reuse the
            names within one inspection, but must not cache them across model
            freshness checks because class state can change.

    Returns:
        A new dictionary of attribute names and values. Unset slots are omitted.
    """
    result = dict(vars(value))
    for name in slot_names(type(value)) if slots is None else slots:
        if hasattr(value, name):
            result[name] = getattr(value, name)
    return result


def freeze(value: object, *, _depth: int = 0) -> object:
    """Convert recognizable configuration into an immutable, comparable value.

    Accepted values are `None`, `str`, `int`, `float`, `bool`, `torch.Size`,
    `torch.device`, `torch.dtype`, `torch.layout`, `torch.memory_format`, and
    nested plain tuples, lists, and dictionaries of these values. Subclasses
    of these types are rejected.

    Args:
        value: Configuration value to freeze.

    Returns:
        `None`, a `FrozenScalar`, a `FrozenList`, a `FrozenDict`, or a tuple of
        frozen values.

    Raises:
        TypeError: The value contains an unrecognized object, or its nesting
            exceeds `config.max_container_depth` (which includes cycles).
    """
    if _depth > config.max_container_depth:
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
    """Rebuild new, independently owned configuration from a frozen value.

    Args:
        value: Value produced by `freeze`.

    Returns:
        The configuration value with fresh lists and dictionaries. Values that
        are not frozen records are returned unchanged.

    Raises:
        ValueError: A frozen PyTorch constant does not name a valid constant.
    """
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


def attributes(
    module: torch.nn.Module, *, raw_attributes: Mapping[str, object] | None = None
) -> tuple[tuple[str, object], ...]:
    """Freeze every recognizable attribute of one module, sorted by name.

    Attributes that `freeze` rejects, such as submodules, tensors, and custom
    objects, are skipped, as is the library's structure attribute.

    Args:
        module: Module to inspect.
        raw_attributes: Current `object_attributes(module)`, if the caller has
            already read them during the same check.

    Returns:
        Pairs of attribute name and frozen value.
    """
    result = []
    values = object_attributes(module) if raw_attributes is None else raw_attributes
    for name, value in sorted(values.items()):
        if name == STRUCTURE_ATTRIBUTE:
            continue
        try:
            result.append((name, freeze(value)))
        except TypeError:
            continue
    return tuple(result)


def forward_hook_paths(model: torch.nn.Module) -> tuple[str, ...]:
    """Return the locations of forward and forward-pre hooks, which analysis does not model.

    Returns:
        `"<global>"` when a process-wide module hook is registered, followed by
        the path of every module with its own hook. The root module is
        reported as `"<root>"`.
    """
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
    """Return whether process-wide parameter, buffer, or module registration hooks exist.

    Such hooks can substitute the registered objects, so the library cannot
    guarantee a transactional model edit while they are installed. PyTorch
    exposes registration but no public inspection API; keep access to these
    process-wide registries alongside the forward-hook compatibility check.
    """
    return bool(
        module_runtime._global_parameter_registration_hooks
        or module_runtime._global_buffer_registration_hooks
        or module_runtime._global_module_registration_hooks
    )
