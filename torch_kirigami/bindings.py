"""Registered tensor references in supported ordinary Python containers."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .configuration import freeze, object_attributes
from .errors import CaptureError

_MODULE_FIELDS = frozenset(vars(nn.Module())) | {"_kirigami_structure"}


def storage_key(tensor: torch.Tensor) -> tuple[str, int] | None:
    """Identify nonempty dense storage for conservative alias checks.

    Args:
        tensor: Tensor whose storage should be inspected.

    Returns:
        A device/pointer pair, or None for an empty tensor.

    Raises:
        CaptureError: The tensor is quantized or does not have strided layout.
    """
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise CaptureError("Only dense, non-quantized strided tensors are supported")
    return (str(tensor.device), tensor.untyped_storage().data_ptr()) if tensor.numel() else None


def has_tensor_hooks(tensor: torch.Tensor) -> bool:
    """Report Tensor hooks that would be silently lost by physical replacement."""
    return bool(tensor._backward_hooks or tensor._post_accumulate_grad_hooks)


@dataclass(frozen=True)
class AttributeEdit:
    """An internal object-state assignment or deletion, prepared before commit."""

    path: str
    value: object = None
    delete: bool = False


def ordinary_attributes(module: nn.Module) -> Iterator[tuple[str, object]]:
    """Yield user-owned attributes, excluding PyTorch registration/hook tables."""
    return ((k, v) for k, v in object_attributes(module).items() if k not in _MODULE_FIELDS)


def copy_module_state(original: nn.Module, shell: nn.Module, memo: dict[int, object]) -> None:
    """Copy dictionary and slot state into a preallocated, memoized module shell."""
    object.__setattr__(shell, "__dict__", copy.deepcopy(vars(original), memo))
    for name, value in object_attributes(original).items():
        if name not in vars(original):
            object.__setattr__(shell, name, copy.deepcopy(value, memo))


def _reference_nodes(
    value: object, seen: set[int] | None = None, *, include_containers: bool = False
) -> Iterator[object]:
    """Yield supported objects, stopping cycles within each container path."""
    if isinstance(value, torch.Tensor):
        yield value
    elif type(value) in (tuple, list, dict):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return
        seen = seen | {id(value)}
        if include_containers:
            yield value
        entries: Iterable[tuple[object, object]]
        if type(value) is dict:
            entries = cast(dict[object, object], value).items()
        else:
            entries = enumerate(cast(tuple[object, ...] | list[object], value))
        for key, item in entries:
            # Validate keys even when values contain no tensors: a Tensor or
            # Module key can itself hide a binding used by the original forward.
            try:
                freeze(key)
            except TypeError as error:
                raise CaptureError("Unsupported key in ordinary model container") from error
            yield from _reference_nodes(item, seen, include_containers=include_containers)


def reference_nodes(value: object) -> tuple[object, ...]:
    """Return supported Tensor and container objects, including empty containers."""
    return tuple(_reference_nodes(value, include_containers=True))


def ordinary_tensors(model: nn.Module) -> Iterator[torch.Tensor]:
    """Yield Tensor leaves of supported ordinary model attributes."""
    for module in model.modules():
        for _, value in ordinary_attributes(module):
            yield from _reference_nodes(value)


def reference_signature(model: nn.Module) -> tuple[object, ...]:
    """Record tensor aliases and ordinary-constant premises; reject storage views.

    Plain lists, tuples, dictionaries, and direct Tensor attributes are supported.
    Arbitrary custom objects are not traversed and must not hide tensor bindings.
    """
    registered = {id(t): (p, t) for p, t in (*model.named_parameters(), *model.named_buffers())}
    storages = {
        (str(t.device), t.untyped_storage().data_ptr()): id(t)
        for _, t in registered.values()
        if t.layout == torch.strided and t.numel()
    }
    result = []
    modules = {id(m): path for path, m in model.named_modules()}

    def describe(value: object, active: tuple[int, ...] = ()) -> tuple[object, bool]:
        """Encode a supported ordinary value and whether it contains references."""
        if id(value) in registered:
            return ("tensor", registered[id(value)][0]), True
        if id(value) in modules:
            return ("module", modules[id(value)]), True
        if isinstance(value, torch.Tensor):
            # Ordinary constants need portable structural guards too. Their
            # numerical values remain constructor-owned unless saved as extra state.
            return (
                "constant_tensor",
                f"{type(value).__module__}.{type(value).__qualname__}",
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                str(value.device),
                value.requires_grad,
                value.is_conj(),
                value.is_neg(),
            ), True
        if type(value) in (tuple, list, dict):
            if id(value) in active:
                return ("cycle", active.index(id(value))), False
            if type(value) is dict:
                entries = cast(
                    Iterator[tuple[object, object]], cast(dict[object, object], value).items()
                )
            else:
                entries = cast(
                    Iterator[tuple[object, object]],
                    enumerate(cast(tuple[object, ...] | list[object], value)),
                )
            items = [(key, *describe(item, (*active, id(value)))) for key, item in entries]
            contains = any(found for _, _, found in items)
            try:
                tree = tuple((freeze(key), item) for key, item, _ in items)
            except TypeError as error:
                if contains:
                    raise CaptureError("Unsupported key in ordinary reference container") from error
                return ("opaque", type(value).__name__), False
            return (type(value).__name__, tree), contains
        try:
            return freeze(value), False
        except TypeError:
            return ("opaque", type(value).__module__, type(value).__qualname__), False

    for path, module in model.named_modules():
        for name, value in ordinary_attributes(module):
            for tensor in _reference_nodes(value):
                tensor = cast(torch.Tensor, tensor)
                binding = registered.get(id(tensor))
                if binding is None and (
                    tensor.layout == torch.strided
                    and tensor.numel()
                    and (str(tensor.device), tensor.untyped_storage().data_ptr()) in storages
                ):
                    raise CaptureError(
                        f"Ordinary attribute {path}.{name} contains a separate view of registered storage"
                    )
            tree, contains = describe(value)
            if contains:
                result.append((path, name, tree))
    return tuple(result)


def reference_edits(model: nn.Module, replacements: dict[int, object]) -> tuple[AttributeEdit, ...]:
    """Prepare container rebinding with one memo, preserving shared containers."""
    memo = {id(m): m for m in model.modules()}
    memo.update((id(t), t) for t in (*model.parameters(), *model.buffers()))
    memo.update(replacements)
    edits: list[AttributeEdit] = []
    for path, module in model.named_modules():
        for name, value in ordinary_attributes(module):
            nodes = reference_nodes(value)
            if not any(id(t) in replacements for t in nodes):
                continue
            # Unrelated constant tensors must retain identity/storage too.
            memo.update(
                (id(t), t)
                for t in nodes
                if isinstance(t, torch.Tensor) and id(t) not in replacements
            )
            edits.append(AttributeEdit(f"{path}.{name}".lstrip("."), copy.deepcopy(value, memo)))
    return tuple(edits)


def final_state_edits(model: nn.Module, prepared: nn.Module) -> tuple[AttributeEdit, ...]:
    """Transfer ordinary data from validated shells without replacing modules.

    Checkpoint preparation has already verified that every registered module and
    tensor keeps its prepared identity. Remap shell references to original modules
    while preserving restored tensor aliases and shared ordinary containers.
    """
    memo = {id(prepared.get_submodule(path)): module for path, module in model.named_modules()}
    memo.update((id(t), t) for t in (*prepared.parameters(), *prepared.buffers()))
    edits: list[AttributeEdit] = []
    for path, module in model.named_modules():
        restored = prepared.get_submodule(path)
        old, new = dict(ordinary_attributes(module)), dict(ordinary_attributes(restored))
        edits.extend(
            AttributeEdit(f"{path}.{name}".lstrip("."), delete=True)
            for name in old.keys() - new.keys()
        )
        edits.extend(
            AttributeEdit(f"{path}.{name}".lstrip("."), copy.deepcopy(value, memo))
            for name, value in new.items()
        )
    return tuple(edits)


def reference_devices_match(expected: object, actual: object, devices: dict[str, str]) -> bool:
    """Compare ordinary references under a deterministic source/device mapping."""
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual):
            return False
        if len(expected) == 9 and expected[0] == actual[0] == "constant_tensor":
            return (
                actual[5] == devices.get(expected[5], expected[5])
                and expected[:5] == actual[:5]
                and expected[6:] == actual[6:]
            )
        return all(
            reference_devices_match(a, b, devices) for a, b in zip(expected, actual, strict=True)
        )
    return expected == actual
