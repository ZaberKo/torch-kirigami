"""Tensor references held in ordinary module attributes and plain Python containers.

Modules can keep registered parameters, buffers, or submodules in ordinary
attributes, lists, tuples, and dictionaries. These helpers find such
references, record them as structural guards, and prepare rebinding edits
when pruning replaces the referenced objects.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .configuration import STRUCTURE_ATTRIBUTE, freeze, object_attributes
from .errors import CaptureError

_MODULE_FIELDS = frozenset(vars(nn.Module())) | {STRUCTURE_ATTRIBUTE}


def storage_key(tensor: torch.Tensor) -> tuple[str, int] | None:
    """Identify the storage of a nonempty dense tensor for conservative alias checks.

    Args:
        tensor: Tensor whose storage should be inspected.

    Returns:
        The device name and storage data pointer, or `None` for an empty tensor.

    Raises:
        torch_kirigami.CaptureError: The tensor is quantized or does not have strided layout.
    """
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise CaptureError("Only dense, non-quantized strided tensors are supported")
    return (str(tensor.device), tensor.untyped_storage().data_ptr()) if tensor.numel() else None


def has_tensor_hooks(tensor: torch.Tensor) -> bool:
    """Return whether a tensor has hooks that physical replacement would silently drop.

    Both backward hooks and post-accumulate-grad hooks are checked.
    """
    # PyTorch can register tensor hooks publicly but has no public inspection API.
    return bool(tensor._backward_hooks or tensor._post_accumulate_grad_hooks)


@dataclass(frozen=True)
class AttributeEdit:
    """An internal attribute assignment or deletion, prepared before the model edit commits.

    Attributes:
        path: Qualified path to the object attribute.
        value: Replacement value when `delete` is false.
        delete: Whether to remove the attribute instead of assigning a value.
    """

    path: str
    value: object = None
    delete: bool = False


def ordinary_attributes(
    module: nn.Module, *, raw_attributes: Mapping[str, object] | None = None
) -> Iterator[tuple[str, object]]:
    """Yield attributes set by user code rather than by `nn.Module` itself.

    Args:
        module: Module to inspect.
        raw_attributes: Current `object_attributes(module)`, if the caller has
            already read them during the same check.

    Returns:
        Name and value pairs, excluding the default `nn.Module` fields and the
        library's structure attribute.
    """
    values = object_attributes(module) if raw_attributes is None else raw_attributes
    return ((k, v) for k, v in values.items() if k not in _MODULE_FIELDS)


def copy_module_state(original: nn.Module, shell: nn.Module, memo: dict[int, object]) -> None:
    """Deep-copy instance dictionary and slot state into a preallocated module shell.

    Args:
        original: Module whose state is copied.
        shell: Uninitialized module object that receives the copy.
        memo: `copy.deepcopy` memo shared across the whole model copy.
    """
    object.__setattr__(shell, "__dict__", copy.deepcopy(vars(original), memo))
    for name, value in object_attributes(original).items():
        if name not in vars(original):
            object.__setattr__(shell, name, copy.deepcopy(value, memo))


def _reference_nodes(
    value: object, seen: set[int] | None = None, *, include_containers: bool = False
) -> Iterator[object]:
    """Yield tensors, and optionally containers, stopping at cycles along each path."""
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
    """Return every tensor and plain container reachable from a value.

    Only plain `tuple`, `list`, and `dict` containers are traversed. Empty
    containers are included.

    Raises:
        torch_kirigami.CaptureError: A traversed dictionary has a key that
            `freeze` does not accept.
    """
    return tuple(_reference_nodes(value, include_containers=True))


def ordinary_tensors(model: nn.Module) -> Iterator[torch.Tensor]:
    """Yield tensors held in ordinary attributes and plain containers of every module.

    Raises:
        torch_kirigami.CaptureError: A traversed dictionary has an unsupported key.
    """
    for module in model.modules():
        for _, value in ordinary_attributes(module):
            yield from _reference_nodes(value)


def reference_signature(
    model: nn.Module, *, module_attributes: Mapping[int, Mapping[str, object]] | None = None
) -> tuple[object, ...]:
    """Describe how ordinary module attributes refer to tensors and modules.

    Plain lists, tuples, dictionaries, and direct tensor attributes are
    traversed. Custom objects are not traversed and must not hide tensor
    bindings. References to registered tensors and modules are recorded by
    path. Unregistered constant tensors are recorded by type, shape, stride,
    dtype, device, and flags, but not by value.

    Args:
        model: Model to inspect.
        module_attributes: Current `object_attributes()` of every module, keyed
            by module identity, if the caller has already read them during the
            same check. Nothing is retained between checks.

    Returns:
        One `(module_path, attribute_name, description)` entry for each
        attribute that contains a tensor or module reference.

    Raises:
        torch_kirigami.CaptureError: An attribute holds a separate view of
            registered storage, or a container with references has a key that
            `freeze` does not accept.
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
        raw_attributes = None if module_attributes is None else module_attributes[id(module)]
        for name, value in ordinary_attributes(module, raw_attributes=raw_attributes):
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
    """Prepare edits that rebind ordinary attributes to replacement objects.

    One deep-copy memo is shared by every edit, so containers shared between
    attributes stay shared. Modules, registered tensors, and unrelated constant
    tensors keep their identity.

    Args:
        model: Model whose ordinary attributes are rebound.
        replacements: Replacement objects keyed by the identity of the objects
            they replace.

    Returns:
        One assignment for each attribute that references a replaced object.
    """
    memo = {id(m): m for m in model.modules()}
    memo.update((id(t), t) for t in (*model.parameters(), *model.buffers()))
    memo.update(replacements)
    edits: list[AttributeEdit] = []
    for path, module in model.named_modules():
        for name, value in ordinary_attributes(module):
            nodes = reference_nodes(value)
            if not any(id(t) in replacements for t in nodes):
                continue
            # Preserve unrelated tensors and containers, including a config
            # dict that another attribute may also reference.
            for node in nodes:
                if node is value or id(node) in memo:
                    continue
                if isinstance(node, torch.Tensor) or not any(
                    id(child) in replacements
                    for child in _reference_nodes(node, include_containers=True)
                ):
                    memo[id(node)] = node
            edits.append(AttributeEdit(f"{path}.{name}".lstrip("."), copy.deepcopy(value, memo)))
    return tuple(edits)


def final_state_edits(model: nn.Module, prepared: nn.Module) -> tuple[AttributeEdit, ...]:
    """Prepare edits that copy ordinary attributes from a prepared model into `model`.

    Checkpoint preparation has already verified that every registered module
    and tensor keeps its prepared identity. References to prepared modules are
    remapped to the corresponding original modules, while restored tensor
    aliases and shared ordinary containers are preserved. Modules themselves
    are not replaced.

    Args:
        model: Live model that receives the attributes.
        prepared: Validated model copy with the same module paths.

    Returns:
        Deletions for attributes missing from `prepared`, followed by
        assignments for every ordinary attribute of `prepared`.
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
    """Compare two reference descriptions, allowing constant tensors to change device.

    Args:
        expected: Description recorded by `reference_signature`.
        actual: Description of the current model.
        devices: Mapping from recorded device names to current device names.
            Devices without an entry must match exactly.

    Returns:
        Whether the descriptions are equal after mapping constant-tensor devices.
    """
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
