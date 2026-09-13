"""Registered tensor references in supported ordinary Python containers."""

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .configuration import freeze, object_attributes
from .errors import CaptureError

_MODULE_FIELDS = frozenset(vars(nn.Module())) | {"_kirigami_structure"}


def storage_key(tensor):
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


def has_tensor_hooks(tensor):
    """Report Tensor hooks that would be silently lost by physical replacement."""
    return bool(tensor._backward_hooks or tensor._post_accumulate_grad_hooks)


@dataclass(frozen=True)
class AttributeEdit:
    """An internal object-state assignment or deletion, prepared before commit."""

    path: str
    value: object = None
    delete: bool = False


def ordinary_attributes(module):
    """Yield user-owned attributes, excluding PyTorch registration/hook tables."""
    return ((k, v) for k, v in object_attributes(module).items() if k not in _MODULE_FIELDS)


def copy_module_state(original, shell, memo):
    """Copy dictionary and slot state into a preallocated, memoized module shell."""
    object.__setattr__(shell, "__dict__", copy.deepcopy(vars(original), memo))
    for name, value in object_attributes(original).items():
        if name not in vars(original):
            object.__setattr__(shell, name, copy.deepcopy(value, memo))


def _leaves(value, path=(), seen=None):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif type(value) in (tuple, list, dict):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return
        seen = seen | {id(value)}
        entries = value.items() if type(value) is dict else enumerate(value)
        for key, item in entries:
            children = tuple(_leaves(item, (), seen))
            if not children:
                continue
            # Only immutable keys can become portable reference paths.
            try:
                token = freeze(key)
            except TypeError as error:
                raise CaptureError("Unsupported key in ordinary model container") from error
            for suffix, tensor in children:
                yield (*path, (type(value).__name__, token), *suffix), tensor


def reference_signature(model):
    """Record container paths to registered tensors; reject separate storage views.

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

    def describe(value, active=()):
        if id(value) in registered:
            return ("tensor", registered[id(value)][0]), True
        if id(value) in modules:
            return ("module", modules[id(value)]), True
        if type(value) in (tuple, list, dict):
            if id(value) in active:
                return ("cycle", active.index(id(value))), False
            entries = value.items() if type(value) is dict else enumerate(value)
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
            for _, tensor in _leaves(value):
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


def reference_edits(model, replacements):
    """Prepare container rebinding with one memo, preserving shared containers."""
    memo = {id(m): m for m in model.modules()}
    memo.update((id(t), t) for t in (*model.parameters(), *model.buffers()))
    memo.update(replacements)
    edits = []
    for path, module in model.named_modules():
        for name, value in ordinary_attributes(module):
            leaves = tuple(_leaves(value))
            if not any(id(t) in replacements for _, t in leaves):
                continue
            # Unrelated constant tensors must retain identity/storage too.
            memo.update((id(t), t) for _, t in leaves if id(t) not in replacements)
            edits.append(AttributeEdit(f"{path}.{name}".lstrip("."), copy.deepcopy(value, memo)))
    return tuple(edits)


def final_state_edits(model, prepared):
    """Transfer ordinary state from the final prepared graph, remapping modules.

    A parent's extra-state setter may update or replace a child. The validated
    final graph, rather than the original shell graph or setter ownership,
    determines the object mapping and all assignments/deletions.
    """
    memo = {id(prepared.get_submodule(path)): module for path, module in model.named_modules()}
    memo.update((id(t), t) for t in (*prepared.parameters(), *prepared.buffers()))
    edits = []
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
