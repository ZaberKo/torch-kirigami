"""Final-structure checkpoints, independent of graph capture and pruning history."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from .serialization import decode, encode
from .state import (
    STRUCTURE_ATTRIBUTE,
    ModelStructure,
    attribute,
    commit,
    managed_record,
    snapshot,
    validate_managed,
)
from .types import AttributeRecipe, ExecutionError


@dataclass(frozen=True)
class _StateAttribute:
    """Prepared ordinary Python state; used only inside the loading transaction."""

    path: str
    new: object


def save_checkpoint(model, path):
    """Save final structure and current values, without retaining pruning history.

    Args:
        model: A pruned, restored, or ordinary Module. Do not mutate it concurrently.
        path: Filename or file-like object accepted by torch.save.

    Registered nonpersistent buffers are included explicitly. Model code and its
    original constructor configuration remain the caller's responsibility.
    """
    structure = snapshot(model)
    validate_managed(model, structure)
    if any(len(s.storage_aliases) > 1 for s in structure.tensors):
        raise ExecutionError("Checkpoint cannot reconstruct distinct tensors sharing storage")
    for state_ in structure.tensors:
        owner, name = attribute(model, state_.paths[0])
        expected = nn.Parameter if state_.kind == "parameter" else torch.Tensor
        if type(getattr(owner, name)) is not expected:
            raise ExecutionError("Checkpoint cannot reconstruct custom tensor subclasses")
    state = model.state_dict()
    buffers = {}
    for item in structure.tensors:
        for name, persistent in zip(item.paths, item.persistent, strict=True):
            if item.kind == "buffer" and not persistent:
                owner, field = attribute(model, name)
                buffers[name] = getattr(owner, field).detach()
    torch.save(
        {
            "format": "torch-kirigami.checkpoint",
            "version": 1,
            "structure": encode(structure),
            "managed": getattr(model, STRUCTURE_ATTRIBUTE, {}).get("attributes", ()),
            "state_dict": state,
            "nonpersistent_buffers": buffers,
        },
        path,
    )


def load_checkpoint(model, path, *, map_location=None):
    """Restore final tensor sizes, sharing, configuration, and values into a skeleton.

    Args:
        model: Newly constructed compatible original Module; returned unchanged
            in identity. No example input, operator registry, or FX graph is needed.
        path: Filename or file-like object accepted by torch.load.
        map_location: PyTorch storage relocation, such as "cpu".

    Returns:
        The supplied model with its final pruned structure and checkpoint values.

    Raises:
        ExecutionError: The schema, skeleton, aliases, or state payload is incompatible.

    Ordinary registered state is prepared before committing. Custom state hooks
    run on an isolated module shell; external side effects in user hooks remain
    outside the transaction contract.
    """
    data = torch.load(path, map_location=map_location, weights_only=True)
    if (
        not isinstance(data, dict)
        or set(data)
        != {
            "format",
            "version",
            "structure",
            "managed",
            "state_dict",
            "nonpersistent_buffers",
        }
        or data["format"] != "torch-kirigami.checkpoint"
        or data["version"] != 1
    ):
        raise ExecutionError("Unsupported checkpoint schema/version")
    try:
        structure = decode(data["structure"])
    except (TypeError, ValueError, KeyError) as error:
        raise ExecutionError(f"Invalid checkpoint structure: {error}") from error
    if not isinstance(structure, ModelStructure):
        raise ExecutionError("Checkpoint requires a final ModelStructure")
    original = snapshot(model)
    expected_types = {"parameter": "torch.nn.parameter.Parameter", "buffer": "torch.Tensor"}
    if any(
        s.type_name != expected_types.get(s.kind) for s in (*original.tensors, *structure.tensors)
    ):
        raise ExecutionError("Checkpoint requires ordinary parameters and buffers")

    def skeleton(s):
        return tuple((m.paths, m.type_name, m.slots) for m in s.modules)

    if skeleton(original) != skeleton(structure):
        raise ExecutionError("Module skeleton, aliases, or registered slots differ")
    if tuple((s.paths, s.kind, s.persistent) for s in original.tensors) != tuple(
        (s.paths, s.kind, s.persistent) for s in structure.tensors
    ):
        raise ExecutionError("Tensor binding/alias/persistence schema differs")
    managed = tuple(data["managed"])
    if not all(isinstance(p, str) for p in managed):
        raise ExecutionError("Invalid managed attribute paths")
    changes = []
    for old, new in zip(original.modules, structure.modules, strict=True):
        before, after = dict(old.attributes), dict(new.attributes)
        if before.keys() != after.keys():
            raise ExecutionError("Module configuration fields differ")
        for name, value in after.items():
            if before[name] != value:
                paths = [f"{p}.{name}".lstrip(".") for p in old.paths]
                owner = model.get_submodule(old.paths[0]) if old.paths[0] else model
                custom_state = type(owner).set_extra_state is not nn.Module.set_extra_state
                if name != "training" and not any(p in managed for p in paths) and not custom_state:
                    raise ExecutionError(f"Original constructor configuration differs: {paths[0]}")
                changes.append(AttributeRecipe(paths[0], before[name], value))
    values = data["state_dict"]
    buffers = data["nonpersistent_buffers"]
    if not isinstance(values, dict) or not isinstance(buffers, dict):
        raise ExecutionError("Checkpoint requires state dictionaries")
    expected_buffers = {
        p
        for s in structure.tensors
        for p, persistent in zip(s.paths, s.persistent, strict=True)
        if not persistent
    }
    if set(buffers) != expected_buffers:
        raise ExecutionError("Nonpersistent buffer keys differ from the structure")
    replacements = []
    with torch.inference_mode(False), torch.no_grad():
        for old, state in zip(original.tensors, structure.tensors, strict=True):
            if len(state.storage_aliases) > 1:
                raise ExecutionError("Unsupported checkpoint storage sharing")
            entries = []
            for name, persistent in zip(state.paths, state.persistent, strict=True):
                source = values if persistent else buffers
                value = source.get(name)
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != state.shape
                    or str(value.dtype) != state.dtype
                ):
                    raise ExecutionError(f"Invalid checkpoint tensor: {name}")
                entries.append(value)
            if any(not torch.equal(entries[0], v) for v in entries[1:]):
                raise ExecutionError(f"Conflicting values for shared aliases: {state.paths}")
            tensor = torch.empty_strided(
                state.shape, state.stride, dtype=entries[0].dtype, device=entries[0].device
            )
            tensor.copy_(entries[0])
            new = (
                nn.Parameter(tensor, requires_grad=state.requires_grad)
                if state.kind == "parameter"
                else tensor.requires_grad_(state.requires_grad)
            )
            owner, name = attribute(model, old.paths[0])
            replacements.append((old, getattr(owner, name), new))

    # Shallow module shells isolate hooks and extra state from original tensor
    # bindings; tensors are already newly allocated, and module aliases persist.
    shells = {id(m): copy.copy(m) for m in model.modules()}
    memo = dict(shells)
    memo.update((id(old), new) for _, old, new in replacements)
    for module in model.modules():
        shell = shells[id(module)]
        shell.__dict__ = {
            key: copy.deepcopy(value, memo)
            for key, value in module.__dict__.items()
            if key not in ("_modules", "_parameters", "_buffers")
        }
        shell._modules = {
            k: shells[id(v)] if v is not None else None for k, v in module._modules.items()
        }
        shell._parameters = dict(module._parameters)
        shell._buffers = dict(module._buffers)
    prepared = shells[id(model)]
    for state, _, tensor in replacements:
        for path_ in state.paths:
            owner, name = attribute(prepared, path_)
            table = owner._parameters if state.kind == "parameter" else owner._buffers
            table[name] = tensor
    for edit in changes:
        owner, name = attribute(prepared, edit.path)
        object.__setattr__(owner, name, edit.new)
    try:
        payload = OrderedDict(values)
        if hasattr(values, "_metadata"):
            payload._metadata = values._metadata
        with torch.inference_mode(False), torch.no_grad():
            prepared.load_state_dict(payload, strict=True)
    except Exception as error:
        raise ExecutionError(f"State loading failed before commit: {error}") from error
    final = snapshot(prepared)
    if skeleton(final) != skeleton(structure) or tuple(
        (s.paths, s.kind, s.shape, s.stride, s.dtype, s.requires_grad, s.persistent)
        for s in final.tensors
    ) != tuple(
        (s.paths, s.kind, s.shape, s.stride, s.dtype, s.requires_grad, s.persistent)
        for s in structure.tensors
    ):
        raise ExecutionError("Loading hooks changed the declared final structure")
    if tuple(m.attributes for m in final.modules) != tuple(m.attributes for m in structure.modules):
        raise ExecutionError("Restored configuration differs from the checkpoint")
    # Transfer prepared custom state as binding assignments, preserving rollback
    # for ordinary failures without invoking set_extra_state on the original.
    extra_changes = []
    module_fields = set(vars(nn.Module()))
    for module_state in original.modules:
        path_ = module_state.paths[0]
        original_module = model.get_submodule(path_) if path_ else model
        restored_module = prepared.get_submodule(path_) if path_ else prepared
        if type(original_module).set_extra_state is not nn.Module.set_extra_state:
            for name, value in vars(restored_module).items():
                if name not in module_fields and name != STRUCTURE_ATTRIBUTE:
                    extra_changes.append(_StateAttribute(f"{path_}.{name}".lstrip("."), value))
    replacements = [
        (state, old, getattr(*attribute(prepared, state.paths[0])))
        for state, old, _ in replacements
    ]
    record = managed_record(prepared, final, tuple(AttributeRecipe(p, None, None) for p in managed))
    commit(model, replacements, (*changes, *extra_changes), record)
    return model
