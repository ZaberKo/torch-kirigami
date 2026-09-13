"""Final-structure checkpoints, independent of graph capture and pruning history."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace

import torch
from torch import nn

from ..bindings import (
    copy_module_state,
    final_state_edits,
    has_tensor_hooks,
    reference_devices_match,
    reference_signature,
    storage_key,
)
from ..configuration import thaw
from .serialization import decode, encode
from .state import (
    STRUCTURE_ATTRIBUTE,
    attribute,
    commit,
    managed_record,
    snapshot,
    validate_managed,
)
from .types import AttributeRecipe, ExecutionError, ModelStructure


def _check_state_contract(model):
    # This format saves raw registered slots, not the output of state_dict codecs.
    # Hooks may maintain additional external state and are explicitly unsupported.
    for path, module in model.named_modules():
        if (type(module).get_extra_state is nn.Module.get_extra_state) != (
            type(module).set_extra_state is nn.Module.set_extra_state
        ):
            raise ExecutionError(
                f"Checkpoint extra state requires paired get_extra_state/set_extra_state: "
                f"{path or '<root>'}"
            )
        if type(module).get_extra_state is not nn.Module.get_extra_state and any(
            table.get("_extra_state") is not None for table in (module._parameters, module._buffers)
        ):
            raise ExecutionError(
                f"Extra state key collides with a registered tensor: "
                f"{path + '.' if path else ''}_extra_state"
            )
        if any(
            getattr(module, name)
            for name in (
                "_state_dict_pre_hooks",
                "_state_dict_hooks",
                "_load_state_dict_pre_hooks",
                "_load_state_dict_post_hooks",
            )
        ):
            raise ExecutionError(f"Checkpoint state_dict hooks are unsupported: {path or '<root>'}")


def _check_nonoverlap(state):
    # Sufficient proof for dense/permuted/strided-with-gaps layouts using public
    # size/stride facts. No private overlap checker or elementwise index table.
    span = 1
    for stride, size in sorted(zip(state.stride, state.shape, strict=True)):
        if size <= 1:
            continue
        if stride < span:
            raise ExecutionError(f"Checkpoint cannot prove non-overlapping layout: {state.paths}")
        span += (size - 1) * stride


def _same_values(left, right):
    if (
        left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
        and left.storage_offset() == right.storage_offset()
        and left.is_conj() == right.is_conj()
        and left.is_neg() == right.is_neg()
        and storage_key(left) == storage_key(right)
    ):
        return True
    if left.is_complex():
        return _same_values(left.real, right.real) and _same_values(left.imag, right.imag)
    if left.is_floating_point():
        missing = torch.isnan(left)
        return torch.equal(missing, torch.isnan(right)) and torch.equal(
            left.masked_fill(missing, 0), right.masked_fill(missing, 0)
        )
    return torch.equal(left, right)


def _validate_extra_state(model, values, extra_keys):
    """Accept only weights-only data without hidden registered storage references."""
    tensors = (
        *model.parameters(),
        *model.buffers(),
        *(
            t
            for name, t in values.items()
            if name not in extra_keys and isinstance(t, torch.Tensor)
        ),
    )
    registered_ids = {id(t) for t in tensors}
    registered = {storage_key(t) for t in tensors if t.numel()}

    def visit(value, active=(), *, allow_registered=False):
        if len(active) > 50:
            raise ExecutionError("Checkpoint extra state is too deeply nested")
        if value is None or type(value) in (
            bool,
            int,
            float,
            str,
            bytes,
            torch.dtype,
            torch.device,
        ):
            return
        if type(value) in (torch.Tensor, nn.Parameter):
            if vars(value) or has_tensor_hooks(value):
                raise ExecutionError(
                    "Checkpoint tensor payload has unsupported attributes or hooks"
                )
            if not allow_registered and (
                id(value) in registered_ids or (value.numel() and storage_key(value) in registered)
            ):
                raise ExecutionError(
                    "Checkpoint extra state cannot contain registered tensor references"
                )
            return
        if type(value) in (tuple, list, dict, OrderedDict, torch.Size):
            if type(value) is OrderedDict and vars(value):
                raise ExecutionError("Checkpoint extra state container has unsupported attributes")
            if id(value) in active:
                raise ExecutionError("Checkpoint extra state must be an acyclic data tree")
            items = (
                (v for pair in value.items() for v in pair) if isinstance(value, dict) else value
            )
            for item in items:
                visit(item, (*active, id(value)))
            return
        raise ExecutionError(f"Unsupported checkpoint extra state type: {type(value).__name__}")

    if type(values) not in (dict, OrderedDict) or (type(values) is OrderedDict and vars(values)):
        raise ExecutionError("Unsupported checkpoint state dictionary container")
    for name, value in values.items():
        if name in extra_keys:
            visit(value)
        elif isinstance(value, torch.Tensor):
            if type(value) not in (torch.Tensor, nn.Parameter):
                raise ExecutionError("Checkpoint tensor payload has unsupported subclass")
            visit(value, allow_registered=True)


def _validate_payload(model, structure, values, buffers):
    """Validate raw tensor slots, entity aliases, and canonical extra-state keys."""
    if not isinstance(values, dict) or not isinstance(buffers, dict):
        raise ExecutionError("Checkpoint requires state dictionaries")
    expected = {
        p
        for s in structure.tensors
        for p, persistent in zip(s.paths, s.persistent, strict=True)
        if persistent
    }
    extra_keys = {
        f"{path}._extra_state".lstrip(".")
        for path, module in model.named_modules()
        if type(module).get_extra_state is not nn.Module.get_extra_state
    }
    expected.update(extra_keys)
    expected_buffers = {
        p
        for s in structure.tensors
        for p, persistent in zip(s.paths, s.persistent, strict=True)
        if not persistent
    }
    if set(values) != expected or set(buffers) != expected_buffers:
        raise ExecutionError("Checkpoint payload keys disagree with registered structure")
    _validate_extra_state(model, values, extra_keys)
    for state in structure.tensors:
        entries = []
        for name, persistent in zip(state.paths, state.persistent, strict=True):
            value = (values if persistent else buffers)[name]
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != state.shape
                or str(value.dtype) != state.dtype
            ):
                raise ExecutionError(f"Invalid checkpoint tensor: {name}")
            entries.append(value)
        if any(not _same_values(entries[0], v) for v in entries[1:]):
            raise ExecutionError(f"Conflicting values for shared aliases: {state.paths}")


def save_checkpoint(model, path):
    """Save final structure and current values, without retaining pruning history.

    Args:
        model: A pruned, restored, or ordinary Module. Do not mutate it concurrently.
        path: Filename or file-like object accepted by torch.save.

    Raw registered tensors and nonpersistent buffers are included explicitly;
    state_dict methods and custom codecs are not invoked. Model code and its
    original constructor configuration remain the caller's responsibility.
    Custom get_extra_state methods must be read-only; arbitrary getter side
    effects are not isolated or rolled back.
    """
    _check_state_contract(model)
    structure = snapshot(model)
    validate_managed(model, structure)
    if any(len(s.storage_aliases) > 1 for s in structure.tensors):
        raise ExecutionError("Checkpoint cannot reconstruct distinct tensors sharing storage")
    for state_ in structure.tensors:
        _check_nonoverlap(state_)
        owner, name = attribute(model, state_.paths[0])
        expected = nn.Parameter if state_.kind == "parameter" else torch.Tensor
        if type(getattr(owner, name)) is not expected:
            raise ExecutionError("Checkpoint cannot reconstruct custom tensor subclasses")
    state, buffers = {}, {}
    for item in structure.tensors:
        owner, field = attribute(model, item.paths[0])
        value = getattr(owner, field).detach()
        for name, persistent in zip(item.paths, item.persistent, strict=True):
            (state if persistent else buffers)[name] = value
    # Extra state belongs to a module entity, so aliases neither repeat a callback
    # nor create conflicting payloads for the same setter. Keep the canonical path.
    for name, module in model.named_modules():
        if type(module).get_extra_state is not nn.Module.get_extra_state:
            key = f"{name}._extra_state".lstrip(".")
            state[key] = module.get_extra_state()
    _validate_payload(model, structure, state, buffers)
    torch.save(
        {
            "format": "torch-kirigami.checkpoint",
            "structure": encode(structure),
            "managed": getattr(model, STRUCTURE_ATTRIBUTE, {}).get("attributes", ()),
            "state_dict": state,
            "nonpersistent_buffers": buffers,
        },
        path,
    )


@torch.inference_mode(False)
def load_checkpoint(model, path, *, map_location=None):
    """Restore final tensor sizes, sharing, configuration, and values into a skeleton.

    Args:
        model: Newly constructed compatible original Module; returned unchanged
            in identity. No example input, operator registry, or FX graph is needed.
        path: Filename or file-like object accepted by torch.load.
        map_location: None, one destination device, or a source/destination device
            dictionary. Per-storage callable relocation is unsupported.

    Returns:
        The supplied model with its final pruned structure and checkpoint values.

    Raises:
        ExecutionError: The schema, skeleton, aliases, or state payload is incompatible.

    Ordinary registered state is prepared before committing. set_extra_state
    runs on an isolated module shell; external side effects remain outside the
    transaction contract. Extra state may restore ordinary data, but cannot change
    registered module or tensor bindings. Registered tensor values are restored
    directly, without calling load_state_dict methods or custom codecs. Registered
    state_dict hooks are explicitly unsupported.
    """
    _check_state_contract(model)
    if any(has_tensor_hooks(t) for t in (*model.parameters(), *model.buffers())):
        raise ExecutionError(
            "Remove Tensor gradient hooks before loading; register them on restored tensors"
        )
    if map_location is not None and not isinstance(map_location, (str, torch.device, dict)):
        raise ExecutionError("Checkpoint map_location requires a device or device dictionary")
    if isinstance(map_location, dict) and not all(
        isinstance(source, str) and isinstance(destination, (str, torch.device))
        for source, destination in map_location.items()
    ):
        raise ExecutionError(
            "Checkpoint map_location requires a source/destination device dictionary"
        )
    mapped_devices = {}

    def relocate(storage, location):
        # Observe PyTorch's actual device spelling, including implicit CUDA indices
        # and storages present only in extra state. One source has one destination.
        destination = (
            map_location.get(location, location)
            if isinstance(map_location, dict)
            else map_location
            if map_location is not None
            else location
        )
        mapped = torch.serialization.default_restore_location(storage, str(destination))
        mapped_devices[location] = str(mapped.device)
        return mapped

    data = torch.load(path, map_location=relocate, weights_only=True)
    if (
        not isinstance(data, dict)
        or set(data)
        != {
            "format",
            "structure",
            "managed",
            "state_dict",
            "nonpersistent_buffers",
        }
        or data["format"] != "torch-kirigami.checkpoint"
    ):
        raise ExecutionError("Invalid checkpoint schema")
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
    custom_state = any(
        type(m).set_extra_state is not nn.Module.set_extra_state for m in model.modules()
    )
    for old, new in zip(original.modules, structure.modules, strict=True):
        before, after = dict(old.attributes), dict(new.attributes)
        if before.keys() != after.keys() and not custom_state:
            raise ExecutionError("Module configuration fields differ")
        for name, value in after.items():
            if name not in before or before[name] != value:
                paths = [f"{p}.{name}".lstrip(".") for p in old.paths]
                if name != "training" and not any(p in managed for p in paths) and not custom_state:
                    raise ExecutionError(f"Original constructor configuration differs: {paths[0]}")
                if name == "training" or any(p in managed for p in paths):
                    changes.append(AttributeRecipe(paths[0], thaw(before.get(name)), thaw(value)))
                # Extra state must itself restore other fields, including creation
                # and deletion. Validate its final configuration before committing.
    values = data["state_dict"]
    buffers = data["nonpersistent_buffers"]
    _validate_payload(model, structure, values, buffers)
    replacements = []
    with torch.inference_mode(False), torch.no_grad():
        for old, state in zip(original.tensors, structure.tensors, strict=True):
            _check_nonoverlap(state)
            if len(state.storage_aliases) > 1:
                raise ExecutionError("Unsupported checkpoint storage sharing")
            source = values if state.persistent[0] else buffers
            value = source[state.paths[0]]
            tensor = torch.empty_strided(
                state.shape, state.stride, dtype=value.dtype, device=value.device
            )
            tensor.copy_(value)
            new = (
                nn.Parameter(tensor, requires_grad=state.requires_grad)
                if state.kind == "parameter"
                else tensor.requires_grad_(state.requires_grad)
            )
            owner, name = attribute(model, old.paths[0])
            replacements.append((old, getattr(owner, name), new))

    # Shallow module shells isolate hooks and extra state from original tensor
    # bindings; tensors are already newly allocated, and module aliases persist.
    # User-defined __copy__ may return the original module. Never invoke it:
    # changing such a "shell" would corrupt the destination before commit.
    shells = {id(m): object.__new__(type(m)) for m in model.modules()}
    memo = dict(shells)
    memo.update((id(old), new) for _, old, new in replacements)
    for module in model.modules():
        shell = shells[id(module)]
        copy_module_state(module, shell, memo)
    prepared = shells[id(model)]
    for state, _, tensor in replacements:
        for path_ in state.paths:
            owner, name = attribute(prepared, path_)
            table = owner._parameters if state.kind == "parameter" else owner._buffers
            table[name] = tensor
    for edit in changes:
        owner, name = attribute(prepared, edit.path)
        object.__setattr__(owner, name, thaw(edit.new))
    # Allocation already applied map_location. Compare the final callback state
    # against these actual devices, rather than the source checkpoint devices.
    prepared_devices = tuple(s.device for s in snapshot(prepared).tensors)
    # Keep direct references to every prepared module and slot. Inspect these
    # tables before recursively traversing callbacks' result, so introduced cycles
    # and same-shape replacement objects are rejected without walking a bad graph.
    registrations = tuple(
        (
            path,
            module,
            tuple(
                (name, tuple((key, id(value)) for key, value in getattr(module, name).items()))
                for name in (
                    "_modules",
                    "_parameters",
                    "_buffers",
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_backward_pre_hooks",
                )
            ),
            frozenset(module._non_persistent_buffers_set),
        )
        for path, module in prepared.named_modules()
    )
    try:
        with torch.inference_mode(False), torch.no_grad():
            # Visit the original prepared module list, not a graph that a callback
            # might mutate. Every Tensor already contains its final saved values.
            for path, module, _, _ in registrations:
                if type(module).set_extra_state is not nn.Module.set_extra_state:
                    module.set_extra_state(values[f"{path}._extra_state".lstrip(".")])
    except Exception as error:
        raise ExecutionError(f"State loading failed before commit: {error}") from error
    finally:
        for _, module, tables, persistence in registrations:
            if (
                any(
                    tuple((key, id(value)) for key, value in getattr(module, name).items())
                    != entries
                    for name, entries in tables
                )
                or frozenset(module._non_persistent_buffers_set) != persistence
            ):
                raise ExecutionError(
                    "Extra state changed registered structure, tensor bindings, or runtime hooks"
                )
    _check_state_contract(prepared)
    try:
        final = snapshot(prepared)
    except Exception as error:
        raise ExecutionError(f"Extra state produced an invalid structure: {error}") from error
    if final.tensors != tuple(
        replace(state, device=device)
        for state, device in zip(structure.tensors, prepared_devices, strict=True)
    ):
        raise ExecutionError(
            "Extra state changed the declared final structure or mapped tensor devices"
        )
    if final.modules != structure.modules:
        raise ExecutionError(
            "Restored module structure or configuration differs from the checkpoint"
        )

    if custom_state:
        for state in structure.tensors:
            tensor = getattr(*attribute(prepared, state.paths[0]))
            source = values if state.persistent[0] else buffers
            if not _same_values(tensor, source[state.paths[0]]) or has_tensor_hooks(tensor):
                raise ExecutionError("Extra state changed registered tensor values or hooks")
    if not reference_devices_match(
        structure.references, reference_signature(prepared), mapped_devices
    ):
        raise ExecutionError("Restored ordinary tensor references differ from the checkpoint")
    # Transfer prepared custom state as binding assignments, preserving rollback
    # for ordinary failures without invoking set_extra_state on the original.
    extra_changes = final_state_edits(model, prepared)
    record = managed_record(prepared, final, managed)
    commit(model, replacements, (*changes, *extra_changes), record, expected=final)
    return model
