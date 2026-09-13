"""Final-structure checkpoints, independent of graph capture and pruning history."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from ..bindings import copy_module_state, final_state_edits, has_tensor_hooks, storage_key
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


def _check_state_hooks(model):
    # The format binds payload names to registered slots before allocation.
    # Arbitrary state_dict rewrites need a separate storage/schema contract.
    for path, module in model.named_modules():
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


def _validate_extra_state(model, values):
    """Accept only weights-only data without hidden registered storage references."""
    registered = {storage_key(t) for t in (*model.parameters(), *model.buffers()) if t.numel()}
    registered.update(
        storage_key(t)
        for name, t in values.items()
        if not name.endswith("_extra_state") and isinstance(t, torch.Tensor) and t.numel()
    )

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
            if not allow_registered and value.numel() and storage_key(value) in registered:
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

    if type(values) not in (dict, OrderedDict) or (
        type(values) is OrderedDict and set(vars(values)) - {"_metadata"}
    ):
        raise ExecutionError("Unsupported checkpoint state dictionary container")
    visit(getattr(values, "_metadata", None))
    for name, value in values.items():
        if name.endswith("_extra_state"):
            visit(value)
        elif isinstance(value, torch.Tensor):
            if type(value) not in (torch.Tensor, nn.Parameter):
                raise ExecutionError("Checkpoint tensor payload has unsupported subclass")
            visit(value, allow_registered=True)


def _validate_payload(model, structure, values, buffers):
    """Validate actual saved values, independently of how state_dict produced them."""
    if not isinstance(values, dict) or not isinstance(buffers, dict):
        raise ExecutionError("Checkpoint requires state dictionaries")
    expected = {
        p
        for s in structure.tensors
        for p, persistent in zip(s.paths, s.persistent, strict=True)
        if persistent
    }
    expected.update(
        f"{path}._extra_state".lstrip(".")
        for path, module in model.named_modules(remove_duplicate=False)
        if type(module).get_extra_state is not nn.Module.get_extra_state
    )
    expected_buffers = {
        p
        for s in structure.tensors
        for p, persistent in zip(s.paths, s.persistent, strict=True)
        if not persistent
    }
    if set(values) != expected or set(buffers) != expected_buffers:
        raise ExecutionError("Checkpoint payload keys disagree with registered structure")
    _validate_extra_state(model, values)
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

    Registered nonpersistent buffers are included explicitly. Model code and its
    original constructor configuration remain the caller's responsibility.
    """
    _check_state_hooks(model)
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
    state = model.state_dict()
    buffers = {}
    for item in structure.tensors:
        for name, persistent in zip(item.paths, item.persistent, strict=True):
            if item.kind == "buffer" and not persistent:
                owner, field = attribute(model, name)
                buffers[name] = getattr(owner, field).detach()
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

    Ordinary registered state is prepared before committing. get/set_extra_state
    runs on an isolated module shell; external side effects remain outside the
    transaction contract. Registered state_dict hooks are explicitly unsupported.
    """
    _check_state_hooks(model)
    if any(has_tensor_hooks(t) for t in (*model.parameters(), *model.buffers())):
        raise ExecutionError(
            "Remove Tensor gradient hooks before loading; register them on restored tensors"
        )
    data = torch.load(path, map_location=map_location, weights_only=True)
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
    standard_load = not custom_state and all(
        type(m).load_state_dict is nn.Module.load_state_dict
        and type(m)._load_from_state_dict is nn.Module._load_from_state_dict
        for m in model.modules()
    )
    custom_loader = any(
        not getattr(method, "__module__", "").startswith("torch.nn.")
        for m in model.modules()
        for method in (type(m).load_state_dict, type(m)._load_from_state_dict)
    )
    pending_resize = {}
    with torch.inference_mode(False), torch.no_grad():
        for old, state in zip(original.tensors, structure.tensors, strict=True):
            _check_nonoverlap(state)
            if len(state.storage_aliases) > 1:
                raise ExecutionError("Unsupported checkpoint storage sharing")
            entries = []
            for name, persistent in zip(state.paths, state.persistent, strict=True):
                source = values if persistent else buffers
                value = source.get(name)
                entries.append(value)
            tensor = torch.empty_strided(
                state.shape, state.stride, dtype=entries[0].dtype, device=entries[0].device
            )
            if custom_loader and any(state.persistent):
                # A custom loader may use constructor values (e.g. delta encoding).
                # Checkpoint payloads are input to the loader, never initial state.
                initial = getattr(*attribute(model, old.paths[0]))
                if old.shape != state.shape:
                    # Keep constructor values until the native loader has run
                    # custom decoding and reaches its ordinary slot-loading step.
                    pending_resize[state.paths[0]] = tensor
                    tensor = torch.empty_strided(
                        old.shape, old.stride, dtype=tensor.dtype, device=tensor.device
                    )
                tensor.copy_(initial)
            elif not standard_load or not any(state.persistent):
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
    handles = []
    resize_by_owner = {}
    for path, tensor in pending_resize.items():
        owner, name = attribute(prepared, path)
        resize_by_owner.setdefault(owner, []).append((name, tensor))

    def install_targets(owner, *args):
        # A public pre-hook runs inside Module._load_from_state_dict, after an
        # override's decoding. set_ retains shared object/container identities.
        for name, target in resize_by_owner[owner]:
            getattr(owner, name).set_(target)

    for owner in resize_by_owner:
        handles.append(owner.register_load_state_dict_pre_hook(install_targets))
    try:
        payload = OrderedDict(values)
        if hasattr(values, "_metadata"):
            payload._metadata = values._metadata
        with torch.inference_mode(False), torch.no_grad():
            prepared.load_state_dict(payload, strict=True)
    except Exception as error:
        raise ExecutionError(f"State loading failed before commit: {error}") from error
    finally:
        for handle in handles:
            handle.remove()
    try:
        final = snapshot(prepared)
    except Exception as error:
        raise ExecutionError(f"Loading hooks produced an invalid structure: {error}") from error
    if skeleton(final) != skeleton(structure) or tuple(
        (
            s.paths,
            s.kind,
            s.shape,
            s.stride,
            s.dtype,
            s.requires_grad,
            s.persistent,
            s.storage_aliases,
            s.type_name,
        )
        for s in final.tensors
    ) != tuple(
        (
            s.paths,
            s.kind,
            s.shape,
            s.stride,
            s.dtype,
            s.requires_grad,
            s.persistent,
            s.storage_aliases,
            s.type_name,
        )
        for s in structure.tensors
    ):
        raise ExecutionError("Loading hooks changed the declared final structure")
    if tuple(s.device for s in final.tensors) != prepared_devices:
        raise ExecutionError("Loading hooks changed the mapped tensor devices")
    if tuple(m.attributes for m in final.modules) != tuple(m.attributes for m in structure.modules):
        raise ExecutionError("Restored configuration differs from the checkpoint")
    if final.references != structure.references:
        raise ExecutionError("Restored ordinary tensor references differ from the checkpoint")
    # Transfer prepared custom state as binding assignments, preserving rollback
    # for ordinary failures without invoking set_extra_state on the original.
    extra_changes = final_state_edits(model, prepared)
    replacements = [
        (state, old, getattr(*attribute(prepared, state.paths[0])))
        for state, old, _ in replacements
    ]
    record = managed_record(prepared, final, tuple(AttributeRecipe(p, None, None) for p in managed))
    commit(model, replacements, (*changes, *extra_changes), record, expected=final)
    return model
