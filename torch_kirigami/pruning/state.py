"""Portable structural preconditions and atomic binding transactions."""

from __future__ import annotations

from dataclasses import replace

from ..bindings import AttributeEdit, reference_edits, reference_signature, storage_key
from ..configuration import attributes as configuration_attributes
from ..configuration import forward_hook_paths, freeze, has_registration_hooks, thaw
from .recipes import compact_stride, validate_recipe
from .types import ExecutionError, ModelStructure, ModuleState, TensorState

STRUCTURE_ATTRIBUTE = "_kirigami_structure"


def attribute(model, path):
    """Resolve a registered or declared attribute to its original owner."""
    parent, _, name = path.rpartition(".")
    return model.get_submodule(parent) if parent else model, name


def snapshot(model, *, guarded=()):
    """Read structural state; ordinary weight updates do not change this record."""
    modules = {}
    for path, module in model.named_modules(remove_duplicate=False):
        modules.setdefault(id(module), (module, []))[1].append(path)
    module_states = []
    for module, paths in modules.values():
        attrs = configuration_attributes(module)
        slots = tuple(
            (kind, name, name not in module._non_persistent_buffers_set)
            for kind, table in (("parameter", module._parameters), ("buffer", module._buffers))
            for name in table
        )
        module_states.append(
            ModuleState(
                tuple(paths),
                f"{type(module).__module__}.{type(module).__qualname__}",
                attrs,
                slots,
            )
        )
    entities, storage = {}, {}
    for kind, items in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for path, tensor in items:
            if id(tensor) in entities and entities[id(tensor)][1] != kind:
                raise ValueError("A tensor cannot share parameter and buffer registration kinds")
            entities.setdefault(id(tensor), (tensor, kind, []))[2].append(path)
    for tensor, _, paths in entities.values():
        key = storage_key(tensor)
        if key is not None:
            storage.setdefault(key, []).append(paths[0])
    tensors = []
    for tensor, kind, paths in entities.values():
        persistence = []
        for path in paths:
            owner, name = attribute(model, path)
            persistence.append(kind == "parameter" or name not in owner._non_persistent_buffers_set)
        tensors.append(
            TensorState(
                tuple(paths),
                kind,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tuple(persistence),
                tuple(storage.get(storage_key(tensor), paths[:1])),
                f"{type(tensor).__module__}.{type(tensor).__qualname__}",
                tuple(tensor.detach().cpu().reshape(-1).tolist())
                if any(p in guarded for p in paths)
                else None,
            )
        )
    return ModelStructure(tuple(module_states), tuple(tensors), reference_signature(model))


def transformed(before, recipes, attributes):
    """Derive the final structural state from validated declarative modifications."""
    replacements = {r.tensor.paths[0]: r for r in recipes}
    known = {p for state in before.tensors for p in state.paths}
    if any(p not in known for p in replacements):
        raise ValueError("Recipe refers to an unknown tensor")
    tensors = []
    for state in before.tensors:
        recipe = replacements.get(state.paths[0])
        tensors.append(
            state
            if recipe is None
            else replace(
                state,
                shape=recipe.shape,
                stride=compact_stride(recipe.shape, recipe.memory_format),
                storage_aliases=(state.paths[0],),
            )
        )
    edits, modules = {a.path: a for a in attributes}, []
    consumed = set()
    for state in before.modules:
        attrs = dict(state.attributes)
        for path in state.paths:
            for name in tuple(attrs):
                key = f"{path}.{name}".lstrip(".")
                if key in edits:
                    edit = edits[key]
                    if dict(state.attributes)[name] != freeze(thaw(edit.old)):
                        raise ValueError(f"Attribute precondition mismatch: {key}")
                    if name in {k for k, v in state.attributes if v != attrs[k]} and attrs[
                        name
                    ] != freeze(thaw(edit.new)):
                        raise ValueError("Shared attribute edits disagree")
                    attrs[name] = freeze(thaw(edit.new))
                    consumed.add(key)
        modules.append(replace(state, attributes=tuple(sorted(attrs.items()))))
    if consumed != set(edits):
        raise ValueError("Attribute recipe has no portable original binding")
    return ModelStructure(tuple(modules), tuple(tensors), before.references)


def validate_plan(plan):
    """Check recipe consistency independently of issuance identity or live Impact."""
    if not isinstance(plan.before, ModelStructure) or not isinstance(plan.after, ModelStructure):
        raise ValueError("Plan requires structural preconditions and postconditions")
    if plan.analysis.status != "resolved":
        raise ValueError("Only resolved plans can execute")
    states = {s.paths[0]: s for s in plan.before.tensors}
    seen = set()
    for recipe in plan.recipes:
        ref = recipe.tensor
        if not ref.paths or ref.paths[0] in seen:
            raise ValueError("Duplicate or unbound tensor recipe")
        seen.add(ref.paths[0])
        state = states.get(ref.paths[0])
        if state is None or (state.paths, state.kind, state.shape) != (
            ref.paths,
            ref.kind,
            ref.shape,
        ):
            raise ValueError("Tensor recipe disagrees with its original binding")
        if len(state.storage_aliases) > 1:
            raise ValueError("Compacting distinct tensors sharing storage is unsupported")
        validate_recipe(recipe, plan.analysis)
        compact_stride(recipe.shape, recipe.memory_format)
    required = {
        s.tensor.paths[0]
        for s in plan.analysis.selections
        if s and s.tensor.kind in ("parameter", "buffer")
    }
    if required != seen:
        raise ValueError("Recipes do not cover the frozen parameter/buffer selections")
    if len({a.path for a in plan.attributes}) != len(plan.attributes):
        raise ValueError("Duplicate attribute recipe")
    if transformed(plan.before, plan.recipes, plan.attributes) != plan.after:
        raise ValueError("Plan postconditions disagree with modification recipes")


def check_structure(model, expected):
    """Reject detectable type, binding, mode, configuration, or layout changes."""
    hooks = forward_hook_paths(model)
    if hooks:
        raise ExecutionError(f"Forward hooks are outside plan semantics: {hooks}")
    guarded = tuple(s.paths[0] for s in expected.tensors if s.values is not None)
    if snapshot(model, guarded=guarded) != expected:
        raise ExecutionError("Model structure/mode/configuration does not match plan preconditions")


def commit(model, replacements, attributes, record):
    """Commit prepared tensor bindings and attributes, restoring ordinary failures.

    Args:
        model: Original module to modify.
        replacements: Tuples of TensorState, original tensor, and prepared tensor.
        attributes: Validated AttributeRecipe objects.
        record: Pure structural metadata to attach only on successful application.
    """
    if has_registration_hooks():
        raise ExecutionError("Global registration hooks are unsupported during structural commit")
    # Every lifecycle uses the same reference rebinding. Explicit final-state
    # edits from checkpoint take precedence over constructor container copies.
    implicit = reference_edits(model, {id(old): new for _, old, new in replacements})
    edits = {edit.path: edit for edit in implicit}
    for edit in attributes:
        normalized = (
            edit if isinstance(edit, AttributeEdit) else AttributeEdit(edit.path, thaw(edit.new))
        )
        edits[normalized.path] = normalized
    missing = object()
    journal = []
    try:
        for state, old, new in replacements:
            for path in state.paths:
                owner, name = attribute(model, path)
                journal.append((owner, name, old, state.kind))
                setattr(owner, name, new)
        for edit in edits.values():
            owner, name = attribute(model, edit.path)
            journal.append((owner, name, getattr(owner, name, missing), "attribute"))
            if edit.delete:
                delattr(owner, name)
            else:
                setattr(owner, name, edit.value)
        journal.append(
            (model, STRUCTURE_ATTRIBUTE, getattr(model, STRUCTURE_ATTRIBUTE, missing), "attribute")
        )
        setattr(model, STRUCTURE_ATTRIBUTE, record)
        for state, _, new in replacements:
            for path in state.paths:
                owner, name = attribute(model, path)
                if getattr(owner, name) is not new:
                    raise ExecutionError(
                        "Committed tensor binding differs from the prepared object"
                    )
    except Exception as error:
        for owner, name, old, kind in reversed(journal):
            if kind == "parameter":
                owner._parameters[name] = old
            elif kind == "buffer":
                owner._buffers[name] = old
            elif old is missing:
                owner.__dict__.pop(name, None)
            else:
                object.__setattr__(owner, name, old)
        raise ExecutionError(
            "Commit failed; bindings, attributes, and structural metadata restored"
        ) from error


def managed_record(model, structure, attributes=()):
    """Track managed structure without retaining weights, runtime graphs, or history."""
    previous = getattr(model, STRUCTURE_ATTRIBUTE, {})
    paths = set(previous.get("attributes", ())) | {a.path for a in attributes}
    values = []
    for path in sorted(paths):
        parent, _, name = path.rpartition(".")
        state = next(s for s in structure.modules if parent in s.paths)
        values.append((path, dict(state.attributes)[name]))
    return {
        "attributes": tuple(sorted(paths)),
        "values": tuple(values),
        "tensors": tuple((s.paths, s.kind, s.shape) for s in structure.tensors),
        "modules": tuple((s.paths, s.type_name, s.slots) for s in structure.modules),
        "references": structure.references,
    }


def validate_managed(model, structure):
    """Detect external structural changes while allowing training and device moves."""
    previous = getattr(model, STRUCTURE_ATTRIBUTE, None)
    if previous is not None and previous != managed_record(model, structure):
        raise ExecutionError(
            "Managed model structure changed outside pruning; cannot infer repairs"
        )
