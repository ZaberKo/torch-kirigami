"""Portable structural preconditions and atomic binding transactions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from math import prod
from typing import Protocol

import torch
from torch import nn

from ..bindings import AttributeEdit, reference_edits, reference_signature, storage_key
from ..configuration import attributes as configuration_attributes
from ..configuration import forward_hook_paths, freeze, has_registration_hooks, thaw
from .recipes import compact_stride, validate_recipe
from .types import (
    AnalysisSummary,
    AttributeRecipe,
    ExecutionError,
    ModelStructure,
    ModuleState,
    ParameterReport,
    SelectionReport,
    TensorRecipe,
    TensorState,
)

STRUCTURE_ATTRIBUTE = "_kirigami_structure"


class _PlanContract(Protocol):
    """Structural plan fields required for independent validation."""

    @property
    def before(self) -> ModelStructure:
        """Return original structural preconditions."""
        ...

    @property
    def after(self) -> ModelStructure:
        """Return expected compact structure."""
        ...

    @property
    def analysis(self) -> AnalysisSummary:
        """Return the portable dependency analysis."""
        ...

    @property
    def recipes(self) -> tuple[TensorRecipe, ...]:
        """Return tensor replacement descriptions."""
        ...

    @property
    def attributes(self) -> tuple[AttributeRecipe, ...]:
        """Return attribute replacement descriptions."""
        ...

    @property
    def selection_report(self) -> SelectionReport | ParameterReport:
        """Return the completed selection and budget report."""
        ...


def attribute(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    """Resolve a registered or declared attribute to its original owner."""
    parent, _, name = path.rpartition(".")
    return model.get_submodule(parent) if parent else model, name


def snapshot(model: nn.Module, *, guarded: tuple[str, ...] = ()) -> ModelStructure:
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


def transformed(
    before: ModelStructure,
    recipes: tuple[TensorRecipe, ...],
    attributes: tuple[AttributeRecipe, ...],
) -> ModelStructure:
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
        assigned = {}
        for path in state.paths:
            for name, old in attrs.items():
                key = f"{path}.{name}".lstrip(".")
                if key not in edits:
                    continue
                edit = edits[key]
                if old != freeze(thaw(edit.old)):
                    raise ValueError(f"Attribute precondition mismatch: {key}")
                new = freeze(thaw(edit.new))
                # All aliases describe one final assignment, including a no-op.
                if name in assigned and assigned[name] != new:
                    raise ValueError("Shared attribute edits disagree")
                assigned[name] = new
                consumed.add(key)
        attrs.update(assigned)
        modules.append(replace(state, attributes=tuple(sorted(attrs.items()))))
    if consumed != set(edits):
        raise ValueError("Attribute recipe has no portable original binding")
    return ModelStructure(tuple(modules), tuple(tensors), before.references)


def validate_plan(plan: _PlanContract) -> None:
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
    report = plan.selection_report
    if isinstance(report, ParameterReport):
        before, after = (
            sum(prod(s.shape) for s in structure.tensors if s.kind == "parameter")
            for structure in (plan.before, plan.after)
        )
        if (report.before_params, report.after_params) != (before, after) or not report.target_met:
            raise ValueError(
                "Parameter report disagrees with plan structures or exceeds its target"
            )


def check_structure(model: nn.Module, expected: ModelStructure) -> None:
    """Reject detectable type, binding, mode, configuration, or layout changes."""
    hooks = forward_hook_paths(model)
    if hooks:
        raise ExecutionError(f"Forward hooks are outside plan semantics: {hooks}")
    guarded = tuple(s.paths[0] for s in expected.tensors if s.values is not None)
    if snapshot(model, guarded=guarded) != expected:
        raise ExecutionError("Model structure/mode/configuration does not match plan preconditions")


def commit(
    model: nn.Module,
    replacements: Sequence[tuple[TensorState, torch.Tensor, torch.Tensor]],
    attributes: Iterable[AttributeRecipe | AttributeEdit],
    record: dict[str, object],
    *,
    expected: ModelStructure,
) -> None:
    """Commit prepared tensor bindings and attributes, restoring ordinary failures.

    Args:
        model: Original module to modify.
        replacements: Tuples of TensorState, original tensor, and prepared tensor.
        attributes: Validated AttributeRecipe objects.
        record: Pure structural metadata to attach only on successful application.
        expected: Final structural postcondition, checked inside the transaction.
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
    registrations = [
        (
            m,
            dict(m._parameters),
            dict(m._buffers),
            dict(m._modules),
            set(m._non_persistent_buffers_set),
        )
        for m in model.modules()
    ]
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
                object.__delattr__(owner, name)
            elif isinstance(edit.value, (torch.Tensor, nn.Module)):
                # Ordinary references must never enter Module's auto-registration path.
                object.__setattr__(owner, name, edit.value)
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
        # Check tables before recursively traversing the final graph: a faulty
        # setter could otherwise introduce a registration cycle.
        for owner, parameters, buffers, modules, persistence in registrations:
            if (
                owner._parameters.keys() != parameters.keys()
                or owner._buffers.keys() != buffers.keys()
                or owner._modules != modules
                or owner._non_persistent_buffers_set != persistence
            ):
                raise ExecutionError("Commit changed registration categories or module bindings")
        check_structure(model, expected)
    except Exception as error:
        for owner, parameters, buffers, modules, persistence in registrations:
            owner._parameters.clear()
            owner._parameters.update(parameters)
            owner._buffers.clear()
            owner._buffers.update(buffers)
            owner._modules.clear()
            owner._modules.update(modules)
            owner._non_persistent_buffers_set.clear()
            owner._non_persistent_buffers_set.update(persistence)
        for owner, name, old, kind in reversed(journal):
            if kind == "parameter":
                owner._parameters[name] = old
            elif kind == "buffer":
                owner._buffers[name] = old
            elif old is missing:
                if hasattr(owner, name):
                    object.__delattr__(owner, name)
            else:
                object.__setattr__(owner, name, old)
        raise ExecutionError(
            "Commit failed; bindings, attributes, and structural metadata restored"
        ) from error


def managed_record(
    model: nn.Module, structure: ModelStructure, attribute_paths: Iterable[str] = ()
) -> dict[str, object]:
    """Track managed structure without retaining weights, runtime graphs, or history."""
    previous = getattr(model, STRUCTURE_ATTRIBUTE, {})
    paths = set(previous.get("attributes", ())) | set(attribute_paths)
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


def validate_managed(model: nn.Module, structure: ModelStructure) -> None:
    """Detect external structural changes while allowing training and device moves."""
    previous = getattr(model, STRUCTURE_ATTRIBUTE, None)
    if previous is not None and previous != managed_record(model, structure):
        raise ExecutionError(
            "Managed model structure changed outside pruning; cannot infer repairs"
        )
