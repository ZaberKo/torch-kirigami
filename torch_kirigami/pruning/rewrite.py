"""Declarative compaction recipes and conservative original-forward checks."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch
from torch import nn

from ..bindings import attribute_owner, has_tensor_hooks
from ..configuration import freeze, thaw
from ..dep_graph.contracts import AttributeBinding, Impact, Requirement
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.operation import Operation, OperatorSpec
from ..dep_graph.operators.coordinates import retained_indices as _keep
from ..dep_graph.operators.defaults import is_native_rule
from ..dep_graph.operators.shapes import evaluate
from ..dep_graph.operators.shapes import reevaluate as _reevaluate
from ..dep_graph.selection import IndexSet, Region, Selection, TensorRef
from ..errors import CaptureError, UnsupportedOperation
from ..operator_recipes import AttributeRecipe, OperatorRecipe, TensorRecipe
from .recipes import compact_stride, memory_format, same_mapping, validate_recipe
from .types import PlanningError, require_compact_shape
from .validation import check_forward


def ordinary_recipe(selection: Selection) -> TensorRecipe:
    """Describe a single Cartesian compaction, or reject its nonrectangular layout."""
    if selection.compact_shape() is None:
        raise PlanningError(
            f"Partitioned tensor needs an explicit rewrite: {selection.tensor.paths}"
        )
    ref = selection.tensor
    return TensorRecipe(
        ref,
        (
            Region(
                tuple(
                    IndexSet.span(0, n).subtract(selection.fully_selected_indices(d))
                    for d, n in enumerate(ref.shape)
                )
            ),
        ),
    )


def _bound_attribute(
    graph: DependencyGraph, binding: AttributeBinding, impact: Impact
) -> AttributeRecipe:
    """Evaluate one call's attribute demand, including an unchanged value."""
    owner, name = attribute_owner(graph.model, binding.target)
    old = getattr(owner, name)
    new = evaluate(binding.expression, lambda ref: require_compact_shape(impact, ref))
    if type(old) in (list, tuple, torch.Size) and isinstance(new, tuple):
        new = type(old)(new)
    return AttributeRecipe(binding.target, old, new)


def default_recipe(
    graph: DependencyGraph,
    op: Operation,
    spec: OperatorSpec,
    impact: Impact,
    requirements: tuple[Requirement | AttributeBinding, ...],
) -> OperatorRecipe:
    """Lower shared layouts and bound attributes without dispatching on operators."""

    def compact_shape(ref: TensorRef) -> tuple[int, ...]:
        """Evaluate existing size expressions against the joint removal."""
        return require_compact_shape(impact, ref)

    recipes, attributes, notes = [], [], []
    for req in requirements:
        if isinstance(req, AttributeBinding):
            # An unchanged binding is still a demand. A shared owner may have
            # another call requiring a different value for the same attribute.
            attributes.append(_bound_attribute(graph, req, impact))
            continue
        check = req.check
        if check is None:
            raise PlanningError(f"Unhandled requirement {req.kind}: {req.target}")
        if check == "argument":
            data = dict(req.data)
            if len(req.arguments) != 1:
                raise PlanningError(f"Expected one functional argument binding for {req.target}")
            name, position = req.arguments[0].name, req.arguments[0].position
            raw = op.raw_argument(name, position)
            current = _reevaluate(raw, op.argument(name, position), op.expressions, compact_shape)
            expected = (
                len(_keep(impact, data["axis"]))
                if "axis" in data
                else tuple(len(_keep(impact, a)) for a in data["axes"])
            )
            if isinstance(current, list):
                current = tuple(current)
            if current != expected:
                raise PlanningError(
                    f"{req.target}: changing a functional argument requires editing original forward"
                )
        elif check == "note":
            notes.append(req.detail)
    for descriptor in spec.layouts:
        if impact.selection(descriptor.tensor):
            recipes.append(
                TensorRecipe(
                    descriptor.tensor,
                    descriptor.retained_regions(impact.selection(descriptor.tensor)),
                    descriptor.concat_dim,
                )
            )
    return OperatorRecipe(tuple(recipes), tuple(attributes), notes=tuple(notes))


def _requirements_by_operation(
    model: nn.Module,
    module_paths: Mapping[str, Mapping[int, str]],
    requirements: Sequence[Requirement | AttributeBinding],
) -> dict[str, tuple[Requirement | AttributeBinding, ...]]:
    """Index each requirement by call name or the identity of its attribute owner."""
    by_name: dict[str, list[Requirement | AttributeBinding]] = {name: [] for name in module_paths}
    calls_by_owner: dict[int, list[str]] = defaultdict(list)
    for call, paths in module_paths.items():
        for owner in paths:
            calls_by_owner[owner].append(call)
    for requirement in requirements:
        if isinstance(requirement, AttributeBinding):
            owner, _ = attribute_owner(model, requirement.target)
            for call in calls_by_owner.get(id(owner), ()):
                by_name[call].append(requirement)
        elif requirement.check != "tensor_metadata" and requirement.target in by_name:
            by_name[requirement.target].append(requirement)
    return {name: tuple(items) for name, items in by_name.items()}


def compile_recipes(
    graph: DependencyGraph,
    operations: Sequence[Operation],
    impact: Impact,
    *,
    attribute_checks: OrderedDict[tuple[tuple[str, object], ...], str | None] | None = None,
    meta_outputs: dict[str, tuple[object, object]] | None = None,
) -> tuple[tuple[TensorRecipe, ...], tuple[AttributeRecipe, ...], tuple[str, ...]]:
    """Prove all affected requirements and combine per-use recipes without weights."""
    if impact.status != "resolved":
        raise PlanningError("; ".join(map(str, impact.diagnostics)))
    bindings = dict(graph.tensor_bindings())
    for selection in (*impact.parameters, *impact.buffers):
        if has_tensor_hooks(bindings[selection.tensor]):
            raise PlanningError("Remove Tensor gradient hooks before replacing their tensors")
        expected_type = nn.Parameter if selection.tensor.kind == "parameter" else torch.Tensor
        if type(bindings[selection.tensor]) is not expected_type:
            raise PlanningError("Physical replacement of custom tensor subclasses is unsupported")
    for ref, path in graph.constants():
        if impact.selection(ref):
            raise PlanningError(
                f"Unregistered captured constant {path} has no proved original binding"
            )
    recipes, notes, handled, strides = {}, [], set(), {}
    active = []
    attribute_bindings = {}
    affected = graph.affected_operations(impact)
    affected_calls = tuple(op for op in operations if op.name in affected)
    # An opaque call owns the attributes of its registered descendants, even
    # when a requirement names a path outside the call to the same descendant.
    module_paths = {
        op.name: (
            {id(module): path for path, module in op.module.named_modules()}
            if op.module is not None
            else {}
        )
        for op in affected_calls
    }
    requirements = _requirements_by_operation(graph.model, module_paths, impact.requirements)
    for op in affected_calls:
        reqs = requirements[op.node.name]
        rule = graph.operator_rule(op)
        if rule is None:
            raise PlanningError(f"No execution rule for {op.node.target}")
        spec = graph.operator_spec(op)
        if spec is None:
            raise PlanningError(f"No structural specification for {op.name}")
        custom = None
        if not is_native_rule(op.node, op.module, rule):
            try:
                custom = rule.build_recipe(op, spec, impact)
            except UnsupportedOperation as error:
                raise PlanningError(str(error)) from error
            finally:
                graph.validate()
        result = default_recipe(graph, op, spec, impact, reqs)
        if custom is not None:
            if not isinstance(custom, OperatorRecipe):
                raise TypeError("build_recipe must return OperatorRecipe or None")
            # Explicit partition layouts remain authoritative, even for custom edits.
            declared = {r.tensor.id: r for r in result.tensors}
            for recipe in custom.tensors:
                if recipe.tensor.id in declared and not same_mapping(
                    declared[recipe.tensor.id], recipe
                ):
                    raise PlanningError("Custom recipe contradicts declared partition layout")
            result = OperatorRecipe(
                tuple(
                    r for r in result.tensors if r.tensor not in {c.tensor for c in custom.tensors}
                )
                + custom.tensors,
                result.attributes + custom.attributes,
                result.notes + custom.notes,
                custom.output_strides,
            )
        handled.update(id(r) for r in reqs)
        notes.extend(result.notes)
        for ref, stride in result.output_strides:
            if ref not in op.outputs:
                raise PlanningError("Output stride declarations must name this operation's outputs")
            strides[ref.id] = stride
        explicit = {r.tensor.id: r for r in result.tensors}
        if len(explicit) != len(result.tensors):
            raise PlanningError("A rewrite rule returned duplicate tensor recipes")
        for ref in dict.fromkeys((*op.inputs, *op.bindings.values())):
            if (
                ref.kind in ("parameter", "buffer")
                and impact.selection(ref)
                and ref.id not in explicit
            ):
                explicit[ref.id] = ordinary_recipe(impact.selection(ref))
        for recipe in explicit.values():
            graph.metadata(recipe.tensor)
            validate_recipe(recipe, impact)
            previous = recipes.get(recipe.tensor.id)
            if previous is not None and not same_mapping(previous, recipe):
                raise PlanningError(
                    f"Shared tensor requires incompatible coordinate mappings: {recipe.tensor.paths}"
                )
            recipes[recipe.tensor.id] = recipe
        for attr in result.attributes:
            owner, name = attribute_owner(graph.model, attr.path)
            if freeze(getattr(owner, name)) != attr.old:
                raise PlanningError(f"Rewrite attribute precondition disagrees: {attr.path}")
            identity = (id(owner), name)
            previous = attribute_bindings.setdefault(identity, attr)
            if previous.new != attr.new:
                raise PlanningError(f"Shared attribute update disagrees: {attr.path}")
        active.append((op, spec, reqs, rule))
    attributes = []
    edits_by_owner: dict[int, list[tuple[str, object]]] = defaultdict(list)
    for (owner_id, name), attr in attribute_bindings.items():
        if attr.old != attr.new:
            attributes.append(attr)
            edits_by_owner[owner_id].append((name, attr.new))
    if attributes:
        # An independent branch can call the same module without selecting any
        # of its tensors. Its declared demand still applies to a shared edit.
        changed_names = {name for edits in edits_by_owner.values() for name, _ in edits}
        for op in operations:
            if op.name in affected:
                continue
            spec = graph.operator_spec(op)
            if spec is None:
                continue
            for req in spec.requirements:
                if not isinstance(req, AttributeBinding):
                    continue
                name = req.target.rsplit(".", 1)[-1]
                if name not in changed_names:
                    continue
                owner, name = attribute_owner(graph.model, req.target)
                shared = attribute_bindings.get((id(owner), name))
                if (
                    shared is not None
                    and shared.old != shared.new
                    and shared.new != _bound_attribute(graph, req, impact).new
                ):
                    raise PlanningError(f"Shared attribute update disagrees: {req.target}")
    for selection in (*impact.parameters, *impact.buffers):
        if selection.tensor.id not in recipes:
            recipe = ordinary_recipe(selection)
            validate_recipe(recipe, impact)
            recipes[selection.tensor.id] = recipe
    # Validate precisely the recipes that apply will execute, including layout.
    recipes = {
        key: replace(recipe, memory_format=memory_format(bindings[recipe.tensor]))
        for key, recipe in recipes.items()
    }
    for req in impact.requirements:
        if not isinstance(req, Requirement) or req.check != "tensor_metadata":
            continue
        ref = req.tensors[0]
        recipe = recipes[ref.id]
        proposed = torch.empty_strided(
            recipe.shape,
            compact_stride(recipe.shape, recipe.memory_format),
            device="meta",
            dtype=graph.metadata(ref).dtype,
        )
        data = dict(req.data)
        value = (
            (proposed.stride() if data["argument"] is None else proposed.stride(data["argument"]))
            if data["kind"] == "stride"
            else proposed.is_contiguous(
                memory_format=getattr(torch, data["argument"].removeprefix("torch."))
            )
        )
        if value != data["observed"]:
            raise PlanningError(f"Compaction changes metadata {req.target}.{data['kind']}")
        handled.add(id(req))
    if any(id(r) not in handled for r in impact.requirements):
        raise PlanningError("Impact contains an unhandled execution requirement")

    call_edits = {
        call: tuple(
            (f"{path}.{name}".lstrip("."), value)
            for owner_id, path in paths.items()
            for name, value in edits_by_owner.get(owner_id, ())
        )
        for call, paths in module_paths.items()
        if edits_by_owner and paths
    }
    check_forward(
        graph, operations, active, impact, recipes, call_edits, strides, meta_outputs=meta_outputs
    )
    # Attribute validation depends on configuration, not which same-width channels
    # were selected. The context owns/invalidate this bounded cache; manual and final
    # independent compilation still validate without sharing another context's cache.
    key = tuple(sorted((a.path, a.new) for a in attributes))
    if attribute_checks is not None and key in attribute_checks:
        error = attribute_checks[key]
        attribute_checks.move_to_end(key)
        if error is not None:
            raise PlanningError(error)
    else:
        error = None
        try:
            graph.validate_attribute_changes((a.path, thaw(a.new)) for a in attributes)
        except CaptureError as cause:
            error = str(cause)
        if attribute_checks is not None:
            attribute_checks[key] = error
        if error is not None:
            raise PlanningError(error)
    # tensor_bindings() checked entry state; check exit even when the cached
    # attribute proof avoided re-tracing. Generic lowering needs no per-call scan.
    graph.validate()
    return tuple(recipes.values()), tuple(attributes), tuple(dict.fromkeys(notes))
