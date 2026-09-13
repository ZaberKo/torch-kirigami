"""Declarative compaction recipes and conservative original-forward checks."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ..bindings import has_tensor_hooks
from ..configuration import freeze, thaw
from ..errors import CaptureError
from ..operators.coordinates import retained_indices as _keep
from ..operators.shapes import evaluate
from ..operators.shapes import reevaluate as _reevaluate
from ..selection import IndexSet, Region
from .recipes import compact_stride, memory_format, same_mapping, validate_recipe
from .types import AttributeRecipe, PlanningError, RewriteContext, RewriteResult, TensorRecipe
from .validation import check_forward


def ordinary_recipe(selection):
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


def lower_spec(ctx):
    """Lower shared layouts and bound attributes without dispatching on operators."""
    op, impact = ctx.operation, ctx.impact
    recipes, attributes, notes = [], [], []
    known = {
        "attribute",
        "operator_argument",
        "partitioned_compaction",
        "shape_arguments",
        "dimension_transform",
        "partition_arguments",
        "slice_arguments",
        "reduction_domain",
        "call_arguments",
        "index_arguments",
    }
    for req in ctx.requirements:
        if req.kind not in known:
            raise PlanningError(f"Unhandled requirement {req.kind}: {req.target}")
        data = dict(req.data)
        if req.kind == "attribute":
            path, _, name = req.target.rpartition(".")
            owner = ctx.graph.model.get_submodule(path) if path else ctx.graph.model
            old = getattr(owner, name)
            new = (
                evaluate(data["expression"], ctx.compact_shape)
                if "expression" in data
                else len(_keep(impact, data["axis"]))
                if "axis" in data
                else tuple(len(_keep(impact, a)) for a in data["axes"])
            )
            if type(old) in (list, tuple, torch.Size) and isinstance(new, tuple):
                new = type(old)(new)
            if freeze(new) != freeze(old):
                attributes.append(AttributeRecipe(req.target, old, new))
        elif req.kind == "operator_argument":
            if len(req.arguments) != 1:
                raise PlanningError(f"Expected one functional argument binding for {req.target}")
            name, position = req.arguments[0].name, req.arguments[0].position
            raw = op.raw_argument(name, position)
            current = _reevaluate(
                raw, op.argument(name, position), op.expressions, ctx.compact_shape
            )
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
        elif req.kind == "reduction_domain":
            notes.append(req.detail)
    for descriptor in ctx.spec.layouts:
        if impact.selection(descriptor.tensor):
            recipes.append(
                TensorRecipe(
                    descriptor.tensor,
                    descriptor.retained_regions(impact.selection(descriptor.tensor)),
                    descriptor.concat_dim,
                )
            )
    return RewriteResult(tuple(recipes), tuple(attributes), ctx.requirements, tuple(notes))


def compile_recipes(graph, operations, impact, *, attribute_checks=None):
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
    recipes, attributes, notes, handled, strides = {}, {}, [], set(), {}
    active = []
    attribute_bindings = {}
    affected = graph.affected_operations(impact)
    for op in operations:
        if op.node.name not in affected:
            continue
        reqs = tuple(
            r
            for r in impact.requirements
            if r.kind != "metadata_layout"
            and (
                r.target == op.node.name
                or (
                    op.module is not None
                    and r.kind == "attribute"
                    and (
                        r.target.rpartition(".")[0] == (op.module_path or "")
                        or r.target.startswith(f"{op.module_path}." if op.module_path else "")
                    )
                )
            )
        )
        rule = graph.operator_rule(op)
        if rule is None:
            raise PlanningError(f"No execution rule for {op.node.target}")
        ctx = RewriteContext(graph, op, impact, reqs)
        result = rule.lower(ctx)
        if result is None:
            result = lower_spec(ctx)
        graph.validate()  # Lowering is an extension boundary, despite its pure contract.
        if not isinstance(result, RewriteResult):
            raise PlanningError("Rewrite rule must return RewriteResult")
        if any(r not in result.handled for r in reqs):
            raise PlanningError(f"{op.node.name}: execution rule left requirements unhandled")
        handled.update(id(r) for r in reqs)
        notes.extend(result.notes)
        strides.update((r.id, s) for r, s in result.output_strides)
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
            path, _, name = attr.path.rpartition(".")
            owner = graph.model.get_submodule(path) if path else graph.model
            if freeze(getattr(owner, name)) != freeze(thaw(attr.old)):
                raise PlanningError(f"Rewrite attribute precondition disagrees: {attr.path}")
            identity = (id(owner), name)
            previous = attribute_bindings.get(identity)
            if previous is not None and (previous.old != attr.old or previous.new != attr.new):
                raise PlanningError(f"Shared attribute update disagrees: {attr.path}")
            attribute_bindings[identity] = attr
            attributes[attr.path] = attr
        active.append((ctx, rule.evaluate_on_meta))
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
        if req.kind != "metadata_layout":
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
    check_forward(graph, operations, active, impact, recipes, attributes, strides)
    # Attribute validation depends on configuration, not which same-width channels
    # were selected. The context owns/invalidate this bounded cache; manual and final
    # independent compilation still validate without sharing another context's cache.
    key = tuple(sorted((a.path, freeze(thaw(a.new))) for a in attributes.values()))
    if attribute_checks is not None and key in attribute_checks:
        error = attribute_checks[key]
        attribute_checks.move_to_end(key)
        if error is not None:
            raise PlanningError(error)
    else:
        error = None
        try:
            graph.validate_attribute_changes((a.path, thaw(a.new)) for a in attributes.values())
        except CaptureError as cause:
            error = str(cause)
        if attribute_checks is not None:
            attribute_checks[key] = error
            if len(attribute_checks) > 32:
                attribute_checks.popitem(last=False)
        if error is not None:
            raise PlanningError(error)
    return tuple(recipes.values()), tuple(attributes.values()), tuple(dict.fromkeys(notes))
