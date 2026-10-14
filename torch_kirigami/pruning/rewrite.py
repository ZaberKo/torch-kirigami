"""Declarative compaction recipes and conservative original-forward checks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import nn

from ..bindings import has_tensor_hooks
from ..configuration import freeze, thaw
from ..dep_graph.contracts import AttributeBinding, Impact, Requirement
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.operation import Operation, OperatorSpec
from ..dep_graph.operators.coordinates import retained_indices as _keep
from ..dep_graph.operators.rules import is_native_rule
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
    known = {
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
    for req in requirements:
        if isinstance(req, AttributeBinding):
            path, _, name = req.target.rpartition(".")
            owner = graph.model.get_submodule(path) if path else graph.model
            old = getattr(owner, name)
            new = evaluate(req.expression, compact_shape)
            if type(old) in (list, tuple, torch.Size) and isinstance(new, tuple):
                new = type(old)(new)
            # An unchanged binding is still a demand. A shared owner may have
            # another call requiring a different value for the same attribute.
            attributes.append(AttributeRecipe(req.target, old, new))
            continue
        if req.kind not in known:
            raise PlanningError(f"Unhandled requirement {req.kind}: {req.target}")
        data = dict(req.data)
        if req.kind == "operator_argument":
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
        elif req.kind == "reduction_domain":
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
    operations: Sequence[Operation], requirements: Sequence[Requirement | AttributeBinding]
) -> dict[str, tuple[Requirement | AttributeBinding, ...]]:
    """Index call and descendant-attribute requirements in declaration order."""
    by_name: dict[str, list[Requirement | AttributeBinding]] = {
        op.node.name: [] for op in operations
    }
    by_path: dict[str, list[str]] = {}
    for op in operations:
        if op.module is not None:
            for path in op.module_paths:
                by_path.setdefault(path, []).append(op.node.name)
    for requirement in requirements:
        if isinstance(requirement, Requirement) and requirement.kind == "metadata_layout":
            continue
        names = {requirement.target} if requirement.target in by_name else set()
        if isinstance(requirement, AttributeBinding):
            # An opaque module owns requirements for its descendants too. Root
            # modules use the empty path; repeated calls must all receive them.
            path = requirement.target.rpartition(".")[0]
            while True:
                names.update(by_path.get(path, ()))
                if not path:
                    break
                path = path.rpartition(".")[0]
        for name in names:
            by_name[name].append(requirement)
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
    recipes, attributes, notes, handled, strides = {}, {}, [], set(), {}
    active = []
    attribute_bindings = {}
    affected = graph.affected_operations(impact)
    requirements = _requirements_by_operation(
        tuple(op for op in operations if op.node.name in affected), impact.requirements
    )
    for op in operations:
        if op.node.name not in affected:
            continue
        reqs = requirements[op.node.name]
        rule = graph.operator_rule(op)
        if rule is None:
            raise PlanningError(f"No execution rule for {op.node.target}")
        spec = graph.operator_spec(op)
        if spec is None:
            raise PlanningError(f"No structural specification for {op.name}")
        custom = None
        if not is_native_rule(rule):
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
            path, _, name = attr.path.rpartition(".")
            owner = graph.model.get_submodule(path) if path else graph.model
            if freeze(getattr(owner, name)) != freeze(thaw(attr.old)):
                raise PlanningError(f"Rewrite attribute precondition disagrees: {attr.path}")
            identity = (id(owner), name)
            previous = attribute_bindings.get(identity)
            if previous is not None and (previous.old != attr.old or previous.new != attr.new):
                raise PlanningError(f"Shared attribute update disagrees: {attr.path}")
            attribute_bindings[identity] = attr
            if attr.old != attr.new:
                attributes[attr.path] = attr
        active.append((op, spec, reqs, rule.evaluate_on_meta))
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
        if not isinstance(req, Requirement) or req.kind != "metadata_layout":
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
    check_forward(
        graph, operations, active, impact, recipes, attributes, strides, meta_outputs=meta_outputs
    )
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
        if error is not None:
            raise PlanningError(error)
    # tensor_bindings() checked entry state; check exit even when the cached
    # attribute proof avoided re-tracing. Generic lowering needs no per-call scan.
    graph.validate()
    return tuple(recipes.values()), tuple(attributes.values()), tuple(dict.fromkeys(notes))
