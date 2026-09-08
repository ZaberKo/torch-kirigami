"""Declarative compaction recipes and conservative original-forward checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..operators.shapes import evaluate
from ..operators.shapes import reevaluate as _reevaluate
from ..operators.validation import check_forward
from ..selection import IndexSet, Region, Selection, TensorRef, full_region
from .types import AttributeRecipe, PlanningError, TensorRecipe


@dataclass(frozen=True)
class RewriteContext:
    """One captured call, its affected requirements, and the combined Impact."""

    graph: object
    operation: object
    impact: object
    requirements: tuple

    @property
    def spec(self):
        """Return the operator's single shared structural description."""
        return self.graph.operator_spec(self.operation)

    def shape(self, ref):
        """Return the ordinary compact shape, rejecting partitioned layouts."""
        result = self.impact.selection(ref).compact_shape()
        if result is None:
            raise PlanningError(f"No rectangular activation layout for {ref.id}")
        return result


@dataclass(frozen=True)
class RewriteResult:
    """Pure extension result; every supplied requirement must be acknowledged.

    Custom rules are responsible for proving original-forward compatibility.
    Output strides can be supplied when proved, allowing downstream view checks.
    """

    tensors: tuple[TensorRecipe, ...] = ()
    attributes: tuple[AttributeRecipe, ...] = ()
    handled: tuple = ()
    notes: tuple[str, ...] = ()
    output_strides: tuple[tuple[TensorRef, tuple[int, ...]], ...] = ()


def _keep(impact, axis):
    return IndexSet.span(0, axis.tensor.shape[axis.dim]).subtract(
        impact.selection(axis.tensor).project(axis.dim)
    )


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
                    IndexSet.span(0, n).subtract(selection.project(d))
                    for d, n in enumerate(ref.shape)
                )
            ),
        ),
    )


def lower_spec(ctx):
    """Lower shared layouts and bound attributes without dispatching on operators."""
    op, impact = ctx.operation, ctx.impact
    if op.module is not None and (op.module._forward_hooks or op.module._forward_pre_hooks):
        raise PlanningError(f"{op.node.name}: built-in rewrite cannot prove forward-hook semantics")
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
                evaluate(data["expression"], ctx.shape)
                if "expression" in data
                else len(_keep(impact, data["axis"]))
                if "axis" in data
                else tuple(len(_keep(impact, a)) for a in data["axes"])
            )
            if new != old:
                attributes.append(AttributeRecipe(req.target, old, new))
        elif req.kind == "operator_argument":
            name = data.get("name", "normalized_shape")
            position = data.get("position")
            if position is None:
                raise PlanningError(f"No functional argument binding for {name}")
            raw = op.node.kwargs.get(
                name, op.node.args[position] if len(op.node.args) > position else None
            )
            current = _reevaluate(raw, op.argument(name, position), op.expressions, ctx.shape)
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
                    descriptor.retained(impact.selection(descriptor.tensor)),
                    descriptor.concat_dim,
                )
            )
    return RewriteResult(tuple(recipes), tuple(attributes), ctx.requirements, tuple(notes))


def _validate_recipe(recipe, impact):
    ref = recipe.tensor
    if ref.kind not in ("parameter", "buffer") or not recipe.segments:
        raise PlanningError("Recipes must retain nonempty registered tensor segments")
    if not 0 <= recipe.concat_dim < max(1, len(ref.shape)):
        raise PlanningError("Invalid concatenation dimension")
    if len(recipe.segments) > 1:
        previous_end = -1
        for segment in recipe.segments:
            axis = segment.axes[recipe.concat_dim]
            if not axis or axis.intervals[0][0] < previous_end:
                raise PlanningError("Recipes must preserve original position order")
            previous_end = axis.intervals[-1][1]
    kept = Selection(ref, recipe.segments)
    if sum(Selection(ref, (r,)).count for r in recipe.segments) != kept.count:
        raise PlanningError("Recipe segments overlap")
    full = Selection(ref, (full_region(ref.shape),))
    if kept != full.subtract(impact.selection(ref)):
        raise PlanningError("Recipe retained coordinates disagree with the joint Impact")
    shapes = [tuple(map(len, r.axes)) for r in recipe.segments]
    if any(
        any(
            a != b
            for d, (a, b) in enumerate(zip(shapes[0], shape, strict=True))
            if d != recipe.concat_dim
        )
        for shape in shapes[1:]
    ):
        raise PlanningError("Recipe segments cannot concatenate")


def _mapping(recipe):
    from .types import CoordinateSegment

    offset, result = 0, []
    for region in recipe.segments:
        axes = [IndexSet.span(0, len(a)) for a in region.axes]
        if axes:
            axes[recipe.concat_dim] = axes[recipe.concat_dim].shift(offset)
            offset += len(region.axes[recipe.concat_dim])
        result.append(CoordinateSegment(region, Region(tuple(axes))))
    return tuple(result)


def _same_mapping(left, right):
    # Different segment boundaries can encode the same mapping. Compare each
    # overlap's per-axis rank offsets, never just the removed set or output shape.
    if left.shape != right.shape:
        return False
    for a in _mapping(left):
        for b in _mapping(right):
            intersection = a.source.intersect(b.source)
            if intersection.empty:
                continue
            for dim, indices in enumerate(intersection.axes):
                for start, stop in indices.intervals:
                    cuts = {start, stop}
                    for source in (a.source.axes[dim], b.source.axes[dim]):
                        cuts.update(
                            p for interval in source.intervals for p in interval if start < p < stop
                        )
                    for point in sorted(cuts)[:-1]:
                        mapped = []
                        for segment in (a, b):
                            rank = len(segment.source.axes[dim].intersect(IndexSet.span(0, point)))
                            mapped.append(next(iter(segment.destination.axes[dim])) + rank)
                        if mapped[0] != mapped[1]:
                            return False
    return True


def compile_recipes(graph, operations, impact):
    """Prove all affected requirements and combine per-use recipes without weights."""
    if impact.status != "resolved":
        raise PlanningError("; ".join(f"{d.code}: {d.message}" for d in impact.diagnostics))
    for selection in (*impact.parameters, *impact.buffers):
        expected_type = nn.Parameter if selection.tensor.kind == "parameter" else torch.Tensor
        if type(graph.tensor(selection.tensor)) is not expected_type:
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
            if r.target == op.node.name
            or (
                op.module is not None
                and r.kind == "attribute"
                and (
                    r.target.rpartition(".")[0] == (op.module_path or "")
                    or r.target.startswith(f"{op.module_path}." if op.module_path else "")
                )
            )
        )
        rule = graph.operator_rule(op)
        if rule is None:
            raise PlanningError(f"No execution rule for {op.node.target}")
        ctx = RewriteContext(graph, op, impact, reqs)
        result = rule.lower(ctx)
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
            _validate_recipe(recipe, impact)
            previous = recipes.get(recipe.tensor.id)
            if previous is not None and not _same_mapping(previous, recipe):
                raise PlanningError(
                    f"Shared tensor requires incompatible coordinate mappings: {recipe.tensor.paths}"
                )
            recipes[recipe.tensor.id] = recipe
        for attr in result.attributes:
            path, _, name = attr.path.rpartition(".")
            owner = graph.model.get_submodule(path) if path else graph.model
            if getattr(owner, name) != attr.old:
                raise PlanningError(f"Rewrite attribute precondition disagrees: {attr.path}")
            identity = (id(owner), name)
            previous = attribute_bindings.get(identity)
            if previous is not None and (previous.old != attr.old or previous.new != attr.new):
                raise PlanningError(f"Shared attribute update disagrees: {attr.path}")
            attribute_bindings[identity] = attr
            attributes[attr.path] = attr
        active.append((ctx, rule.evaluate))
    if any(id(r) not in handled for r in impact.requirements):
        raise PlanningError("Impact contains an unhandled execution requirement")
    for selection in (*impact.parameters, *impact.buffers):
        if selection.tensor.id not in recipes:
            recipe = ordinary_recipe(selection)
            _validate_recipe(recipe, impact)
            recipes[selection.tensor.id] = recipe
    check_forward(graph, operations, active, impact, recipes, attributes, strides)
    return tuple(recipes.values()), tuple(attributes.values()), tuple(dict.fromkeys(notes))
