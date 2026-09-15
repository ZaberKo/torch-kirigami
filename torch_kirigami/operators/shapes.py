"""Dimension provenance and dependencies shared by operator contracts."""

from __future__ import annotations

import builtins
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from typing import Any, cast

from torch import fx

from ..contracts import ArgumentRef, Diagnostic, ShapeExpr
from ..errors import UnsupportedOperation
from ..operation import OperationContext, PartitionedLayout, argument_locations
from ..regions import concatenated_shape
from ..selection import Selection, TensorRef


class _PendingLayout(Exception):
    """Known structure whose partition counts still need a legal completion."""


def shape_expression(ctx: OperationContext, value: object) -> ShapeExpr:
    """Resolve dimension provenance without guessing from numeric equality."""
    if isinstance(value, fx.Node):
        return ctx.expressions.get(value, ShapeExpr("unknown", value.name))
    if isinstance(value, (tuple, list)):
        return ShapeExpr("tuple", args=tuple(shape_expression(ctx, v) for v in value))
    if isinstance(value, int):
        return ShapeExpr("infer" if value == -1 else "constant", value)
    return ShapeExpr("unknown", repr(value))


def has_known_provenance(expr: ShapeExpr) -> bool:
    """Check whether every leaf of a shape expression has known provenance."""
    return expr.kind != "unknown" and all(has_known_provenance(arg) for arg in expr.args)


def expression_for(ctx: OperationContext) -> ShapeExpr | None:
    """Recognize public dimension reads and scalar integer arithmetic."""
    node = ctx.node
    x = ctx.args[0] if ctx.args else None
    expression = None
    if (
        node.op == "call_function"
        and node.target is builtins.getattr
        and isinstance(x, TensorRef)
        and ctx.args[1] in ("shape", "ndim")
    ):
        expression = ShapeExpr("shape" if ctx.args[1] == "shape" else "dim", x)
    if node.op == "call_method" and isinstance(x, TensorRef):
        if node.target == "size":
            dim = ctx.argument("dim", 1)
            raw_dim = ctx.raw_argument("dim", 1)
            if isinstance(raw_dim, fx.Node) and raw_dim not in ctx.expressions:
                raise UnsupportedOperation("Dimension selector has unknown provenance")
            expression = (
                ShapeExpr("shape", x)
                if dim is None
                else ShapeExpr("dimension", (x, dim % len(x.shape)))
            )
        elif node.target in ("dim", "numel"):
            expression = ShapeExpr(node.target, x)
    if node.op == "call_function" and node.target is operator.getitem:
        source = ctx.expressions.get(node.args[0])
        if source and source.kind == "shape" and isinstance(ctx.args[1], int):
            if isinstance(node.args[1], fx.Node) and node.args[1] not in ctx.expressions:
                raise UnsupportedOperation("Dimension selector has unknown provenance")
            expression = ShapeExpr(
                "dimension", (source.value, ctx.args[1] % len(source.value.shape))
            )
    arithmetic = {
        operator.add: "add",
        operator.sub: "sub",
        operator.mul: "mul",
        operator.floordiv: "floordiv",
        operator.mod: "mod",
    }
    if node.op == "call_function" and node.target in arithmetic and not ctx.outputs:
        expressions = []
        for arg in node.args:
            if isinstance(arg, int):
                expressions.append(ShapeExpr("constant", arg))
            elif arg in ctx.expressions:
                expressions.append(ctx.expressions[arg])
            else:
                break
        else:
            expression = ShapeExpr(arithmetic[node.target], args=tuple(expressions))
    return expression


def evaluate(
    expr: ShapeExpr, shape: Callable[[TensorRef], tuple[int, ...]]
) -> int | tuple[int, ...]:
    """Evaluate supported integer provenance using supplied compact shapes."""
    if expr.kind in ("constant", "infer"):
        return expr.value
    if expr.kind == "tuple":
        return tuple(cast(int, evaluate(e, shape)) for e in expr.args)
    if expr.kind == "dimension":
        ref, dim = expr.value
        return shape(ref)[dim]
    if expr.kind == "shape":
        return shape(expr.value)
    if expr.kind == "dim":
        return len(shape(expr.value))
    if expr.kind == "numel":
        return prod(shape(expr.value))
    functions = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "floordiv": operator.floordiv,
        "mod": operator.mod,
    }
    if expr.kind in functions:
        try:
            return functions[expr.kind](*(evaluate(e, shape) for e in expr.args))
        except ArithmeticError as error:
            raise ValueError("Compact shape arithmetic is invalid") from error
    raise ValueError(f"Unknown shape expression: {expr.kind}")


def reevaluate(
    raw: Any,
    normalized: Any,
    expressions: Mapping[fx.Node, ShapeExpr],
    shape: Callable[[TensorRef], tuple[int, ...]],
) -> Any:
    """Recompute only captured dimension expressions in an argument tree."""
    if isinstance(raw, fx.Node):
        expression = expressions.get(raw)
        return evaluate(expression, shape) if expression is not None else normalized
    if isinstance(raw, (tuple, list)):
        return type(normalized)(
            reevaluate(r, n, expressions, shape) for r, n in zip(raw, normalized, strict=True)
        )
    if isinstance(raw, dict):
        return {k: reevaluate(v, normalized[k], expressions, shape) for k, v in raw.items()}
    return normalized


def dependencies(ctx: OperationContext) -> tuple[TensorRef, ...]:
    """Return dimension-source tensors used by scalar arguments of a call."""
    return tuple(
        dict.fromkeys(
            ref
            for node in ctx.node.all_input_nodes
            if node in ctx.expressions
            for ref in ctx.expressions[node].refs
        )
    )


@dataclass(frozen=True)
class CallArgumentConstraint:
    """Check immutable scalar provenance independently of removal propagation.

    Expressions, observed values, layouts, and an optional conditional hint are
    retained. No FX node, model, or mutable OperationContext escapes through graph
    and impact constraint inspection. The hint is displayed only on a violation.
    """

    node: str
    arguments: tuple[tuple[ShapeExpr, int | tuple[int, ...]], ...]
    layouts: tuple[PartitionedLayout, ...] = ()
    hint: str = ""

    def __post_init__(self) -> None:
        arguments = tuple(
            (expr, tuple(value) if isinstance(value, (list, tuple)) else value)
            for expr, value in self.arguments
        )
        for expr, value in arguments:
            if not isinstance(expr, ShapeExpr) or not (
                isinstance(value, int)
                or (isinstance(value, tuple) and all(isinstance(item, int) for item in value))
            ):
                raise TypeError("Call argument checks require shape expressions and integer values")
        if not isinstance(self.hint, str):
            raise TypeError("Call argument hint must be text")
        object.__setattr__(self, "arguments", arguments)
        object.__setattr__(self, "layouts", tuple(self.layouts))

    @classmethod
    def from_operation(
        cls, context: OperationContext, *, checked_arguments: tuple[ArgumentRef, ...] = ()
    ) -> CallArgumentConstraint:
        """Detach the used shape expressions and observed values from capture."""
        arguments = []

        def visit(raw: object, observed: Any) -> None:
            """Collect expressions found in one raw and normalized value pair."""
            if isinstance(raw, fx.Node):
                expression = context.expressions.get(raw)
                if expression is not None:
                    arguments.append((expression, observed))
            elif isinstance(raw, (tuple, list)):
                for item, value in zip(raw, observed, strict=True):
                    visit(item, value)
            elif isinstance(raw, dict):
                for key, item in raw.items():
                    visit(item, observed[key])

        # Match parameter locations, not expression identity: the same size()
        # node may feed a checked groups argument and an unchecked stride.
        checked = {
            location
            for arg in checked_arguments
            for location in argument_locations(
                context.node.args,
                context.node.kwargs,
                arg.name,
                arg.position,
                target=context.node.target if context.module is None else None,
                variadic=arg.variadic,
            )
        }
        for index, (raw, observed) in enumerate(zip(context.node.args, context.args, strict=True)):
            if ("args", index) not in checked:
                visit(raw, observed)
        for key, raw in context.node.kwargs.items():
            if ("kwargs", key) not in checked:
                visit(raw, context.kwargs[key])
        return cls(context.node.name, tuple(arguments))

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Return tensors whose sizes feed this call's scalar arguments."""
        return tuple(dict.fromkeys(ref for expr, _ in self.arguments for ref in expr.refs))

    def check(self, selections: Mapping[str, Selection]) -> Diagnostic | None:
        """Reject changed semantic arguments while permitting declared shape contracts."""
        if not any(r.id in selections for r in self.refs):
            return None

        def shape(ref: TensorRef) -> tuple[int, ...]:
            """Resolve a reference's compact shape under the current selections."""
            if ref.id not in selections:
                return ref.shape
            selection = selections[ref.id]
            ordinary = selection.compact_shape()
            if ordinary is not None:
                return ordinary
            shapes = []
            for layout in self.layouts:
                if layout.tensor != ref:
                    continue
                segments = layout.retained_regions(selection)
                try:
                    shapes.append(concatenated_shape(segments, layout.concat_dim))
                except ValueError as error:
                    raise _PendingLayout("Partitioned shape needs further balancing") from error
            if shapes and all(s == shapes[0] for s in shapes):
                return shapes[0]
            raise _PendingLayout("Compact layout has not been established")

        try:
            changed = any(evaluate(expr, shape) != value for expr, value in self.arguments)
        except _PendingLayout:
            return Diagnostic(
                "argument_layout",
                "Scalar arguments require a resolved compact layout",
                node=self.node,
            )
        except (TypeError, ValueError, ZeroDivisionError, IndexError):
            return Diagnostic(
                "argument_provenance",
                "Cannot reevaluate scalar arguments",
                node=self.node,
                complete=False,
            )
        if changed:
            return Diagnostic(
                "changed_arguments",
                "Compaction changes a semantic argument in the original forward"
                + (f". {self.hint}" if self.hint else ""),
                "conflict",
                self.node,
            )
        return None
