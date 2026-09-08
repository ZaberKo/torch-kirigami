"""Dimension provenance and dependencies shared by operator contracts."""

import builtins
import operator
from dataclasses import dataclass
from math import prod

from torch import fx

from ..contracts import Diagnostic, ShapeExpr
from ..registry import tensors
from ..selection import TensorRef


class _PendingLayout(Exception):
    """Known structure whose partition counts still need a legal completion."""


def expression_for(ctx):
    """Recognize public dimension reads and scalar integer arithmetic."""
    node = ctx.node
    x = ctx.args[0] if ctx.args else None
    expression = None
    if (
        node.op == "call_function"
        and node.target is builtins.getattr
        and isinstance(x, TensorRef)
        and ctx.args[1] == "shape"
    ):
        expression = ShapeExpr("shape", x)
    if node.op == "call_method" and isinstance(x, TensorRef):
        if node.target == "size":
            dim = ctx.argument("dim", 1)
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


def evaluate(expr, shape):
    """Evaluate supported integer provenance using supplied compact shapes."""
    if expr.kind in ("constant", "infer"):
        return expr.value
    if expr.kind == "tuple":
        return tuple(evaluate(e, shape) for e in expr.args)
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


def reevaluate(raw, normalized, expressions, shape):
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


def dependencies(ctx):
    """Return dimension-source tensors used by scalar arguments of a call."""
    result = []

    def visit(expr):
        if isinstance(expr.value, TensorRef):
            result.append(expr.value)
        elif isinstance(expr.value, tuple):
            result.extend(tensors(expr.value))
        for arg in expr.args:
            visit(arg)

    for node in ctx.node.all_input_nodes:
        if node in ctx.expressions:
            visit(ctx.expressions[node])
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class ArgumentDependencies:
    """Trigger scalar-argument checks independently of removal propagation."""

    context: object
    allow_changes: bool = False
    layouts: tuple = ()

    @property
    def refs(self):
        """Return tensors whose sizes feed this call's scalar arguments."""
        return dependencies(self.context)

    def check(self, selections):
        """Reject changed semantic arguments while permitting declared shape contracts."""
        if self.allow_changes or not any(r.id in selections for r in self.refs):
            return None

        def shape(ref):
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
                segments = layout.retained(selection)
                sizes = [tuple(len(a) for a in r.axes) for r in segments]
                if not sizes or any(
                    a != b
                    for size in sizes[1:]
                    for d, (a, b) in enumerate(zip(sizes[0], size, strict=True))
                    if d != layout.concat_dim
                ):
                    raise _PendingLayout("Partitioned shape needs further balancing")
                result = list(sizes[0])
                result[layout.concat_dim] = sum(s[layout.concat_dim] for s in sizes)
                shapes.append(tuple(result))
            if shapes and all(s == shapes[0] for s in shapes):
                return shapes[0]
            raise _PendingLayout("Compact layout has not been established")

        ctx = self.context
        try:
            args = reevaluate(ctx.node.args, ctx.args, ctx.expressions, shape)
            kwargs = reevaluate(ctx.node.kwargs, ctx.kwargs, ctx.expressions, shape)
        except _PendingLayout:
            return Diagnostic(
                "argument_layout",
                "Scalar arguments require a resolved compact layout",
                node=ctx.node.name,
            )
        except (TypeError, ValueError, ZeroDivisionError, IndexError):
            return Diagnostic(
                "argument_provenance",
                "Cannot reevaluate scalar arguments",
                node=ctx.node.name,
                complete=False,
            )
        if args != ctx.args or kwargs != ctx.kwargs:
            return Diagnostic(
                "changed_arguments",
                "Compaction changes a semantic argument in the original forward",
                "conflict",
                ctx.node.name,
            )
        return None
