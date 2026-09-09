"""Shape provenance arithmetic and nested argument evaluation contracts."""

import operator
from dataclasses import FrozenInstanceError

import pytest
from torch import fx

from torch_kirigami import ArgumentRef, Requirement, ShapeExpr, TensorRef
from torch_kirigami.operators.shapes import evaluate, reevaluate


@pytest.mark.parametrize(
    "kind,function",
    [
        ("add", operator.add),
        ("sub", operator.sub),
        ("mul", operator.mul),
        ("floordiv", operator.floordiv),
        ("mod", operator.mod),
    ],
)
def test_shape_arithmetic_tracks_both_sources_and_compact_dimensions(kind, function):
    a, b = TensorRef("a", (2, 6)), TensorRef("b", (3, 4))
    expression = ShapeExpr(kind, args=(ShapeExpr("dimension", (a, -1)), ShapeExpr("numel", b)))
    assert expression.refs == (a, b)
    for width in range(1, 7):
        for batch in range(1, 4):
            shapes = {a: (2, width), b: (batch, 4)}
            assert evaluate(expression, shapes.__getitem__) == function(width, batch * 4)
    with pytest.raises(FrozenInstanceError):
        expression.kind = "constant"


def test_nested_argument_tree_reevaluation_changes_only_proven_sources():
    ref = TensorRef("x", (2, 6))
    graph = fx.Graph()
    width, unrelated = graph.placeholder("width"), graph.placeholder("unrelated")
    expressions = {width: ShapeExpr("dimension", (ref, 1))}
    raw = {"shape": (2, width), "options": [unrelated, {"constant": 6}]}
    normalized = {"shape": (2, 6), "options": [99, {"constant": 6}]}
    actual = reevaluate(raw, normalized, expressions, lambda _: (2, 4))
    assert actual == {"shape": (2, 4), "options": [99, {"constant": 6}]}
    assert normalized == {"shape": (2, 6), "options": [99, {"constant": 6}]}
    assert evaluate(
        ShapeExpr(
            "tuple", args=(ShapeExpr("shape", ref), ShapeExpr("dim", ref), ShapeExpr("infer", -1))
        ),
        lambda _: (2, 4),
    ) == ((2, 4), 2, -1)


@pytest.mark.parametrize("kind", ["floordiv", "mod"])
def test_invalid_compact_arithmetic_has_a_stable_error(kind):
    expression = ShapeExpr(kind, args=(ShapeExpr("constant", 3), ShapeExpr("constant", 0)))
    with pytest.raises(ValueError, match="arithmetic"):
        evaluate(expression, lambda ref: ref.shape)
    with pytest.raises(ValueError, match="Unknown shape"):
        evaluate(ShapeExpr("unknown", "untracked input"), lambda ref: ref.shape)


def test_requirement_keeps_embedded_sources_and_precise_argument_roles():
    a, b = TensorRef("a", (2, 6)), TensorRef("b", (3, 4))
    expr = ShapeExpr("add", args=(ShapeExpr("dimension", (a, 1)), ShapeExpr("numel", b)))
    arguments = [ArgumentRef("groups", 6), ArgumentRef("sizes", 1, variadic=True)]
    requirement = Requirement(
        "custom", "call", (), "", (("nested", [[a.axis(0), expr]]),), arguments
    )
    arguments.clear()
    assert requirement.refs == (a, b)
    assert requirement.arguments == (ArgumentRef("groups", 6), ArgumentRef("sizes", 1, True))
    for args in [("", 0), ("axis", -1), ("axis", True)]:
        with pytest.raises(ValueError):
            ArgumentRef(*args)
    with pytest.raises(TypeError):
        ArgumentRef("axis", 0, variadic=1)
