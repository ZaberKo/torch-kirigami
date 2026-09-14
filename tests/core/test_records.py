"""core / records contracts."""

from dataclasses import replace

import pytest

from torch_kirigami import (
    Diagnostic,
    IndexSet,
    OperatorSpec,
    Region,
    Requirement,
    ShapeExpr,
    TensorRef,
)
from torch_kirigami.operation import OutputContract
from torch_kirigami.pruning import (
    AnalysisSummary,
    AttributeRecipe,
    CandidateSpace,
    CoordinateSegment,
    RewriteResult,
    SelectionReport,
    TensorRecipe,
)
from torch_kirigami.pruning.types import ModelStructure, ModuleState, TensorState


def test_static_records_validate_and_freeze_construction_inputs():
    ref = TensorRef("p", (4,), "parameter", ("weight",))
    region = Region((IndexSet.of([0, 1]),))
    for options in (
        {"segments": []},
        {"segments": [region], "concat_dim": 1},
        {"segments": [region], "memory_format": "channels_last"},
    ):
        with pytest.raises(ValueError):
            TensorRecipe(ref, **options)
    with pytest.raises(ValueError):
        AttributeRecipe(1, 4, 2)
    with pytest.raises(ValueError):
        CoordinateSegment(region, Region((IndexSet.of([0]),)))
    with pytest.raises(ValueError):
        SelectionReport(
            channel_axes=[ref.axis(0)], widths=[4], removed=[], targets=[1], scope="local"
        )
    with pytest.raises(ValueError):
        RewriteResult(output_strides=[(ref, (1, 1))])
    with pytest.raises(ValueError):
        ModuleState([""], "Model", [("a", 1), ("a", 2)], [])
    with pytest.raises(ValueError):
        TensorState(
            ["weight"],
            "parameter",
            [4],
            [],
            "torch.float32",
            "cpu",
            True,
            [True],
            ["weight"],
            "Parameter",
        )
    attrs = [["configuration", [1, 2]]]
    module = ModuleState([""], "Model", attrs, [])
    attrs[0][1].append(3)
    assert module.attributes == (("configuration", (1, 2)),)
    structure = ModelStructure([module], [], [["weight", [1, 2]]])
    assert isinstance(structure.modules, tuple) and structure.references == (("weight", (1, 2)),)
    assert CandidateSpace((), iter([ref.axis(0), ref.axis(0)])).channel_axes == (ref.axis(0),)
    summary = AnalysisSummary("resolved", [], [ref.axis(0).select([1])], [])
    with pytest.raises(ValueError, match="original tensor"):
        summary.selection(replace(ref, shape=(2,)))


def test_nested_descriptors_are_detached_and_validated():
    ref = TensorRef("source", (2, 4))
    args = [ShapeExpr("dimension", [ref, -1]), ShapeExpr("constant", 2)]
    expr = ShapeExpr("mul", args=args)
    args.clear()
    assert expr.refs == (ref,)
    assert expr.args[0].value == (ref, 1)
    payload = [[ref.axis(0), [expr]]]
    requirement = Requirement("third_party", "call", [ref], "", [("custom", payload)])
    payload.clear()
    assert requirement.refs == (ref,)
    assert requirement.data[0][1] == ((ref.axis(0), (expr,)),)
    for factory in (
        lambda: ShapeExpr("typo"),
        lambda: ShapeExpr("add", args=(expr,)),
        lambda: ShapeExpr("constant", []),
        lambda: Diagnostic("test", "", severity="typo"),
        lambda: OutputContract(output_layout="typo"),
        lambda: Requirement("custom", "call", (), "", (("state", {}),)),
        lambda: Requirement("custom", "call", (), "", (("a", 1), ("a", 2))),
        lambda: OperatorSpec(layouts=(object(),)),
    ):
        with pytest.raises((TypeError, ValueError)):
            factory()
