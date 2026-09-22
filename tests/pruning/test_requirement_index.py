"""Requirement indexes preserve call ownership, declaration order and packing."""

from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisRelation,
    DependencyGraph,
    Impact,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Requirement,
)
from torch_kirigami.operation import OperationContext
from torch_kirigami.pruning import Pruner
from torch_kirigami.pruning import rewrite as rewrite_module
from torch_kirigami.pruning import validation as validation_module
from torch_kirigami.pruning.rewrite import _requirements_by_operation
from torch_kirigami.selection import TensorRef


@pytest.mark.parametrize("function_call", [False, True])
def test_requirement_index_matches_exhaustive_ownership(function_call: bool) -> None:
    """Match root, nested and repeated calls without duplicating dual matches."""
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    calls = graph.operations()
    operations = (
        replace(calls[0], module_path=""),
        replace(
            calls[1],
            module=None if function_call else calls[1].module,
            module_path=None if function_call else "block",
        ),
        replace(calls[2], module_path="block.inner"),
    )
    targets = (
        "width",
        "block.width",
        "block.inner.width",
        "block.inner_more.width",
        "block_more.width",
        "unrelated.width",
        *(op.node.name for op in operations),
    )
    requirements = tuple(
        Requirement(kind, target, (), f"{kind}:{target}")
        for target in targets
        for kind in ("attribute", "call_arguments", "metadata_layout")
    )
    # Equal declarations remain distinct entries, including repeated identities.
    requirements += (requirements[0], replace(requirements[0]))
    indexed = _requirements_by_operation(operations, requirements)
    for op in operations:
        expected = tuple(
            req
            for req in requirements
            if req.kind != "metadata_layout"
            and (
                req.target == op.node.name
                or (
                    op.module is not None
                    and req.kind == "attribute"
                    and (
                        req.target.rpartition(".")[0] == (op.module_path or "")
                        or req.target.startswith(f"{op.module_path}." if op.module_path else "")
                    )
                )
            )
        )
        assert tuple(map(id, indexed[op.node.name])) == tuple(map(id, expected))


class _OpaqueBlock(nn.Module):
    """Expose a descendant attribute through one opaque captured call."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(4, 6, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).reshape(x.size(0), self.inner.out_features)


def _block_spec(ctx: OperationContext) -> OperatorSpec:
    """Bind the opaque block's width to its descendant Linear attribute."""
    source, result, weight = ctx.inputs[0], ctx.outputs[0], ctx.binding("inner.weight")
    return OperatorSpec(
        (
            AxisRelation.equal(source.axis(1), weight.axis(1), "input"),
            AxisRelation.equal(result.axis(1), weight.axis(0), "output"),
        ),
        requirements=(
            Requirement(
                "attribute",
                f"{ctx.module_path}.inner.out_features".lstrip("."),
                (result,),
                "nested width",
                (("axis", result.axis(1)),),
            ),
        ),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_descendant_attribute_requirement_survives_plan_apply(
    nested: bool, execution_device: str
) -> None:
    """Root and shared nested leaf calls update one descendant binding correctly."""

    class Shared(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = _OpaqueBlock()
            self.alias = self.block
            self.out = nn.Linear(6, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(self.block(x) + self.alias(x))

    model = Shared() if nested else _OpaqueBlock()
    x = torch.randn(2, 4)
    block = model.block if nested else model
    keep = [0, 2, 4, 5]
    expected = F.linear(x, block.inner.weight[keep])
    if nested:
        expected = F.linear(expected * 2, model.out.weight[:, keep], model.out.bias)
    operators = OperatorRegistry.default().register(_OpaqueBlock, OperatorRule(_block_spec))
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    path = "block.inner.weight" if nested else "inner.weight"
    old = tuple(model.parameters())
    pruner = Pruner(model, graph=graph, preserve_io=nested)
    plan = pruner.plan_remove([graph.parameter(path).axis(0).select([1, 3])])
    assert block.inner.out_features == 6
    assert all(a is b for a, b in zip(old, model.parameters(), strict=True))
    pruner.apply(plan)
    assert block.inner.out_features == 4
    if nested:
        assert model.alias is model.block
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


def test_forward_shapes_are_computed_once_per_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse shape arithmetic within a proof without sharing proofs across plans."""
    model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    original_shape = validation_module.require_compact_shape
    original_check = rewrite_module.check_forward
    visits: list[dict[TensorRef, int]] = []

    def counted_shape(impact: Impact, ref: TensorRef) -> tuple[int, ...]:
        visits[-1][ref] = visits[-1].get(ref, 0) + 1
        return original_shape(impact, ref)

    def counted_check(*args: Any, **kwargs: Any) -> None:
        visits.append({})
        original_check(*args, **kwargs)

    monkeypatch.setattr(validation_module, "require_compact_shape", counted_shape)
    monkeypatch.setattr(rewrite_module, "check_forward", counted_check)
    pruner = Pruner(model, graph=graph)
    for indices in ([1], [1, 3]):
        pruner.plan_remove([graph.parameter("0.weight").axis(0).select(indices)])
    assert len(visits) >= 2
    assert all(counts and set(counts.values()) == {1} for counts in visits)
