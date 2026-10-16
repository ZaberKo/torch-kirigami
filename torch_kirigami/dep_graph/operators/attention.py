"""Matrix contractions and attention expressed with reusable axis/block relations."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import torch

from ...errors import UnsupportedOperation
from ..contracts import (
    AttributeBinding,
    AxisBarrier,
    Balanced,
    BlockBalance,
    Requirement,
    ShapeExpr,
)
from ..operation import CandidateAxis, Operation, OperatorSpec, OutputContract, tensors
from ..relations import AxisPort, AxisRelation, BlockMap, BroadcastRelation
from ..selection import IndexSet
from .native import equal, layout, matmul, one, requirement


def addmm(operation: Operation) -> OperatorSpec:
    """Share matrix-product relations and connect the broadcast additive operand."""
    additive = one(operation.argument("input", 0))
    a = operation.argument(
        "batch1" if operation.node.target in (torch.baddbmm, "baddbmm") else "mat1", 1
    )
    b = operation.argument(
        "batch2" if operation.node.target in (torch.baddbmm, "baddbmm") else "mat2", 2
    )
    result = matmul(replace(operation, args=(a, b), kwargs={}))
    return replace(
        result,
        relations=(
            *result.relations,
            BroadcastRelation(additive, one(operation.result), operation.node.name),
        ),
    )


def einsum(operation: Operation) -> OperatorSpec:
    """Map einsum labels, including diagonals and broadcast ellipses."""
    equation = operation.argument("equation", 0)
    if not isinstance(equation, str):
        raise UnsupportedOperation("einsum requires a string equation")
    inputs, arrow, output = equation.replace(" ", "").partition("->")
    if not arrow:
        # Native implicit output places ellipses first, followed by labels that
        # appear exactly once in the entire equation, in alphabetical order.
        counts = Counter(inputs.replace(",", "").replace("...", ""))
        output = ("..." if "..." in inputs else "") + "".join(
            sorted(label for label, count in counts.items() if count == 1)
        )
    operands = operation.args[1:]
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = operands[0]
    labels = inputs.split(",")
    if len(labels) != len(operands):
        raise UnsupportedOperation("einsum operands do not match the equation")
    ellipsis = max(
        len(t.shape) - len(label.replace("...", ""))
        for t, label in zip(operands, labels, strict=True)
    )
    extra = [f"@{i}" for i in range(ellipsis)]

    def expand(label: str, rank: int) -> list[str]:
        """Expand an einsum ellipsis into right-aligned synthetic labels."""
        count = rank - len(label.replace("...", ""))
        before, marker, after = label.partition("...")
        result = [*before, *(extra[ellipsis - count :] if marker else ()), *after]
        if len(result) != rank:
            raise UnsupportedOperation("einsum labels do not match the operand rank")
        return result

    occurrences = {}
    y = one(operation.result)
    for tensor, label in zip((*operands, y), (*labels, output), strict=True):
        # Repeated labels in one operand name different physical axes of the
        # same logical coordinate. Existing equal-axis relations couple their
        # retained positions, including when that coordinate is reduced away.
        for dim, key in enumerate(expand(label, len(tensor.shape))):
            occurrences.setdefault(key, []).append(tensor.axis(dim))
    relations = []
    for axes in occurrences.values():
        size = max(a.tensor.shape[a.dim] for a in axes)
        aligned = [a for a in axes if a.tensor.shape[a.dim] == size]
        if any(a.tensor.shape[a.dim] not in (1, size) for a in axes):
            raise UnsupportedOperation("einsum label sizes disagree")
        relations.extend(
            AxisRelation.equal(aligned[0], a, operation.node.name) for a in aligned[1:]
        )
    return OperatorSpec(
        tuple(relations),
        tuple(layout(operation, t) for t in (*operands, y)),
        requirements=(
            Requirement(
                "reduction_domain",
                operation.node.name,
                tuple(operands),
                "Contract the retained einsum domain",
            ),
        ),
    )


def sdpa(operation: Operation) -> OperatorSpec:
    """Bind Q/K contractions, V features, masks, and optional grouped-query heads."""
    q, k, v = (one(operation.argument(name, i)) for i, name in enumerate(("query", "key", "value")))
    y = one(operation.result)
    if min(len(t.shape) for t in (q, k, v)) < 3:
        raise UnsupportedOperation("SDPA requires explicit head, sequence, and feature dimensions")
    relations = [
        equal(q, -1, k, -1, operation.node.name),
        equal(k, -2, v, -2, operation.node.name),
        equal(q, -2, y, -2, operation.node.name),
        equal(v, -1, y, -1, operation.node.name),
    ]
    constraints = []
    if operation.argument("is_causal", 5, False):
        constraints.extend(
            AxisBarrier(
                t.axis(-2), "Implicit causal token coordinates are fixed", operation.node.name
            )
            for t in (q, k)
        )
    gqa = operation.argument("enable_gqa", 7, False)
    if gqa:
        if k.shape[-3] != v.shape[-3] or q.shape[-3] % k.shape[-3]:
            raise UnsupportedOperation("Invalid GQA head grouping")
        multiplier = q.shape[-3] // k.shape[-3]
        relations.extend(
            (
                equal(k, -3, v, -3, operation.node.name),
                equal(q, -3, y, -3, operation.node.name),
                AxisRelation(
                    AxisPort(k.axis(-3)),
                    AxisPort(q.axis(-3)),
                    (BlockMap(0, 0, k.shape[-3], 1, multiplier, require_full_target=True),),
                    operation.node.name,
                ),
            )
        )
        constraints.append(BlockBalance(k.axis(-3), q.axis(-3), multiplier, operation.node.name))
    else:
        for tensor in (q, k, v):
            if tensor.shape[-3] == y.shape[-3]:
                relations.append(equal(tensor, -3, y, -3, operation.node.name))
            elif tensor.shape[-3] != 1:
                raise UnsupportedOperation("Unproven attention head broadcast")
    for tensor in (q, k, v):
        batch = len(tensor.shape) - 3
        for d in range(batch):
            target = len(y.shape) - 3 - batch + d
            if tensor.shape[d] == y.shape[target]:
                relations.append(equal(tensor, d, y, target, operation.node.name))
            elif tensor.shape[d] != 1:
                raise UnsupportedOperation("Unproven attention batch broadcast")
    mask = operation.argument("attn_mask", 3)
    if mask is not None:
        mask = one(mask)
        destinations = [y.axis(d) for d in range(len(y.shape) - 1)] + [k.axis(-2)]
        if len(mask.shape) > len(destinations):
            raise UnsupportedOperation("Attention mask rank exceeds broadcast domain")
        for d, size in enumerate(mask.shape):
            axis = destinations[len(destinations) - len(mask.shape) + d]
            if size == axis.tensor.shape[axis.dim]:
                relations.append(AxisRelation.equal(mask.axis(d), axis, operation.node.name))
            elif size != 1:
                raise UnsupportedOperation("Attention mask cannot broadcast")
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        requirements=(
            Requirement(
                "reduction_domain",
                operation.node.name,
                (q, k, v),
                "Recompute attention logits, default scale, softmax and weighted values on compact domains",
            ),
        ),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def multihead_attention(operation: Operation) -> OperatorSpec:
    """Shrink native MHA width with fixed heads and explicitly tied projection axes.

    Batch/sequence/token pruning is excluded. Packed and separate projections
    retain their original representation; no Python forward or module replacement
    is needed. Dense self- and cross-attention share the same width descriptor.
    """
    module = operation.module
    q, k, v = (one(operation.argument(name, i)) for i, name in enumerate(("query", "key", "value")))
    y = one(operation.outputs[0])
    out = one(operation.bindings.get("out_proj.weight"))
    axis = out.axis(0)
    width, heads = module.embed_dim, module.num_heads
    relations = [
        AxisRelation.equal(axis, out.axis(1), operation.node.name),
        AxisRelation.equal(axis, q.axis(-1), operation.node.name),
        AxisRelation.equal(axis, y.axis(-1), operation.node.name),
    ]
    requirements = [requirement(operation, "embed_dim", y, -1)]
    for name in ("in_features", "out_features"):
        requirements.append(
            AttributeBinding(
                f"{operation.module_path}.out_proj.{name}".lstrip("."),
                ShapeExpr("dimension", (axis.tensor, axis.dim)),
                "Bind output projection width",
            )
        )
    requirements.append(
        AttributeBinding(
            f"{operation.module_path}.head_dim".lstrip("."),
            ShapeExpr(
                "floordiv",
                args=(
                    ShapeExpr("dimension", (out, 0)),
                    ShapeExpr("constant", heads),
                ),
            ),
            "Derive per-head width with fixed head count",
        )
    )
    if module.in_proj_weight is not None:
        packed = one(operation.bindings.get("in_proj_weight"))
        relations.append(AxisRelation.equal(axis, packed.axis(1), operation.node.name))
        relations.append(
            AxisRelation(
                AxisPort(axis),
                AxisPort(packed.axis(0)),
                tuple(BlockMap(0, i * width, width) for i in range(3)),
                operation.node.name,
            )
        )
        relations.extend(
            (
                AxisRelation.equal(axis, k.axis(-1), operation.node.name),
                AxisRelation.equal(axis, v.axis(-1), operation.node.name),
            )
        )
    else:
        for name, input_ in (("q", q), ("k", k), ("v", v)):
            weight = one(operation.bindings.get(f"{name}_proj_weight"))
            relations.extend(
                (
                    AxisRelation.equal(axis, weight.axis(0), operation.node.name),
                    equal(input_, -1, weight, 1, operation.node.name),
                )
            )
    for name, tensor in (("kdim", k), ("vdim", v)):
        requirements.append(requirement(operation, name, tensor, -1))
    for name in ("in_proj_bias", "out_proj.bias", "bias_k", "bias_v"):
        bias = operation.bindings.get(name)
        if bias is not None:
            if name == "in_proj_bias":
                relations.append(
                    AxisRelation(
                        AxisPort(axis),
                        AxisPort(bias.axis(0)),
                        tuple(BlockMap(0, i * width, width) for i in range(3)),
                        operation.node.name,
                    )
                )
            else:
                relations.append(AxisRelation.equal(axis, bias.axis(-1), operation.node.name))
    block = width // heads
    constraints = [
        Balanced(axis, tuple(IndexSet.span(h * block, (h + 1) * block) for h in range(heads)))
    ]
    constraints.extend(
        AxisBarrier(t.axis(d), "Native MHA batch/token positions are fixed", operation.node.name)
        for t in (q, k, v, y)
        for d in range(len(t.shape) - 1)
    )
    masks = tuple(
        tensors((operation.argument("key_padding_mask", 3), operation.argument("attn_mask", 5)))
    )
    for ref in (*masks, *operation.outputs[1:]):
        constraints.extend(
            AxisBarrier(ref.axis(d), "Native MHA mask/weight axes are fixed", operation.node.name)
            for d in range(len(ref.shape))
        )
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        tuple(requirements),
        candidates=(CandidateAxis(f"{out.paths[0]}:0", axis),),
        contract=OutputContract(output_layout="backend_dependent"),
    )
