"""Matrix contractions and attention expressed with reusable axis/block relations."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import AxisBarrier, Balanced, BlockBalance, Requirement, ShapeExpr
from ..registry import CandidateAxis, OperatorSpec, tensors
from ..relations import AxisRelation, BlockMap, BroadcastRelation, Port
from ..selection import IndexSet
from .layouts import CallContract
from .native import UnsupportedOperation, equal, layout, matmul, one, requirement


def addmm(ctx):
    """Share matrix-product relations and connect the broadcast additive operand."""
    additive = one(ctx.argument("input", 0))
    a = ctx.argument("batch1" if ctx.node.target in (torch.baddbmm, "baddbmm") else "mat1", 1)
    b = ctx.argument("batch2" if ctx.node.target in (torch.baddbmm, "baddbmm") else "mat2", 2)
    result = matmul(replace(ctx, args=(a, b), kwargs={}))
    return replace(
        result,
        relations=(*result.relations, BroadcastRelation(additive, one(ctx.output), ctx.node.name)),
    )


def einsum(ctx):
    """Map explicit einsum labels, including right-aligned broadcast ellipses."""
    equation = ctx.argument("equation", 0)
    if not isinstance(equation, str) or "->" not in equation:
        raise UnsupportedOperation("einsum requires an explicit string output equation")
    inputs, output = equation.replace(" ", "").split("->")
    operands = ctx.args[1:]
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

    def expand(label, rank):
        count = rank - len(label.replace("...", ""))
        before, marker, after = label.partition("...")
        result = [*before, *(extra[ellipsis - count :] if marker else ()), *after]
        if len(result) != rank or len(set(result)) != len(result):
            raise UnsupportedOperation("einsum diagonal/repeated operand labels are unsupported")
        return result

    occurrences = {}
    y = one(ctx.output)
    for tensor, label in zip((*operands, y), (*labels, output), strict=True):
        for dim, key in enumerate(expand(label, len(tensor.shape))):
            occurrences.setdefault(key, []).append(tensor.axis(dim))
    relations = []
    for axes in occurrences.values():
        size = max(a.tensor.shape[a.dim] for a in axes)
        aligned = [a for a in axes if a.tensor.shape[a.dim] == size]
        if any(a.tensor.shape[a.dim] not in (1, size) for a in axes):
            raise UnsupportedOperation("einsum label sizes disagree")
        relations.extend(AxisRelation.equal(aligned[0], a, ctx.node.name) for a in aligned[1:])
    return OperatorSpec(
        tuple(relations),
        tuple(layout(ctx, t) for t in (*operands, y)),
        requirements=(
            Requirement(
                "reduction_domain",
                ctx.node.name,
                tuple(operands),
                "Contract the retained einsum domain",
            ),
        ),
    )


def sdpa(ctx):
    """Bind Q/K contractions, V features, masks, and optional grouped-query heads."""
    q, k, v = (one(ctx.argument(name, i)) for i, name in enumerate(("query", "key", "value")))
    y = one(ctx.output)
    if min(len(t.shape) for t in (q, k, v)) < 3:
        raise UnsupportedOperation("SDPA requires explicit head, sequence, and feature dimensions")
    relations = [
        equal(q, -1, k, -1, ctx.node.name),
        equal(k, -2, v, -2, ctx.node.name),
        equal(q, -2, y, -2, ctx.node.name),
        equal(v, -1, y, -1, ctx.node.name),
    ]
    constraints = []
    if ctx.argument("is_causal", 5, False):
        constraints.extend(
            AxisBarrier(t.axis(-2), "Implicit causal token coordinates are fixed", ctx.node.name)
            for t in (q, k)
        )
    gqa = ctx.argument("enable_gqa", 7, False)
    if gqa:
        if k.shape[-3] != v.shape[-3] or q.shape[-3] % k.shape[-3]:
            raise UnsupportedOperation("Invalid GQA head grouping")
        multiplier = q.shape[-3] // k.shape[-3]
        relations.extend(
            (
                equal(k, -3, v, -3, ctx.node.name),
                equal(q, -3, y, -3, ctx.node.name),
                AxisRelation(
                    Port(k.axis(-3)),
                    Port(q.axis(-3)),
                    (BlockMap(0, 0, k.shape[-3], 1, multiplier, require_full_target=True),),
                    ctx.node.name,
                ),
            )
        )
        constraints.append(BlockBalance(k.axis(-3), q.axis(-3), multiplier, ctx.node.name))
    else:
        for tensor in (q, k, v):
            if tensor.shape[-3] == y.shape[-3]:
                relations.append(equal(tensor, -3, y, -3, ctx.node.name))
            elif tensor.shape[-3] != 1:
                raise UnsupportedOperation("Unproven attention head broadcast")
    for tensor in (q, k, v):
        batch = len(tensor.shape) - 3
        for d in range(batch):
            target = len(y.shape) - 3 - batch + d
            if tensor.shape[d] == y.shape[target]:
                relations.append(equal(tensor, d, y, target, ctx.node.name))
            elif tensor.shape[d] != 1:
                raise UnsupportedOperation("Unproven attention batch broadcast")
    mask = ctx.argument("attn_mask", 3)
    if mask is not None:
        mask = one(mask)
        destinations = [y.axis(d) for d in range(len(y.shape) - 1)] + [k.axis(-2)]
        if len(mask.shape) > len(destinations):
            raise UnsupportedOperation("Attention mask rank exceeds broadcast domain")
        for d, size in enumerate(mask.shape):
            axis = destinations[len(destinations) - len(mask.shape) + d]
            if size == axis.tensor.shape[axis.dim]:
                relations.append(AxisRelation.equal(mask.axis(d), axis, ctx.node.name))
            elif size != 1:
                raise UnsupportedOperation("Attention mask cannot broadcast")
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        requirements=(
            Requirement(
                "reduction_domain",
                ctx.node.name,
                (q, k, v),
                "Recompute attention logits, default scale, softmax and weighted values on compact domains",
            ),
        ),
        contract=CallContract(fresh_output=True, output_layout="backend_dependent"),
    )


def multihead_attention(ctx):
    """Shrink native MHA width with fixed heads and explicitly tied projection axes.

    Batch/sequence/token pruning is excluded. Packed and separate projections
    retain their original representation; no Python forward or module replacement
    is needed. Dense self- and cross-attention share the same width descriptor.
    """
    module = ctx.module
    q, k, v = (one(ctx.argument(name, i)) for i, name in enumerate(("query", "key", "value")))
    y = one(ctx.outputs[0])
    out = one(ctx.parameter("out_proj.weight"))
    axis = out.axis(0)
    width, heads = module.embed_dim, module.num_heads
    relations = [
        AxisRelation.equal(axis, out.axis(1), ctx.node.name),
        AxisRelation.equal(axis, q.axis(-1), ctx.node.name),
        AxisRelation.equal(axis, y.axis(-1), ctx.node.name),
    ]
    requirements = [requirement(ctx, "embed_dim", y, -1)]
    for name in ("in_features", "out_features"):
        requirements.append(
            Requirement(
                "attribute",
                f"{ctx.module_path}.out_proj.{name}".lstrip("."),
                (out,),
                "Bind output projection width",
                (("axis", axis),),
            )
        )
    requirements.append(
        Requirement(
            "attribute",
            f"{ctx.module_path}.head_dim".lstrip("."),
            (out,),
            "Derive per-head width with fixed head count",
            (
                (
                    "expression",
                    ShapeExpr(
                        "floordiv",
                        args=(
                            ShapeExpr("dimension", (out, 0)),
                            ShapeExpr("constant", heads),
                        ),
                    ),
                ),
            ),
        )
    )
    if module.in_proj_weight is not None:
        packed = one(ctx.parameter("in_proj_weight"))
        relations.append(AxisRelation.equal(axis, packed.axis(1), ctx.node.name))
        relations.append(
            AxisRelation(
                Port(axis),
                Port(packed.axis(0)),
                tuple(BlockMap(0, i * width, width) for i in range(3)),
                ctx.node.name,
            )
        )
        relations.extend(
            (
                AxisRelation.equal(axis, k.axis(-1), ctx.node.name),
                AxisRelation.equal(axis, v.axis(-1), ctx.node.name),
            )
        )
    else:
        for name, input_ in (("q", q), ("k", k), ("v", v)):
            weight = one(ctx.parameter(f"{name}_proj_weight"))
            relations.extend(
                (
                    AxisRelation.equal(axis, weight.axis(0), ctx.node.name),
                    equal(input_, -1, weight, 1, ctx.node.name),
                )
            )
    for name, tensor in (("kdim", k), ("vdim", v)):
        requirements.append(requirement(ctx, name, tensor, -1))
    for name in ("in_proj_bias", "out_proj.bias", "bias_k", "bias_v"):
        bias = ctx.parameter(name)
        if bias is not None:
            if name == "in_proj_bias":
                relations.append(
                    AxisRelation(
                        Port(axis),
                        Port(bias.axis(0)),
                        tuple(BlockMap(0, i * width, width) for i in range(3)),
                        ctx.node.name,
                    )
                )
            else:
                relations.append(AxisRelation.equal(axis, bias.axis(-1), ctx.node.name))
    block = width // heads
    constraints = [
        Balanced(axis, tuple(IndexSet.span(h * block, (h + 1) * block) for h in range(heads)))
    ]
    constraints.extend(
        AxisBarrier(t.axis(d), "Native MHA batch/token positions are fixed", ctx.node.name)
        for t in (q, k, v, y)
        for d in range(len(t.shape) - 1)
    )
    masks = tuple(tensors((ctx.argument("key_padding_mask", 3), ctx.argument("attn_mask", 5))))
    for ref in (*masks, *ctx.outputs[1:]):
        constraints.extend(
            AxisBarrier(ref.axis(d), "Native MHA mask/weight axes are fixed", ctx.node.name)
            for d in range(len(ref.shape))
        )
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        tuple(requirements),
        candidates=(CandidateAxis(f"{out.paths[0]}:0", axis),),
        contract=CallContract(fresh_output=True, output_layout="backend_dependent"),
    )


def register_attention(modules, functions, methods):
    """Register matrix and attention APIs without extending the executor registry."""
    functions([torch.addmm, torch.baddbmm], addmm)
    methods(["addmm", "baddbmm"], addmm)
    functions([torch.einsum], einsum)
    functions([F.scaled_dot_product_attention], sdpa)
    modules([nn.MultiheadAttention], multihead_attention)
