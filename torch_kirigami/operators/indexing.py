"""Axis, partition, and block-coordinate families without activation label tensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ArgumentRef, AxisBarrier, Balanced, Requirement
from ..errors import UnsupportedOperation
from ..operation import OperationContext, OperatorSpec, OutputContract
from ..relations import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    ReshapeRelation,
    SliceRelation,
)
from ..selection import IndexSet
from .coordinates import narrow_index
from .native import equal, one, split


def stack(ctx: OperationContext) -> OperatorSpec:
    """Align every stacked input and protect the newly introduced port axis."""
    items, y = ctx.argument("tensors", 0), one(ctx.output)
    dim = ctx.argument("dim", 1, 0) % len(y.shape)
    relations = []
    for x in items:
        for d in range(len(x.shape)):
            relations.append(equal(x, d, y, d + (d >= dim), ctx.node.name))
    return OperatorSpec(
        tuple(relations),
        (
            AxisBarrier(
                y.axis(dim), "Removing stack inputs requires editing forward", ctx.node.name
            ),
        ),
    )


def chunk(ctx: OperationContext) -> OperatorSpec:
    """Describe captured chunk ports and retain their original coordinate identity."""
    x = one(ctx.argument("input", 0))
    dim = ctx.argument("dim", 2, 0) % len(x.shape)
    width = ctx.outputs[0].shape[dim]
    result = split(replace(ctx, args=(x, width, dim), kwargs={}))
    requirements = tuple(
        replace(r, data=(*r.data, ("chunks", ctx.argument("chunks", 1))), arguments=())
        for r in result.requirements
    )
    return replace(result, requirements=requirements)


def narrow(ctx: OperationContext) -> OperatorSpec:
    """Map a fixed contiguous interval and verify it after upstream compaction."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    dim, start, length = ctx.argument("dim", 1), ctx.argument("start", 2), ctx.argument("length", 3)
    if not all(type(v) is int for v in (dim, start, length)):
        raise UnsupportedOperation("narrow requires static integer arguments")
    dim %= len(x.shape)
    index = narrow_index(x.shape, dim, start, length)
    return OperatorSpec(
        (SliceRelation(x, y, tuple(index), ctx.node.name),),
        requirements=(
            Requirement(
                "slice_arguments",
                ctx.node.name,
                (x, y),
                "Preserve narrow coordinates",
                (("index", tuple(index)), ("narrow_dim", dim)),
                arguments=(ArgumentRef("start", 2), ArgumentRef("length", 3)),
            ),
        ),
    )


def repeat(ctx: OperationContext) -> OperatorSpec:
    """Map repeated positions through compressed block relations."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    target = ctx.node.target
    if target in ("expand", "expand_as", torch.broadcast_to):
        relations = [BroadcastRelation(x, y, ctx.node.name)]
        if target == "expand_as":
            template = one(ctx.argument("other", 1))
            relations.append(BroadcastRelation(template, y, ctx.node.name))
        return OperatorSpec(
            tuple(relations),
            requirements=(
                Requirement(
                    "call_arguments",
                    ctx.node.name,
                    (x, y),
                    "Validate compact broadcast dimensions",
                    arguments=(
                        ArgumentRef(
                            "size" if target is torch.broadcast_to else "sizes", 1, variadic=True
                        ),
                    )
                    if target != "expand_as"
                    else (),
                ),
            ),
        )
    relations, constraints = [], []
    if target in ("repeat_interleave", torch.repeat_interleave):
        count = ctx.argument("repeats", 1)
        dim = ctx.argument("dim", 2)
        if type(count) is not int or count < 1 or type(dim) is not int:
            raise UnsupportedOperation(
                "repeat_interleave requires scalar repeats and an explicit dim"
            )
        dim %= len(x.shape)
        for d, size in enumerate(x.shape):
            relations.append(
                AxisRelation(
                    AxisPort(x.axis(d)),
                    AxisPort(y.axis(d)),
                    (BlockMap(0, 0, size, 1, count if d == dim else 1),),
                    ctx.node.name,
                )
            )
    else:
        # Repetition factors come from the arguments, never coincidentally equal shapes.
        factors = ctx.argument(
            "dims" if target in (torch.tile, "tile") else "sizes", 1, variadic=True
        )
        if len(factors) == 1 and isinstance(factors[0], (tuple, list)):
            factors = factors[0]
        if not isinstance(factors, (tuple, list)) or not all(
            type(n) is int and n > 0 for n in factors
        ):
            raise UnsupportedOperation("repeat/tile factors must be positive constants")
        factors = (1,) * max(0, len(x.shape) - len(factors)) + tuple(factors)
        leading = len(factors) - len(x.shape)
        for d in range(leading):
            constraints.append(
                AxisBarrier(y.axis(d), "Repeat factor is fixed in forward", ctx.node.name)
            )
        for d, size in enumerate(x.shape):
            relations.append(
                AxisRelation(
                    AxisPort(x.axis(d)),
                    AxisPort(y.axis(d + leading)),
                    tuple(BlockMap(0, k * size, size) for k in range(factors[d + leading])),
                    ctx.node.name,
                )
            )
    return OperatorSpec(tuple(relations), tuple(constraints))


def index_select(ctx: OperationContext) -> OperatorSpec:
    """Map captured constant indices and guard their values for portable replay."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    dim = ctx.argument("dim", 1)
    index = one(ctx.argument("index", 2))
    indices = ctx.constants.get(index.id)
    if indices is None or len(index.shape) != 1:
        raise UnsupportedOperation("index_select requires a captured constant integer vector")
    if not x.shape:
        return OperatorSpec(
            (ReshapeRelation(x, y, ctx.node.name),),
            (
                AxisBarrier(
                    index.axis(0), "Static index vector entries cannot be deleted", ctx.node.name
                ),
            ),
            constants=(index,),
        )
    dim %= len(x.shape)
    if any(type(i) is not int or not 0 <= i < x.shape[dim] for i in indices):
        raise UnsupportedOperation("Invalid static index_select coordinates")
    relations = [equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)) if d != dim]
    relations.append(
        AxisRelation(
            AxisPort(x.axis(dim)),
            AxisPort(y.axis(dim)),
            tuple(BlockMap(i, j, 1) for j, i in enumerate(indices)),
            ctx.node.name,
        )
    )
    return OperatorSpec(
        tuple(relations),
        (
            AxisBarrier(
                index.axis(0), "Static index vector entries cannot be deleted", ctx.node.name
            ),
        ),
        (
            Requirement(
                "index_arguments",
                ctx.node.name,
                (x, y),
                "Preserve static index coordinates",
                (("axis", x.axis(dim)), ("indices", indices)),
            ),
        ),
        constants=(index,),
    )


def glu(ctx: OperationContext) -> OperatorSpec:
    """Tie both gate halves to the same compact output positions."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    dim = (ctx.module.dim if ctx.module is not None else ctx.argument("dim", 1, -1)) % len(x.shape)
    width = y.shape[dim]
    relations = [equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)) if d != dim]
    relations.append(
        AxisRelation(
            AxisPort(y.axis(dim)),
            AxisPort(x.axis(dim)),
            (BlockMap(0, 0, width), BlockMap(0, width, width)),
            ctx.node.name,
        )
    )
    return OperatorSpec(tuple(relations))


def channel_shuffle(ctx: OperationContext) -> OperatorSpec:
    """Represent shuffle as a channel permutation with fixed group partitions."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    groups = ctx.module.groups if ctx.module is not None else ctx.argument("groups", 1)
    if type(groups) is not int:
        raise UnsupportedOperation("ChannelShuffle groups must be constant")
    width = x.shape[1] // groups
    if x.shape[1] > 4096:
        raise UnsupportedOperation("ChannelShuffle compressed map limit exceeded")
    relations = [equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)) if d != 1]
    # Group-local removal patterns must match: equal counts alone would change
    # the original channel correspondence after transposing the compact groups.
    for local in range(width):
        for group in range(groups):
            relations.append(
                AxisRelation(
                    AxisPort(x.axis(1)),
                    AxisPort(y.axis(1)),
                    (BlockMap(group * width + local, local * groups + group, 1),),
                    ctx.node.name,
                )
            )
            if group:
                relations.append(
                    AxisRelation(
                        AxisPort(x.axis(1)),
                        AxisPort(x.axis(1)),
                        (BlockMap(local, group * width + local, 1),),
                        ctx.node.name,
                    )
                )
    return OperatorSpec(
        tuple(relations),
        (
            Balanced(
                x.axis(1), tuple(IndexSet.span(g * width, (g + 1) * width) for g in range(groups))
            ),
        ),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def pixel_shuffle(ctx: OperationContext) -> OperatorSpec:
    """Connect complete channel blocks while fixing spatial coordinates."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    inverse = type(ctx.module) is nn.PixelUnshuffle or ctx.node.target in (
        F.pixel_unshuffle,
        torch.pixel_unshuffle,
    )
    name = "downscale_factor" if inverse else "upscale_factor"
    factor = getattr(ctx.module, name) if ctx.module is not None else ctx.argument(name, 1)
    if type(factor) is not int or factor < 1:
        raise UnsupportedOperation("Pixel shuffle factor must be constant")
    a, b = (y, x) if inverse else (x, y)
    channel = len(x.shape) - 3
    relations = [equal(x, d, y, d, ctx.node.name) for d in range(channel)]
    relations.append(
        AxisRelation(
            AxisPort(b.axis(channel)),
            AxisPort(a.axis(channel)),
            (BlockMap(0, 0, b.shape[channel], 1, factor * factor),),
            ctx.node.name,
        )
    )
    constraints = tuple(
        AxisBarrier(t.axis(d), "Pixel shuffle spatial coordinates are fixed", ctx.node.name)
        for t in (x, y)
        for d in range(channel + 1, len(x.shape))
    )
    return OperatorSpec(
        tuple(relations),
        constraints,
        contract=OutputContract(output_layout="backend_dependent"),
    )


def unfold_fold(ctx: OperationContext) -> OperatorSpec:
    """Connect complete im2col channel blocks; spatial/kernel coordinates stay fixed."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    fold = type(ctx.module) is nn.Fold or ctx.node.target is F.fold
    kernel = (
        ctx.module.kernel_size
        if ctx.module is not None
        else ctx.argument("kernel_size", 2 if fold else 1)
    )
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if not isinstance(kernel, (tuple, list)) or len(kernel) != 2:
        raise UnsupportedOperation("Only static two-dimensional im2col kernels are supported")
    image, columns = (y, x) if fold else (x, y)
    if len(image.shape) != 4 or len(columns.shape) != 3:
        raise UnsupportedOperation("Use batched Unfold/Fold tensors")
    relations = (
        equal(x, 0, y, 0, ctx.node.name),
        AxisRelation(
            AxisPort(image.axis(1)),
            AxisPort(columns.axis(1)),
            (BlockMap(0, 0, image.shape[1], 1, prod(kernel)),),
            ctx.node.name,
        ),
    )
    constraints = tuple(
        AxisBarrier(
            t.axis(d), "Unfold/Fold spatial and kernel coordinates are fixed", ctx.node.name
        )
        for t in (x, y)
        for d in range(2, len(t.shape))
    )
    return OperatorSpec(relations, constraints)


def register_indexing(
    modules: Callable[..., None], functions: Callable[..., None], methods: Callable[..., None]
) -> None:
    """Register exact indexing-family API spellings."""
    functions([torch.stack], stack, fresh=True)
    functions([torch.chunk], chunk, fresh=False)
    methods(["chunk"], chunk, fresh=False)
    functions([torch.narrow], narrow, fresh=False)
    methods(["narrow"], narrow, fresh=False)
    functions([torch.index_select], index_select, fresh=True)
    methods(["index_select"], index_select, fresh=True)
    functions([torch.tile, torch.repeat_interleave], repeat, fresh=True)
    functions([torch.broadcast_to], repeat, fresh=False)
    methods(["repeat", "tile", "repeat_interleave"], repeat, fresh=True)
    methods(["expand", "expand_as"], repeat, fresh=False)
    modules([nn.GLU], glu, fresh=True)
    functions([F.glu], glu, fresh=True)
    modules([nn.ChannelShuffle], channel_shuffle, fresh=True)
    functions([torch.channel_shuffle], channel_shuffle, fresh=True)
    modules([nn.PixelShuffle, nn.PixelUnshuffle], pixel_shuffle, fresh=True)
    functions(
        list(
            dict.fromkeys(
                [F.pixel_shuffle, F.pixel_unshuffle, torch.pixel_shuffle, torch.pixel_unshuffle]
            )
        ),
        pixel_shuffle,
        fresh=True,
    )
    modules([nn.Unfold, nn.Fold], unfold_fold, fresh=False)
    functions([F.unfold, F.fold], unfold_fold, fresh=False)
