"""Axis, partition, and block-coordinate families without activation label tensors."""

from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import AxisBarrier, Balanced, Requirement
from ..registry import OperatorSpec
from ..relations import AxisRelation, BlockMap, BroadcastRelation, Port, SliceRelation
from ..selection import IndexSet
from .coordinates import narrow_index
from .native import UnsupportedOperation, equal, one, split


def stack(ctx):
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


def chunk(ctx):
    """Describe captured chunk ports and retain their original coordinate identity."""
    from dataclasses import replace

    x = one(ctx.argument("input", 0))
    dim = ctx.argument("dim", 2, 0) % len(x.shape)
    width = ctx.outputs[0].shape[dim]
    result = split(replace(ctx, args=(x, width, dim), kwargs={}))
    requirements = tuple(
        replace(r, data=(*r.data, ("chunks", ctx.argument("chunks", 1))))
        for r in result.requirements
    )
    return replace(result, requirements=requirements)


def narrow(ctx):
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
            ),
        ),
    )


def repeat(ctx):
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
                    "call_arguments", ctx.node.name, (x, y), "Validate compact broadcast dimensions"
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
                    Port(x.axis(d)),
                    Port(y.axis(d)),
                    (BlockMap(0, 0, size, 1, count if d == dim else 1),),
                    ctx.node.name,
                )
            )
    else:
        # Repetition factors come from the arguments, never coincidentally equal shapes.
        factors = ctx.argument("dims" if target in (torch.tile, "tile") else "sizes", 1)
        if isinstance(factors, int):
            factors = ctx.args[1:]
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
                    Port(x.axis(d)),
                    Port(y.axis(d + leading)),
                    tuple(BlockMap(0, k * size, size) for k in range(factors[d + leading])),
                    ctx.node.name,
                )
            )
    return OperatorSpec(tuple(relations), tuple(constraints))


def index_select(ctx):
    """Map captured constant indices and guard their values for portable replay."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    dim = ctx.argument("dim", 1) % len(x.shape)
    index = one(ctx.argument("index", 2))
    indices = ctx.constants.get(index.id)
    if indices is None or len(index.shape) != 1:
        raise UnsupportedOperation("index_select requires a captured constant integer vector")
    if any(type(i) is not int or not 0 <= i < x.shape[dim] for i in indices):
        raise UnsupportedOperation("Invalid static index_select coordinates")
    relations = [equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)) if d != dim]
    relations.append(
        AxisRelation(
            Port(x.axis(dim)),
            Port(y.axis(dim)),
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


def glu(ctx):
    """Tie both gate halves to the same compact output positions."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    dim = (ctx.module.dim if ctx.module is not None else ctx.argument("dim", 1, -1)) % len(x.shape)
    width = y.shape[dim]
    relations = [equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape)) if d != dim]
    relations.append(
        AxisRelation(
            Port(y.axis(dim)),
            Port(x.axis(dim)),
            (BlockMap(0, 0, width), BlockMap(0, width, width)),
            ctx.node.name,
        )
    )
    return OperatorSpec(tuple(relations))


def channel_shuffle(ctx):
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
                    Port(x.axis(1)),
                    Port(y.axis(1)),
                    (BlockMap(group * width + local, local * groups + group, 1),),
                    ctx.node.name,
                )
            )
            if group:
                relations.append(
                    AxisRelation(
                        Port(x.axis(1)),
                        Port(x.axis(1)),
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
    )


def pixel_shuffle(ctx):
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
            Port(b.axis(channel)),
            Port(a.axis(channel)),
            (BlockMap(0, 0, b.shape[channel], 1, factor * factor),),
            ctx.node.name,
        )
    )
    constraints = tuple(
        AxisBarrier(t.axis(d), "Pixel shuffle spatial coordinates are fixed", ctx.node.name)
        for t in (x, y)
        for d in range(channel + 1, len(x.shape))
    )
    return OperatorSpec(tuple(relations), constraints)


def unfold_fold(ctx):
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
            Port(image.axis(1)),
            Port(columns.axis(1)),
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


def register_indexing(modules, functions, methods):
    """Register exact indexing-family API spellings."""
    functions([torch.stack], stack)
    functions([torch.chunk], chunk)
    methods(["chunk"], chunk)
    functions([torch.narrow], narrow)
    methods(["narrow"], narrow)
    functions([torch.index_select], index_select)
    methods(["index_select"], index_select)
    functions([torch.tile, torch.repeat_interleave, torch.broadcast_to], repeat)
    methods(["repeat", "tile", "repeat_interleave", "expand", "expand_as"], repeat)
    modules([nn.GLU], glu)
    functions([F.glu], glu)
    modules([nn.ChannelShuffle], channel_shuffle)
    functions([torch.channel_shuffle], channel_shuffle)
    modules([nn.PixelShuffle, nn.PixelUnshuffle], pixel_shuffle)
    functions(
        list(
            dict.fromkeys(
                [F.pixel_shuffle, F.pixel_unshuffle, torch.pixel_shuffle, torch.pixel_unshuffle]
            )
        ),
        pixel_shuffle,
    )
    modules([nn.Unfold, nn.Fold], unfold_fold)
    functions([F.unfold, F.fold], unfold_fold)
