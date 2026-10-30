"""Axis, partition, and block-coordinate families without activation label tensors."""

from __future__ import annotations

from dataclasses import replace
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from ...errors import UnsupportedOperation
from ...settings import config
from ..contracts import ArgumentRef, AxisBarrier, Balanced, Requirement
from ..operation import Operation, OperatorSpec, OutputContract
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


def tensor_unfold(operation: Operation) -> OperatorSpec:
    """Preserve nonwindow axes of Tensor.unfold without expanding overlapping patches.

    The unfolded input axis, output window count, and appended window axis stay
    fixed. The output is a view; its rule must not promise independent storage.
    """
    x, y = one(operation.argument("input", 0)), one(operation.result)
    dim = operation.argument("dimension", 1)
    size, step = operation.argument("size", 2), operation.argument("step", 3)
    if not x.shape or not all(type(v) is int for v in (dim, size, step)):
        raise UnsupportedOperation("Tensor.unfold requires a nonscalar input and integer arguments")
    if size < 1 or step < 1:
        raise UnsupportedOperation("Tensor.unfold requires positive size and step")
    dim %= len(x.shape)
    relations = tuple(
        equal(x, d, y, d, operation.node.name) for d in range(len(x.shape)) if d != dim
    )
    constraints = tuple(
        AxisBarrier(
            axis,
            "Tensor.unfold window coordinates stay fixed; only nonwindow axes can shrink",
            operation.node.name,
        )
        for axis in (x.axis(dim), y.axis(dim), y.axis(-1))
    )
    return OperatorSpec(relations, constraints)


def stack(operation: Operation) -> OperatorSpec:
    """Align every stacked input and protect the newly introduced port axis."""
    items, y = operation.argument("tensors", 0), one(operation.result)
    dim = operation.argument("dim", 1, 0) % len(y.shape)
    relations = []
    for x in items:
        for d in range(len(x.shape)):
            relations.append(equal(x, d, y, d + (d >= dim), operation.node.name))
    return OperatorSpec(
        tuple(relations),
        (
            AxisBarrier(
                y.axis(dim), "Removing stack inputs requires editing forward", operation.node.name
            ),
        ),
    )


def chunk(operation: Operation) -> OperatorSpec:
    """Describe captured chunk ports and retain their original coordinate identity."""
    x = one(operation.argument("input", 0))
    dim = operation.argument("dim", 2, 0) % len(x.shape)
    width = operation.outputs[0].shape[dim]
    result = split(replace(operation, args=(x, width, dim), kwargs={}))
    requirements = tuple(
        replace(r, data=(*r.data, ("chunks", operation.argument("chunks", 1))), arguments=())
        for r in result.requirements
    )
    return replace(result, requirements=requirements)


def narrow(operation: Operation) -> OperatorSpec:
    """Map a fixed contiguous interval and verify it after upstream compaction."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    dim, start, length = (
        operation.argument("dim", 1),
        operation.argument("start", 2),
        operation.argument("length", 3),
    )
    if not all(type(v) is int for v in (dim, start, length)):
        raise UnsupportedOperation("narrow requires static integer arguments")
    dim %= len(x.shape)
    index = narrow_index(x.shape, dim, start, length)
    return OperatorSpec(
        (SliceRelation(x, y, tuple(index), operation.node.name),),
        requirements=(
            Requirement(
                "slice_arguments",
                operation.node.name,
                (x, y),
                "Preserve narrow coordinates",
                (("index", tuple(index)), ("narrow_dim", dim)),
                arguments=(ArgumentRef("start", 2), ArgumentRef("length", 3)),
            ),
        ),
    )


def select(operation: Operation) -> OperatorSpec:
    """Map an integer-selected slice while preserving its indexing axis.

    The selected axis stays fixed, so the original integer cannot acquire a
    different coordinate after compaction. Other axes map through SliceRelation.
    """
    x, y = one(operation.argument("input", 0)), one(operation.result)
    dim, index = operation.argument("dim", 1), operation.argument("index", 2)
    if not x.shape or type(dim) is not int or type(index) is not int:
        raise UnsupportedOperation("select requires a nonscalar tensor and static integer indices")
    dim %= len(x.shape)
    indices = [slice(None)] * len(x.shape)
    indices[dim] = index
    return OperatorSpec(
        (SliceRelation(x, y, tuple(indices), operation.node.name),),
        (
            AxisBarrier(
                x.axis(dim),
                "The select indexing axis stays fixed to preserve original integer coordinates",
                operation.node.name,
            ),
        ),
    )


def repeat(operation: Operation) -> OperatorSpec:
    """Map repeated positions through compressed block relations."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    target = operation.node.target
    if target in ("expand", "expand_as", "broadcast_to", torch.broadcast_to):
        relations = [BroadcastRelation(x, y, operation.node.name)]
        if target == "expand_as":
            template = one(operation.argument("other", 1))
            relations.append(BroadcastRelation(template, y, operation.node.name))
        return OperatorSpec(
            tuple(relations),
            requirements=(
                Requirement(
                    "call_arguments",
                    operation.node.name,
                    (x, y),
                    "Validate compact broadcast dimensions",
                    arguments=(
                        ArgumentRef(
                            "size" if target in ("broadcast_to", torch.broadcast_to) else "sizes",
                            1,
                            variadic=True,
                        ),
                    )
                    if target != "expand_as"
                    else (),
                ),
            ),
        )
    relations, constraints = [], []
    if target in ("repeat_interleave", torch.repeat_interleave):
        count = operation.argument("repeats", 1)
        dim = operation.argument("dim", 2)
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
                    operation.node.name,
                )
            )
    else:
        # Repetition factors come from the arguments, never coincidentally equal shapes.
        factors = operation.argument(
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
                AxisBarrier(y.axis(d), "Repeat factor is fixed in forward", operation.node.name)
            )
        for d, size in enumerate(x.shape):
            relations.append(
                AxisRelation(
                    AxisPort(x.axis(d)),
                    AxisPort(y.axis(d + leading)),
                    tuple(BlockMap(0, k * size, size) for k in range(factors[d + leading])),
                    operation.node.name,
                )
            )
    return OperatorSpec(tuple(relations), tuple(constraints))


def index_select(operation: Operation) -> OperatorSpec:
    """Map captured constant indices and guard their values for portable replay."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    dim = operation.argument("dim", 1)
    index = one(operation.argument("index", 2))
    indices = operation.constants.get(index.id)
    if indices is None or len(index.shape) != 1:
        raise UnsupportedOperation("index_select requires a captured constant integer vector")
    if not x.shape:
        return OperatorSpec(
            (ReshapeRelation(x, y, operation.node.name),),
            (
                AxisBarrier(
                    index.axis(0),
                    "Static index vector entries cannot be deleted",
                    operation.node.name,
                ),
            ),
            constants=(index,),
        )
    dim %= len(x.shape)
    if any(type(i) is not int or not 0 <= i < x.shape[dim] for i in indices):
        raise UnsupportedOperation("Invalid static index_select coordinates")
    relations = [equal(x, d, y, d, operation.node.name) for d in range(len(x.shape)) if d != dim]
    relations.append(
        AxisRelation(
            AxisPort(x.axis(dim)),
            AxisPort(y.axis(dim)),
            tuple(BlockMap(i, j, 1) for j, i in enumerate(indices)),
            operation.node.name,
        )
    )
    return OperatorSpec(
        tuple(relations),
        (
            AxisBarrier(
                index.axis(0), "Static index vector entries cannot be deleted", operation.node.name
            ),
        ),
        (
            Requirement(
                "index_arguments",
                operation.node.name,
                (x, y),
                "Preserve static index coordinates",
                (("axis", x.axis(dim)), ("indices", indices)),
            ),
        ),
        constants=(index,),
    )


def glu(operation: Operation) -> OperatorSpec:
    """Tie both gate halves to the same compact output positions."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    dim = (
        operation.module.dim if operation.module is not None else operation.argument("dim", 1, -1)
    ) % len(x.shape)
    width = y.shape[dim]
    relations = [equal(x, d, y, d, operation.node.name) for d in range(len(x.shape)) if d != dim]
    relations.append(
        AxisRelation(
            AxisPort(y.axis(dim)),
            AxisPort(x.axis(dim)),
            (BlockMap(0, 0, width), BlockMap(0, width, width)),
            operation.node.name,
        )
    )
    return OperatorSpec(tuple(relations))


def channel_shuffle(operation: Operation) -> OperatorSpec:
    """Represent shuffle as a channel permutation with fixed group partitions."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    groups = (
        operation.module.groups if operation.module is not None else operation.argument("groups", 1)
    )
    if type(groups) is not int:
        raise UnsupportedOperation("ChannelShuffle groups must be constant")
    width = x.shape[1] // groups
    if x.shape[1] > config.max_shuffle_channels:
        raise UnsupportedOperation("ChannelShuffle compressed map limit exceeded")
    relations = [equal(x, d, y, d, operation.node.name) for d in range(len(x.shape)) if d != 1]
    # Group-local removal patterns must match: equal counts alone would change
    # the original channel correspondence after transposing the compact groups.
    permutation, tied = [], []
    for local in range(width):
        for group in range(groups):
            permutation.append(BlockMap(group * width + local, local * groups + group, 1))
            if group:
                tied.append(BlockMap(local, group * width + local, 1))
    relations.append(
        AxisRelation(
            AxisPort(x.axis(1)), AxisPort(y.axis(1)), tuple(permutation), operation.node.name
        )
    )
    if tied:
        relations.append(
            AxisRelation(AxisPort(x.axis(1)), AxisPort(x.axis(1)), tuple(tied), operation.node.name)
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


def pixel_shuffle(operation: Operation) -> OperatorSpec:
    """Connect complete channel blocks while fixing spatial coordinates."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    inverse = type(operation.module) is nn.PixelUnshuffle or operation.node.target in (
        F.pixel_unshuffle,
        torch.pixel_unshuffle,
    )
    name = "downscale_factor" if inverse else "upscale_factor"
    factor = (
        getattr(operation.module, name)
        if operation.module is not None
        else operation.argument(name, 1)
    )
    if type(factor) is not int or factor < 1:
        raise UnsupportedOperation("Pixel shuffle factor must be constant")
    a, b = (y, x) if inverse else (x, y)
    channel = len(x.shape) - 3
    relations = [equal(x, d, y, d, operation.node.name) for d in range(channel)]
    relations.append(
        AxisRelation(
            AxisPort(b.axis(channel)),
            AxisPort(a.axis(channel)),
            (BlockMap(0, 0, b.shape[channel], 1, factor * factor),),
            operation.node.name,
        )
    )
    constraints = tuple(
        AxisBarrier(t.axis(d), "Pixel shuffle spatial coordinates are fixed", operation.node.name)
        for t in (x, y)
        for d in range(channel + 1, len(x.shape))
    )
    return OperatorSpec(
        tuple(relations),
        constraints,
        contract=OutputContract(output_layout="backend_dependent"),
    )


def unfold_fold(operation: Operation) -> OperatorSpec:
    """Connect complete im2col channel blocks; spatial/kernel coordinates stay fixed."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    fold = type(operation.module) is nn.Fold or operation.node.target is F.fold
    kernel = (
        operation.module.kernel_size
        if operation.module is not None
        else operation.argument("kernel_size", 2 if fold else 1)
    )
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if not isinstance(kernel, (tuple, list)) or len(kernel) != 2:
        raise UnsupportedOperation("Only static two-dimensional im2col kernels are supported")
    image, columns = (y, x) if fold else (x, y)
    if len(image.shape) != 4 or len(columns.shape) != 3:
        raise UnsupportedOperation("Use batched Unfold/Fold tensors")
    relations = (
        equal(x, 0, y, 0, operation.node.name),
        AxisRelation(
            AxisPort(image.axis(1)),
            AxisPort(columns.axis(1)),
            (BlockMap(0, 0, image.shape[1], 1, prod(kernel)),),
            operation.node.name,
        ),
    )
    constraints = tuple(
        AxisBarrier(
            t.axis(d), "Unfold/Fold spatial and kernel coordinates are fixed", operation.node.name
        )
        for t in (x, y)
        for d in range(2, len(t.shape))
    )
    return OperatorSpec(relations, constraints)
