"""Common channel, normalization, embedding, and elementwise operator families."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ...errors import UnsupportedOperation
from ..contracts import AxisBarrier, Requirement
from ..operation import (
    CandidateAxis,
    Operation,
    OperatorSpec,
    OutputContract,
)
from ..relations import ReshapeRelation
from ..selection import TensorRef
from .native import (
    affine_layouts,
    bind,
    equal,
    layout,
    one,
    pointwise,
    requirement,
)


def channel_operator(operation: Operation) -> OperatorSpec:
    """Preserve batch/channel coordinates while protecting spatial transformations."""
    x = one(operation.argument("input", 0))
    outputs = operation.outputs
    # Public pooling/padding/interpolation APIs place channels immediately before
    # their spatial axes. Batched and unbatched pooling forms share this descriptor.
    name = (
        type(operation.module).__name__.lower()
        if operation.module is not None
        else str(getattr(operation.node.target, "__name__", operation.node.target))
    )
    spatial = next((n for n in (1, 2, 3) if f"{n}d" in name), None)
    channel = len(x.shape) - spatial - 1 if spatial is not None else 1
    if channel not in (0, 1) or len(x.shape) < channel + 2:
        raise UnsupportedOperation("Expected a channel/spatial tensor")
    relations, constraints = [], []
    for y in (*outputs, *(t for t in operation.inputs if t != x)):
        if len(y.shape) != len(x.shape) or y.shape[channel] != x.shape[channel]:
            raise UnsupportedOperation("Only channel-preserving spatial calls are supported")
        for d in range(channel + 1):
            relations.append(equal(x, d, y, d, operation.node.name))
    for tensor in (*operation.inputs, *outputs):
        constraints.extend(
            AxisBarrier(
                tensor.axis(d),
                "Spatial resizing needs a separate coordinate rule",
                operation.node.name,
            )
            for d in range(channel + 1, len(tensor.shape))
        )
    # Pooling/interpolation can select different layouts by backend and grad
    # path. Meta remains useful for shape checks, not for a compact stride proof.
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def padding(operation: Operation) -> OperatorSpec:
    """Preserve only axes untouched by padding, independently of their names/size."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    pad = operation.module.padding if operation.module is not None else operation.argument("pad", 1)
    if isinstance(pad, int) and operation.module is not None:
        spatial = next(n for n in (1, 2, 3) if f"{n}d" in type(operation.module).__name__.lower())
        pad = (pad,) * (2 * spatial)
    if not isinstance(pad, (tuple, list)) or len(pad) % 2 or len(pad) > 2 * len(x.shape):
        raise UnsupportedOperation("Padding requires explicit axis pairs")
    changed = {len(x.shape) - 1 - i // 2 for i, value in enumerate(pad) if value != 0}
    # Equal final sizes do not establish identity: (-1, 1) crops one position
    # and appends another. Protect that axis on both sides, allowing other axes.
    return OperatorSpec(
        tuple(
            equal(x, d, y, d, operation.node.name) for d in range(len(x.shape)) if d not in changed
        ),
        tuple(
            AxisBarrier(t.axis(d), "Padded/cropped coordinates are fixed", operation.node.name)
            for t in (x, y)
            for d in sorted(changed)
        ),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def instance_norm(operation: Operation) -> OperatorSpec:
    """Bind affine/statistic channels for batched and unbatched InstanceNorm."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    if operation.module is None:
        channel = 1
    else:
        spatial = {nn.InstanceNorm1d: 1, nn.InstanceNorm2d: 2, nn.InstanceNorm3d: 3}[
            type(operation.module)
        ]
        channel = len(x.shape) - spatial - 1
    relations = [ReshapeRelation(x, y, operation.node.name)]
    positions = (
        (("weight", 1), ("bias", 2), ("running_mean", 3), ("running_var", 4))
        if operation.node.target is torch.instance_norm
        else (("running_mean", 1), ("running_var", 2), ("weight", 3), ("bias", 4))
    )
    for name, position in positions:
        value = bind(operation, name, position)
        if value is not None:
            relations.append(equal(x, channel, one(value), 0, operation.node.name))
    requirements = (
        () if operation.module is None else (requirement(operation, "num_features", x, channel),)
    )
    return OperatorSpec(
        tuple(relations), (*affine_layouts(operation), layout(operation, y)), requirements
    )


def prelu(operation: Operation) -> OperatorSpec:
    """Bind per-channel PReLU weights; leave the shared scalar slope unchanged."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    weight = one(bind(operation, "weight", 1))
    relations = [ReshapeRelation(x, y, operation.node.name)]
    constraints, requirements = [], []
    if weight.shape == (1,):
        constraints.append(
            AxisBarrier(weight.axis(0), "A scalar PReLU slope is shared", operation.node.name)
        )
    else:
        channel = 1 if len(x.shape) > 1 else 0
        relations.append(equal(x, channel, weight, 0, operation.node.name))
        if operation.module is not None:
            requirements.append(requirement(operation, "num_parameters", x, channel))
    return OperatorSpec(tuple(relations), tuple(constraints), tuple(requirements))


def embedding(operation: Operation) -> OperatorSpec:
    """Expose embedding feature width while fixing vocabulary IDs and row positions."""
    native = operation.node.target is torch.embedding
    x = one(operation.argument("indices", 1) if native else operation.argument("input", 0))
    y = one(operation.result)
    weight = one(bind(operation, "weight", 0 if native else 1))
    relations = [equal(weight, 1, y, -1, operation.node.name)]
    relations.extend(equal(x, d, y, d, operation.node.name) for d in range(len(x.shape)))
    constraints = (
        AxisBarrier(
            weight.axis(0), "Vocabulary/token-ID remapping is unsupported", operation.node.name
        ),
    )
    requirements = (
        () if operation.module is None else (requirement(operation, "embedding_dim", weight, 1),)
    )
    candidates = (
        () if operation.module is None else (CandidateAxis(f"{weight.paths[0]}:1", weight.axis(1)),)
    )
    return OperatorSpec(
        tuple(relations),
        constraints,
        requirements,
        candidates=candidates,
        contract=OutputContract(output_layout="contiguous"),
    )


def normalize(operation: Operation) -> OperatorSpec:
    """Preserve coordinates while recomputing the norm over the compact domain."""
    result = pointwise(operation)
    x = one(operation.argument("input", 0))
    return replace(
        result,
        requirements=(
            Requirement(
                "reduction_domain",
                operation.node.name,
                (x,),
                "Recompute normalization on retained positions; zero-mask equivalence is not assumed",
            ),
        ),
    )


def cast(operation: Operation) -> OperatorSpec:
    """Preserve input coordinates; a dtype/device reference contributes no axes."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    copy_output = False
    if operation.node.target == "to":
        position = (
            3
            if len(operation.args) > 1 and isinstance(operation.args[1], (torch.dtype, TensorRef))
            else 4
        )
        copy_output = operation.argument("copy", position, False)
        if type(copy_output) is not bool:
            raise UnsupportedOperation("Tensor.to copy must be a static boolean")
    return OperatorSpec(
        (ReshapeRelation(x, y, operation.node.name),),
        contract=OutputContract(output_layout="cast", copy_output=copy_output),
    )
