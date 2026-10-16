"""Structural semantics expressed through the same interface as external rules."""

from __future__ import annotations

from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from ...errors import UnsupportedOperation
from ..contracts import (
    ArgumentRef,
    AttributeBinding,
    AxisBarrier,
    Balanced,
    BlockBalance,
    LayoutConstraint,
    NonEmpty,
    Requirement,
    ShapeExpr,
)
from ..operation import (
    CandidateAxis,
    Operation,
    OperatorSpec,
    OutputContract,
    PartitionedLayout,
    tensors,
)
from ..relations import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
)
from ..selection import IndexSet, Region, TensorRef, full_region
from .shapes import has_known_provenance, scalar_spec, shape_expression


def one(value: object) -> TensorRef:
    """Require one tensor reference instead of a container or scalar argument."""
    if not isinstance(value, TensorRef):
        raise UnsupportedOperation("Expected a single tensor")
    return value


def equal(a: TensorRef, da: int, b: TensorRef, db: int, reason: str) -> AxisRelation:
    """Build an identity relation between two specified tensor axes."""
    return AxisRelation.equal(a.axis(da), b.axis(db), reason)


def requirement(
    operation: Operation,
    name: str,
    tensor: TensorRef,
    dim: int,
    *,
    position: int | None = None,
) -> Requirement | AttributeBinding:
    """Bind an axis change to a module attribute or functional operator argument."""
    target = (
        f"{operation.module_path}.{name}".lstrip(".")
        if operation.module is not None
        else operation.node.name
    )
    if operation.module is not None:
        return AttributeBinding(target, ShapeExpr("dimension", (tensor, dim)))
    return Requirement(
        "operator_argument",
        target,
        (tensor,),
        f"Update {name} from the retained axis size",
        (("axis", tensor.axis(dim)),),
        arguments=(ArgumentRef(name, position),)
        if operation.module is None and position is not None
        else (),
    )


def bind(
    operation: Operation, name: str, position: int, default: object = None
) -> TensorRef | None:
    """Resolve a module-local binding or a normalized functional argument."""
    return (
        operation.bindings.get(name)
        if operation.module is not None
        else operation.argument(name, position, default)
    )


def _identity(operation: Operation) -> OperatorSpec:
    """Preserve tensor coordinates for a shape-preserving operation."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    if x.shape != y.shape:
        raise UnsupportedOperation("Identity rule requires equal shapes")
    return OperatorSpec((ReshapeRelation(x, y, operation.node.name),))


def pointwise(operation: Operation) -> OperatorSpec:
    """Connect broadcast-compatible inputs to an elementwise output."""
    scalar = scalar_spec(operation)
    if scalar is not None:
        return scalar
    y = one(operation.result)
    relations = []
    for x in operation.inputs:
        try:
            relations.append(BroadcastRelation(x, y, operation.node.name))
        except ValueError as error:
            raise UnsupportedOperation("Unproven broadcast correspondence") from error
    contract = (
        OutputContract(output_layout="contiguous")
        if operation.node.target == "contiguous"
        else None
    )
    return OperatorSpec(tuple(relations), contract=contract)


def layout(
    operation: Operation, tensor: TensorRef, ports: tuple[object, ...] = ()
) -> LayoutConstraint:
    """Declare the compact layouts accepted by a particular tensor use."""
    return LayoutConstraint(tensor, tuple(ports), node=operation.node.name)


def affine_layouts(operation: Operation) -> tuple[LayoutConstraint, ...]:
    """Require ordinary axis compaction for each input and bound affine tensor."""
    return tuple(
        layout(operation, ref)
        for ref in dict.fromkeys((*operation.inputs, *operation.bindings.values()))
    )


def linear(operation: Operation) -> OperatorSpec:
    """Connect Linear feature axes, batch dimensions, weight, and bias."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    w, b = one(bind(operation, "weight", 1)), bind(operation, "bias", 2)
    if len(w.shape) not in (1, 2):
        raise UnsupportedOperation("Linear requires rank-one or rank-two weight")
    relations = [equal(x, -1, w, -1, operation.node.name)]
    if len(w.shape) == 2:
        relations.append(equal(y, -1, w, 0, operation.node.name))
    relations.extend(equal(x, d, y, d, operation.node.name) for d in range(len(x.shape) - 1))
    if b is not None:
        relations.append(BroadcastRelation(one(b), y, operation.node.name))
    reqs = (
        ()
        if operation.module is None
        else (
            requirement(operation, "in_features", x, -1),
            requirement(operation, "out_features", y, -1),
        )
    )
    candidates = () if operation.module is None else (CandidateAxis(f"{w.paths[0]}:0", w.axis(0)),)
    return OperatorSpec(
        tuple(relations),
        (layout(operation, x), layout(operation, y), layout(operation, w)),
        reqs,
        candidates=candidates,
        contract=OutputContract(output_layout="contiguous"),
    )


def convolution(operation: Operation) -> OperatorSpec:
    """Describe ordinary/transposed groups once for mapping, channel counting, and packing."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    w, b = one(bind(operation, "weight", 1)), bind(operation, "bias", 2)
    transposed = type(operation.module) in (
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ) or operation.node.target in (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
    groups = (
        operation.module.groups
        if operation.module is not None
        else operation.argument("groups", 6, 1)
    )
    if not isinstance(groups, int):
        raise UnsupportedOperation("Convolution groups must be static")
    channel = len(x.shape) - len(w.shape) + 1
    if channel not in (0, 1):
        raise UnsupportedOperation("Unexpected convolution rank")
    cin, cout = x.shape[channel], y.shape[channel]
    depthwise = groups == cin
    global_tensor, global_size = (x, cin) if transposed else (y, cout)
    local_tensor, local_size = (y, cout) if transposed else (x, cin)
    global_block, local_block = global_size // groups, local_size // groups
    relations = [equal(global_tensor, channel, w, 0, operation.node.name)]
    constraints, requirements, partitions = [], [], []
    if channel == 1:
        relations.append(equal(x, 0, y, 0, operation.node.name))
    if b is not None:
        relations.append(equal(y, channel, one(b), 0, operation.node.name))
    for group in range(groups):
        axes = list(full_region(w.shape).axes)
        axes[0] = IndexSet.span(group * global_block, (group + 1) * global_block)
        region = Region(tuple(axes))
        partitions.append(region)
        if not depthwise:
            relations.append(
                AxisRelation(
                    AxisPort(local_tensor.axis(channel)),
                    AxisPort(w.axis(1), region),
                    (BlockMap(group * local_block, 0, local_block),),
                    operation.node.name,
                )
            )
        elif transposed:
            # A retained input row owns its own local multiplier columns.
            relations.append(
                AxisRelation(
                    AxisPort(y.axis(channel)),
                    AxisPort(w.axis(1), region),
                    (BlockMap(group * local_block, 0, local_block),),
                    operation.node.name,
                )
            )
    if depthwise:
        relations.append(
            AxisRelation(
                AxisPort(x.axis(channel)),
                AxisPort(y.axis(channel)),
                (BlockMap(0, 0, cin, 1, cout // cin, require_full_target=True),),
                operation.node.name,
            )
        )
        constraints.append(
            BlockBalance(x.axis(channel), y.axis(channel), cout // cin, operation.node.name)
        )
        requirements.append(requirement(operation, "groups", x, channel, position=6))
    elif groups > 1:
        for tensor, width in ((x, cin // groups), (y, cout // groups)):
            constraints.append(
                Balanced(
                    tensor.axis(channel),
                    tuple(IndexSet.span(g * width, (g + 1) * width) for g in range(groups)),
                )
            )
    for tensor, dims in (
        (x, range(channel + 1, len(x.shape))),
        (y, range(channel + 1, len(y.shape))),
        (w, range(2, len(w.shape))),
    ):
        constraints.extend(
            AxisBarrier(
                tensor.axis(d),
                "Convolution spatial/kernel resizing is unsupported",
                operation.node.name,
            )
            for d in dims
        )
    if operation.module is not None:
        requirements.extend(
            (
                requirement(operation, "in_channels", x, channel),
                requirement(operation, "out_channels", y, channel),
            )
        )
    weight_ports = tuple(
        port
        for relation in relations
        if isinstance(relation, AxisRelation)
        for port in (relation.left, relation.right)
        if port.tensor == w
    )
    constraints.extend(
        (layout(operation, x), layout(operation, y), layout(operation, w, weight_ports))
    )
    # Ordinary kernels expose the logical output as a physical row axis.
    # Transposed groups require the activation axis, not local weight columns.
    axis = y.axis(channel) if transposed else w.axis(0)
    key = f"{w.paths[0]}:{'output' if transposed else '0'}" if w.paths else ""
    candidates = (
        ()
        if operation.module is None
        else (
            CandidateAxis(
                key,
                axis,
                cout // cin if depthwise else 1,
                w if transposed else None,
                alignment_axis=y.axis(channel),
            ),
        )
    )
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        tuple(requirements),
        candidates=candidates,
        layouts=(PartitionedLayout(w, tuple(partitions)),),
        contract=OutputContract(output_layout="backend_dependent"),
    )


def batch_norm(operation: Operation) -> OperatorSpec:
    """Connect channel positions to affine parameters and running statistics."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    relations = [ReshapeRelation(x, y, operation.node.name)]
    positions = {"running_mean": 1, "running_var": 2, "weight": 3, "bias": 4}
    for name, position in positions.items():
        value = bind(operation, name, position)
        if value is not None:
            relations.append(equal(x, 1, one(value), 0, operation.node.name))
    reqs = (requirement(operation, "num_features", x, 1),) if operation.module is not None else ()
    return OperatorSpec(
        tuple(relations),
        (*affine_layouts(operation), layout(operation, y)),
        reqs,
        contract=OutputContract(output_layout="backend_dependent"),
    )


def layer_norm(operation: Operation) -> OperatorSpec:
    """Connect normalized axes and record normalized-shape attribute changes."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    normalized = (
        operation.module.normalized_shape
        if operation.module is not None
        else operation.argument("normalized_shape", 1)
    )
    if isinstance(normalized, int):
        normalized = (normalized,)
    if not isinstance(normalized, (tuple, list)) or not all(isinstance(v, int) for v in normalized):
        raise UnsupportedOperation("normalized_shape must have a static provenance")
    rank = len(normalized)
    relations = [ReshapeRelation(x, y, operation.node.name)]
    positions = (
        (("weight", 2),)
        if type(operation.module) is nn.RMSNorm or operation.node.target is F.rms_norm
        else (("weight", 2), ("bias", 3))
    )
    for name, position in positions:
        value = bind(operation, name, position)
        if value is not None:
            value = one(value)
            relations.extend(
                equal(x, len(x.shape) - rank + d, value, d, operation.node.name)
                for d in range(rank)
            )
    axes = tuple(x.axis(len(x.shape) - rank + d) for d in range(rank))
    detail = "Update normalized_shape; compact normalization is not zero-mask equivalence"
    if operation.module is not None:
        binding = AttributeBinding(
            f"{operation.module_path}.normalized_shape".lstrip("."),
            ShapeExpr("tuple", args=tuple(ShapeExpr("dimension", (x, a.dim)) for a in axes)),
            detail,
        )
    else:
        binding = Requirement(
            "operator_argument",
            operation.node.name,
            (x,),
            detail,
            (("axes", axes),),
            arguments=(ArgumentRef("normalized_shape", 1),),
        )
    requirements = (binding,)
    return OperatorSpec(
        tuple(relations), (*affine_layouts(operation), layout(operation, y)), requirements
    )


def group_norm(operation: Operation) -> OperatorSpec:
    """Connect normalization channels and preserve fixed group balance."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    groups = (
        operation.module.num_groups
        if operation.module is not None
        else operation.argument("num_groups", 1)
    )
    if not isinstance(groups, int):
        raise UnsupportedOperation("GroupNorm groups must be static")
    relations = [ReshapeRelation(x, y, operation.node.name)]
    for name, position in (("weight", 2), ("bias", 3)):
        value = bind(operation, name, position)
        if value is not None:
            relations.append(equal(x, 1, one(value), 0, operation.node.name))
    width = x.shape[1] // groups
    constraints = (
        Balanced(
            x.axis(1), tuple(IndexSet.span(g * width, (g + 1) * width) for g in range(groups))
        ),
    )
    reqs = (requirement(operation, "num_channels", x, 1),) if operation.module is not None else ()
    return OperatorSpec(
        tuple(relations),
        (*constraints, *affine_layouts(operation), layout(operation, y)),
        reqs,
        contract=OutputContract(output_layout="backend_dependent"),
    )


def matmul(operation: Operation) -> OperatorSpec:
    """Connect contracting, free, and broadcast batch axes of matrix products."""
    a = one(operation.argument("input", 0))
    b = one(operation.argument("other", 1, operation.kwargs.get("mat2")))
    y = one(operation.result)
    ra, rb = len(a.shape), len(b.shape)
    if not ra or not rb:
        raise UnsupportedOperation("matmul requires tensor ranks >= 1")
    relations = [equal(a, -1, b, -2 if rb > 1 else 0, operation.node.name)]
    free = int(ra > 1) + int(rb > 1)
    if ra > 1:
        relations.append(equal(a, -2, y, -free, operation.node.name))
    if rb > 1:
        relations.append(equal(b, -1, y, -1, operation.node.name))
    batchrank = len(y.shape) - free
    for tensor in (a, b):
        batch = max(0, len(tensor.shape) - 2)
        for d in range(batch):
            target = batchrank - batch + d
            if tensor.shape[d] == y.shape[target]:
                relations.append(equal(tensor, d, y, target, operation.node.name))
            elif tensor.shape[d] != 1:
                raise UnsupportedOperation("Unproven matmul batch broadcast")
    return OperatorSpec(
        tuple(relations), (layout(operation, a), layout(operation, b), layout(operation, y))
    )


def permute(operation: Operation) -> OperatorSpec:
    """Normalize static transpose and permutation forms into one axis relation."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    target = operation.node.target
    rank = len(x.shape)
    if target in ("movedim", "moveaxis", torch.movedim, torch.moveaxis):
        if not rank:
            # PyTorch permits moving the virtual scalar dimension 0 (or -1).
            return OperatorSpec((PermuteRelation(x, y, (), operation.node.name),))
        source = operation.argument("source", 1)
        destination = operation.argument("destination", 2)
        source = (source,) if isinstance(source, int) else source
        destination = (destination,) if isinstance(destination, int) else destination
        if not isinstance(source, (tuple, list)) or not isinstance(destination, (tuple, list)):
            raise UnsupportedOperation("Moved dimensions must be static")
        source = tuple(d % rank for d in source)
        destination = tuple(d % rank for d in destination)
        dims = [d for d in range(rank) if d not in source]
        for dest, src in sorted(zip(destination, source, strict=True)):
            dims.insert(dest, src)
    elif target in (
        "transpose",
        torch.transpose,
        "swapaxes",
        torch.swapaxes,
        "swapdims",
        torch.swapdims,
    ):
        d0 = operation.argument("dim0", 1)
        d1 = operation.argument("dim1", 2)
        dims = list(range(rank))
        if rank:
            dims[d0 % rank], dims[d1 % rank] = dims[d1 % rank], dims[d0 % rank]
    elif target in ("t", torch.t):
        dims = list(reversed(range(rank)))
    else:
        dims = operation.argument("dims", 1, variadic=True)
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        if not isinstance(dims, (tuple, list)):
            raise UnsupportedOperation("Permutation must be static")
    dims = tuple(d % rank for d in dims)
    if sorted(dims) != list(range(rank)):
        raise UnsupportedOperation("Invalid dimension permutation")
    return OperatorSpec((PermuteRelation(x, y, dims, operation.node.name),))


def reshape(operation: Operation) -> OperatorSpec:
    """Map reshape coordinates and record dimension-source update requirements."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    if prod(x.shape) != prod(y.shape):
        raise UnsupportedOperation("Reshape must preserve the original element count")
    relation = ReshapeRelation(x, y, operation.node.name)
    if isinstance(operation.module, nn.Unflatten):
        dim = operation.module.dim
        if not isinstance(dim, int):
            raise UnsupportedOperation("Named unflatten dimensions are unsupported")
        axes = tuple(
            y.axis(dim % len(x.shape) + i) for i in range(len(operation.module.unflattened_size))
        )
        requirement_ = AttributeBinding(
            f"{operation.module_path}.unflattened_size".lstrip("."),
            ShapeExpr(
                "tuple",
                args=tuple(ShapeExpr("dimension", (axis.tensor, axis.dim)) for axis in axes),
            ),
            "Update explicitly bound unflatten factors from retained output dimensions",
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    target = operation.node.target
    if target in ("unflatten", torch.unflatten):
        dim = operation.argument("dim", 1)
        if not isinstance(dim, int):
            raise UnsupportedOperation("Named unflatten dimensions are unsupported")
        expression = shape_expression(operation, operation.raw_argument("sizes", 2))
        if not has_known_provenance(expression):
            raise UnsupportedOperation(
                "Cannot prove unflatten size argument provenance. Use tensor.size(dim), "
                "supported integer arithmetic, or -1, and rebuild the dependency graph."
            )
        # Reuse original-call evaluation: sizes may follow proven shape expressions,
        # while dim stays guarded. Unflatten only splits one axis, so unknown input
        # strides do not require the merge proof used by view.
        requirement_ = Requirement(
            "dimension_transform",
            operation.node.name,
            (x, y),
            "Check original unflatten sizes against retained output dimensions. "
            "Fixed factors need original-forward editing or an explicit attribute binding",
            (("input", x), ("output", y)),
            arguments=(ArgumentRef("sizes", 2),),
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    if target in ("reshape", "view", torch.reshape):
        raw = operation.raw_argument("shape", 1, variadic=True)
        expression = shape_expression(operation, raw)
        if not has_known_provenance(expression):
            raise UnsupportedOperation(
                "Cannot prove reshape size argument provenance. Express dimensions using "
                "tensor.size(dim), supported integer arithmetic, or -1 where appropriate, "
                "then rebuild the dependency graph. Keep algorithmic constants fixed."
            )
        requirement_ = Requirement(
            "shape_arguments",
            operation.node.name,
            (x, y),
            "Recompute shape arguments from retained dimensions; source Python attributes "
            "are not inferred from equal integers",
            (
                ("expression", expression),
                ("input", x),
                ("output", y),
                ("requires_view", target == "view"),
                ("unpack_shape", target is not torch.reshape),
            ),
            arguments=(ArgumentRef("shape", 1, variadic=True),),
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    if target in ("squeeze", torch.squeeze):
        requirement_ = Requirement(
            "dimension_transform",
            operation.node.name,
            (x, y),
            "Preserve captured output rank: newly singleton axes may require replacing squeeze "
            "with an explicit reshape of the retained output shape",
            (("input", x), ("output", y)),
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    # Flatten and unsqueeze derive their output dimensions from the compact input.
    return OperatorSpec((relation,))


def concatenate(operation: Operation) -> OperatorSpec:
    """Connect each input to its original offset region in the concatenation."""
    items = operation.argument("tensors", 0)
    y = one(operation.result)
    dim = operation.argument("dim", 1, 0) % len(y.shape)
    relations, offset = [], 0
    for x in items:
        x = one(x)
        index = [slice(None)] * len(y.shape)
        index[dim] = slice(offset, offset + x.shape[dim])
        relations.append(SliceRelation(y, x, tuple(index), operation.node.name))
        offset += x.shape[dim]
    return OperatorSpec(tuple(relations))


def split(operation: Operation) -> OperatorSpec:
    """Connect static split or unbind outputs to their source partitions."""
    x = one(operation.argument("input", 0))
    dim = operation.argument("dim", 2, 0) % len(x.shape)
    target = operation.node.target
    if target in ("unbind", torch.unbind):
        dim = operation.argument("dim", 1, 0) % len(x.shape)
        relations = []
        for i, y in enumerate(operation.outputs):
            index = [slice(None)] * len(x.shape)
            index[dim] = i
            relations.append(SliceRelation(x, y, tuple(index), operation.node.name))
        # Removing an unbound position changes the number and identity of output ports.
        constraints = (
            AxisBarrier(
                x.axis(dim),
                "Deleting unbind output ports needs an explicit graph rewrite",
                operation.node.name,
            ),
        )
    else:
        sections = operation.argument("split_size_or_sections", 1)
        if not isinstance(sections, (int, tuple, list)):
            raise UnsupportedOperation("Split sections must be static")
        relations, offset, constraints = [], 0, ()
        for y in operation.outputs:
            index = [slice(None)] * len(x.shape)
            index[dim] = slice(offset, offset + y.shape[dim])
            relations.append(SliceRelation(x, y, tuple(index), operation.node.name))
            offset += y.shape[dim]
    requirement_ = Requirement(
        "partition_arguments",
        operation.node.name,
        (x, *operation.outputs),
        "Update split sizes/port bindings while preserving original partition order",
        (
            ("axis", x.axis(dim)),
            ("outputs", operation.outputs),
            ("unbound", target in ("unbind", torch.unbind)),
        ),
        arguments=()
        if target in ("unbind", torch.unbind)
        else (ArgumentRef("split_size_or_sections", 1),),
    )
    return OperatorSpec(tuple(relations), tuple(constraints), (requirement_,))


def getitem(operation: Operation) -> OperatorSpec:
    """Normalize container access and supported basic tensor indexing."""
    scalar = scalar_spec(operation)
    if scalar is not None:
        return scalar
    source, index = operation.args
    if not isinstance(source, TensorRef):
        selected = source[index]
        a, b = tuple(tensors(selected)), operation.outputs
        if len(a) != len(b):
            raise UnsupportedOperation("Container indexing mismatch")
        return OperatorSpec(
            tuple(ReshapeRelation(x, y, operation.node.name) for x, y in zip(a, b, strict=False))
        )
    if not isinstance(index, tuple):
        index = (index,)
    if any(item is None for item in index):
        raise UnsupportedOperation(
            "New-axis indexing is unsupported; use unsqueeze(dim) to insert singleton "
            "dimensions separately from indexing, then rebuild the dependency graph. "
            "Other indexing restrictions still apply."
        )
    if sum(item is Ellipsis for item in index) > 1:
        raise UnsupportedOperation("Multiple ellipses")
    expanded = []
    for item in index:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(source.shape) - len(index) + 1))
        else:
            expanded.append(item)
    expanded.extend([slice(None)] * (len(source.shape) - len(expanded)))
    if len(expanded) != len(source.shape):
        raise UnsupportedOperation("Advanced indexing has no supported coordinate rule")
    normalized = []
    for size, item in zip(source.shape, expanded, strict=False):
        if isinstance(item, bool):
            raise UnsupportedOperation("Boolean indexing is data-dependent")
        if isinstance(item, int):
            normalized.append(item % size)
        elif isinstance(item, slice) and all(
            v is None or isinstance(v, int) for v in (item.start, item.stop, item.step)
        ):
            start, stop, step = item.indices(size)
            if step <= 0:
                raise UnsupportedOperation("Negative slice steps are unsupported")
            normalized.append(slice(start, max(start, stop), step))
        else:
            raise UnsupportedOperation("Only static basic tensor indexing is supported")
    y = one(operation.result)
    req = Requirement(
        "slice_arguments",
        operation.node.name,
        (source, y),
        "Remap slice coordinates after upstream compaction",
        (("index", tuple(normalized)),),
        arguments=(ArgumentRef("index", 1),),
    )
    return OperatorSpec(
        (SliceRelation(source, y, tuple(normalized), operation.node.name),), requirements=(req,)
    )


def reduction(operation: Operation) -> OperatorSpec:
    """Connect retained axes while recording changes to the reduction domain."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    dims = operation.argument("dim", 1)
    if dims is None or (isinstance(dims, (tuple, list)) and not dims):
        dims = tuple(range(len(x.shape)))
    elif isinstance(dims, int):
        dims = (dims,)
    if not isinstance(dims, (tuple, list)) or not all(isinstance(d, int) for d in dims):
        raise UnsupportedOperation("Reduction axes must be static")
    dims = tuple(d % len(x.shape) for d in dims) if x.shape else ()
    keep = operation.argument("keepdim", 2, False)
    relations, outdim = [], 0
    for dim in range(len(x.shape)):
        if dim not in dims:
            relations.append(equal(x, dim, y, outdim, operation.node.name))
            outdim += 1
        elif keep:
            outdim += 1
    constraints = tuple(NonEmpty(x.axis(d)) for d in dims)
    req = Requirement(
        "reduction_domain",
        operation.node.name,
        (x,),
        "Recompute the reduction on retained positions; zero masking may differ",
        (("axes", dims),),
    )
    return OperatorSpec(tuple(relations), constraints, (req,))


def getattr_rule(operation: Operation) -> OperatorSpec:
    """Handle supported tensor attributes and the transpose property."""
    scalar = scalar_spec(operation)
    if scalar is not None:
        return scalar
    name = operation.args[1]
    if name == "T":
        x, y = one(operation.argument("input", 0)), one(operation.result)
        return OperatorSpec(
            (PermuteRelation(x, y, tuple(reversed(range(len(x.shape)))), operation.node.name),)
        )
    if name in ("shape", "ndim", "dtype", "device"):
        return OperatorSpec()
    raise UnsupportedOperation(f"Tensor attribute {name} has no structural rule")


def shape_only(operation: Operation) -> OperatorSpec:
    """Leave supported dimension-only operations without tensor relationships."""
    scalar = scalar_spec(operation)
    if scalar is not None:
        return scalar
    return OperatorSpec()


def softmax(operation: Operation) -> OperatorSpec:
    """Preserve coordinates and record changes to the normalization domain."""
    result = pointwise(operation)
    x = one(operation.argument("input", 0))
    dim = operation.module.dim if operation.module is not None else operation.argument("dim", 1)
    if dim is None:
        # PyTorch's legacy default depends only on rank. Compaction preserves
        # rank, so the same axis remains valid. Native execution emits its warning.
        dim = 0 if len(x.shape) in (0, 1, 3) else 1
    if not isinstance(dim, int):
        raise UnsupportedOperation("Softmax requires a static reduction dimension")
    requirement_ = Requirement(
        "reduction_domain",
        operation.node.name,
        (x,),
        "Recompute softmax/log_softmax over retained positions; zero masks are not equivalent",
        (("axes", (dim % len(x.shape),) if x.shape else ()),),
    )
    return OperatorSpec(result.relations, result.constraints, (requirement_,))
