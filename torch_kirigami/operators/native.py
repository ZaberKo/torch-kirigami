"""Structural semantics expressed through the same interface as external rules."""

from __future__ import annotations

import builtins
import operator
from dataclasses import replace
from math import prod

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import (
    AxisBarrier,
    Balanced,
    BlockBalance,
    Layout,
    NonEmpty,
    Requirement,
    ShapeExpr,
)
from ..registry import CandidateAxis, OperatorRule, OperatorSpec, tensors
from ..relations import (
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    Port,
    ReshapeRelation,
    SliceRelation,
)
from ..selection import IndexSet, Region, TensorRef, full_region
from .effects import native_effects
from .layouts import CallContract, PartitionedLayout
from .shapes import expression_for


class UnsupportedOperation(Exception):
    """A captured operation has no proven rule for these arguments."""


def one(value):
    """Require one tensor reference instead of a container or scalar argument."""
    if not isinstance(value, TensorRef):
        raise UnsupportedOperation("Expected a single tensor")
    return value


def equal(a, da, b, db, reason):
    """Build an identity relation between two specified tensor axes."""
    return AxisRelation.equal(a.axis(da), b.axis(db), reason)


def requirement(ctx, name, tensor, dim, *, kind="attribute", position=None):
    """Bind an axis change to a module attribute or functional operator argument."""
    target = f"{ctx.module_path}.{name}".lstrip(".") if ctx.module is not None else ctx.node.name
    return Requirement(
        kind if ctx.module is not None else "operator_argument",
        target,
        (tensor,),
        f"Update {name} from the retained axis size",
        (("name", name), ("axis", tensor.axis(dim)), ("position", position)),
    )


def bind(ctx, name, position, default=None):
    """Resolve a module-local binding or a normalized functional argument."""
    return ctx.parameter(name) if ctx.module is not None else ctx.argument(name, position, default)


def _identity(ctx):
    """Preserve tensor coordinates for a shape-preserving operation."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    if x.shape != y.shape:
        raise UnsupportedOperation("Identity rule requires equal shapes")
    return OperatorSpec((ReshapeRelation(x, y, ctx.node.name),))


def pointwise(ctx):
    """Connect broadcast-compatible inputs to an elementwise output."""
    y = one(ctx.output)
    relations = []
    for x in ctx.inputs:
        try:
            relations.append(BroadcastRelation(x, y, ctx.node.name))
        except ValueError as error:
            raise UnsupportedOperation("Unproven broadcast correspondence") from error
    return OperatorSpec(tuple(relations))


def layout(ctx, tensor, ports=()):
    """Declare the compact layouts accepted by a particular tensor use."""
    return Layout(tensor, tuple(ports), node=ctx.node.name)


def affine_layouts(ctx):
    """Require ordinary axis compaction for each input and bound affine tensor."""
    return tuple(layout(ctx, ref) for ref in dict.fromkeys((*ctx.inputs, *ctx.bindings.values())))


def linear(ctx):
    """Connect Linear feature axes, batch dimensions, weight, and bias."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    w, b = one(bind(ctx, "weight", 1)), bind(ctx, "bias", 2)
    relations = [equal(x, -1, w, 1, ctx.node.name), equal(y, -1, w, 0, ctx.node.name)]
    relations.extend(equal(x, d, y, d, ctx.node.name) for d in range(len(x.shape) - 1))
    if b is not None:
        relations.append(equal(y, -1, one(b), 0, ctx.node.name))
    reqs = (requirement(ctx, "in_features", x, -1), requirement(ctx, "out_features", y, -1))
    if ctx.module is None:
        reqs = ()
    candidates = () if ctx.module is None else (CandidateAxis(f"{w.paths[0]}:0", w.axis(0)),)
    return OperatorSpec(
        tuple(relations),
        (layout(ctx, x), layout(ctx, y), layout(ctx, w)),
        reqs,
        candidates=candidates,
        contract=CallContract(fresh_output=True, output_layout="contiguous"),
    )


def convolution(ctx):
    """Describe ordinary/transposed groups once for mapping, budgets, and packing."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    w, b = one(bind(ctx, "weight", 1)), bind(ctx, "bias", 2)
    transposed = type(ctx.module) in (
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ) or ctx.node.target in (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
    groups = ctx.module.groups if ctx.module is not None else ctx.argument("groups", 6, 1)
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
    relations = [equal(global_tensor, channel, w, 0, ctx.node.name)]
    constraints, requirements, partitions = [], [], []
    if channel == 1:
        relations.append(equal(x, 0, y, 0, ctx.node.name))
    if b is not None:
        relations.append(equal(y, channel, one(b), 0, ctx.node.name))
    for group in range(groups):
        axes = list(full_region(w.shape).axes)
        axes[0] = IndexSet.span(group * global_block, (group + 1) * global_block)
        region = Region(tuple(axes))
        partitions.append(region)
        if not depthwise:
            relations.append(
                AxisRelation(
                    Port(local_tensor.axis(channel)),
                    Port(w.axis(1), region),
                    (BlockMap(group * local_block, 0, local_block),),
                    ctx.node.name,
                )
            )
        elif transposed:
            # A retained input row owns its own local multiplier columns.
            relations.append(
                AxisRelation(
                    Port(y.axis(channel)),
                    Port(w.axis(1), region),
                    (BlockMap(group * local_block, 0, local_block),),
                    ctx.node.name,
                )
            )
    if depthwise:
        relations.append(
            AxisRelation(
                Port(x.axis(channel)),
                Port(y.axis(channel)),
                (BlockMap(0, 0, cin, 1, cout // cin, require_full_target=True),),
                ctx.node.name,
            )
        )
        constraints.append(
            BlockBalance(x.axis(channel), y.axis(channel), cout // cin, ctx.node.name)
        )
        requirements.append(requirement(ctx, "groups", x, channel, position=6))
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
                tensor.axis(d), "Convolution spatial/kernel resizing is unsupported", ctx.node.name
            )
            for d in dims
        )
    if ctx.module is not None:
        requirements.extend(
            (
                requirement(ctx, "in_channels", x, channel),
                requirement(ctx, "out_channels", y, channel),
            )
        )
    weight_ports = tuple(
        port
        for relation in relations
        if isinstance(relation, AxisRelation)
        for port in (relation.left, relation.right)
        if port.tensor == w
    )
    constraints.extend((layout(ctx, x), layout(ctx, y), layout(ctx, w, weight_ports)))
    # Ordinary kernels expose the logical output as a physical row axis.
    # Transposed groups require the activation axis, not local weight columns.
    axis = y.axis(channel) if transposed else w.axis(0)
    key = f"{w.paths[0]}:{'output' if transposed else '0'}" if w.paths else ""
    candidates = (
        () if ctx.module is None else (CandidateAxis(key, axis, cout // cin if depthwise else 1),)
    )
    return OperatorSpec(
        tuple(relations),
        tuple(constraints),
        tuple(requirements),
        candidates=candidates,
        layouts=(PartitionedLayout(w, tuple(partitions)),),
        contract=CallContract(fresh_output=True, output_layout="convolution"),
    )


def batch_norm(ctx):
    """Connect channel positions to affine parameters and running statistics."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    relations = [ReshapeRelation(x, y, ctx.node.name)]
    positions = {"running_mean": 1, "running_var": 2, "weight": 3, "bias": 4}
    for name, position in positions.items():
        value = bind(ctx, name, position)
        if value is not None:
            relations.append(equal(x, 1, one(value), 0, ctx.node.name))
    reqs = (requirement(ctx, "num_features", x, 1),) if ctx.module is not None else ()
    return OperatorSpec(tuple(relations), (*affine_layouts(ctx), layout(ctx, y)), reqs)


def layer_norm(ctx):
    """Connect normalized axes and record normalized-shape attribute changes."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    normalized = (
        ctx.module.normalized_shape
        if ctx.module is not None
        else ctx.argument("normalized_shape", 1)
    )
    if isinstance(normalized, int):
        normalized = (normalized,)
    if not isinstance(normalized, (tuple, list)) or not all(isinstance(v, int) for v in normalized):
        raise UnsupportedOperation("normalized_shape must have a static provenance")
    rank = len(normalized)
    relations = [ReshapeRelation(x, y, ctx.node.name)]
    positions = (
        (("weight", 2),)
        if type(ctx.module) is nn.RMSNorm or ctx.node.target is F.rms_norm
        else (("weight", 2), ("bias", 3))
    )
    for name, position in positions:
        value = bind(ctx, name, position)
        if value is not None:
            value = one(value)
            relations.extend(
                equal(x, len(x.shape) - rank + d, value, d, ctx.node.name) for d in range(rank)
            )
    requirements = (
        Requirement(
            "attribute" if ctx.module is not None else "operator_argument",
            f"{ctx.module_path}.normalized_shape".lstrip(".")
            if ctx.module is not None
            else ctx.node.name,
            (x,),
            "Update normalized_shape; compact normalization is not zero-mask equivalence",
            (
                ("axes", tuple(x.axis(len(x.shape) - rank + d) for d in range(rank))),
                ("position", 1),
            ),
        ),
    )
    return OperatorSpec(tuple(relations), (*affine_layouts(ctx), layout(ctx, y)), requirements)


def group_norm(ctx):
    """Connect normalization channels and preserve fixed group balance."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    groups = ctx.module.num_groups if ctx.module is not None else ctx.argument("num_groups", 1)
    if not isinstance(groups, int):
        raise UnsupportedOperation("GroupNorm groups must be static")
    relations = [ReshapeRelation(x, y, ctx.node.name)]
    for name, position in (("weight", 2), ("bias", 3)):
        value = bind(ctx, name, position)
        if value is not None:
            relations.append(equal(x, 1, one(value), 0, ctx.node.name))
    width = x.shape[1] // groups
    constraints = (
        Balanced(
            x.axis(1), tuple(IndexSet.span(g * width, (g + 1) * width) for g in range(groups))
        ),
    )
    reqs = (requirement(ctx, "num_channels", x, 1),) if ctx.module is not None else ()
    return OperatorSpec(
        tuple(relations), (*constraints, *affine_layouts(ctx), layout(ctx, y)), reqs
    )


def matmul(ctx):
    """Connect contracting, free, and broadcast batch axes of matrix products."""
    a = one(ctx.argument("input", 0))
    b = one(ctx.argument("other", 1, ctx.kwargs.get("mat2")))
    y = one(ctx.output)
    ra, rb = len(a.shape), len(b.shape)
    if not ra or not rb:
        raise UnsupportedOperation("matmul requires tensor ranks >= 1")
    relations = [equal(a, -1, b, -2 if rb > 1 else 0, ctx.node.name)]
    free = int(ra > 1) + int(rb > 1)
    if ra > 1:
        relations.append(equal(a, -2, y, -free, ctx.node.name))
    if rb > 1:
        relations.append(equal(b, -1, y, -1, ctx.node.name))
    batchrank = len(y.shape) - free
    for tensor in (a, b):
        batch = max(0, len(tensor.shape) - 2)
        for d in range(batch):
            target = batchrank - batch + d
            if tensor.shape[d] == y.shape[target]:
                relations.append(equal(tensor, d, y, target, ctx.node.name))
            elif tensor.shape[d] != 1:
                raise UnsupportedOperation("Unproven matmul batch broadcast")
    return OperatorSpec(tuple(relations), (layout(ctx, a), layout(ctx, b), layout(ctx, y)))


def permute(ctx):
    """Normalize static transpose and permutation forms into one axis relation."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    target = ctx.node.target
    rank = len(x.shape)
    if target in (
        "transpose",
        torch.transpose,
        "swapaxes",
        torch.swapaxes,
        "swapdims",
        torch.swapdims,
    ):
        d0 = ctx.argument("dim0", 1)
        d1 = ctx.argument("dim1", 2)
        dims = list(range(rank))
        dims[d0 % rank], dims[d1 % rank] = dims[d1 % rank], dims[d0 % rank]
    elif target in ("t", torch.t):
        dims = list(reversed(range(rank)))
    else:
        dims = ctx.argument("dims", 1)
        if isinstance(dims, int):
            dims = ctx.args[1:]
        if not isinstance(dims, (tuple, list)):
            raise UnsupportedOperation("Permutation must be static")
    dims = tuple(d % rank for d in dims)
    if sorted(dims) != list(range(rank)):
        raise UnsupportedOperation("Invalid dimension permutation")
    return OperatorSpec((PermuteRelation(x, y, dims, ctx.node.name),))


def _expr(ctx, value):
    """Resolve dimension provenance without guessing from numeric equality."""
    from torch import fx

    if isinstance(value, fx.Node):
        return ctx.expressions.get(value, ShapeExpr("unknown", value.name))
    if isinstance(value, (tuple, list)):
        return ShapeExpr("tuple", args=tuple(_expr(ctx, v) for v in value))
    if isinstance(value, int):
        return ShapeExpr("infer" if value == -1 else "constant", value)
    return ShapeExpr("unknown", repr(value))


def _known(expr):
    """Check whether every leaf of a shape expression has known provenance."""
    return expr.kind != "unknown" and all(_known(arg) for arg in expr.args)


def reshape(ctx):
    """Map reshape coordinates and record dimension-source update requirements."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    if prod(x.shape) != prod(y.shape):
        raise UnsupportedOperation("Reshape must preserve the original element count")
    relation = ReshapeRelation(x, y, ctx.node.name)
    if isinstance(ctx.module, nn.Unflatten):
        dim = ctx.module.dim
        if not isinstance(dim, int):
            raise UnsupportedOperation("Named unflatten dimensions are unsupported")
        axes = tuple(
            y.axis(dim % len(x.shape) + i) for i in range(len(ctx.module.unflattened_size))
        )
        requirement_ = Requirement(
            "attribute",
            f"{ctx.module_path}.unflattened_size".lstrip("."),
            (x, y),
            "Update explicitly bound unflatten factors from retained output dimensions",
            (("axes", axes),),
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    target = ctx.node.target
    if target in ("reshape", "view", torch.reshape):
        raw = ctx.node.kwargs.get("shape", ctx.node.args[1:])
        expression = _expr(ctx, raw)
        if not _known(expression):
            raise UnsupportedOperation("Cannot prove reshape size argument provenance")
        requirement_ = Requirement(
            "shape_arguments",
            ctx.node.name,
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
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    if target in ("squeeze", torch.squeeze):
        requirement_ = Requirement(
            "dimension_transform",
            ctx.node.name,
            (x, y),
            "Preserve captured output rank: newly singleton axes may require replacing squeeze "
            "with an explicit reshape of the retained output shape",
            (("input", x), ("output", y)),
        )
        return OperatorSpec((relation,), requirements=(requirement_,))
    # Flatten and unsqueeze derive their output dimensions from the compact input.
    return OperatorSpec((relation,))


def concatenate(ctx):
    """Connect each input to its original offset region in the concatenation."""
    items = ctx.argument("tensors", 0)
    y = one(ctx.output)
    dim = ctx.argument("dim", 1, 0) % len(y.shape)
    relations, offset = [], 0
    for x in items:
        x = one(x)
        index = [slice(None)] * len(y.shape)
        index[dim] = slice(offset, offset + x.shape[dim])
        relations.append(SliceRelation(y, x, tuple(index), ctx.node.name))
        offset += x.shape[dim]
    return OperatorSpec(tuple(relations))


def split(ctx):
    """Connect static split or unbind outputs to their source partitions."""
    x = one(ctx.argument("input", 0))
    dim = ctx.argument("dim", 2, 0) % len(x.shape)
    target = ctx.node.target
    if target in ("unbind", torch.unbind):
        dim = ctx.argument("dim", 1, 0) % len(x.shape)
        relations = []
        for i, y in enumerate(ctx.outputs):
            index = [slice(None)] * len(x.shape)
            index[dim] = i
            relations.append(SliceRelation(x, y, tuple(index), ctx.node.name))
        # Removing an unbound position changes the number and identity of output ports.
        constraints = (
            AxisBarrier(
                x.axis(dim),
                "Deleting unbind output ports needs an explicit graph rewrite",
                ctx.node.name,
            ),
        )
    else:
        sections = ctx.argument("split_size_or_sections", 1)
        if not isinstance(sections, (int, tuple, list)):
            raise UnsupportedOperation("Split sections must be static")
        relations, offset, constraints = [], 0, ()
        for y in ctx.outputs:
            index = [slice(None)] * len(x.shape)
            index[dim] = slice(offset, offset + y.shape[dim])
            relations.append(SliceRelation(x, y, tuple(index), ctx.node.name))
            offset += y.shape[dim]
    requirement_ = Requirement(
        "partition_arguments",
        ctx.node.name,
        (x, *ctx.outputs),
        "Update split sizes/port bindings while preserving original partition order",
        (
            ("axis", x.axis(dim)),
            ("outputs", ctx.outputs),
            ("unbound", target in ("unbind", torch.unbind)),
            ("argument", ("split_size_or_sections", 1)),
        ),
    )
    return OperatorSpec(tuple(relations), tuple(constraints), (requirement_,))


def getitem(ctx):
    """Normalize container access and supported basic tensor indexing."""
    source, index = ctx.args
    if not isinstance(source, TensorRef):
        selected = source[index]
        a, b = tuple(tensors(selected)), ctx.outputs
        if len(a) != len(b):
            raise UnsupportedOperation("Container indexing mismatch")
        return OperatorSpec(
            tuple(ReshapeRelation(x, y, ctx.node.name) for x, y in zip(a, b, strict=False))
        )
    if not isinstance(index, tuple):
        index = (index,)
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
        raise UnsupportedOperation("New-axis/advanced indexing is unsupported; use unsqueeze")
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
    y = one(ctx.output)
    req = Requirement(
        "slice_arguments",
        ctx.node.name,
        (source, y),
        "Remap slice coordinates after upstream compaction",
        (("index", tuple(normalized)),),
    )
    return OperatorSpec(
        (SliceRelation(source, y, tuple(normalized), ctx.node.name),), requirements=(req,)
    )


def reduction(ctx):
    """Connect retained axes while recording changes to the reduction domain."""
    x, y = one(ctx.argument("input", 0)), one(ctx.output)
    dims = ctx.argument("dim", 1)
    if dims is None:
        dims = tuple(range(len(x.shape)))
    elif isinstance(dims, int):
        dims = (dims,)
    if not isinstance(dims, (tuple, list)) or not all(isinstance(d, int) for d in dims):
        raise UnsupportedOperation("Reduction axes must be static")
    dims = tuple(d % len(x.shape) for d in dims)
    keep = ctx.argument("keepdim", 2, False)
    relations, outdim = [], 0
    for dim in range(len(x.shape)):
        if dim not in dims:
            relations.append(equal(x, dim, y, outdim, ctx.node.name))
            outdim += 1
        elif keep:
            outdim += 1
    constraints = tuple(NonEmpty(x.axis(d)) for d in dims)
    req = Requirement(
        "reduction_domain",
        ctx.node.name,
        (x,),
        "Recompute the reduction on retained positions; zero masking may differ",
        (("axes", dims),),
    )
    return OperatorSpec(tuple(relations), constraints, (req,))


def getattr_rule(ctx):
    """Handle supported tensor attributes and the transpose property."""
    name = ctx.args[1]
    if name == "T":
        x, y = one(ctx.argument("input", 0)), one(ctx.output)
        return OperatorSpec(
            (PermuteRelation(x, y, tuple(reversed(range(len(x.shape)))), ctx.node.name),)
        )
    if name in ("shape", "ndim", "dtype", "device"):
        return OperatorSpec()
    raise UnsupportedOperation(f"Tensor attribute {name} has no structural rule")


def shape_only(ctx):
    """Leave supported dimension-only operations without tensor relationships."""
    return OperatorSpec()


def softmax(ctx):
    """Preserve coordinates and record changes to the normalization domain."""
    result = pointwise(ctx)
    x = one(ctx.argument("input", 0))
    dim = ctx.module.dim if ctx.module is not None else ctx.argument("dim", 1)
    if not isinstance(dim, int):
        raise UnsupportedOperation("Softmax requires an explicit static reduction dimension")
    requirement_ = Requirement(
        "reduction_domain",
        ctx.node.name,
        (x,),
        "Recompute softmax/log_softmax over retained positions; zero masks are not equivalent",
        (("axes", (dim % len(x.shape),)),),
    )
    return OperatorSpec(result.relations, result.constraints, (requirement_,))


def register_defaults(registry):
    """Populate a local registry with exact built-in operation matches."""

    def native(rule):
        def analyze(ctx):
            expression = expression_for(ctx)
            if expression is not None and not ctx.outputs:
                return OperatorSpec(expression=expression)
            result = rule(ctx)
            if result.contract is None:
                fresh = native_effects(ctx.node, ctx.module).fresh_output
                dense = ctx.node.target == "contiguous"
                result = replace(
                    result,
                    contract=CallContract(
                        fresh_output=fresh,
                        output_layout="contiguous" if dense else "unknown",
                    ),
                )
            return result

        return OperatorRule(analyze, evaluate=True, effects=native_effects)

    def modules(types, rule):
        for target in types:
            registry.register(target, native(rule))

    def functions(targets, rule):
        for target in targets:
            registry.register(target, native(rule), opaque=False)

    def methods(names, rule):
        for name in names:
            registry.register_method(name, native(rule))

    modules([nn.Softmax, nn.LogSoftmax], softmax)
    functions([F.softmax, F.log_softmax, torch.softmax, torch.log_softmax], softmax)
    methods(["softmax", "log_softmax"], softmax)
    modules([nn.Linear], linear)
    functions([F.linear], linear)
    modules(
        [
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ],
        convolution,
    )
    functions(
        [F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d],
        convolution,
    )
    modules([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d], batch_norm)
    functions([F.batch_norm], batch_norm)
    modules([nn.LayerNorm], layer_norm)
    functions([F.layer_norm], layer_norm)
    modules([nn.GroupNorm], group_norm)
    functions([F.group_norm], group_norm)
    modules([nn.Identity], _identity)
    modules([nn.Flatten, nn.Unflatten], reshape)
    modules(
        [
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.Sigmoid,
            nn.Tanh,
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
        ],
        pointwise,
    )
    functions(
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.neg,
            torch.add,
            torch.sub,
            torch.mul,
            torch.div,
            torch.neg,
            torch.relu,
            torch.sigmoid,
            torch.tanh,
            torch.clone,
            F.relu,
            F.relu6,
            F.gelu,
            F.silu,
            F.dropout,
        ],
        pointwise,
    )
    methods(
        [
            "add",
            "sub",
            "mul",
            "div",
            "neg",
            "relu",
            "sigmoid",
            "tanh",
            "clone",
            "detach",
            "contiguous",
            "relu_",
            "add_",
            "mul_",
        ],
        pointwise,
    )
    functions([operator.matmul, torch.matmul, torch.mm, torch.bmm], matmul)
    methods(["matmul", "mm", "bmm"], matmul)
    functions([torch.permute, torch.transpose, torch.swapaxes, torch.swapdims, torch.t], permute)
    methods(["permute", "transpose", "swapaxes", "swapdims", "t"], permute)
    functions([torch.reshape, torch.flatten, torch.squeeze, torch.unsqueeze], reshape)
    methods(["reshape", "view", "flatten", "squeeze", "unsqueeze"], reshape)
    functions([torch.cat, torch.concat, torch.concatenate], concatenate)
    functions([torch.split, torch.unbind], split)
    methods(["split", "unbind"], split)
    functions([operator.getitem], getitem)
    # The wrapper extracts proven scalar size expressions first. Tensor overloads
    # must retain pointwise dependencies instead of silently emitting an empty rule.
    functions([operator.floordiv, operator.mod, torch.floor_divide, torch.remainder], pointwise)
    methods(["floor_divide", "remainder"], pointwise)
    functions([builtins.getattr], getattr_rule)
    functions([torch.sum, torch.mean], reduction)
    methods(["sum", "mean"], reduction)
    methods(["size", "dim", "numel"], shape_only)
    from .extended import register_extended

    register_extended(registry, modules, functions, methods)
    from .indexing import register_indexing

    register_indexing(modules, functions, methods)
    from .attention import register_attention

    register_attention(modules, functions, methods)
