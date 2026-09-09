"""Concrete native calls and independently specified axis expectations.

Every registry entry has an explicit case family. The registry is only the
inventory; it never supplies expected axes or removal coordinates.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NativeCase:
    args: tuple
    seed: torch.Tensor
    axis: int
    output_axes: tuple[int | None, ...] = (1,)
    removed: tuple[int, ...] = (1,)
    output_removed: tuple[tuple[int, ...], ...] = ((1,),)
    kwargs: dict = field(default_factory=dict)
    module: nn.Module | None = None
    companions: tuple[tuple[torch.Tensor, int, tuple[int, ...]], ...] = ()
    expression: bool = False


def sample(shape, *, integer=False):
    count = 1
    for size in shape:
        count *= size
    values = torch.arange(count).reshape(shape)
    return values % 3 if integer else (values.to(torch.float64) % 17 + 1) / 20


UNARY = {
    "neg",
    "relu",
    "sigmoid",
    "tanh",
    "clone",
    "detach",
    "contiguous",
    "relu_",
    "sin",
    "cos",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "abs",
    "square",
    "reciprocal",
    "floor",
    "ceil",
    "round",
    "expm1",
    "log1p",
    "erf",
    "sign",
    "sinh",
    "cosh",
    "pos",
}
BINARY = {
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "logical_and",
    "logical_or",
    "logical_xor",
    "add",
    "sub",
    "mul",
    "div",
    "truediv",
    "add_",
    "mul_",
    "floor_divide",
    "floordiv",
    "remainder",
    "mod",
    "pow",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and_",
    "or_",
    "xor",
}
ACTIVATIONS = {
    "relu",
    "relu6",
    "gelu",
    "silu",
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "elu",
    "celu",
    "selu",
    "leaky_relu",
    "mish",
    "hardswish",
    "hardsigmoid",
    "hardtanh",
    "softplus",
    "softsign",
    "tanhshrink",
    "threshold",
}
REDUCTIONS = {"sum", "mean", "amax", "amin", "prod", "logsumexp"}
CASTS = {
    "to",
    "type_as",
    "float",
    "double",
    "half",
    "bfloat16",
    "bool",
    "long",
    "int",
    "cpu",
    "cuda",
}


def module_case(target):
    name = target.__name__
    lowered = name.lower()
    spatial = next((n for n in (1, 2, 3) if f"{n}d" in lowered), 2)
    if name == "Linear":
        x, module = sample((2, 6)), target(6, 4).double()
        return NativeCase(
            (x,), x, 1, (None,), module=module, companions=((module.weight, 1, (1,)),)
        )
    if name.startswith("Conv"):
        x = sample((2, 6, *((4,) * spatial)))
        module = target(6, 4, 1).double()
        weight_axis = 0 if "Transpose" in name else 1
        return NativeCase(
            (x,), x, 1, (None,), module=module, companions=((module.weight, weight_axis, (1,)),)
        )
    if name.startswith(("BatchNorm", "InstanceNorm")):
        x = sample((2, 6, *((3,) * spatial)))
        options = {} if name.startswith("Batch") else {"affine": True, "track_running_stats": True}
        module = target(6, **options).double().eval()
        return NativeCase(
            (x,),
            x,
            1,
            module=module,
            companions=((module.weight, 0, (1,)), (module.running_mean, 0, (1,))),
        )
    if name in {"LayerNorm", "RMSNorm", "GroupNorm", "PReLU"}:
        x = sample((2, 6))
        module = target(2, 6) if name == "GroupNorm" else target(6)
        removed = (1, 4) if name == "GroupNorm" else (1,)
        module = module.double()
        return NativeCase(
            (x,),
            x,
            1,
            removed=removed,
            output_removed=(removed,),
            module=module,
            companions=((module.weight, 0, removed),),
        )
    if name == "Embedding":
        x, module = sample((2, 3), integer=True), target(5, 6).double()
        return NativeCase((x,), module.weight, 1, (2,), module=module)
    if "pool" in lowered or name == "Upsample":
        x = sample((2, 6, *((4,) * spatial)))
        if "Unpool" in name:
            pooled, indices = getattr(F, f"max_pool{spatial}d")(x, 2, return_indices=True)
            return NativeCase(
                (pooled, indices), pooled, 1, module=target(2), companions=((indices, 1, (1,)),)
            )
        module = target(scale_factor=2) if name == "Upsample" else target(2)
        return NativeCase((x,), x, 1, module=module)
    if "Pad" in name:
        x = sample((2, 6, *((3,) * spatial)))
        module = target(1, 0.25) if name.startswith("Constant") else target(1)
        return NativeCase((x,), x, 1, module=module)
    if name in {"Softmax", "LogSoftmax"}:
        x = sample((2, 6, 3))
        return NativeCase((x,), x, 1, module=target(dim=2))
    if name == "Flatten":
        x = sample((2, 3, 2))
        return NativeCase((x,), x, 1, output_removed=((2, 3),), module=target(1))
    if name == "Unflatten":
        x = sample((2, 6))
        return NativeCase((x,), x, 1, (2,), (1, 4), ((1,),), module=target(1, (2, 3)))
    if name == "GLU":
        x = sample((2, 6, 3))
        return NativeCase((x,), x, 1, module=target(dim=1))
    if name == "ChannelShuffle":
        x = sample((2, 6, 3))
        return NativeCase((x,), x, 1, output_removed=((2, 3),), module=target(2))
    if name in {"PixelShuffle", "PixelUnshuffle"}:
        inverse = name == "PixelUnshuffle"
        x = sample((2, 3 if inverse else 12, 4, 4))
        removed = (1,) if inverse else (4, 5, 6, 7)
        output = (4, 5, 6, 7) if inverse else (1,)
        return NativeCase((x,), x, 1, removed=removed, output_removed=(output,), module=target(2))
    if name in {"Unfold", "Fold"}:
        fold = name == "Fold"
        x = sample((2, 12, 4) if fold else (2, 3, 3, 3))
        module = target((3, 3), 2) if fold else target(2)
        return NativeCase(
            (x,),
            x,
            1,
            removed=(4, 5, 6, 7) if fold else (1,),
            output_removed=((1,) if fold else (4, 5, 6, 7),),
            module=module,
        )
    if name == "MultiheadAttention":
        x = sample((2, 3, 6))
        module = target(6, 2, batch_first=True).double().eval()
        return NativeCase(
            (x, x, x),
            x,
            2,
            (2, None),
            (1, 4),
            ((1, 4), ()),
            module=module,
            companions=((module.out_proj.weight, 0, (1, 4)), (module.out_proj.weight, 1, (1, 4))),
        )
    if (
        lowered in UNARY
        or lowered in {name.replace("_", "") for name in ACTIVATIONS}
        or name == "Identity"
    ):
        shape = (
            (2, 6, 3, 3, 3)
            if name == "Dropout3d"
            else (2, 6, 3)
            if name == "Dropout1d"
            else (2, 6, 3, 3)
        )
        x = sample(shape)
        module = target(0.3, -0.1) if name == "Threshold" else target()
        return NativeCase((x,), x, 1, module=module.double().eval())
    raise AssertionError(f"Missing module case: {name}")


def callable_case(kind, target):
    name = target if kind == "method" else target.__name__
    public_name = name.removeprefix("_")
    x = sample((2, 6, 3))
    if public_name in UNARY:
        return NativeCase((x,), x, 1)
    if public_name in BINARY:
        if public_name in {"and_", "or_", "xor", "bitwise_and", "bitwise_or", "bitwise_xor"}:
            x = sample((2, 6, 3), integer=True)
        other = torch.full_like(x, 2)
        return NativeCase((x, other), x, 1, companions=((other, 1, (1,)),))
    if public_name in {"where", "masked_fill"}:
        condition = x > 0.4
        if public_name == "masked_fill":
            return NativeCase((x, condition, 0.0), x, 1, companions=((condition, 1, (1,)),))
        args = (x, condition, x + 1) if kind == "method" else (condition, x, x + 1)
        return NativeCase(args, x, 1, companions=((condition, 1, (1,)),))
    if public_name in {"clamp", "clamp_min", "clamp_max"}:
        args = (x, 0.2, 0.7) if public_name == "clamp" else (x, 0.4)
        return NativeCase(args, x, 1)
    if public_name in ACTIVATIONS:
        kwargs = {"p": 0.0, "training": False} if "dropout" in public_name else {}
        if public_name == "dropout3d":
            x = sample((2, 6, 3, 3, 3))
        elif public_name == "dropout2d":
            x = sample((2, 6, 3, 3))
        args = (x, 0.3, -0.1) if public_name == "threshold" else (x,)
        return NativeCase(args, x, 1, kwargs=kwargs)
    if public_name in REDUCTIONS or public_name in {"softmax", "log_softmax"}:
        return NativeCase((x,), x, 1, kwargs={"dim": 2})
    if public_name in CASTS:
        args = (x, torch.float32) if public_name == "to" else (x,)
        if public_name == "type_as":
            args = (x, sample((7,)).float())
        return NativeCase(args, x, 1)
    if public_name in {"size", "dim", "numel", "getattr"}:
        args = (
            (x, 1) if public_name == "size" else (x, "shape") if public_name == "getattr" else (x,)
        )
        return NativeCase(args, x, 1, (), output_removed=(), expression=True)
    if public_name == "linear":
        x, weight, bias = sample((2, 6)), sample((4, 6)), sample((4,))
        return NativeCase((x, weight, bias), x, 1, (None,), companions=((weight, 1, (1,)),))
    if public_name.startswith("conv"):
        spatial = next(n for n in (1, 2, 3) if f"{n}d" in public_name)
        transpose = "transpose" in public_name
        x = sample((2, 6, *((4,) * spatial)))
        weight = sample((6, 4, *((1,) * spatial)) if transpose else (4, 6, *((1,) * spatial)))
        return NativeCase(
            (x, weight), x, 1, (None,), companions=((weight, 0 if transpose else 1, (1,)),)
        )
    if public_name in {"batch_norm", "instance_norm"}:
        mean, var, weight, bias = sample((6,)), sample((6,)) + 1, sample((6,)), sample((6,))
        return NativeCase(
            (x, mean, var, weight, bias), x, 1, companions=((mean, 0, (1,)), (weight, 0, (1,)))
        )
    if public_name in {"layer_norm", "rms_norm", "group_norm", "prelu", "normalize"}:
        x, weight = sample((2, 6)), sample((6,))
        removed = (1, 4) if public_name == "group_norm" else (1,)
        if public_name in {"layer_norm", "rms_norm"}:
            args = (x, (6,), weight)
        elif public_name == "group_norm":
            args = (x, 2, weight)
        elif public_name == "prelu":
            args = (x, weight)
        else:
            return NativeCase((x,), x, 1)
        return NativeCase(
            args,
            x,
            1,
            removed=removed,
            output_removed=(removed,),
            companions=((weight, 0, removed),),
        )
    if public_name == "embedding":
        indices, weight = sample((2, 3), integer=True), sample((5, 6))
        return NativeCase((indices, weight), weight, 1, (2,))
    if "pool" in public_name or public_name == "interpolate":
        spatial = next((n for n in (1, 2, 3) if f"{n}d" in public_name), 2)
        x = sample((2, 6, *((4,) * spatial)))
        if "unpool" in public_name:
            pooled, indices = getattr(F, f"max_pool{spatial}d")(x, 2, return_indices=True)
            return NativeCase((pooled, indices, 2), pooled, 1, companions=((indices, 1, (1,)),))
        kwargs = {"scale_factor": 2} if public_name == "interpolate" else {}
        args = (x,) if kwargs else (x, 2)
        if public_name.endswith("with_indices"):
            return NativeCase(args, x, 1, (1, 1), output_removed=((1,), (1,)), kwargs=kwargs)
        return NativeCase(args, x, 1, kwargs=kwargs)
    if public_name == "pad":
        return NativeCase((x, (1, 1)), x, 1)
    if public_name in {"matmul", "mm", "bmm", "addmm", "baddbmm", "einsum"}:
        batched = public_name in {"bmm", "baddbmm"}
        x, right = (
            sample((2, 3, 6) if batched else (3, 6)),
            sample((2, 6, 4) if batched else (6, 4)),
        )
        args = (x, right)
        if public_name in {"addmm", "baddbmm"}:
            args = (sample((2, 3, 4) if batched else (3, 4)), x, right)
        elif public_name == "einsum":
            args = ("ik,kj->ij", x, right)
        return NativeCase(
            args, x, 2 if batched else 1, (None,), companions=((right, 1 if batched else 0, (1,)),)
        )
    if public_name in {"permute", "transpose", "swapaxes", "swapdims", "t"}:
        if public_name == "t":
            x = sample((2, 6))
            return NativeCase((x,), x, 1, (0,))
        args = (x, (0, 2, 1)) if public_name == "permute" else (x, 1, 2)
        return NativeCase(args, x, 1, (2,))
    if public_name in {"reshape", "view", "flatten", "squeeze", "unsqueeze"}:
        if public_name in {"reshape", "view"}:
            x = sample((2, 6))
            return NativeCase((x, (2, 2, 3)), x, 1, (2,), (1, 4))
        if public_name == "flatten":
            x = sample((2, 3, 2))
            return NativeCase((x, 1), x, 1, output_removed=((2, 3),))
        x = sample((2, 6, 1) if public_name == "squeeze" else (2, 6))
        return NativeCase(
            (x, 2 if public_name == "squeeze" else 1), x, 1, (1 if public_name == "squeeze" else 2,)
        )
    if public_name in {"cat", "concat", "concatenate", "stack"}:
        x = sample((2, 6))
        other = sample((2, 6) if public_name == "stack" else (2, 3))
        return NativeCase(((x, other), 1), x, 1, (2 if public_name == "stack" else 1,))
    if public_name in {"split", "chunk", "unbind"}:
        if public_name == "unbind":
            return NativeCase((x, 2), x, 1, (1, 1, 1), output_removed=((1,), (1,), (1,)))
        x = sample((2, 6))
        return NativeCase(
            (x, 3 if public_name == "split" else 2, 1), x, 1, (1, None), output_removed=((1,), ())
        )
    if public_name in {"getitem", "narrow", "index_select"}:
        if public_name == "getitem":
            args = (x, (slice(None), slice(None), slice(0, 2)))
        elif public_name == "narrow":
            args = (x, 2, 0, 2)
        else:
            args = (x, 2, torch.tensor([0, 2]))
        return NativeCase(args, x, 1)
    if public_name in {
        "repeat",
        "tile",
        "repeat_interleave",
        "expand",
        "expand_as",
        "broadcast_to",
    }:
        x = sample((1, 6))
        if public_name == "repeat_interleave":
            return NativeCase((x, 2, 1), x, 1, output_removed=((2, 3),))
        if public_name in {"repeat", "tile"}:
            return NativeCase((x, (1, 2)), x, 1, output_removed=((1, 7),))
        args = (x, sample((3, 6))) if public_name == "expand_as" else (x, (3, 6))
        return NativeCase(args, x, 1)
    if public_name in {
        "glu",
        "channel_shuffle",
        "pixel_shuffle",
        "pixel_unshuffle",
        "unfold",
        "fold",
    }:
        cls = {
            "glu": nn.GLU,
            "channel_shuffle": nn.ChannelShuffle,
            "pixel_shuffle": nn.PixelShuffle,
            "pixel_unshuffle": nn.PixelUnshuffle,
            "unfold": nn.Unfold,
            "fold": nn.Fold,
        }[public_name]
        case = module_case(cls)
        if public_name == "fold":
            case.args = (*case.args, (3, 3), 2)
        else:
            case.args = (*case.args, 1 if public_name == "glu" else 2)
        case.module = None
        return case
    if public_name == "scaled_dot_product_attention":
        q, k, v = sample((2, 2, 3, 6)), sample((2, 2, 4, 6)), sample((2, 2, 4, 5))
        return NativeCase((q, k, v), q, 3, (None,), companions=((k, 3, (1,)),))
    raise AssertionError(f"Missing {kind} case: {name}")


def make_case(kind, target):
    return module_case(target) if kind == "module" else callable_case(kind, target)
