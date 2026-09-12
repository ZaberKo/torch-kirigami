"""Capture-time call effects, also used by compact-forward validation."""

import inspect
import operator

import torch
from torch import nn
from torch.nn import functional as F

from ..operation import CallEffects


def named_inplace(node):
    """Recognize mutating API names without confusing Python keyword escapes."""
    if node.op not in ("call_function", "call_method"):
        return False
    if node.target in (operator.and_, operator.or_, operator.not_):
        return False
    mutators = (
        operator.iadd,
        operator.isub,
        operator.imul,
        operator.imatmul,
        operator.itruediv,
        operator.ifloordiv,
        operator.imod,
        operator.ipow,
        operator.iand,
        operator.ior,
        operator.ixor,
        operator.ilshift,
        operator.irshift,
        operator.setitem,
        operator.delitem,
    )
    if node.target in mutators:
        return True
    name = getattr(node.target, "__name__", str(node.target))
    if name in {f"__{target.__name__}__" for target in mutators}:
        return True
    return name.endswith("_") and not name.endswith("__")


def native_effects(node, module):
    """Declare public in-place arguments and guaranteed allocations in one place."""
    target = node.target
    mutates = named_inplace(node)
    mutates = mutates or bool(getattr(module, "inplace", False))
    if node.op == "call_function":
        try:
            bound = inspect.signature(target).bind_partial(*node.args, **node.kwargs)
            value = bound.arguments.get("inplace", False)
            mutates = mutates or (value is not False and value is not None)
        except (TypeError, ValueError):
            mutates = mutates or node.kwargs.get("inplace") is True
    # Allocation belongs to the call-effects contract, shared by capture and
    # compact execution. Views and conditional aliases (e.g. dropout) stay unknown.
    fresh_modules = (
        nn.MultiheadAttention,
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
        nn.MaxUnpool1d,
        nn.MaxUnpool2d,
        nn.MaxUnpool3d,
        nn.Upsample,
        nn.ReflectionPad1d,
        nn.ReflectionPad2d,
        nn.ReflectionPad3d,
        nn.ReplicationPad1d,
        nn.ReplicationPad2d,
        nn.ReplicationPad3d,
        nn.ConstantPad1d,
        nn.ConstantPad2d,
        nn.ConstantPad3d,
        nn.ZeroPad2d,
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.RMSNorm,
        nn.Embedding,
        nn.PReLU,
        nn.ReLU,
        nn.ReLU6,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
        nn.Tanh,
        nn.Softmax,
        nn.LogSoftmax,
    )
    fresh_functions = (
        F.max_pool1d,
        F.max_pool2d,
        F.max_pool3d,
        F.max_pool1d_with_indices,
        F.max_pool2d_with_indices,
        F.max_pool3d_with_indices,
        F.avg_pool1d,
        F.avg_pool2d,
        F.avg_pool3d,
        F.adaptive_avg_pool1d,
        F.adaptive_avg_pool2d,
        F.adaptive_avg_pool3d,
        F.adaptive_max_pool1d,
        F.adaptive_max_pool2d,
        F.adaptive_max_pool3d,
        F.adaptive_max_pool1d_with_indices,
        F.adaptive_max_pool2d_with_indices,
        F.adaptive_max_pool3d_with_indices,
        F.max_unpool1d,
        F.max_unpool2d,
        F.max_unpool3d,
        F.interpolate,
        F.pad,
        F.linear,
        F.conv1d,
        F.conv2d,
        F.conv3d,
        F.conv_transpose1d,
        F.conv_transpose2d,
        F.conv_transpose3d,
        F.batch_norm,
        F.layer_norm,
        F.group_norm,
        F.instance_norm,
        F.rms_norm,
        F.embedding,
        F.prelu,
        F.relu,
        F.relu6,
        F.gelu,
        F.silu,
        F.softmax,
        F.log_softmax,
        F.scaled_dot_product_attention,
        torch.clone,
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.neg,
        operator.matmul,
        torch.add,
        torch.sub,
        torch.mul,
        torch.div,
        torch.neg,
        torch.matmul,
        torch.mm,
        torch.bmm,
        torch.relu,
        torch.sigmoid,
        torch.tanh,
        torch.softmax,
        torch.log_softmax,
        torch.cat,
    )
    fresh_methods = (
        "clone",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "matmul",
        "mm",
        "bmm",
        "relu",
        "sigmoid",
        "tanh",
        "softmax",
        "log_softmax",
    )
    fresh = not mutates and (
        type(module) in fresh_modules
        or (node.op == "call_function" and target in fresh_functions)
        or (node.op == "call_method" and target in fresh_methods)
    )
    return CallEffects(mutates, fresh)
