"""Assemble exact built-in matches from independently defined operator families."""

import builtins
import operator
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ArgumentRef
from ..operation import OperatorRule, OperatorSpec, OutputContract
from .attention import register_attention
from .effects import native_effects
from .extended import register_extended
from .indexing import register_indexing
from .native import (
    _identity,
    batch_norm,
    concatenate,
    convolution,
    getattr_rule,
    getitem,
    group_norm,
    layer_norm,
    linear,
    matmul,
    permute,
    pointwise,
    reduction,
    reshape,
    shape_only,
    softmax,
    split,
)
from .shapes import CallArgumentConstraint, expression_for


def register_defaults(registry):
    """Populate a local registry with exact built-in operation matches."""

    def native(rule):
        def analyze(ctx):
            expression = expression_for(ctx)
            if expression is not None and not ctx.outputs:
                constraints = ()
                if expression.kind == "dimension":
                    # A dimension read can shrink, but its axis selector must not
                    # silently switch to another axis after upstream compaction.
                    guard = CallArgumentConstraint.from_operation(
                        ctx, checked_arguments=(ArgumentRef("input", 0),)
                    )
                    if guard.refs:
                        constraints = (guard,)
                return OperatorSpec(expression=expression, constraints=constraints)
            result = rule(ctx)
            if result.contract is None:
                fresh = native_effects(ctx.node, ctx.module).fresh_output
                dense = ctx.node.target == "contiguous"
                result = replace(
                    result,
                    contract=OutputContract(
                        fresh_output=fresh,
                        output_layout="contiguous" if dense else "unknown",
                    ),
                )
            return result

        return OperatorRule(analyze, evaluate_on_meta=True, effects=native_effects)

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
    register_extended(registry, modules, functions, methods)
    register_indexing(modules, functions, methods)
    register_attention(modules, functions, methods)
