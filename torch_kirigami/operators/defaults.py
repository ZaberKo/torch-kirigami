"""Assemble exact built-in matches from independently defined operator families."""

from __future__ import annotations

import builtins
import operator
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ArgumentRef
from ..operation import (
    OperationContext,
    OperatorRegistrar,
    OperatorRule,
    OperatorSpec,
    OutputContract,
)
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


def register_defaults(registry: OperatorRegistrar) -> None:
    """Populate a local registry with exact built-in operation matches."""

    def native(rule: Callable[[OperationContext], OperatorSpec], fresh: bool) -> OperatorRule:
        """Wrap a rule with shape-expression handling and native effects."""

        def analyze(ctx: OperationContext) -> OperatorSpec:
            """Analyze one captured operation using the wrapped rule."""
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
                dense = ctx.node.target == "contiguous"
                result = replace(
                    result,
                    contract=OutputContract(
                        output_layout="contiguous" if dense else "unknown",
                    ),
                )
            return result

        return OperatorRule(
            analyze, evaluate_on_meta=True, effects=partial(native_effects, fresh_output=fresh)
        )

    def modules(
        types: list[type[nn.Module]],
        rule: Callable[[OperationContext], OperatorSpec],
        *,
        fresh: bool,
    ) -> None:
        """Register one rule for each listed module class."""
        for target in types:
            registry.register(target, native(rule, fresh))

    def functions(
        targets: list[object], rule: Callable[[OperationContext], OperatorSpec], *, fresh: bool
    ) -> None:
        """Register one rule for each listed function target."""
        for target in targets:
            registry.register(target, native(rule, fresh), opaque=False)

    def methods(
        names: list[str], rule: Callable[[OperationContext], OperatorSpec], *, fresh: bool
    ) -> None:
        """Register one rule for each listed method name."""
        for name in names:
            registry.register_method(name, native(rule, fresh))

    modules([nn.Softmax, nn.LogSoftmax], softmax, fresh=True)
    functions([F.softmax, F.log_softmax, torch.softmax, torch.log_softmax], softmax, fresh=True)
    methods(["softmax", "log_softmax"], softmax, fresh=True)
    modules([nn.Linear], linear, fresh=True)
    functions([F.linear], linear, fresh=True)
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
        fresh=True,
    )
    functions(
        [F.conv1d, F.conv2d, F.conv3d, F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d],
        convolution,
        fresh=True,
    )
    modules([nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d], batch_norm, fresh=True)
    functions([F.batch_norm], batch_norm, fresh=True)
    modules([nn.LayerNorm], layer_norm, fresh=True)
    functions([F.layer_norm], layer_norm, fresh=True)
    modules([nn.GroupNorm], group_norm, fresh=True)
    functions([F.group_norm], group_norm, fresh=True)
    modules([nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d], pointwise, fresh=False)
    functions([F.dropout], pointwise, fresh=False)
    methods(["detach", "contiguous"], pointwise, fresh=False)
    modules([nn.Identity], _identity, fresh=False)
    modules([nn.Flatten, nn.Unflatten], reshape, fresh=False)
    modules(
        [
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
            nn.SiLU,
            nn.Sigmoid,
            nn.Tanh,
        ],
        pointwise,
        fresh=True,
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
        ],
        pointwise,
        fresh=True,
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
            "relu_",
            "add_",
            "mul_",
        ],
        pointwise,
        fresh=True,
    )
    functions([operator.matmul, torch.matmul, torch.mm, torch.bmm], matmul, fresh=True)
    methods(["matmul", "mm", "bmm"], matmul, fresh=True)
    functions(
        [torch.permute, torch.transpose, torch.swapaxes, torch.swapdims, torch.t],
        permute,
        fresh=False,
    )
    methods(["permute", "transpose", "swapaxes", "swapdims", "t"], permute, fresh=False)
    functions([torch.reshape, torch.flatten, torch.squeeze, torch.unsqueeze], reshape, fresh=False)
    methods(["reshape", "view", "flatten", "squeeze", "unsqueeze"], reshape, fresh=False)
    functions([torch.cat, torch.concat, torch.concatenate], concatenate, fresh=True)
    functions([torch.split, torch.unbind], split, fresh=False)
    methods(["split", "unbind"], split, fresh=False)
    functions([operator.getitem], getitem, fresh=False)
    # The wrapper extracts proven scalar size expressions first. Tensor overloads
    # must retain pointwise dependencies instead of silently emitting an empty rule.
    functions(
        [operator.floordiv, operator.mod, torch.floor_divide, torch.remainder],
        pointwise,
        fresh=True,
    )
    methods(["floor_divide", "remainder"], pointwise, fresh=True)
    functions([builtins.getattr], getattr_rule, fresh=False)
    functions([torch.sum, torch.mean], reduction, fresh=True)
    methods(["sum", "mean"], reduction, fresh=True)
    methods(["size", "dim", "numel"], shape_only, fresh=False)
    register_extended(registry, modules, functions, methods)
    register_indexing(modules, functions, methods)
    register_attention(modules, functions, methods)
