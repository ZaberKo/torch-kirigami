"""Parameter counts, nominal FLOPs, dense MACs, and coverage diagnostics."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.autograd.profiler_util import FunctionEvent
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.flop_counter import FlopCounterMode, flop_registry

from ..dep_graph.capture import tensor_leaves
from .execution import isolated_evaluation
from .formulas import (
    FLOP_COMPOSITES,
    FLOP_FREE_OPS,
    MAC_COMPOSITES,
    MAC_FORMULAS,
    MAC_FREE_OPS,
    arithmetic_formulas,
    biased_matrix_vector_flops,
    cudnn_convolution_flops,
    dot_flops,
    efficient_attention_flops,
    matrix_extra_flops,
    matrix_vector_flops,
    slow_conv2d_flops,
)


def count_parameters(model: nn.Module) -> int:
    """Count elements in unique registered Parameters without executing the model.

    Includes frozen and unused parameters; excludes buffers. Multiple names for
    the same Parameter count once. Distinct Parameters sharing storage remain
    distinct entities. No device transfer, sample input, or FX graph is needed.
    Counts tensor sizes, not nonzero values: masking weights does not lower it.
    """
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class ModelComplexity:
    """Recorded FLOPs, dense MACs and unique registered parameter elements.

    `flops` includes registered formulas and local nominal arithmetic formulas.
    Both operation counts cover the entire supplied batch and executed path.

    One multiply-accumulate counts as one MAC (two FLOPs). Matrix products and
    convolutions count toward MACs; bias, normalization and elementwise
    arithmetic contribute only to FLOPs. `unsupported_ops` lists unverified MAC
    coverage, not these explicit exclusions. `unsupported_flop_ops` separately
    lists unverified FLOP coverage. A nonempty tuple means completeness is unknown
    for the corresponding count.
    Parameters include frozen weights, deduplicated by Parameter identity.

    Attributes:
        macs: Count of audited matrix and convolution multiply-accumulates.
        params: Count of unique registered Parameter elements.
        flops: Nominal floating-point arithmetic under the registered formulas.
        unsupported_ops: Captured operations with unverified MAC coverage.
        unsupported_flop_ops: Captured operations with unverified FLOP coverage.
    """

    macs: int
    params: int
    flops: int
    unsupported_ops: tuple[str, ...] = ()
    unsupported_flop_ops: tuple[str, ...] = ()


@contextmanager
def counting_attention() -> Iterator[None]:
    """Expose attention matrix work and restore public backend switches on exit."""
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def find_coverage_gaps(
    events: list[FunctionEvent], mac_free: frozenset[str]
) -> tuple[set[str], set[str]]:
    """Match formula markers to actual calls, not all calls with the same name.

    A C++ composite can invoke an operation below Python dispatch. A separate
    counted invocation of that operation does not cover the hidden invocation.
    Profiler parents connect each formula to its own call and backend children.
    """
    prefix = "kirigami.formula/"
    counted: dict[int, str] = {}
    recorded: set[str] = set()
    for event in events:
        if not event.name.startswith(prefix):
            continue
        name = event.name.removeprefix(prefix)
        recorded.add(name)
        parent = event.cpu_parent
        while parent is not None and parent.name != name:
            parent = parent.cpu_parent
        if (
            parent is None
            and not name.startswith("aten::")
            and event.cpu_parent is not None
            and event.cpu_parent.name == "PythonDispatchMode"
        ):
            # Python custom operators can lack their own profiler event. The
            # immediate dispatch scope still contains this call and its formula.
            parent = event.cpu_parent
        if parent is not None:
            counted[parent.id] = name
    unsupported = recorded - MAC_FORMULAS - mac_free
    flop_gaps: set[str] = set()
    mac_ignored = (mac_free | MAC_COMPOSITES) - unsupported
    flop_ignored = FLOP_FREE_OPS | MAC_COMPOSITES | FLOP_COMPOSITES
    for event in events:
        if "::" not in event.name or event.name.startswith(prefix):
            continue
        mac_covered = flop_covered = False
        parent = event
        while parent is not None:
            if parent.id in counted:
                flop_covered = True
                mac_covered |= counted[parent.id] in MAC_FORMULAS
            parent = parent.cpu_parent
        if not mac_covered and event.name not in mac_ignored:
            unsupported.add(event.name)
        if not flop_covered and event.name not in flop_ignored:
            flop_gaps.add(event.name)
    return unsupported, flop_gaps


def calculate_model_complexity(
    target: nn.Module,
    input_args: Any,
    device: torch.device | str | None = None,
    input_kwargs: dict[str, Any] | None = None,
) -> ModelComplexity:
    """Record eager inference FLOPs, MACs and parameters without changing training state.

    Args:
        target: Eager module already on the chosen device; do not pass a compiled
            wrapper. Parameter values must not be mutated by its forward.
        input_args: One input or positional argument tuple/list. Nested tensor
            containers are moved to device and isolated from forward writes.
        device: CPU/CUDA device, inferred from model tensors when omitted.
        input_kwargs: Keyword inputs, including nested tensor containers.

    Returns:
        Counts for the entire supplied batch and explicit unsupported operations.
        FLOPs include supported bias, arithmetic, reduction and normalization work.
        Check `unsupported_flop_ops` independently of MAC coverage diagnostics.
        Dense MACs exclude this additional arithmetic. SDPA uses the math
        backend here, and MHA fast paths are disabled during counting.

    Model modes, registered buffer bindings/values, inputs and torch RNG state
    are restored even on failure. This does not isolate arbitrary Python side
    effects or parameter writes. Counting runs outside inference_mode; latency
    uses a separate, uninstrumented forward and the original attention backend.
    """
    with (
        torch.inference_mode(False),
        # Disabling inference mode enables gradients, including under an outer
        # no_grad decorator. Establish no_grad after that mode transition.
        torch.no_grad(),
        isolated_evaluation(target, input_args, input_kwargs, device) as (args, kwargs, _),
    ):
        # Preserve external non-MAC formulas. The mappings and validation
        # wrappers belong to this invocation, never the global registry.
        arithmetic = arithmetic_formulas()
        additions = {op: formula for op, formula in arithmetic.items() if op not in flop_registry}
        counter = FlopCounterMode(
            display=False,
            custom_mapping={
                **additions,
                torch.ops.aten.dot: dot_flops,
                torch.ops.aten.vdot: dot_flops,
                torch.ops.aten.mv: matrix_vector_flops,
                torch.ops.aten.addmv: biased_matrix_vector_flops,
                torch.ops.aten.addmv_: biased_matrix_vector_flops,
                torch.ops.aten.cudnn_convolution: cudnn_convolution_flops,
                torch.ops.aten._slow_conv2d_forward: slow_conv2d_flops,
            },
        )
        # Reuse the existing dense formulas for identical operand layouts.
        # Assign wrapped functions directly; custom_mapping wraps shape formulas.
        for alias, source in (
            (torch.ops.aten.addmm_, torch.ops.aten.addmm),
            (torch.ops.aten.baddbmm_, torch.ops.aten.baddbmm),
            (torch.ops.aten.addbmm, torch.ops.aten.baddbmm),
            (torch.ops.aten.addbmm_, torch.ops.aten.baddbmm),
            (
                torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
                torch.ops.aten._scaled_dot_product_flash_attention,
            ),
        ):
            counter.flop_registry[alias] = counter.flop_registry[source]
        counter.flop_registry[torch.ops.aten._efficient_attention_forward] = (
            efficient_attention_flops
        )
        macs = 0
        uncovered_flops: set[str] = set()

        def checked_formula(op: object, formula: Callable, added: bool) -> Callable:
            """Validate each invocation before aggregation and retain its dense MAC count."""
            name = str(op).replace(".", "::", 1)

            def record(*values: Any, out_val: Any = None, **options: Any) -> int:
                """Check the formula result and add separately defined non-MAC arithmetic."""
                nonlocal macs
                output_tensors = tensor_leaves(out_val)
                if added and any(t.is_complex() for t in tensor_leaves((values, options))):
                    uncovered_flops.add(name)
                    return 0
                if added and output_tensors and not output_tensors[0].is_floating_point():
                    if output_tensors[0].is_complex():
                        uncovered_flops.add(name)
                    with record_function(f"kirigami.formula/{name}"):
                        pass
                    return 0
                value = formula(*values, out_val=out_val, **options)
                if value is None and added:
                    uncovered_flops.add(name)
                    return 0
                is_mac = name in MAC_FORMULAS
                if type(value) is not int or value < 0 or (is_mac and value % 2):
                    expected = "nonnegative even" if is_mac else "nonnegative integer"
                    raise ValueError(f"{name}: expected a {expected} FLOP count")
                if (
                    name in ("aten::_scaled_mm", "aten::convolution_backward")
                    or (name.startswith("aten::") and "attention" in name)
                    or name in ("flex_attention", "flex_attention_backward")
                ):
                    # Native fused formulas omit arithmetic outside their
                    # matrix products, such as softmax or bias gradients.
                    uncovered_flops.add(name)
                if is_mac:
                    macs += value // 2
                    value += matrix_extra_flops(name, values, options, out_val)
                    if output_tensors and not output_tensors[0].is_floating_point():
                        if output_tensors[0].is_complex():
                            uncovered_flops.add(name)
                        else:
                            value = 0  # Integer MACs are not floating-point arithmetic.
                with record_function(f"kirigami.formula/{name}"):
                    pass
                return value

            return record

        counter.flop_registry = {
            op: checked_formula(op, formula, op in additions)
            for op, formula in counter.flop_registry.items()
        }
        with (
            profile(activities=[ProfilerActivity.CPU]) as trace,
            counting_attention(),
            counter,
        ):
            target(*args, **kwargs)
        mac_free = (
            MAC_FREE_OPS
            | FLOP_FREE_OPS
            | {str(op).replace("aten.", "aten::", 1) for op in arithmetic}
        )
        unsupported, flop_gaps = find_coverage_gaps(trace.events(), mac_free)
        uncovered_flops.update(flop_gaps)
        return ModelComplexity(
            macs=macs,
            params=count_parameters(target),
            flops=counter.get_total_flops(),
            unsupported_ops=tuple(sorted(unsupported)),
            unsupported_flop_ops=tuple(sorted(uncovered_flops)),
        )
