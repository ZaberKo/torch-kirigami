"""Region-union weight scores with explicit gradient ownership."""

from __future__ import annotations

import math

import torch

from ..regions import gather_region
from .types import PlanningError


class Magnitude:
    """Compute L1 or L2 over the union of all affected parameter regions.

    Args:
        p: One or two; L2 includes the final square root.
        parameter_filter: Optional predicate accepting (TensorRef, Parameter).
            Bias and normalization parameters are included by default.
    """

    def __init__(self, p=2, *, parameter_filter=None):
        if p not in (1, 2):
            raise ValueError("Magnitude supports p=1 or p=2")
        self.p = p
        self.parameter_filter = parameter_filter

    def __call__(self, context, candidate_batch):
        """Return aligned scores without changing gradients or retaining graphs."""
        return _scores(self, context, candidate_batch)


class WeightTaylor:
    """Score real weights using caller-provided current, unscaled gradients.

    The caller owns loss reduction, gradient accumulation, and AMP unscaling.
    This is not a per-example Fisher estimator and never invokes backward.

    Args:
        mode: elementwise_abs sums absolute products; joint_abs takes one
            absolute value after summing all signed products in the candidate.
        parameter_filter: Optional (TensorRef, Parameter) predicate.
    """

    def __init__(self, mode="elementwise_abs", *, parameter_filter=None):
        if mode not in ("elementwise_abs", "joint_abs"):
            raise ValueError("Unknown Taylor mode")
        self.mode = mode
        self.parameter_filter = parameter_filter

    def __call__(self, context, candidate_batch):
        """Return aligned scores; missing or nonfinite statistics are errors."""
        return _scores(self, context, candidate_batch)


def _scores(metric, context, batch):
    result = [0.0] * len(batch)
    device_scores = {}
    l2 = isinstance(metric, Magnitude) and metric.p == 2
    bindings = dict(context.graph.tensor_bindings())
    with torch.no_grad():
        for index, candidate in enumerate(batch):
            impact = context.impact(candidate.remove)
            context.require_complete(impact)
            totals = {}
            for selection in impact.parameters:
                weight = bindings[selection.tensor]
                if metric.parameter_filter is not None:
                    include = metric.parameter_filter(selection.tensor, weight)
                    context.graph.validate()  # A user callback cannot invalidate cached bindings.
                    if not include:
                        continue
                taylor = isinstance(metric, WeightTaylor)
                if taylor and (weight.is_complex() or weight.grad is None or weight.grad.is_sparse):
                    raise PlanningError(
                        "WeightTaylor requires real parameters and dense current gradients"
                    )
                dtype = (
                    torch.float64
                    if weight.dtype in (torch.float64, torch.complex128)
                    else torch.float32
                )
                for region in selection.regions:  # Selection normalizes to a disjoint union.
                    values = gather_region(weight.detach(), region)
                    if taylor:
                        # Promote before multiplication: a float64 reduction cannot
                        # recover products already underflowed/overflowed in float32.
                        values = values.to(torch.float64) * gather_region(
                            weight.grad.detach(), region
                        ).to(torch.float64)
                        if metric.mode == "elementwise_abs":
                            values = values.abs()
                    elif l2:
                        # Scale before squaring, including float64 extremes. Real
                        # components avoid overflowing complex64 abs prematurely.
                        values = (
                            torch.view_as_real(values.resolve_conj())
                            if values.is_complex()
                            else values
                        )
                        values = values.to(dtype).abs()
                        scale = values.amax()
                        divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
                        value = (values / divisor).square().sum(
                            dtype=torch.float64
                        ).sqrt() * scale.to(torch.float64)
                    else:
                        values = values.to(
                            torch.complex128 if values.is_complex() else torch.float64
                        ).abs()
                    if not l2:
                        value = values.sum(dtype=torch.float64)
                    previous = totals.get(weight.device)
                    totals[weight.device] = (
                        value
                        if previous is None
                        else torch.hypot(previous, value)
                        if l2
                        else previous + value
                    )
            for device, value in totals.items():
                device_scores.setdefault(device, []).append((index, value))
        # One transfer per device/batch, rather than one synchronization per
        # selected region. Different parameter devices can still contribute.
        for entries in device_scores.values():
            values = torch.stack([value for _, value in entries]).cpu().tolist()
            for (index, _), value in zip(entries, values, strict=True):
                result[index] = math.hypot(result[index], value) if l2 else result[index] + value
    if isinstance(metric, WeightTaylor) and metric.mode == "joint_abs":
        result = [abs(value) for value in result]
    return result
