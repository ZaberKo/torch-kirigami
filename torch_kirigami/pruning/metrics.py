"""Region-union weight scores with explicit gradient ownership."""

from __future__ import annotations

import torch

from .types import PlanningError


def gather_region(tensor, region):
    """Gather a Cartesian region without creating a full-tensor mask."""
    for dim, indices in enumerate(region.axes):
        if len(indices) != tensor.shape[dim]:
            index = torch.tensor(tuple(indices), dtype=torch.long, device=tensor.device)
            tensor = tensor.index_select(dim, index)
    return tensor


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
    result = []
    bindings = dict(context.graph.tensor_bindings())
    with torch.no_grad():
        for candidate in batch:
            impact = context.impact(candidate.remove)
            context.require_complete(impact)
            total = 0.0
            for selection in impact.parameters:
                weight = bindings[selection.tensor]
                if metric.parameter_filter:
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
                        values = values.to(dtype) * gather_region(weight.grad.detach(), region).to(
                            dtype
                        )
                        if metric.mode == "elementwise_abs":
                            values = values.abs()
                    else:
                        values = values.abs().to(dtype)
                        if metric.p == 2:
                            values = values.square()
                    total += values.sum(dtype=dtype).item()
            if isinstance(metric, Magnitude) and metric.p == 2:
                total = total**0.5
            elif isinstance(metric, WeightTaylor) and metric.mode == "joint_abs":
                total = abs(total)
            result.append(total)
    return result
