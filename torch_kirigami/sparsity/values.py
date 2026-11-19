"""Live parameter-group access and numerically stable norm kernels.

Regularizers and parameter operations share these helpers. They read current
parameter values on every call and never cache values or autograd graphs.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.autograd.function import FunctionCtx, once_differentiable

from ..dep_graph.selection import IndexSet, Selection, TensorRef, gather_region
from ..pruning.groups import ParameterGroup, checked_groups


def group_bindings(
    groups: tuple[ParameterGroup, ...],
    *,
    refs: tuple[TensorRef, ...] | None = None,
) -> tuple[dict[TensorRef, torch.Tensor], torch.dtype]:
    """Resolve live parameters for groups and choose the computation dtype.

    No parameter values are gathered.

    Args:
        groups: Nonempty groups that share one dependency graph.
        refs: Parameters to validate. `None` validates every parameter that
            the groups select.

    Returns:
        The graph's tensor bindings and the computation dtype: float64 if any
        validated parameter is float64, otherwise float32.

    Raises:
        TypeError: An item of `groups` is not a `ParameterGroup`.
        ValueError: `groups` is empty or spans several graphs, or the validated
            parameters are on several devices or are not dense real
            floating-point tensors.
    """
    groups = checked_groups(groups)
    if not groups:
        raise ValueError("Expected a nonempty ParameterGroup sequence")
    graph = groups[0].graph
    if any(g.graph is not graph for g in groups):
        raise ValueError("Groups must share one dependency graph")
    bindings = dict(graph.tensor_bindings())
    refs = (
        refs
        if refs is not None
        else tuple(dict.fromkeys(s.tensor for g in groups for s in g.selections))
    )
    return bindings, parameter_dtype([bindings[ref] for ref in refs])


def parameter_dtype(parameters: Sequence[torch.Tensor]) -> torch.dtype:
    """Validate sparse-operation bindings and choose float32 or float64 accumulation."""
    if len({p.device for p in parameters}) != 1:
        raise ValueError("Sparse operations require groups on one device")
    if any(not p.is_floating_point() or p.layout != torch.strided for p in parameters):
        raise ValueError("Sparse operations require dense real floating parameters")
    return torch.float64 if any(p.dtype == torch.float64 for p in parameters) else torch.float32


def gather_group(
    selections: tuple[Selection, ...],
    bindings: Mapping[TensorRef, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Read one group from bindings already validated for this operation."""
    return torch.cat(
        [
            gather_region(bindings[selection.tensor], region).reshape(-1).to(dtype)
            for selection in selections
            for region in selection.regions
        ]
    )


def group_values(
    groups: tuple[tuple[Selection, ...], ...],
    bindings: Mapping[TensorRef, torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Gather the current selected values of each group, preserving autograd.

    Distributed or sharded parameters are unsupported.

    Args:
        groups: Selections for each validated group in parameter order.
        bindings: Live tensors returned by `group_bindings` for this operation.
        dtype: Common computation dtype returned with the bindings.

    Returns:
        One flattened tensor per group, in the dtype chosen by `group_bindings`.

    Raises:
        ValueError: A selected value is nonfinite.
    """
    result = tuple(gather_group(group, bindings, dtype) for group in groups)
    if not torch.stack([torch.isfinite(v).all() for v in result]).all():
        raise ValueError("Sparse operations require finite selected values")
    return result


def scaled_product(*factors: torch.Tensor, divide: tuple[torch.Tensor, ...] = ()) -> torch.Tensor:
    """Multiply factors and divide by divisors without intermediate overflow or underflow.

    Mantissas and exponents are combined separately, so only the final exponent
    determines whether the result is representable. The final `ldexp` is split
    in two to avoid overflowing an intermediate power of two. Divisors must be
    nonzero. A zero factor gives an exact zero.
    """
    mantissa, exponent = torch.frexp(factors[0])
    for factor in factors[1:]:
        part, power = torch.frexp(factor)
        mantissa, exponent = mantissa * part, exponent + power
    for divisor in divide:
        part, power = torch.frexp(divisor)
        mantissa, exponent = mantissa / part, exponent - power
    half = torch.div(exponent, 2, rounding_mode="floor")
    result = torch.ldexp(torch.ldexp(mantissa, half), exponent - half)
    return torch.where(mantissa == 0, torch.zeros_like(result), result)


def _norm_divisors(
    values: torch.Tensor, dimensions: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the nonzero max-abs scale and scaled length used by norm backward."""
    scale = values.detach().abs().amax(dim=dimensions, keepdim=True)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    scaled = values / divisor
    length = torch.linalg.vector_norm(scaled, dim=dimensions, keepdim=True)
    length = torch.where(length == 0, torch.ones_like(length), length)
    return divisor, length


class _StableNorm(torch.autograd.Function):
    """Coefficient-weighted L2 norm whose gradient avoids tiny scaled intermediates."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, values: torch.Tensor, dimensions: tuple[int, ...], coefficient: float
    ) -> torch.Tensor:
        """Evaluate weighted L2 norms over `dimensions` and save the inputs for backward."""
        ctx.save_for_backward(values)
        ctx.dimensions = dimensions
        ctx.coefficient = coefficient
        scale = values.abs().amax(dim=dimensions, keepdim=True)
        divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
        length = torch.linalg.vector_norm(values / divisor, dim=dimensions)
        scale = divisor.squeeze(dimensions)
        if coefficient == 1:
            return length * scale
        if coefficient == 0:
            return torch.zeros_like(length)
        return scaled_product(length, scale, values.new_tensor(coefficient))

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx, gradient: torch.Tensor
    ) -> tuple[torch.Tensor | None, None, None]:
        """Propagate the stable L2 gradient through saved values."""
        (values,) = ctx.saved_tensors
        scale, length = _norm_divisors(values, ctx.dimensions)
        for dimension in sorted(ctx.dimensions):
            gradient = gradient.unsqueeze(dimension)
        return (
            scaled_product(
                values, gradient, values.new_tensor(ctx.coefficient), divide=(scale, length)
            ),
            None,
            None,
        )


def stable_norm(values: torch.Tensor, *, coefficient: float = 1.0) -> torch.Tensor:
    """Return `coefficient` times the L2 norm of all elements, with a stable gradient.

    The norm and its gradient stay accurate for nonzero subnormal values. An
    all-zero input has a zero gradient. Only first-order autograd is supported.
    """
    return _StableNorm.apply(values, tuple(range(values.ndim)), coefficient)


def _scaled_segments(
    values: torch.Tensor, owner: torch.Tensor, count: int
) -> tuple[torch.Tensor, ...]:
    """Return scaled values, scaled roots, scales, and nonzero flags for each owner."""
    zero = values.new_zeros(count)
    scales = zero.scatter_reduce(0, owner, values.detach(), reduce="amax")
    divisors = torch.where(scales == 0, torch.ones_like(scales), scales)
    scaled = values / divisors[owner]
    squared = zero.index_add(0, owner, scaled.square())
    root = torch.sqrt(torch.where(squared > 0, squared, torch.ones_like(squared)))
    return scaled, root, divisors, squared > 0


class _SegmentNorm(torch.autograd.Function):
    """Stable L2 norm over each owner's ragged set of nonnegative row norms."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, values: torch.Tensor, owner: torch.Tensor, count: int
    ) -> torch.Tensor:
        """Evaluate stable norms for ragged owner groups."""
        ctx.save_for_backward(values, owner)
        ctx.count = count
        _, root, divisors, nonzero = _scaled_segments(values, owner, count)
        return torch.where(nonzero, root, torch.zeros_like(root)) * divisors

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Propagate gradients to each ragged group's source values."""
        values, owner = ctx.saved_tensors
        scaled, root, _, _ = _scaled_segments(values, owner, ctx.count)
        return gradient[owner] * (scaled / root[owner]), None, None


class _HalfSquared(torch.autograd.Function):
    """Half squares without a spurious square overflow or half-gradient underflow."""

    @staticmethod
    def forward(ctx: FunctionCtx, values: torch.Tensor) -> torch.Tensor:
        """Evaluate half squares without an intermediate square operation."""
        ctx.save_for_backward(values)
        return (values * 0.5) * values

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gradient: torch.Tensor) -> torch.Tensor:
        """Return the exact first derivative of half squares."""
        (values,) = ctx.saved_tensors
        return gradient * values


def half_squared(values: torch.Tensor) -> torch.Tensor:
    """Return `0.5 * values**2` elementwise, with gradient `gradient * values`."""
    return _HalfSquared.apply(values)


class _SquaredNorm(torch.autograd.Function):
    """Half the weighted squared norm of several tensors, with an exact first derivative."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, coefficients: tuple[float, ...], *values: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate half the weighted squared norm in float64 and save the inputs."""
        ctx.save_for_backward(*values)
        ctx.coefficients = coefficients
        scaled = torch.cat(
            [
                value.double() * math.sqrt(coefficient)
                for value, coefficient in zip(values, coefficients, strict=True)
            ]
        )
        norm = stable_norm(scaled)
        return ((norm * 0.5) * norm).to(values[0].dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: FunctionCtx, gradient: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Propagate weighted squared-norm gradients to each group tensor."""
        return (
            None,
            *(
                scaled_product(
                    value.double(),
                    gradient.double(),
                    value.new_tensor(coefficient, dtype=torch.float64),
                ).to(value.dtype)
                for value, coefficient in zip(ctx.saved_tensors, ctx.coefficients, strict=True)
            ),
        )


def squared_norm(
    *values: torch.Tensor, coefficients: tuple[float, ...] | None = None
) -> torch.Tensor:
    """Return `0.5 * sum(a_i * ||v_i||**2)` without per-element square underflow.

    Args:
        *values: Tensors `v_i` to include.
        coefficients: One weight `a_i` per tensor. `None` uses one for each.

    Returns:
        A scalar in the dtype of the first tensor.
    """
    weights = (1.0,) * len(values) if coefficients is None else coefficients
    return _SquaredNorm.apply(weights, *values)


def _axis_sections(group: ParameterGroup) -> list[tuple[TensorRef, int, IndexSet]] | None:
    """Return `(tensor, axis, indices)` sections that select whole slices along one axis.

    Returns `None` when the group selects a scalar parameter or a region that
    restricts more than one axis, because such a group needs general gathering.
    """
    sections = []
    for selection in group.selections:
        ref = selection.tensor
        if not ref.shape:
            return None
        for region in selection.regions:
            partial = [
                dimension
                for dimension, (indices, width) in enumerate(
                    zip(region.axes, ref.shape, strict=True)
                )
                if len(indices) != width
            ]
            if len(partial) > 1:
                return None
            axis = partial[0] if partial else 0
            sections.append((ref, axis, region.axes[axis]))
    return sections


@dataclass(frozen=True, eq=False)
class ReductionPlan:
    """Precomputed coordinate indices for batched group penalties.

    The plan retains no parameter values or autograd graphs.

    Attributes:
        batches: Entries containing a parameter reference, axis, optional selected
            positions, and indices into the reduced axis vector. `None` selects
            the whole axis. Contribution indices preserve repeated groups.
        fallback: Indices of groups requiring general Cartesian gathering.
        owners: Group index for each contribution, in batch-then-fallback order.
        refs: Distinct parameter references requiring live validation each call.
    """

    batches: tuple[tuple[TensorRef, int, torch.Tensor | None, torch.Tensor], ...]
    fallback: tuple[int, ...]
    owners: torch.Tensor
    refs: tuple[TensorRef, ...]


def reduction_plan(
    groups: tuple[ParameterGroup, ...], bindings: Mapping[TensorRef, torch.Tensor]
) -> ReductionPlan:
    """Plan batched penalty evaluation for groups.

    Groups made only of whole slices along one axis are batched: each
    parameter and axis pair is reduced once per call, and the reduced axis
    vector is shared by every group that uses it. Other groups use general
    region gathering. The plan holds original-coordinate indices only.

    Args:
        groups: Nonempty groups that share one dependency graph and device.
        bindings: Live tensors returned by `group_bindings` for these groups.

    Returns:
        A plan for evaluating the same groups in `batched_slices`.
    """
    batches: dict[tuple[TensorRef, int], list[tuple[int, int]]] = {}
    fallback: list[int] = []
    for number, group in enumerate(groups):
        sections = _axis_sections(group)
        if sections is None:
            fallback.append(number)
            continue
        # Register only after every region qualifies, so a mixed group cannot
        # contribute through both the batched and general paths.
        for ref, axis, indices in sections:
            entries = batches.setdefault((ref, axis), [])
            entries.extend((number, i) for i in indices)
    prepared: list[tuple[TensorRef, int, torch.Tensor | None, torch.Tensor]] = []
    owners: list[int] = []
    refs = tuple(dict.fromkeys(s.tensor for group in groups for s in group.selections))
    # Index tensors must remain usable by autograd outside inference mode.
    with torch.inference_mode(False):
        for (ref, axis), entries in batches.items():
            device = bindings[ref].device
            selected = sorted({index for _, index in entries})
            positions = {old: new for new, old in enumerate(selected)}
            selected_positions = (
                torch.tensor(selected, dtype=torch.long, device=device)
                if len(selected) != ref.shape[axis]
                else None
            )
            contribution_indices = torch.tensor(
                [positions[i] for _, i in entries], dtype=torch.long, device=device
            )
            prepared.append((ref, axis, selected_positions, contribution_indices))
            owners.extend(number for number, _ in entries)
        owners.extend(fallback)
        owner = torch.tensor(owners, dtype=torch.long, device=bindings[refs[0]].device)
    return ReductionPlan(tuple(prepared), tuple(fallback), owner, refs)


def slice_norm(values: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Return stable L2 norms over `dimensions`, or absolute values when there are none."""
    if not dimensions:
        return values.abs()
    return _StableNorm.apply(values, dimensions, 1.0)


def batched_slices(
    plan: ReductionPlan, bindings: Mapping[TensorRef, torch.Tensor], dtype: torch.dtype
) -> Iterator[tuple[torch.Tensor, tuple[int, ...], torch.Tensor]]:
    """Yield each selected tensor, reduction dimensions, and contribution indices."""
    for ref, axis, selected_positions, contribution_indices in plan.batches:
        tensor = bindings[ref].to(dtype)
        if selected_positions is not None:
            tensor = tensor.index_select(axis, selected_positions)
        yield tensor, tuple(d for d in range(tensor.ndim) if d != axis), contribution_indices


def segment_norms(values: torch.Tensor, owner: torch.Tensor, count: int) -> torch.Tensor:
    """Reduce row norms by group, without losing subnormal relative magnitudes."""
    return _SegmentNorm.apply(values, owner, count)


def needs_direct_norm(values: torch.Tensor, owner: torch.Tensor, count: int) -> bool:
    """Detect when combining rounded row norms would lose the original magnitudes."""
    zero = values.new_zeros(count)
    maxima = zero.scatter_reduce(0, owner, values.detach(), reduce="amax")
    divisor = torch.where(maxima > 0, maxima, torch.ones_like(maxima))
    tiny = torch.finfo(values.dtype).tiny
    return bool((((values < tiny) | (values / divisor[owner] < tiny)) & (values != 0)).any())
