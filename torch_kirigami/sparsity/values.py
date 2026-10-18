"""Live sparse parameter access below losses and mutation operations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.autograd.function import FunctionCtx, once_differentiable

from ..dep_graph.selection import IndexSet, TensorRef, gather_region
from ..pruning.groups import ParameterGroup


def group_bindings(
    groups: tuple[ParameterGroup, ...],
    *,
    refs: tuple[TensorRef, ...] | None = None,
) -> tuple[dict[TensorRef, torch.Tensor], torch.dtype]:
    """Validate live group ownership, device and dtype without gathering weights."""
    groups = tuple(groups)
    if not groups or any(not isinstance(g, ParameterGroup) for g in groups):
        raise ValueError("Expected nonempty ParameterGroup sequence")
    graph = groups[0].graph
    if any(g.graph is not graph for g in groups):
        raise ValueError("Groups must share one dependency graph")
    bindings = dict(graph.tensor_bindings())
    refs = (
        refs
        if refs is not None
        else tuple(dict.fromkeys(s.tensor for g in groups for s in g.selections))
    )
    parameters = [bindings[ref] for ref in refs]
    if len({p.device for p in parameters}) != 1:
        raise ValueError("Sparse operations require groups on one device")
    if any(not p.is_floating_point() or p.layout != torch.strided for p in parameters):
        raise ValueError("Sparse operations require dense real floating parameters")
    dtype = torch.float64 if any(p.dtype == torch.float64 for p in parameters) else torch.float32
    return bindings, dtype


def _gather_group(
    group: ParameterGroup, bindings: Mapping[TensorRef, torch.Tensor], dtype: torch.dtype
) -> torch.Tensor:
    """Read one group from bindings already validated for this operation."""
    return torch.cat(
        [
            gather_region(bindings[selection.tensor], region).reshape(-1).to(dtype)
            for selection in group.selections
            for region in selection.regions
        ]
    )


def group_values(groups: tuple[ParameterGroup, ...]) -> tuple[torch.Tensor, ...]:
    """Gather current dense real values on one device, preserving autograd.

    All groups belong to one graph. Distributed/sharded parameter gathering is
    unsupported. No values or autograd graphs are cached.
    """
    groups = tuple(groups)
    bindings, dtype = group_bindings(groups)
    result = tuple(_gather_group(group, bindings, dtype) for group in groups)
    if not torch.stack([torch.isfinite(v).all() for v in result]).all():
        raise ValueError("Sparse operations require finite selected values")
    return result


def scaled_product(*factors: torch.Tensor, divide: tuple[torch.Tensor, ...] = ()) -> torch.Tensor:
    """Combine floating factors without losing range in intermediate products.

    Mantissas stay near one; only the final exponent determines output range.
    Split ldexp to avoid a premature power-of-two intermediate overflow.
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
    """Return the nonzero scale and normalized length used by norm backward."""
    scale = values.detach().abs().amax(dim=dimensions, keepdim=True)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    scaled = values / divisor
    length = torch.linalg.vector_norm(scaled, dim=dimensions, keepdim=True)
    length = torch.where(length == 0, torch.ones_like(length), length)
    return divisor, length


class _StableNorm(torch.autograd.Function):
    """Differentiate normalized directions without tiny scaled intermediates."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, values: torch.Tensor, dimensions: tuple[int, ...], coefficient: float
    ) -> torch.Tensor:
        """Evaluate normalized L2 magnitudes while saving backward inputs."""
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
    """Compute scaled L2 with a stable gradient, including nonzero subnormals."""
    return _StableNorm.apply(values, tuple(range(values.ndim)), coefficient)


def _scaled_segments(
    values: torch.Tensor, owner: torch.Tensor, count: int
) -> tuple[torch.Tensor, ...]:
    """Compute stable per-owner scales and squared magnitudes."""
    zero = values.new_zeros(count)
    scales = zero.scatter_reduce(0, owner, values.detach(), reduce="amax")
    divisors = torch.where(scales == 0, torch.ones_like(scales), scales)
    scaled = values / divisors[owner]
    squared = zero.index_add(0, owner, scaled.square())
    root = torch.sqrt(torch.where(squared > 0, squared, torch.ones_like(squared)))
    return scaled, root, divisors, squared > 0


class _SegmentNorm(torch.autograd.Function):
    """Stable norms and directions for ragged groups of batched row norms."""

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
    """Return elementwise half squares, differentiating as gradient * values."""
    return _HalfSquared.apply(values)


class _SquaredNorm(torch.autograd.Function):
    """Aggregate weighted squares and preserve original factors for first derivatives."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, coefficients: tuple[float, ...], *values: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate a weighted squared norm from concatenated group values."""
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
    """Return half the weighted squared norm without per-element square underflow."""
    weights = (1.0,) * len(values) if coefficients is None else coefficients
    return _SquaredNorm.apply(weights, *values)


def _axis_sections(group: ParameterGroup) -> list[tuple[TensorRef, int, IndexSet]] | None:
    """Return whole-axis sections, or None when any region needs general gathering."""
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
    """Cache coordinate indices without retaining values or autograd graphs.

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


def reduction_plan(groups: tuple[ParameterGroup, ...]) -> ReductionPlan:
    """Batch whole-axis sections; retain the region path for irregular groups.

    The plan holds original-coordinate indices only. Each parameter/axis is
    reduced once, then small axis vectors are shared by all participating groups.
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
    bindings, _ = group_bindings(groups)
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


def _row_norm(values: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
    """Stable vector norms along axes, with zero subgradients at all-zero rows."""
    if not dimensions:
        return values.abs()
    return _StableNorm.apply(values, dimensions, 1.0)


def group_penalty(
    groups: tuple[ParameterGroup, ...], kind: str, plan: ReductionPlan
) -> torch.Tensor:
    """Evaluate the scalar batched L1/L2/squared-L2 objective."""
    bindings, dtype = group_bindings(groups, refs=plan.refs)
    contributions: list[torch.Tensor] = []
    tiny_squares = []
    for ref, axis, selected_positions, contribution_indices in plan.batches:
        tensor = bindings[ref].to(dtype)
        if selected_positions is not None:
            tensor = tensor.index_select(axis, selected_positions)
        dims = tuple(d for d in range(tensor.ndim) if d != axis)
        if kind == "l2":
            reduced = _row_norm(tensor, dims)
        else:
            if kind == "squared_l2":
                magnitude = tensor.detach().abs()
                tiny_squares.append(
                    ((magnitude != 0) & (magnitude < torch.finfo(dtype).tiny ** 0.5)).any()
                )
            values = tensor.abs() if kind == "l1" else half_squared(tensor)
            reduced = values.sum(dim=dims) if dims else values
        contributions.append(reduced.index_select(0, contribution_indices))
    for number in plan.fallback:
        values = _gather_group(groups[number], bindings, dtype)
        if kind == "squared_l2":
            magnitude = values.detach().abs()
            tiny_squares.append(
                ((magnitude != 0) & (magnitude < torch.finfo(dtype).tiny ** 0.5)).any()
            )
        if kind == "l2":
            value = stable_norm(values)
        elif kind == "l1":
            value = values.abs().sum()
        else:
            value = squared_norm(values)
        contributions.append(value.reshape(1))
    values = torch.cat(contributions)
    if not torch.isfinite(values).all():
        raise ValueError("Sparse operations require finite selected values and penalties")
    if tiny_squares and torch.stack(tiny_squares).any():
        # A sum can be representable although every individual square rounds
        # to zero. Reduce each original group magnitude before squaring it.
        gathered = torch.cat([_gather_group(group, bindings, dtype) for group in groups])
        return squared_norm(gathered)
    if kind != "l2":
        return values.sum()

    owner = plan.owners
    zero = values.new_zeros(len(groups))
    maxima = zero.scatter_reduce(0, owner, values.detach(), reduce="amax")
    divisor = torch.where(maxima > 0, maxima, torch.ones_like(maxima))
    unstable = ((values != 0) & (values.abs() < torch.finfo(dtype).tiny)) | (
        (values != 0) & (values / divisor[owner] < torch.finfo(dtype).tiny)
    )
    if unstable.any():
        # Rounded subnormal row norms lose the relative magnitudes of rows.
        # Recompute directly from original values, and
        # differentiate the direction rather than multiply/divide tiny scales.
        return torch.stack(
            [stable_norm(_gather_group(group, bindings, dtype)) for group in groups]
        ).sum()
    return _SegmentNorm.apply(values, owner, len(groups)).sum()
