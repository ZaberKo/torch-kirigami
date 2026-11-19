"""Coordinate and index normalization shared by operator rules."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..contracts import Impact
from ..operation import argument
from ..selection import AxisRef, IndexSet


def retained_indices(impact: Impact, axis: AxisRef) -> IndexSet:
    """Return retained original axis coordinates without reordering."""
    return IndexSet.span(0, axis.tensor.shape[axis.dim]).subtract(
        impact.selection(axis.tensor).fully_selected_indices(axis.dim)
    )


def expand_index(index: object, rank: int) -> tuple[object, ...]:
    """Expand an ellipsis and append trailing full slices to a bracket index.

    Every item other than None and an ellipsis consumes one axis.

    Raises:
        ValueError: The index has more than one ellipsis or consumes more axes
            than `rank`.
    """
    items = index if isinstance(index, tuple) else (index,)
    missing = rank - sum(item is not None and item is not Ellipsis for item in items)
    ellipses = sum(item is Ellipsis for item in items)
    if ellipses > 1:
        raise ValueError("Use at most one ellipsis in a bracket index")
    if missing < 0:
        raise ValueError(f"Index consumes more axes than the tensor rank of {rank}")
    full = (slice(None),) * missing
    if not ellipses:
        return (*items, *full)
    return tuple(part for item in items for part in (full if item is Ellipsis else (item,)))


def narrow_index(shape: tuple[int, ...], dim: int, start: int, length: int) -> tuple[slice, ...]:
    """Convert `narrow` arguments into an equivalent index with a nonnegative slice.

    Raises:
        ValueError: The interval lies outside the axis.
    """
    dim %= len(shape)
    if start < 0:
        start += shape[dim]
    if start < 0 or length < 0 or start + length > shape[dim]:
        raise ValueError("narrow interval is outside the compact axis")
    index = [slice(None)] * len(shape)
    index[dim] = slice(start, start + length)
    return tuple(index)


def slice_index(
    target: object,
    shape: tuple[int, ...],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> tuple[object, ...]:
    """Read a bracket or narrow call as an expanded basic index."""
    if target in ("narrow", torch.narrow):
        dim, start, length = (
            argument(args, kwargs, name, position, target=target)
            for position, name in enumerate(("dim", "start", "length"), 1)
        )
        if not all(type(value) is int for value in (dim, start, length)):
            raise ValueError("narrow requires static integer arguments")
        return narrow_index(shape, dim, start, length)
    return expand_index(args[1], len(shape))
