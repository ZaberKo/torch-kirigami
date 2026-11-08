"""Shared original/compact coordinate normalization for operator contracts."""

from __future__ import annotations

from ..contracts import Impact
from ..selection import AxisRef, IndexSet


def retained_indices(impact: Impact, axis: AxisRef) -> IndexSet:
    """Return retained original axis coordinates without reordering."""
    return IndexSet.span(0, axis.tensor.shape[axis.dim]).subtract(
        impact.selection(axis.tensor).fully_selected_indices(axis.dim)
    )


def expand_index(index: object, rank: int) -> tuple:
    """Expand ellipsis and trailing slices for indices that consume one axis each."""
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
    """Convert narrow's signed start into an equivalent positive Python slice."""
    dim %= len(shape)
    if start < 0:
        start += shape[dim]
    if start < 0 or length < 0 or start + length > shape[dim]:
        raise ValueError("narrow interval is outside the compact axis")
    index = [slice(None)] * len(shape)
    index[dim] = slice(start, start + length)
    return tuple(index)
