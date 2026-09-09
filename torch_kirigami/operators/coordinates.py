"""Shared original/compact coordinate normalization for operator contracts."""

from ..selection import IndexSet


def retained_indices(impact, axis):
    """Return retained original axis coordinates without reordering."""
    return IndexSet.span(0, axis.tensor.shape[axis.dim]).subtract(
        impact.selection(axis.tensor).fully_selected_indices(axis.dim)
    )


def narrow_index(shape, dim, start, length):
    """Convert narrow's signed start into an equivalent positive Python slice."""
    dim %= len(shape)
    if start < 0:
        start += shape[dim]
    if start < 0 or length < 0 or start + length > shape[dim]:
        raise ValueError("narrow interval is outside the compact axis")
    index = [slice(None)] * len(shape)
    index[dim] = slice(start, start + length)
    return tuple(index)
