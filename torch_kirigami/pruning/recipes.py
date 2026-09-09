"""Pure recipe coordinate and memory-layout checks shared across lifecycles."""

import torch

from ..selection import IndexSet, Region, Selection, full_region
from .types import CoordinateSegment, PlanningError


def compact_stride(shape, memory_format):
    """Calculate supported dense strides without allocating a tensor."""
    if memory_format == "contiguous":
        order = tuple(reversed(range(len(shape))))
    elif memory_format == "channels_last" and len(shape) == 4:
        order = (1, 3, 2, 0)
    elif memory_format == "channels_last_3d" and len(shape) == 5:
        order = (1, 4, 3, 2, 0)
    else:
        raise ValueError("Unsupported compact memory format")
    stride, result = 1, [0] * len(shape)
    for dim in order:
        result[dim] = stride
        stride *= max(1, shape[dim])
    return tuple(result)


def memory_format(tensor):
    """Preserve an unambiguous channels-last layout; gather other layouts densely."""
    if not tensor.is_contiguous():
        if tensor.ndim == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
            return "channels_last"
        if tensor.ndim == 5 and tensor.is_contiguous(memory_format=torch.channels_last_3d):
            return "channels_last_3d"
    return "contiguous"


def validate_recipe(recipe, impact):
    """Verify retained regions against the complete original-coordinate selection."""
    ref = recipe.tensor
    if ref.kind not in ("parameter", "buffer") or not recipe.segments:
        raise PlanningError("Recipes must retain nonempty registered tensor segments")
    if not 0 <= recipe.concat_dim < max(1, len(ref.shape)):
        raise PlanningError("Invalid concatenation dimension")
    if len(recipe.segments) > 1:
        previous_end = -1
        for segment in recipe.segments:
            axis = segment.axes[recipe.concat_dim]
            if not axis or axis.intervals[0][0] < previous_end:
                raise PlanningError("Recipes must preserve original position order")
            previous_end = axis.intervals[-1][1]
    kept = Selection(ref, recipe.segments)
    if sum(Selection(ref, (r,)).count for r in recipe.segments) != kept.count:
        raise PlanningError("Recipe segments overlap")
    full = Selection(ref, (full_region(ref.shape),))
    if kept != full.subtract(impact.selection(ref)):
        raise PlanningError("Recipe retained coordinates disagree with the joint Impact")
    shapes = [tuple(map(len, r.axes)) for r in recipe.segments]
    if any(
        any(
            a != b
            for d, (a, b) in enumerate(zip(shapes[0], shape, strict=True))
            if d != recipe.concat_dim
        )
        for shape in shapes[1:]
    ):
        raise PlanningError("Recipe segments cannot concatenate")


def coordinate_mapping(recipe):
    """Describe each retained segment in original and compact coordinates."""
    offset, result = 0, []
    for region in recipe.segments:
        axes = [IndexSet.span(0, len(a)) for a in region.axes]
        if axes:
            axes[recipe.concat_dim] = axes[recipe.concat_dim].shift(offset)
            offset += len(region.axes[recipe.concat_dim])
        result.append(CoordinateSegment(region, Region(tuple(axes))))
    return tuple(result)


def same_mapping(left, right):
    """Compare position mappings independently of segment boundaries."""
    # Different segment boundaries can encode the same mapping. Compare each
    # overlap's per-axis rank offsets, never just the removed set or output shape.
    if (
        left.tensor != right.tensor
        or left.shape != right.shape
        or Selection(left.tensor, left.segments) != Selection(right.tensor, right.segments)
    ):
        return False
    for a in coordinate_mapping(left):
        for b in coordinate_mapping(right):
            intersection = a.source.intersect(b.source)
            if intersection.empty:
                continue
            for dim, indices in enumerate(intersection.axes):
                for start, stop in indices.intervals:
                    cuts = {start, stop}
                    for source in (a.source.axes[dim], b.source.axes[dim]):
                        cuts.update(
                            p for interval in source.intervals for p in interval if start < p < stop
                        )
                    for point in sorted(cuts)[:-1]:
                        mapped = []
                        for segment in (a, b):
                            rank = len(segment.source.axes[dim].intersect(IndexSet.span(0, point)))
                            mapped.append(next(iter(segment.destination.axes[dim])) + rank)
                        if mapped[0] != mapped[1]:
                            return False
    return True
