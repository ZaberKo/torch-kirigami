"""Cartesian region shapes and tensor access shared across analysis and execution."""

import torch


def concatenated_shape(regions, dim):
    """Compute a concatenation shape from retained Cartesian regions.

    Regions must have equal ranks and matching sizes on every non-concatenated
    axis. One scalar region represents a scalar; scalars cannot be concatenated.
    No tensor data or pruning policy is involved.

    Args:
        regions: Nonempty sequence of retained regions in concatenation order.
        dim: Nonnegative concatenation axis, or zero for a single scalar.

    Returns:
        The resulting shape as a tuple of integers.

    Raises:
        ValueError: The regions do not describe a valid concatenation.
    """
    shapes = [tuple(len(axis) for axis in region.axes) for region in regions]
    if not shapes:
        raise ValueError("Concatenation requires at least one region")
    rank = len(shapes[0])
    if type(dim) is not int or not 0 <= dim < max(1, rank):
        raise ValueError("Invalid concatenation dimension")
    if any(len(shape) != rank for shape in shapes):
        raise ValueError("Region ranks differ")
    if rank == 0:
        if len(shapes) != 1:
            raise ValueError("Scalar concatenation requires exactly one region")
        return ()
    result = list(shapes[0])
    for shape in shapes[1:]:
        if any(shape[axis] != result[axis] for axis in range(rank) if axis != dim):
            raise ValueError("Region sizes differ on a non-concatenated axis")
        result[dim] += shape[dim]
    return tuple(result)


def gather_region(tensor, region):
    """Gather a region without a full-tensor mask, preserving autograd."""
    # Shrink the most selective axis first to bound intermediate allocation.
    order = sorted(range(tensor.ndim), key=lambda d: len(region.axes[d]) / max(1, tensor.shape[d]))
    for dim in order:
        indices = region.axes[dim]
        if len(indices) != tensor.shape[dim]:
            index = torch.tensor(tuple(indices), dtype=torch.long, device=tensor.device)
            tensor = tensor.index_select(dim, index)
    return tensor
