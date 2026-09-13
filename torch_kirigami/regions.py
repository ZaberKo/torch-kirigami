"""Tensor access for Cartesian regions, shared by scoring and sparse training."""

import torch


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
