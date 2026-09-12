"""Tensor access for Cartesian regions, shared by scoring and sparse training."""

import torch


def gather_region(tensor, region):
    """Gather a region without a full-tensor mask, preserving autograd."""
    for dim, indices in enumerate(region.axes):
        if len(indices) != tensor.shape[dim]:
            index = torch.tensor(tuple(indices), dtype=torch.long, device=tensor.device)
            tensor = tensor.index_select(dim, index)
    return tensor
