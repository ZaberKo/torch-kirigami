"""Live sparse parameter access below losses and mutation operations."""

import torch

from ..pruning.groups import ParameterGroup
from ..regions import gather_region


def group_values(groups):
    """Gather current dense real values on one device, preserving autograd.

    All groups belong to one graph. Distributed/sharded parameter gathering is
    unsupported. No values or autograd graphs are cached.
    """
    groups = tuple(groups)
    if not groups or any(not isinstance(g, ParameterGroup) for g in groups):
        raise ValueError("Expected nonempty ParameterGroup sequence")
    graph = groups[0].graph
    if any(g.graph is not graph for g in groups):
        raise ValueError("Groups must share one dependency graph")
    bindings = dict(graph.tensor_bindings())
    parameters = [bindings[s.tensor] for g in groups for s in g.selections]
    if len({p.device for p in parameters}) != 1:
        raise ValueError("Sparse operations require groups on one device")
    if any(not p.is_floating_point() or p.layout != torch.strided for p in parameters):
        raise ValueError("Sparse operations require dense real floating parameters")
    dtype = torch.float64 if any(p.dtype == torch.float64 for p in parameters) else torch.float32
    result = tuple(
        torch.cat(
            [
                gather_region(bindings[s.tensor], region).reshape(-1).to(dtype)
                for s in group.selections
                for region in s.regions
            ]
        )
        for group in groups
    )
    if any(not torch.isfinite(v).all() for v in result):
        raise ValueError("Sparse operations require finite selected values")
    return result


def stable_norm(values):
    """Compute L2 with scaling and the zero subgradient at the origin."""
    scale = values.detach().abs().amax()
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.linalg.vector_norm(values / divisor) * divisor
