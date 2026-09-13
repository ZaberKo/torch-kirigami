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


def reduction_plan(groups):
    """Batch whole-axis sections; retain the region path for irregular groups.

    The plan holds original-coordinate indices only. Each parameter/axis is
    reduced once, then small axis vectors are shared by all participating groups.
    """
    batches, fallback = {}, []
    for number, group in enumerate(groups):
        sections = []
        for selection in group.selections:
            ref = selection.tensor
            for region in selection.regions:
                partial = [
                    d
                    for d, (indices, n) in enumerate(zip(region.axes, ref.shape, strict=True))
                    if len(indices) != n
                ]
                if len(partial) > 1 or not ref.shape:
                    break
                axis = partial[0] if partial else 0
                sections.append((ref, axis, region.axes[axis]))
            else:
                continue
            break
        else:
            for ref, axis, indices in sections:
                entries = batches.setdefault((ref, axis), [])
                entries.extend((number, i) for i in indices)
            continue
        fallback.append(number)
    return tuple((ref, axis, tuple(entries)) for (ref, axis), entries in batches.items()), tuple(
        fallback
    )


def _row_norm(values, dimensions):
    """Stable vector norms along axes, with zero subgradients at all-zero rows."""
    if not dimensions:
        return values.abs()
    scale = values.detach().abs().amax(dim=dimensions, keepdim=True)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.linalg.vector_norm(values / divisor, dim=dimensions) * divisor.squeeze(dimensions)


def group_penalties(groups, kind, plan):
    """Evaluate batched L1/L2/squared-L2 penalties without per-group weight gathers."""
    bindings = dict(groups[0].graph.tensor_bindings())
    parameters = [bindings[s.tensor] for g in groups for s in g.selections]
    if len({p.device for p in parameters}) != 1 or any(
        not p.is_floating_point() or p.layout != torch.strided for p in parameters
    ):
        raise ValueError("Sparse operations require dense real parameters on one device")
    dtype = torch.float64 if any(p.dtype == torch.float64 for p in parameters) else torch.float32
    contributions, owners = [], []
    batches, fallback = plan
    for ref, axis, entries in batches:
        tensor = bindings[ref].to(dtype)
        selected = sorted({index for _, index in entries})
        if len(selected) != tensor.shape[axis]:
            tensor = tensor.index_select(axis, torch.tensor(selected, device=tensor.device))
        positions = {old: new for new, old in enumerate(selected)}
        dims = tuple(d for d in range(tensor.ndim) if d != axis)
        if kind == "l2":
            reduced = _row_norm(tensor, dims)
        else:
            values = tensor.abs() if kind == "l1" else tensor.square() * 0.5
            reduced = values.sum(dim=dims) if dims else values
        index = torch.tensor([positions[i] for _, i in entries], device=tensor.device)
        contributions.append(reduced.index_select(0, index))
        owners.extend(number for number, _ in entries)
    for number in fallback:
        values = group_values((groups[number],))[0].to(dtype)
        value = (
            stable_norm(values)
            if kind == "l2"
            else values.abs().sum()
            if kind == "l1"
            else values.square().sum() * 0.5
        )
        contributions.append(value.reshape(1))
        owners.append(number)
    values = torch.cat(contributions)
    if not torch.isfinite(values).all():
        raise ValueError("Sparse operations require finite selected values and penalties")
    owner = torch.tensor(owners, device=values.device)
    zero = values.new_zeros(len(groups))
    if kind != "l2":
        return zero.index_add(0, owner, values)
    # Scales are numerical guards, not differentiable statistics. Reduction is
    # over per-axis norms, never a dense per-element group-label tensor.
    scales = zero.scatter_reduce(0, owner, values.detach(), reduce="amax")
    divisors = torch.where(scales == 0, torch.ones_like(scales), scales)
    squared = zero.index_add(0, owner, (values / divisors[owner]).square())
    nonzero = squared > 0
    root = torch.sqrt(torch.where(nonzero, squared, torch.ones_like(squared)))
    return torch.where(nonzero, root, torch.zeros_like(root)) * divisors
