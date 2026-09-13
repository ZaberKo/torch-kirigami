"""Explicit parameter operations, independent of optimizer and regularizer policy."""

import math

import torch

from ..bindings import storage_key
from ..pruning.groups import ParameterGroup, group_equivalence_classes
from .values import group_bindings, group_values, scaled_product


def _write(tensor, region, values):
    if not region.axes:
        tensor.copy_(values)
    else:
        indices = [torch.tensor(tuple(a), device=tensor.device) for a in region.axes]
        tensor[torch.meshgrid(*indices, indexing="ij")] = values


def _commit_values(groups, vectors):
    graph = groups[0].graph
    bindings = dict(graph.tensor_bindings())
    storage = {}
    for tensor in bindings.values():
        key = storage_key(tensor)
        if key is not None:
            storage.setdefault(key, set()).add(id(tensor))
    # Even an unselected alias could be changed by committing a selected tensor.
    # Distinct registrations of the same Parameter remain supported.
    if any(
        len(storage.get(storage_key(bindings[s.tensor]), ())) > 1
        for group in groups
        for s in group.selections
    ):
        raise ValueError("Parameter operations do not support distinct tensors sharing storage")
    originals, prepared = {}, {}
    for group, vector in zip(groups, vectors, strict=True):
        offset = 0
        for selection in group.selections:
            parameter = bindings[selection.tensor]
            if parameter not in originals:
                originals[parameter] = parameter.detach().clone()
                prepared[parameter] = parameter.detach().clone()
            for region in selection.regions:
                shape = tuple(len(axis) for axis in region.axes)
                count = math.prod(shape)
                values = vector[offset : offset + count].reshape(shape).to(parameter.dtype)
                offset += count
                if not torch.isfinite(values).all():
                    raise ValueError("Parameter operation would produce nonfinite values")
                _write(prepared[parameter], region, values)
    graph.validate()
    try:
        for parameter, value in prepared.items():
            parameter.copy_(value)
    except (RuntimeError, ValueError):
        for parameter, value in originals.items():
            parameter.copy_(value)
        raise


@torch.no_grad()
def scale_groups_(groups, factor):
    """Scale a union once, keeping optimizer state and gradients untouched.

    Args:
        groups: Nonempty groups belonging to one graph/device.
        factor: Finite nonnegative scalar. Zero allows subsequent regrowth.

    Call outside a live forward/backward graph, normally after a successful
    optimizer step. All values are prepared before committing changes. Distinct
    tensors sharing storage are rejected, including unselected registered aliases.
    """
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or factor < 0
    ):
        raise ValueError("factor must be finite and nonnegative")
    groups = tuple(groups)
    group_bindings(groups)
    union = ParameterGroup(groups[0].graph, tuple(s for g in groups for s in g.selections))
    values = group_values((union,))[0]
    _commit_values((union,), (values.to(torch.float64) * factor,))


def zero_groups_(groups):
    """Zero regions once; no persistent mask or optimizer-state edits are installed."""
    scale_groups_(groups, 0.0)


@torch.no_grad()
def set_group_norms_(groups, targets):
    """Rescale disjoint groups to explicit nonnegative L2 norms.

    Equivalent groups with identical targets count once; other overlaps are
    rejected. Zero vectors cannot acquire positive norms without a new direction.
    All validation precedes mutation. Momentum and algorithm progress are external.
    """
    groups, targets = tuple(groups), tuple(targets)
    if len(groups) != len(targets) or any(
        isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0
        for t in targets
    ):
        raise ValueError("Expected one finite nonnegative target per group")
    classes = group_equivalence_classes(groups)
    unique = tuple(groups[indices[0]] for indices in classes)
    unique_targets = []
    for indices in classes:
        values = [targets[i] for i in indices]
        if len(set(values)) != 1:
            raise ValueError("Equivalent groups have conflicting targets")
        unique_targets.append(values[0])
    values = group_values(unique)
    for i, left in enumerate(unique):
        for right in unique[i + 1 :]:
            for a in left.selections:
                for b in right.selections:
                    if a.tensor == b.tensor and a.subtract(b).count != a.count:
                        raise ValueError("Norm targets require disjoint groups")
    projected = []
    for vector, target in zip(values, unique_targets, strict=True):
        # Normalize in two stages, without forming target/norm. That ratio can
        # overflow for nonzero subnormals even when every final value is finite.
        vector = vector.to(torch.float64)
        scale = vector.abs().amax()
        if scale == 0 and target != 0:
            raise ValueError("Cannot give a zero group a positive norm")
        if target == 0:
            projected.append(torch.zeros_like(vector))
        else:
            direction = vector / scale
            projected.append(
                scaled_product(
                    vector,
                    vector.new_tensor(target),
                    divide=(scale, torch.linalg.vector_norm(direction)),
                )
            )
    _commit_values(unique, projected)
