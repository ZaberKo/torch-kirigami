"""Explicit parameter operations, independent of optimizer and regularizer policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..bindings import storage_key
from ..dep_graph.selection import Region, Selection, TensorRef, gather_region
from ..pruning.groups import ParameterGroup, group_equivalence_classes
from .values import group_bindings, group_values, scaled_product


def _region_index(tensor: torch.Tensor, region: Region) -> tuple[slice | torch.Tensor, ...]:
    """Allocate write indices before mutation, using slices for rectangles."""
    if all(len(axis.intervals) == 1 for axis in region.axes):
        return tuple(slice(*axis.intervals[0]) for axis in region.axes)
    indices = [
        torch.tensor(tuple(axis), dtype=torch.long, device=tensor.device) for axis in region.axes
    ]
    return torch.meshgrid(*indices, indexing="ij")


def _write(
    tensor: torch.Tensor, index: tuple[slice | torch.Tensor, ...], values: torch.Tensor
) -> None:
    """Write prepared values without constructing coordinate tensors."""
    if not index:
        tensor.copy_(values)
    elif all(isinstance(axis, slice) for axis in index):
        tensor[index].copy_(values)
    else:
        tensor[index] = values


def _proves_nonoverlap(tensor: torch.Tensor) -> bool:
    """Prove nonoverlap for dense, transposed and regularly strided layouts."""
    # Each stride must step past the address span of all smaller-stride axes.
    # Failure is not proof of overlap. The caller retains full-copy preparation.
    extent = 1
    for size, stride in sorted(
        zip(tensor.shape, tensor.stride(), strict=True), key=lambda item: item[1]
    ):
        if size <= 1:
            continue
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return True


@dataclass(frozen=True, eq=False)
class _PreparedWrite:
    """Hold one tensor assignment and the values required to undo it.

    Attributes:
        tensor: Original registered parameter to update in place.
        index: Prepared slices or coordinate tensors. Empty means the whole tensor.
        before: Copied values at these coordinates before any transaction writes.
        after: Prepared replacement values, independent of live parameter storage.
    """

    tensor: torch.Tensor
    index: tuple[slice | torch.Tensor, ...]
    before: torch.Tensor
    after: torch.Tensor


def _prepare_writes(
    groups: tuple[ParameterGroup, ...], vectors: tuple[torch.Tensor, ...]
) -> list[_PreparedWrite]:
    """Validate bindings and allocate every value and index without changing parameters."""
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
    prepared: list[_PreparedWrite] = []
    finite = []
    full_copies: dict[torch.Tensor, torch.Tensor] = {}
    for group, vector in zip(groups, vectors, strict=True):
        offset = 0
        for selection in group.selections:
            parameter = bindings[selection.tensor]
            if not _proves_nonoverlap(parameter) and parameter not in full_copies:
                original, edited = parameter.detach().clone(), parameter.detach().clone()
                full_copies[parameter] = edited
                prepared.append(_PreparedWrite(parameter, (), original, edited))
            for region in selection.regions:
                shape = tuple(len(axis) for axis in region.axes)
                count = math.prod(shape)
                values = vector[offset : offset + count].reshape(shape).to(parameter.dtype)
                offset += count
                # Validate the final dtype, including overflow introduced by casting.
                finite.append(torch.isfinite(values).all())
                index = _region_index(parameter, region)
                if parameter in full_copies:
                    # Preserve the original full-copy behavior when sparse writes
                    # cannot prove that other logical positions remain unchanged.
                    _write(full_copies[parameter], index, values)
                else:
                    original = gather_region(parameter.detach(), region).clone()
                    prepared.append(_PreparedWrite(parameter, index, original, values.clone()))
    if finite and not torch.stack(finite).all():
        raise ValueError("Parameter operation would produce nonfinite values")
    return prepared


def _commit_values(groups: tuple[ParameterGroup, ...], vectors: tuple[torch.Tensor, ...]) -> None:
    """Prepare all writes, revalidate the graph, then commit with rollback on failure."""
    prepared = _prepare_writes(groups, vectors)
    graph = groups[0].graph
    graph.validate()
    try:
        for edit in prepared:
            _write(edit.tensor, edit.index, edit.after)
    except (RuntimeError, ValueError):
        for edit in prepared:
            _write(edit.tensor, edit.index, edit.before)
        raise


def _require_disjoint(groups: tuple[ParameterGroup, ...]) -> None:
    """Sweep sparse axis bounds, using exact region checks for possible overlaps.

    Disjoint rows or columns avoid quadratic pair comparisons. General regions
    retain exact intersection checks without constructing a large union.
    """
    by_tensor: dict[TensorRef, list[Selection]] = {}
    for group in groups:
        for selection in group.selections:
            by_tensor.setdefault(selection.tensor, []).append(selection)
    for ref, selections in by_tensor.items():
        if len(selections) < 2:
            continue
        if not ref.shape:
            raise ValueError("Norm targets require disjoint groups")
        bounds = [
            tuple(
                (
                    min(r.axes[d].intervals[0][0] for r in s.regions),
                    max(r.axes[d].intervals[-1][1] for r in s.regions),
                )
                for d in range(len(ref.shape))
            )
            for s in selections
        ]
        # Axis choice only reduces comparisons. Exact intersections decide overlap.
        axis = min(
            range(len(ref.shape)),
            key=lambda d: sum(b[d][1] - b[d][0] for b in bounds) / ref.shape[d],
        )
        ordered = sorted(range(len(selections)), key=lambda i: bounds[i][axis])
        active: list[int] = []
        for index in ordered:
            start = bounds[index][axis][0]
            active = [i for i in active if bounds[i][axis][1] > start]
            for previous in active:
                if any(
                    not a.intersect(b).empty
                    for a in selections[index].regions
                    for b in selections[previous].regions
                ):
                    raise ValueError("Norm targets require disjoint groups")
            active.append(index)


@torch.no_grad()
def scale_groups_(groups: tuple[ParameterGroup, ...], factor: int | float) -> None:
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
    if not groups or any(not isinstance(group, ParameterGroup) for group in groups):
        raise ValueError("Expected nonempty ParameterGroup sequence")
    if len(groups) == 1:
        # group_values validates this existing union before reading any values.
        union = groups[0]
    else:
        # Validate all source graphs/devices before combining their coordinates.
        group_bindings(groups)
        union = ParameterGroup(groups[0].graph, tuple(s for g in groups for s in g.selections))
    values = group_values((union,))[0]
    _commit_values((union,), (values.to(torch.float64) * factor,))


def zero_groups_(groups: tuple[ParameterGroup, ...]) -> None:
    """Zero regions once; no persistent mask or optimizer-state edits are installed.

    Args:
        groups: Nonempty groups on one graph and device.

    Raises:
        ValueError: Groups do not share a graph or device, use unsupported
            tensors, or include distinct registered tensors that share storage.
    """
    scale_groups_(groups, 0.0)


@torch.no_grad()
def set_group_norms_(groups: tuple[ParameterGroup, ...], targets: tuple[int | float, ...]) -> None:
    """Rescale disjoint groups to explicit nonnegative L2 norms.

    Equivalent groups with identical targets count once; other overlaps are
    rejected. Zero vectors cannot acquire positive norms without a new direction.
    All validation precedes mutation. Momentum and algorithm progress are external.

    Args:
        groups: Nonempty groups on one graph and device.
        targets: Finite nonnegative L2 norm for each group.

    Raises:
        ValueError: Targets are invalid, groups overlap incompatibly, or a zero
            vector has a positive target.
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
    _require_disjoint(unique)
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
