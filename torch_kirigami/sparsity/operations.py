"""In-place operations on parameter groups, independent of optimizer and regularizer policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..bindings import storage_key
from ..dep_graph.selection import Region, Selection, TensorRef, gather_region
from ..pruning.groups import ParameterGroup, group_equivalence_classes
from .values import group_bindings, group_values, scaled_product


def _region_index(tensor: torch.Tensor, region: Region) -> tuple[slice | torch.Tensor, ...]:
    """Build write indices for a region: slices for one interval per axis, otherwise a mesh."""
    if all(len(axis.intervals) == 1 for axis in region.axes):
        return tuple(slice(*axis.intervals[0]) for axis in region.axes)
    indices = [
        torch.tensor(tuple(axis), dtype=torch.long, device=tensor.device) for axis in region.axes
    ]
    return torch.meshgrid(*indices, indexing="ij")


def _write(
    tensor: torch.Tensor, index: tuple[slice | torch.Tensor, ...], values: torch.Tensor
) -> None:
    """Write prepared values at a prepared index; an empty index writes the whole tensor."""
    if not index:
        tensor.copy_(values)
    elif all(isinstance(axis, slice) for axis in index):
        tensor[index].copy_(values)
    else:
        tensor[index] = values


def _proves_nonoverlap(tensor: torch.Tensor) -> bool:
    """Return whether the strides prove that no two elements share memory; False is unproven."""
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
    """Reject groups that select a common coordinate of any parameter.

    A sweep over per-axis bounds finds candidate pairs, and exact region
    intersections decide whether a pair overlaps. Disjoint rows or columns
    therefore avoid quadratic pair comparisons, and general regions keep exact
    checks without constructing a large union.
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
    """Multiply the union of the selected parameter values by one factor in place.

    A coordinate selected by several groups is scaled once. Values change under
    `torch.no_grad()`. Gradients and optimizer state are not changed, so
    optimizer momentum can regrow values that were scaled down or set to zero.
    Call this outside a live forward and backward pass, normally after
    `optimizer.step()`. Every new value is computed and validated before any
    parameter changes, and completed writes are restored if a later write fails.

    Args:
        groups: Nonempty groups that share one dependency graph and device.
        factor: Finite, nonnegative Python number.

    Raises:
        TypeError: An item of `groups` is not a `ParameterGroup`.
        ValueError: `factor` is invalid; `groups` is empty or spans several
            graphs or devices; a selected parameter is not a dense real
            floating-point tensor; a selected tensor shares storage with another
            registered tensor, selected or not; or a selected or resulting value
            is nonfinite.
    """
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(factor)
        or factor < 0
    ):
        raise ValueError("factor must be finite and nonnegative")
    groups = tuple(groups)
    if len(groups) == 1:
        # group_values validates this existing union before reading any values.
        union = groups[0]
    else:
        # Validate the groups, graphs and devices before combining their coordinates.
        group_bindings(groups)
        union = ParameterGroup(groups[0].graph, tuple(s for g in groups for s in g.selections))
    values = group_values((union,))[0]
    _commit_values((union,), (values.to(torch.float64) * factor,))


def zero_groups_(groups: tuple[ParameterGroup, ...]) -> None:
    """Set the union of the selected parameter values to zero in place.

    This is `scale_groups_(groups, 0.0)`. It installs no persistent mask and
    does not change gradients or optimizer state, so later optimizer steps,
    including momentum, can regrow the zeroed values.

    Args:
        groups: Nonempty groups that share one dependency graph and device.

    Raises:
        TypeError: An item of `groups` is not a `ParameterGroup`.
        ValueError: The groups violate the requirements of `scale_groups_`.
    """
    scale_groups_(groups, 0.0)


@torch.no_grad()
def set_group_norms_(groups: tuple[ParameterGroup, ...], targets: tuple[int | float, ...]) -> None:
    """Rescale each group in place so that its values have the target L2 norm.

    Each group keeps its direction. Equivalent groups count once if their
    targets are equal; any other overlap between groups is rejected. An
    all-zero group can only receive target zero because it has no direction.
    Values change under `torch.no_grad()`, and every check runs before any
    parameter changes. Gradients and optimizer state are not changed, so
    momentum can move the norms again on later steps.

    Args:
        groups: Nonempty groups that share one dependency graph and device.
        targets: One finite, nonnegative Python number per group.

    Raises:
        TypeError: An item of `groups` is not a `ParameterGroup`.
        ValueError: The targets are invalid or do not match the groups in
            number, equivalent groups have different targets, groups overlap
            in any other way, an all-zero group has a positive target, or the
            groups violate the requirements of `scale_groups_`.
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
