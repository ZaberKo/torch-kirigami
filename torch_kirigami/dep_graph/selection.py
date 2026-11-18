"""Finite index sets and unions of Cartesian regions, in original coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from math import prod
from typing import TypeVar

import torch

from ..errors import AnalysisLimitError
from ..settings import config


@dataclass(frozen=True)
class IndexSet:
    """A normalized union of nonnegative, half-open integer intervals.

    Construction sorts the intervals, drops empty ones, and merges overlapping
    or adjacent ones. It never allocates masks.

    Attributes:
        intervals: Sorted, disjoint pairs with inclusive starts and exclusive stops.

    Raises:
        ValueError: A bound is not a nonnegative integer or a stop precedes its start.
        torch_kirigami.AnalysisLimitError: The normalized set has more intervals
            than the active `max_parts` limit.
    """

    intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        merged: list[tuple[int, int]] = []
        for lo, hi in sorted(self.intervals):
            if (
                not isinstance(lo, int)
                or not isinstance(hi, int)
                or isinstance(lo, bool)
                or isinstance(hi, bool)
                or lo < 0
                or hi < lo
            ):
                raise ValueError("Invalid index interval")
            if lo == hi:
                continue
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        if len(merged) > config.max_parts:
            raise AnalysisLimitError("Too many index intervals")
        object.__setattr__(self, "intervals", tuple(merged))

    @classmethod
    def of(cls, indices: Iterable[int]) -> IndexSet:
        """Construct a set from individual indices, discarding order and duplicates.

        Args:
            indices: Nonnegative Python integers; booleans are rejected.

        Returns:
            A normalized interval set.

        Raises:
            ValueError: An index is negative or is not a Python integer.
            torch_kirigami.AnalysisLimitError: The normalized set exceeds the interval limit.
        """
        values = []
        for i in indices:
            if not isinstance(i, int) or isinstance(i, bool) or i < 0:
                raise ValueError("Indices must be nonnegative Python integers")
            values.append((i, i + 1))
        return cls(tuple(values))

    @classmethod
    def span(cls, start: int, stop: int) -> IndexSet:
        """Construct the half-open interval `[start, stop)`.

        Args:
            start: Inclusive first index.
            stop: Exclusive end index.

        Returns:
            A set containing the interval, or an empty set if `start == stop`.

        Raises:
            ValueError: A bound is not a nonnegative integer or `stop < start`.
        """
        return cls(((start, stop),))

    def __bool__(self) -> bool:
        return bool(self.intervals)

    def __len__(self) -> int:
        return sum(b - a for a, b in self.intervals)

    def __iter__(self) -> Iterator[int]:
        for a, b in self.intervals:
            yield from range(a, b)

    def __contains__(self, index: object) -> bool:
        """Check an integer position without expanding interval elements."""
        if type(self) is not IndexSet or type(index) is not int:
            # Preserve iterable equality and extension behavior outside native ints.
            return any(value == index for value in self)
        for start, stop in self.intervals:
            if type(start) is not int or type(stop) is not int:
                return any(value == index for value in self)
            if index < start:
                return False
            if index < stop:
                return True
        return False

    def union(self, other: IndexSet) -> IndexSet:
        """Return indices contained in either set.

        Args:
            other: Set to combine with this set.

        Returns:
            The normalized union.

        Raises:
            torch_kirigami.AnalysisLimitError: The union exceeds the interval limit.
        """
        if type(self) is type(other) is IndexSet:
            if not self.intervals or self.intervals == other.intervals:
                return other
            if not other.intervals:
                return self
        return IndexSet(self.intervals + other.intervals)

    def intersect(self, other: IndexSet) -> IndexSet:
        """Return indices contained in both sets.

        Args:
            other: Set to intersect with this set.

        Returns:
            The normalized intersection.

        Raises:
            torch_kirigami.AnalysisLimitError: The intersection exceeds the interval limit.
        """
        if type(self) is type(other) is IndexSet:
            if not self.intervals or self.intervals == other.intervals:
                return self
            if not other.intervals:
                return other
        result, i, j = [], 0, 0
        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i]
            c, d = other.intervals[j]
            if max(a, c) < min(b, d):
                result.append((max(a, c), min(b, d)))
            if b <= d:
                i += 1
            else:
                j += 1
        return IndexSet(tuple(result))

    def subtract(self, other: IndexSet) -> IndexSet:
        """Return indices in this set that are absent from another set.

        Args:
            other: Indices to remove.

        Returns:
            The normalized difference.

        Raises:
            torch_kirigami.AnalysisLimitError: The difference exceeds the interval limit.
        """
        if type(self) is type(other) is IndexSet:
            if not self.intervals or not other.intervals:
                return self
            if self.intervals == other.intervals:
                return IndexSet()
        result = []
        right = 0
        for lo, hi in self.intervals:
            cursor = lo
            while right < len(other.intervals):
                a, b = other.intervals[right]
                if b <= cursor:
                    right += 1
                    continue
                if a >= hi:
                    break
                if cursor < a:
                    result.append((cursor, a))
                cursor = max(cursor, b)
                if b >= hi:
                    break  # This right interval may also overlap the next left interval.
                right += 1
            if cursor < hi:
                result.append((cursor, hi))
        return IndexSet(tuple(result))

    def shift(self, offset: int) -> IndexSet:
        """Translate every interval bound by an offset.

        Args:
            offset: Integer amount added to each start and stop.

        Returns:
            The translated set.

        Raises:
            ValueError: The offset makes a bound negative or produces a non-integer bound.
            TypeError: The offset cannot be added to interval bounds.
        """
        return IndexSet(tuple((a + offset, b + offset) for a, b in self.intervals))

    def __repr__(self) -> str:
        return "IndexSet(" + ", ".join(f"{a}:{b}" for a, b in self.intervals) + ")"


@dataclass(frozen=True)
class Region:
    """A Cartesian product of index sets in tensor-axis order.

    Attributes:
        axes: One IndexSet per physical tensor axis. An empty tuple denotes
            the single coordinate of a scalar tensor.

    Raises:
        TypeError: An axis is not an IndexSet.
    """

    axes: tuple[IndexSet, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if any(not isinstance(axis, IndexSet) for axis in axes):
            raise TypeError("Region axes must be IndexSet instances")
        object.__setattr__(self, "axes", axes)

    @property
    def empty(self) -> bool:
        """Whether some axis is empty, which makes the Cartesian product empty."""
        return any(not a for a in self.axes)

    def intersect(self, other: Region) -> Region:
        """Return the Cartesian intersection.

        Args:
            other: A region of the same rank.

        Returns:
            The axis-wise intersection, which may be empty.

        Raises:
            ValueError: The regions have different ranks.
        """
        if len(self.axes) != len(other.axes):
            raise ValueError("Region rank mismatch")
        return Region(tuple(a.intersect(b) for a, b in zip(self.axes, other.axes, strict=False)))

    def subtract(self, other: Region) -> tuple[Region, ...]:
        """Subtract a region and return disjoint Cartesian pieces.

        Args:
            other: A region of the same rank.

        Returns:
            Nonoverlapping pieces covering exactly the difference.

        Raises:
            ValueError: Region ranks differ.
        """
        overlap = self.intersect(other)
        if overlap.empty:
            return (self,)
        prefix = list(self.axes)
        result = []
        for axis, inside in enumerate(overlap.axes):
            outside = prefix[axis].subtract(inside)
            if outside:
                piece = prefix.copy()
                piece[axis] = outside
                result.append(Region(tuple(piece)))
            prefix[axis] = inside
        return tuple(result)


def full_region(shape: tuple[int, ...]) -> Region:
    """Return the complete coordinate domain for a tensor shape."""
    return Region(tuple(IndexSet.span(0, n) for n in shape))


def normalize(regions: Iterable[Region]) -> tuple[Region, ...]:
    """Build a disjoint region union and coalesce compatible rectangles.

    Args:
        regions: Cartesian regions in a common coordinate system.

    Returns:
        Sorted, nonoverlapping regions with empty pieces removed.

    Raises:
        torch_kirigami.AnalysisLimitError: The intermediate region representation exceeds its limit.
    """
    result: list[Region] = []
    for region in sorted(set(regions), key=lambda r: tuple(a.intervals for a in r.axes)):
        if region.empty:
            continue
        pending = [region]
        for old in result:
            pending = [piece for item in pending for piece in item.subtract(old)]
            if len(pending) + len(result) > config.max_parts:
                raise AnalysisLimitError("Too many Cartesian regions")
        result.extend(pending)
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(result):
            for j in range(i + 1, len(result)):
                right = result[j]
                different = [
                    k for k, (a, b) in enumerate(zip(left.axes, right.axes, strict=False)) if a != b
                ]
                if len(different) == 1:
                    axis = different[0]
                    axes = list(left.axes)
                    axes[axis] = axes[axis].union(right.axes[axis])
                    result[i] = Region(tuple(axes))
                    del result[j]
                    changed = True
                    break
            if changed:
                break
    return tuple(sorted(result, key=lambda r: tuple(a.intervals for a in r.axes)))


@dataclass(frozen=True)
class TensorRef:
    """The identity and original shape of a tensor in an analysis snapshot.

    Attributes:
        id: Snapshot-qualified identity.
        shape: Original physical dimensions.
        kind: Tensor category: `parameter`, `buffer`, `input`, or `value` for
            other captured values.
        paths: Original registered paths, including aliases, for parameters
            and buffers.

    Raises:
        ValueError: `id` is empty, a dimension is not a nonnegative integer, or
            `kind` or a path is not a string.
    """

    id: str
    shape: tuple[int, ...]
    kind: str = "value"
    paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shape, paths = tuple(self.shape), tuple(self.paths)
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Tensor identity must be a nonempty string")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape):
            raise ValueError("Tensor dimensions must be nonnegative integers")
        if (
            not isinstance(self.kind, str)
            or not self.kind
            or any(not isinstance(path, str) for path in paths)
        ):
            raise ValueError("Tensor kind and paths must be strings")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "paths", paths)

    def portable(self) -> TensorRef:
        """Return a copy without the graph UUID prefix, for saved-plan queries.

        Portable labels do not establish ownership of a live dependency graph.

        Returns:
            A reference whose `id` omits the snapshot UUID, or this reference
            if its `id` has no UUID prefix.
        """
        prefix, separator, suffix = self.id.partition(":")
        if separator and len(prefix) == 32 and all(c in "0123456789abcdef" for c in prefix):
            return replace(self, id=suffix)
        return self

    def axis(self, dim: int) -> AxisRef:
        """Refer to a physical axis, normalizing negative dimension indices.

        Args:
            dim: Axis index in the original tensor shape.

        Returns:
            An `AxisRef` with a nonnegative dimension index.

        Raises:
            TypeError: `dim` is not an integer.
            IndexError: `dim` is outside the tensor rank.
        """
        return AxisRef(self, dim)

    def select(self, regions: Iterable[Region]) -> Selection:
        """Construct a removal selection from Cartesian regions in original coordinates.

        Args:
            regions: Regions with one index set per tensor axis.

        Returns:
            A normalized selection on this tensor.

        Raises:
            ValueError: A region rank differs from the tensor rank.
            IndexError: A region exceeds the original shape.
            torch_kirigami.AnalysisLimitError: Normalization exceeds the region limit.
        """
        return Selection(self, tuple(regions))


def resolve_reference(query: TensorRef, references: Iterable[TensorRef]) -> TensorRef:
    """Find the reference matching a live or portable label.

    A reference matches if its portable ID equals the query's portable ID, or if
    the two share a registered path. A `KeyError` means the label is unknown to
    this catalog, which differs from a known but unaffected tensor.

    Raises:
        TypeError: `query` is not a `TensorRef`.
        KeyError: No reference matches the label.
        ValueError: A reference matches but its shape, kind, or paths disagree.
    """
    if not isinstance(query, TensorRef):
        raise TypeError("Expected a TensorRef")
    label = query.portable()
    for ref in references:
        if ref.portable().id == label.id or (ref.paths and set(ref.paths) & set(label.paths)):
            if (ref.shape, ref.kind, ref.paths) != (label.shape, label.kind, label.paths):
                raise ValueError("Query does not match the original tensor shape/kind/aliases")
            return ref
    raise KeyError(query.id)


_Value = TypeVar("_Value")


@dataclass(frozen=True)
class TensorRefMap(Mapping[TensorRef, tuple[_Value, ...]]):
    """Map tensor references to tuples while accepting compatible portable labels.

    Attributes:
        entries: Ordered reference/value pairs. Portable reference labels must be
            unique. Values are converted to tuples during construction.

    Raises:
        ValueError: Two entries have the same portable reference label.
    """

    entries: tuple[tuple[TensorRef, tuple[_Value, ...]], ...]

    def __post_init__(self) -> None:
        entries = tuple((ref, tuple(value)) for ref, value in self.entries)
        if len({ref.portable().id for ref, _ in entries}) != len(entries):
            raise ValueError("Duplicate tensor mapping labels")
        object.__setattr__(self, "entries", entries)

    def __getitem__(self, query: TensorRef) -> tuple[_Value, ...]:
        """Resolve a live or portable tensor label through this mapping.

        Raises:
            KeyError: No entry has a compatible tensor label.
            ValueError: A matching label has different metadata.
        """
        ref = resolve_reference(query, self)
        return next(value for key, value in self.entries if key == ref)

    def __iter__(self) -> Iterator[TensorRef]:
        return (ref for ref, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class AxisRef:
    """A physical axis of a tensor reference.

    Attributes:
        tensor: Owning tensor reference.
        dim: Nonnegative axis index. Construction normalizes negative indices.

    Raises:
        TypeError: `tensor` is not a `TensorRef` or `dim` is not an integer.
        IndexError: `dim` is outside the tensor rank.
    """

    tensor: TensorRef
    dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, TensorRef):
            raise TypeError("Axis owner must be a TensorRef")
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise TypeError("Axis must be an integer")
        rank = len(self.tensor.shape)
        if not -rank <= self.dim < rank:
            raise IndexError(f"Axis {self.dim} outside rank {rank}")
        object.__setattr__(self, "dim", self.dim % rank)

    def select(self, indices: Iterable[int] | IndexSet) -> Selection:
        """Select complete cross-sections at the given original axis positions.

        Args:
            indices: Individual indices or an `IndexSet`.

        Returns:
            A selection spanning every other tensor axis.

        Raises:
            ValueError: An individual index is invalid.
            IndexError: A selected position exceeds the original axis size.
        """
        selected = indices if isinstance(indices, IndexSet) else IndexSet.of(indices)
        axes = list(full_region(self.tensor.shape).axes)
        axes[self.dim] = selected
        return Selection(self.tensor, (Region(tuple(axes)),))


def _intersection_count(left: IndexSet, right: IndexSet) -> int:
    """Count common indices without constructing a potentially fragmented set."""
    if left == right:
        return len(left)
    i = j = count = 0
    while i < len(left.intervals) and j < len(right.intervals):
        a, b = left.intervals[i], right.intervals[j]
        count += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        if b[1] <= a[1]:
            j += 1
    return count


@dataclass(frozen=True, eq=False)
class Selection:
    """An immutable union of selected tensor regions in original coordinates.

    Attributes:
        tensor: Owning tensor reference.
        regions: Normalized, disjoint Cartesian regions.

    Raises:
        ValueError: A region rank differs from the tensor rank.
        IndexError: A region exceeds the original tensor shape.
        torch_kirigami.AnalysisLimitError: Normalization exceeds the region limit.

    Notes:
        Equality compares selected coordinates, not a particular rectangle
        decomposition. Selections are therefore intentionally unhashable.
    """

    tensor: TensorRef
    regions: tuple[Region, ...] = ()

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        if not regions and type(self.tensor) is TensorRef:
            object.__setattr__(self, "regions", ())
            return
        if type(self.tensor) is not TensorRef:
            # Extensions may supply their own shape behavior; retain the full
            # interval construction and subtraction used for their validation.
            bounds = full_region(self.tensor.shape)
            for region in regions:
                if len(region.axes) != len(bounds.axes):
                    raise ValueError("Selection rank does not match tensor")
                if any(a.subtract(b) for a, b in zip(region.axes, bounds.axes, strict=True)):
                    raise IndexError(f"Selection outside {self.tensor.shape}")
            object.__setattr__(self, "regions", normalize(regions))
            return
        for region in regions:
            if len(region.axes) != len(self.tensor.shape):
                raise ValueError("Selection rank does not match tensor")
            # Canonical intervals are nonnegative and sorted, so their last end
            # is enough to check the bound. Preserve extension method behavior.
            if any(
                (indices.intervals and indices.intervals[-1][1] > size)
                if type(indices) is IndexSet
                else indices.subtract(IndexSet.span(0, size))
                for indices, size in zip(region.axes, self.tensor.shape, strict=True)
            ):
                raise IndexError(f"Selection outside {self.tensor.shape}")
        object.__setattr__(self, "regions", normalize(regions))

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection):
            return NotImplemented
        if self.tensor != other.tensor:
            return False
        if self.regions == other.regions:
            return True
        count = self.count
        return count == other.count and self._shares(other, count)

    def issubset(self, other: Selection) -> bool:
        """Check that another selection contains every position selected here.

        Unlike `subtract`, this never builds the difference, so it cannot reach
        the region limit.

        Args:
            other: Selection on the same tensor reference.

        Returns:
            Whether every selected position is also selected by `other`.

        Raises:
            ValueError: The selections refer to different tensor references.
        """
        self._same(other)
        if not self.regions or self.regions == other.regions:
            return True
        return self._shares(other, self.count)

    def _shares(self, other: Selection, count: int) -> bool:
        """Check that `self` and `other` have `count` common positions, or more."""
        # Each side's regions are disjoint, so pairwise intersection volumes
        # count every common coordinate exactly once.
        covered = 0
        for left in self.regions:
            for right in other.regions:
                volume = 1
                for a, b in zip(left.axes, right.axes, strict=True):
                    volume *= _intersection_count(a, b)
                    if not volume:
                        break
                covered += volume
                if covered >= count:
                    return True
        return False

    def __bool__(self) -> bool:
        return bool(self.regions)

    @property
    def count(self) -> int:
        """Number of selected tensor elements, each counted once."""
        return sum(prod(len(a) for a in r.axes) for r in self.regions)

    def union(self, other: Selection) -> Selection:
        """Combine selections on the same tensor.

        Args:
            other: Selection on the same tensor reference.

        Returns:
            A normalized selection containing positions from either selection.

        Raises:
            ValueError: The selections refer to different tensor references.
            torch_kirigami.AnalysisLimitError: Their normalized union exceeds the region limit.
        """
        self._same(other)
        return Selection(self.tensor, self.regions + other.regions)

    def subtract(self, other: Selection) -> Selection:
        """Return positions selected here but absent from another selection.

        Args:
            other: Selection on the same tensor reference.

        Returns:
            A normalized selection containing the difference.

        Raises:
            ValueError: The selections refer to different tensor references.
            torch_kirigami.AnalysisLimitError: The difference exceeds the region limit.
        """
        self._same(other)
        pending = self.regions
        for region in other.regions:
            pending = tuple(p for r in pending for p in r.subtract(region))
        return Selection(self.tensor, pending)

    def _same(self, other: Selection) -> None:
        """Reject operations combining different tensor references."""
        if self.tensor != other.tensor:
            raise ValueError("Selections refer to different tensors")

    def fully_selected_indices(self, dim: int, scope: Region | None = None) -> IndexSet:
        """Find axis positions whose entire scoped cross-section is selected.

        Args:
            dim: Physical axis whose cross-sections are inspected. Negative
                values count from the last axis.
            scope: Region restricting the cross-sections, or None for the full tensor.

        Returns:
            Positions inside the scope whose scoped cross-sections are fully
            selected. Coverage inside a partial scope does not imply that the
            corresponding physical axis position can be removed.

        Raises:
            TypeError: `dim` is not an integer.
            IndexError: `dim` is outside the tensor rank, or `scope` exceeds the tensor.
            ValueError: `scope` has a different rank from the tensor.
            torch_kirigami.AnalysisLimitError: The result exceeds the interval
                limit. The result depends only on the selected coordinates,
                never on their region decomposition.
        """
        dim = self.tensor.axis(dim).dim
        if not self and scope is None:
            return IndexSet()
        if scope is None and len(self.regions) == 1:
            # A single Cartesian region contains complete fibers precisely when
            # every other axis spans its full original extent. Normalization and
            # bounds checks already establish the interval representation.
            axes = self.regions[0].axes
            complete = all(
                axis == dim or indices.intervals == ((0, size),)
                for axis, (indices, size) in enumerate(zip(axes, self.tensor.shape, strict=True))
            )
            return IndexSet(axes[dim].intervals) if complete else IndexSet()
        scope = scope or full_region(self.tensor.shape)
        Selection(self.tensor, (scope,))  # Validates the scope rank and bounds.
        if not self:
            return IndexSet()
        uncovered = canonical_boxes(region_boxes((scope,)), region_boxes(self.regions))
        return IndexSet(complete_intervals(scope.axes[dim], uncovered, dim))

    def compact_shape(self) -> tuple[int, ...] | None:
        """Return the shape after deleting complete axis cross-sections.

        Returns:
            Remaining dimensions, or None if the selection is not a union of
            complete axis cross-sections (for example, when partition-specific
            packing is needed or a scalar is removed). None does not by itself
            mean that the selection is structurally invalid.

        Raises:
            torch_kirigami.AnalysisLimitError: The complete positions of one
                axis exceed the interval limit. Like the result, this depends
                only on the selected coordinates.
        """
        if not self:
            return self.tensor.shape
        if (
            type(self) is Selection
            and type(self.tensor) is TensorRef
            and len(self.regions) == 1
            and type(self.regions[0]) is Region
            and all(type(axis) is IndexSet for axis in self.regions[0].axes)
        ):
            axes = self.regions[0].axes
            if not axes:
                return None  # A removed scalar has no surviving tensor shape.
            partial = [
                dim
                for dim, (indices, size) in enumerate(zip(axes, self.tensor.shape, strict=True))
                if indices.intervals != ((0, size),)
            ]
            if not partial:
                return (0,) * len(axes)
            if len(partial) == 1:
                sizes = list(self.tensor.shape)
                sizes[partial[0]] -= len(axes[partial[0]])
                return tuple(sizes)
            return None
        shape = self.tensor.shape
        selected = region_boxes(self.regions)
        uncovered = canonical_boxes(region_boxes((full_region(shape),)), selected)
        sizes, slices = [], []
        for dim, size in enumerate(shape):
            indices = IndexSet(complete_intervals(IndexSet.span(0, size), uncovered, dim))
            sizes.append(size - len(indices))
            if indices:
                box = [((0, n),) for n in shape]
                box[dim] = indices.intervals
                slices.append(tuple(box))
        return tuple(sizes) if not canonical_boxes(selected, slices) else None


def coalesce_selections(selections: tuple[Selection, ...]) -> tuple[Selection, ...]:
    """Batch simple same-axis seeds without changing intermediate limit failures.

    Only exact built-in records consisting of one rectangle per seed qualify.
    A group's rectangles must differ on at most one axis, and the combined
    interval count must fit even before normalization. Otherwise the original
    request order is retained for the ordinary incremental union algorithm.
    The caller must first verify that all references belong to its snapshot,
    so matching tensor identities have identical metadata.
    """
    groups: dict[str, list[Selection]] = {}
    for selection in selections:
        if (
            type(selection) is not Selection
            or type(selection.tensor) is not TensorRef
            or len(selection.regions) > 1
            or any(
                type(region) is not Region
                or any(type(axis) is not IndexSet for axis in region.axes)
                for region in selection.regions
            )
        ):
            return selections
        if selection:
            groups.setdefault(selection.tensor.id, []).append(selection)
    # Eligibility is checked for every group before constructing anything. This
    # keeps generic-path validation/limit failures in their original order.
    recipes = []
    for group in groups.values():
        first = group[0]
        axes = first.regions[0].axes
        different = {
            dim
            for item in group[1:]
            for dim, (left, right) in enumerate(zip(axes, item.regions[0].axes, strict=True))
            if left != right
        }
        if len(different) > 1 or config.max_parts < 2:
            return selections
        dim = next(iter(different), None)
        if (
            dim is not None
            and sum(len(item.regions[0].axes[dim].intervals) for item in group) > config.max_parts
        ):
            return selections
        recipes.append((group, dim))
    result = []
    for group, dim in recipes:
        first = group[0]
        if dim is None:
            result.append(first)
            continue
        axes = list(first.regions[0].axes)
        axes[dim] = IndexSet(
            tuple(interval for item in group for interval in item.regions[0].axes[dim].intervals)
        )
        result.append(Selection(first.tensor, (Region(tuple(axes)),)))
    return tuple(result)


def linear_indices(region: Region, shape: tuple[int, ...]) -> IndexSet:
    """Convert a Cartesian region into row-major logical offset intervals.

    Args:
        region: Selected positions inside shape.
        shape: Original logical tensor dimensions.

    Returns:
        Flat offset intervals, collapsing complete trailing dimensions.

    Raises:
        torch_kirigami.AnalysisLimitError: The exact offset representation exceeds its limit.
    """
    if not shape:
        return IndexSet.span(0, 1)
    trailing = 1
    split = len(shape)
    while split and region.axes[split - 1] == IndexSet.span(0, shape[split - 1]):
        split -= 1
        trailing *= shape[split]
    if split == 0:
        return IndexSet.span(0, prod(shape))
    last = split - 1
    count = prod(len(a) for a in region.axes[:last])
    if count * len(region.axes[last].intervals) > config.max_parts:
        raise AnalysisLimitError("Reshape selection needs too many offset intervals")
    strides = [prod(shape[i + 1 :]) for i in range(last)]
    parts: list[tuple[int, int]] = []
    for prefix in product(*(iter(a) for a in region.axes[:last])):
        offset = sum(i * stride for i, stride in zip(prefix, strides, strict=False))
        parts.extend(
            (offset + a * trailing, offset + b * trailing) for a, b in region.axes[last].intervals
        )
    return IndexSet(tuple(parts))


def offset_regions(indices: IndexSet, shape: tuple[int, ...]) -> tuple[Region, ...]:
    """Convert row-major offset intervals back into Cartesian regions.

    Args:
        indices: Flat offsets inside the target tensor domain.
        shape: Target logical tensor dimensions.

    Returns:
        A disjoint region representation with aligned blocks kept symbolic.
    """

    def visit(start: int, stop: int, dims: tuple[int, ...]) -> list[Region]:
        """Recursively split flat offsets into Cartesian regions."""
        if not dims:
            return [Region(())] if start < stop else []
        stride = prod(dims[1:])
        if stride == 0:
            return []
        first, last = start // stride, (stop - 1) // stride
        if first == last:
            return [
                Region((IndexSet.span(first, first + 1), *r.axes))
                for r in visit(start % stride, (stop - 1) % stride + 1, dims[1:])
            ]
        result: list[Region] = []
        middle_start = first
        if start % stride:
            result.extend(
                Region((IndexSet.span(first, first + 1), *r.axes))
                for r in visit(start % stride, stride, dims[1:])
            )
            middle_start += 1
        middle_stop = last if stop % stride else last + 1
        if middle_start < middle_stop:
            result.append(
                Region((IndexSet.span(middle_start, middle_stop), *full_region(dims[1:]).axes))
            )
        if stop % stride:
            result.extend(
                Region((IndexSet.span(last, last + 1), *r.axes))
                for r in visit(0, stop % stride, dims[1:])
            )
        return result

    return normalize(r for a, b in indices.intervals for r in visit(a, b, shape))


def concatenated_shape(regions: Sequence[Region], dim: int) -> tuple[int, ...]:
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


IntervalBox = tuple[tuple[tuple[int, int], ...], ...]
"""Cartesian product with one tuple of half-open intervals per tensor axis.

Unlike a `Region`, a box is not limited by `max_parts`.
"""


def region_boxes(regions: Iterable[Region]) -> tuple[IntervalBox, ...]:
    """Return the interval boxes of regions."""
    return tuple(tuple(axis.intervals for axis in region.axes) for region in regions)


_FULL_TAIL: tuple[IntervalBox, ...] = ((),)


def canonical_boxes(
    plus: Sequence[IntervalBox], minus: Sequence[IntervalBox] = ()
) -> tuple[IntervalBox, ...]:
    """Decompose the positions in `plus` and outside `minus` by their coordinates alone.

    Positions on the first axis are grouped by their cross-sections, which
    are decomposed the same way on the remaining axes. Equal position sets
    therefore give equal boxes in the same order, whatever boxes describe
    them. The intervals of a box are not limited by `max_parts`, so this never
    raises `AnalysisLimitError`. The work is proportional to the number of
    distinct interval bounds of each axis, times the number of distinct
    cross-sections found on the preceding axes.

    Args:
        plus: Boxes of one rank, which can overlap. Each axis holds sorted,
            disjoint, nonadjacent, nonempty intervals, as in
            `IndexSet.intervals`.
        minus: Boxes of the same rank whose positions are excluded.

    Returns:
        Sorted, disjoint, nonempty boxes with sorted, disjoint, nonadjacent
        intervals.
    """
    plus, minus = tuple(plus), tuple(minus)
    if not plus:
        return ()
    if not minus and len(plus) == 1 and all(plus[0]):
        return plus
    rank = len(plus[0])
    memo: dict[tuple[int, tuple[int, ...], tuple[int, ...]], tuple[IntervalBox, ...]] = {}

    def sweep(
        depth: int, active_plus: tuple[int, ...], active_minus: tuple[int, ...]
    ) -> tuple[IntervalBox, ...]:
        """Decompose the cross-section that the active boxes give from axis `depth` on."""
        if depth == rank:
            return () if active_minus else _FULL_TAIL
        key = (depth, active_plus, active_minus)
        found = memo.get(key)
        if found is not None:
            return found
        changes: dict[int, list[tuple[int, int]]] = {}
        for sign, boxes, active in ((1, plus, active_plus), (-1, minus, active_minus)):
            for index in active:
                for start, stop in boxes[index][depth]:
                    changes.setdefault(start, []).append((sign * (index + 1), 1))
                    changes.setdefault(stop, []).append((sign * (index + 1), -1))
        # Box identifiers are signed: positive for `plus` and negative for `minus`.
        counts: dict[int, int] = {}
        groups: dict[tuple[IntervalBox, ...], list[tuple[int, int]]] = {}
        # Equal active sets return one memoized object; hash each tail once.
        rows_by_tail: dict[int, list[tuple[int, int]]] = {}
        previous = 0
        for bound in sorted(changes):
            if any(identifier > 0 for identifier in counts):
                tail = sweep(
                    depth + 1,
                    tuple(sorted(i - 1 for i in counts if i > 0)),
                    tuple(sorted(-i - 1 for i in counts if i < 0)),
                )
                if tail:
                    rows = rows_by_tail.get(id(tail))
                    if rows is None:
                        rows = rows_by_tail[id(tail)] = groups.setdefault(tail, [])
                    if rows and rows[-1][1] == previous:
                        rows[-1] = (rows[-1][0], bound)
                    else:
                        rows.append((previous, bound))
            for identifier, delta in changes[bound]:
                count = counts.get(identifier, 0) + delta
                if count:
                    counts[identifier] = count
                else:
                    del counts[identifier]
            previous = bound
        result = tuple(sorted((tuple(rows), *box) for tail, rows in groups.items() for box in tail))
        memo[key] = result
        return result

    return sweep(0, tuple(range(len(plus))), tuple(range(len(minus))))


def complete_intervals(
    indices: IndexSet, uncovered: Sequence[IntervalBox], dim: int
) -> tuple[tuple[int, int], ...]:
    """Return the positions of `indices` that no uncovered box reaches on axis `dim`.

    The result can be used directly as a box axis for coverage checks.
    """
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(interval for box in uncovered for interval in box[dim]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    result, right = [], 0
    for lo, hi in indices.intervals:
        cursor = lo
        while right < len(merged) and merged[right][1] <= cursor:
            right += 1
        index = right
        while index < len(merged) and merged[index][0] < hi:
            start, stop = merged[index]
            if cursor < start:
                result.append((cursor, start))
            cursor = max(cursor, stop)
            index += 1
        if cursor < hi:
            result.append((cursor, hi))
    return tuple(result)


def gather_intervals(
    tensor: torch.Tensor, axes: Sequence[tuple[tuple[int, int], ...]]
) -> torch.Tensor:
    """Gather a Cartesian product of axis intervals without a full-tensor mask, preserving autograd.

    The most selective axis is gathered first. Each step still allocates the
    product of the gathered and the remaining full axes, so temporary memory
    can exceed the result, for example a `[2, n]` slice for a `[2, 2]` result.

    Args:
        tensor: Tensor to read.
        axes: Sorted, disjoint intervals for each axis of `tensor`.

    Returns:
        The selected elements, in axis order.
    """
    counts = [sum(stop - start for start, stop in intervals) for intervals in axes]
    order = sorted(range(tensor.ndim), key=lambda d: counts[d] / max(1, tensor.shape[d]))
    for dim in order:
        intervals = axes[dim]
        if counts[dim] != tensor.shape[dim]:
            if len(intervals) == 1:
                start, stop = intervals[0]
                tensor = tensor.narrow(dim, start, stop - start)
            else:
                index = torch.tensor(
                    [i for start, stop in intervals for i in range(start, stop)],
                    dtype=torch.long,
                    device=tensor.device,
                )
                tensor = tensor.index_select(dim, index)
    return tensor


def gather_region(tensor: torch.Tensor, region: Region) -> torch.Tensor:
    """Gather a region without a full-tensor mask, preserving autograd.

    Temporary memory follows `gather_intervals`.
    """
    return gather_intervals(tensor, tuple(axis.intervals for axis in region.axes))
