"""Finite index sets and unions of Cartesian regions, in original coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from math import prod

from .errors import AnalysisLimitError

MAX_PARTS = 4096


@dataclass(frozen=True)
class IndexSet:
    """A normalized union of nonnegative, half-open integer intervals.

    Overlapping and adjacent intervals are merged. Construction rejects invalid
    bounds and representations exceeding MAX_PARTS; it does not allocate masks.

    Attributes:
        intervals: Sorted pairs of inclusive starts and exclusive stops.
    """

    intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
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
        if len(merged) > MAX_PARTS:
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
            AnalysisLimitError: The normalized set exceeds the interval budget.
        """
        values = []
        for i in indices:
            if not isinstance(i, int) or isinstance(i, bool) or i < 0:
                raise ValueError("Indices must be nonnegative Python integers")
            values.append((i, i + 1))
        return cls(tuple(values))

    @classmethod
    def span(cls, start: int, stop: int) -> IndexSet:
        """Construct the half-open interval [start, stop)."""
        return cls(((start, stop),))

    def __bool__(self):
        return bool(self.intervals)

    def __len__(self):
        return sum(b - a for a, b in self.intervals)

    def __iter__(self):
        for a, b in self.intervals:
            yield from range(a, b)

    def union(self, other: IndexSet) -> IndexSet:
        """Return indices contained in either set."""
        return IndexSet(self.intervals + other.intervals)

    def intersect(self, other: IndexSet) -> IndexSet:
        """Return indices contained in both sets."""
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
        """Return indices in this set that are absent from other."""
        result = []
        for lo, hi in self.intervals:
            cursor = lo
            for a, b in other.intervals:
                if b <= cursor:
                    continue
                if a >= hi:
                    break
                if cursor < a:
                    result.append((cursor, a))
                cursor = max(cursor, b)
            if cursor < hi:
                result.append((cursor, hi))
        return IndexSet(tuple(result))

    def shift(self, offset: int) -> IndexSet:
        """Translate interval bounds, rejecting any resulting negative indices."""
        return IndexSet(tuple((a + offset, b + offset) for a, b in self.intervals))

    def __repr__(self):
        return "IndexSet(" + ", ".join(f"{a}:{b}" for a, b in self.intervals) + ")"


@dataclass(frozen=True)
class Region:
    """A Cartesian product of index sets in tensor-axis order.

    Attributes:
        axes: One IndexSet per physical tensor axis. An empty tuple denotes
            the single coordinate of a scalar tensor.
    """

    axes: tuple[IndexSet, ...]

    @property
    def empty(self):
        """Return whether any axis makes the Cartesian product empty."""
        return any(not a for a in self.axes)

    def intersect(self, other: Region) -> Region:
        """Return the Cartesian intersection, rejecting different ranks."""
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
        AnalysisLimitError: The intermediate region representation exceeds its budget.
    """
    result: list[Region] = []
    for region in sorted(set(regions), key=lambda r: tuple(a.intervals for a in r.axes)):
        if region.empty:
            continue
        pending = [region]
        for old in result:
            pending = [piece for item in pending for piece in item.subtract(old)]
            if len(pending) + len(result) > MAX_PARTS:
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
        kind: Parameter, buffer, input, or intermediate value category.
        paths: Original registered aliases, when applicable.
    """

    id: str
    shape: tuple[int, ...]
    kind: str = "value"
    paths: tuple[str, ...] = ()

    def axis(self, dim: int) -> AxisRef:
        """Refer to a physical axis, normalizing negative dimension indices.

        Args:
            dim: Axis index in the original tensor shape.

        Returns:
            An AxisRef with a nonnegative dimension index.

        Raises:
            IndexError: dim is outside the tensor rank.
        """
        if not -len(self.shape) <= dim < len(self.shape):
            raise IndexError(f"Axis {dim} outside rank {len(self.shape)}")
        return AxisRef(self, dim % len(self.shape))

    def select(self, regions: Iterable[Region]) -> Selection:
        """Construct a removal selection from Cartesian regions in original coordinates."""
        return Selection(self, tuple(regions))


@dataclass(frozen=True)
class AxisRef:
    """A physical axis of a tensor reference.

    Attributes:
        tensor: Owning tensor reference.
        dim: Nonnegative axis index when constructed through TensorRef.axis().
    """

    tensor: TensorRef
    dim: int

    def select(self, indices: Iterable[int] | IndexSet) -> Selection:
        """Select complete cross-sections at the given original axis positions.

        Args:
            indices: Individual indices or a symbolic IndexSet.

        Returns:
            A Selection spanning every other tensor axis.

        Raises:
            ValueError: Individual indices are invalid.
            IndexError: Selected positions exceed the original axis size.
        """
        selected = indices if isinstance(indices, IndexSet) else IndexSet.of(indices)
        axes = list(full_region(self.tensor.shape).axes)
        axes[self.dim] = selected
        return Selection(self.tensor, (Region(tuple(axes)),))


@dataclass(frozen=True, eq=False)
class Selection:
    """An immutable union of selected tensor regions in original coordinates.

    Attributes:
        tensor: Owning tensor reference.
        regions: Normalized, disjoint Cartesian regions.

    Notes:
        Equality compares selected coordinates, not a particular rectangle
        decomposition. Selections are therefore intentionally unhashable.
    """

    tensor: TensorRef
    regions: tuple[Region, ...] = ()

    def __post_init__(self):
        bounds = full_region(self.tensor.shape)
        for region in self.regions:
            if len(region.axes) != len(bounds.axes):
                raise ValueError("Selection rank does not match tensor")
            if any(a.subtract(b) for a, b in zip(region.axes, bounds.axes, strict=False)):
                raise IndexError(f"Selection outside {self.tensor.shape}")
        object.__setattr__(self, "regions", normalize(self.regions))

    __hash__ = None

    def __eq__(self, other):
        if not isinstance(other, Selection):
            return NotImplemented
        if self.tensor != other.tensor:
            return False
        return self.regions == other.regions or (
            not self.subtract(other) and not other.subtract(self)
        )

    def __bool__(self):
        return bool(self.regions)

    @property
    def count(self):
        """Return the number of selected tensor elements without overlap."""
        return sum(prod(len(a) for a in r.axes) for r in self.regions)

    def union(self, other: Selection) -> Selection:
        """Combine selections of the same tensor without double-counting positions."""
        self._same(other)
        return Selection(self.tensor, self.regions + other.regions)

    def subtract(self, other: Selection) -> Selection:
        """Return selected positions absent from another selection of the same tensor."""
        self._same(other)
        pending = self.regions
        for region in other.regions:
            pending = tuple(p for r in pending for p in r.subtract(region))
        return Selection(self.tensor, pending)

    def _same(self, other):
        """Reject operations combining different tensor references."""
        if self.tensor != other.tensor:
            raise ValueError("Selections refer to different tensors")

    def project(self, axis: int, scope: Region | None = None) -> IndexSet:
        """Find axis positions whose entire scoped cross-section is selected.

        Args:
            axis: Physical axis to project onto.
            scope: Region restricting the cross-section, or None for the full tensor.

        Returns:
            Positions fully covered inside the scope. Partial coverage does not
            imply that the corresponding physical axis position can be removed.
        """
        if not self:
            return IndexSet()
        scope = scope or full_region(self.tensor.shape)
        remaining = Selection(self.tensor, (scope,)).subtract(self)
        uncovered = IndexSet()
        for region in remaining.regions:
            uncovered = uncovered.union(region.axes[axis])
        return scope.axes[axis].subtract(uncovered)

    def compact_shape(self) -> tuple[int, ...] | None:
        """Return the shape after deleting complete axis cross-sections.

        Returns:
            Remaining dimensions, or None if partition-specific packing is needed.
            None does not by itself mean that the selection is structurally invalid.
        """
        covered = Selection(self.tensor)
        sizes = []
        for dim, size in enumerate(self.tensor.shape):
            indices = self.project(dim)
            covered = covered.union(self.tensor.axis(dim).select(indices))
            sizes.append(size - len(indices))
        return tuple(sizes) if not self.subtract(covered) else None


def linear_indices(region: Region, shape: tuple[int, ...]) -> IndexSet:
    """Convert a Cartesian region into row-major logical offset intervals.

    Args:
        region: Selected positions inside shape.
        shape: Original logical tensor dimensions.

    Returns:
        Flat offset intervals, collapsing complete trailing dimensions.

    Raises:
        AnalysisLimitError: The exact offset representation exceeds its budget.
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
    if count * len(region.axes[last].intervals) > MAX_PARTS:
        raise AnalysisLimitError("Reshape selection needs too many offset intervals")
    strides = [prod(shape[i + 1 :]) for i in range(last)]
    parts = []
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

    def visit(start, stop, dims):
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
        result = []
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
