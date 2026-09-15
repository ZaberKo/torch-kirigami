"""Finite index sets and unions of Cartesian regions, in original coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
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
            AnalysisLimitError: The normalized set exceeds the interval limit.
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

    def __bool__(self) -> bool:
        return bool(self.intervals)

    def __len__(self) -> int:
        return sum(b - a for a, b in self.intervals)

    def __iter__(self) -> Iterator[int]:
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
        """Translate interval bounds, rejecting any resulting negative indices."""
        return IndexSet(tuple((a + offset, b + offset) for a, b in self.intervals))

    def __repr__(self) -> str:
        return "IndexSet(" + ", ".join(f"{a}:{b}" for a, b in self.intervals) + ")"


@dataclass(frozen=True)
class Region:
    """A Cartesian product of index sets in tensor-axis order.

    Attributes:
        axes: One IndexSet per physical tensor axis. An empty tuple denotes
            the single coordinate of a scalar tensor.
    """

    axes: tuple[IndexSet, ...]

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if any(not isinstance(axis, IndexSet) for axis in axes):
            raise TypeError("Region axes must be IndexSet instances")
        object.__setattr__(self, "axes", axes)

    @property
    def empty(self) -> bool:
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
        AnalysisLimitError: The intermediate region representation exceeds its limit.
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
        """Remove a graph UUID, retaining a structural label for saved-plan queries.

        Portable labels do not establish ownership of a live dependency graph.
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
            An AxisRef with a nonnegative dimension index.

        Raises:
            IndexError: dim is outside the tensor rank.
        """
        return AxisRef(self, dim)

    def select(self, regions: Iterable[Region]) -> Selection:
        """Construct a removal selection from Cartesian regions in original coordinates."""
        return Selection(self, tuple(regions))


def resolve_reference(query: str | TensorRef, references: Iterable[TensorRef]) -> TensorRef:
    """Resolve compatible portable labels; distinguish unknown from unaffected.

    Raises:
        KeyError: The label is absent from this snapshot's catalog.
        ValueError: The label exists but its shape, kind, or aliases disagree.
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


@dataclass(frozen=True)
class TensorRefMap(Mapping):
    """Immutable mapping accepting compatible live or portable tensor labels."""

    entries: tuple

    def __post_init__(self) -> None:
        entries = tuple((ref, tuple(value)) for ref, value in self.entries)
        if len({ref.portable().id for ref, _ in entries}) != len(entries):
            raise ValueError("Duplicate tensor mapping labels")
        object.__setattr__(self, "entries", entries)

    def __getitem__(self, query: str | TensorRef) -> TensorRef:
        """Resolve a live or portable tensor label through this mapping."""
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
        dim: Canonical nonnegative axis index, including direct construction.
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

    Notes:
        Equality compares selected coordinates, not a particular rectangle
        decomposition. Selections are therefore intentionally unhashable.
    """

    tensor: TensorRef
    regions: tuple[Region, ...] = ()

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        bounds = full_region(self.tensor.shape)
        for region in regions:
            if len(region.axes) != len(bounds.axes):
                raise ValueError("Selection rank does not match tensor")
            if any(a.subtract(b) for a, b in zip(region.axes, bounds.axes, strict=False)):
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
        if count != other.count:
            return False
        # Each side's regions are disjoint, so pairwise intersection volumes
        # count every common coordinate exactly once. Equality must not fail
        # because a hypothetical difference exceeds the representation limit.
        covered = 0
        for left in self.regions:
            for right in other.regions:
                volume = 1
                for a, b in zip(left.axes, right.axes, strict=True):
                    volume *= _intersection_count(a, b)
                    if not volume:
                        break
                covered += volume
                if covered == count:
                    return True
        return False

    def __bool__(self) -> bool:
        return bool(self.regions)

    @property
    def count(self) -> int:
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

    def _same(self, other: Selection) -> None:
        """Reject operations combining different tensor references."""
        if self.tensor != other.tensor:
            raise ValueError("Selections refer to different tensors")

    def fully_selected_indices(self, dim: int, scope: Region | None = None) -> IndexSet:
        """Find axis positions whose entire scoped cross-section is selected.

        Args:
            dim: Physical axis whose complete cross sections are inspected.
            scope: Region restricting the cross-section, or None for the full tensor.

        Returns:
            Positions fully covered inside the scope. Partial coverage does not
            imply that the corresponding physical axis position can be removed.
        """
        dim = self.tensor.axis(dim).dim
        scope = scope or full_region(self.tensor.shape)
        scoped = Selection(self.tensor, (scope,))
        if not self:
            return IndexSet()
        remaining = scoped.subtract(self)
        uncovered = IndexSet()
        for region in remaining.regions:
            uncovered = uncovered.union(region.axes[dim])
        return scope.axes[dim].subtract(uncovered)

    def compact_shape(self) -> tuple[int, ...] | None:
        """Return the shape after deleting complete axis cross-sections.

        Returns:
            Remaining dimensions, or None if partition-specific packing is needed.
            None does not by itself mean that the selection is structurally invalid.
        """
        covered = Selection(self.tensor)
        sizes = []
        for dim, size in enumerate(self.tensor.shape):
            indices = self.fully_selected_indices(dim)
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
        AnalysisLimitError: The exact offset representation exceeds its limit.
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
