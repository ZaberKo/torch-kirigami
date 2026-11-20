"""Reusable index relations. They describe structure, never execute pruning."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol

from ..errors import AnalysisLimitError
from ..settings import config
from .selection import (
    AxisRef,
    IndexSet,
    IntervalBox,
    Region,
    Selection,
    TensorRef,
    canonical_boxes,
    complete_intervals,
    full_region,
    linear_indices,
    offset_regions,
    region_boxes,
)


class Relation(Protocol):
    """Monotone selection propagation between tensor entities."""

    reason: str

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """All tensor endpoints, used to schedule propagation."""
        ...

    def propagate(self, source: Selection) -> tuple[Selection, ...]:
        """Map an accumulated selection through the relation.

        Args:
            source: Current selection on one of the relation's endpoints.

        Returns:
            Implied endpoint selections in original coordinates. Implementations must
            be deterministic and monotone and must not mutate the source or execute model code.
        """
        ...


@dataclass(frozen=True)
class AxisPort:
    """Expose one logical axis within an optional tensor partition.

    Attributes:
        axis: Physical tensor axis used for the logical positions.
        scope: Original-coordinate region limiting the axis cross-sections, or
            None to use the whole tensor.

    Raises:
        TypeError: `axis` is not an `AxisRef`.
        ValueError: `scope` has a different rank from the tensor.
        IndexError: `scope` exceeds the tensor shape.
    """

    axis: AxisRef
    scope: Region | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.axis, AxisRef):
            raise TypeError("Axis port requires an AxisRef")
        if self.scope is not None:
            Selection(self.tensor, (self.scope,))

    @property
    def tensor(self) -> TensorRef:
        """Tensor containing this logical port."""
        return self.axis.tensor

    @property
    def region(self) -> Region:
        """Explicit partition scope, or the full tensor region if there is none."""
        return self.scope or full_region(self.tensor.shape)

    @property
    def is_full(self) -> bool:
        """Whether this port covers the entire original tensor region."""
        return self.scope is None or self.scope == full_region(self.tensor.shape)

    def fully_selected_indices(self, selection: Selection) -> IndexSet:
        """Return axis positions whose entire scoped cross-sections are selected.

        Args:
            selection: Selection on this port's tensor.

        Returns:
            Original axis positions fully covered inside the port's region.

        Raises:
            ValueError: The selection belongs to another tensor.
        """
        if selection.tensor != self.tensor:
            raise ValueError("Selection belongs to another port tensor")
        if self.scope is not None or len(selection.regions) != 1:
            return selection.fully_selected_indices(self.axis.dim, self.region)
        # Only exact native records can bypass the general scoped calculation.
        scope = self.scope if self._native_coordinates(selection) else self.region
        return selection.fully_selected_indices(self.axis.dim, scope)

    def _native_coordinates(self, selection: Selection) -> bool:
        """Check whether fiber geometry can use native interval operations."""
        regions = selection.regions if self.scope is None else (*selection.regions, self.scope)
        return (
            type(self) is AxisPort
            and type(self.axis) is AxisRef
            and type(self.axis.dim) is int
            and type(selection) is Selection
            and type(selection.tensor) is TensorRef
            and all(type(size) is int for size in self.tensor.shape)
            and all(
                type(region) is Region
                and all(
                    type(indices) is IndexSet
                    and all(
                        type(bound) is int for interval in indices.intervals for bound in interval
                    )
                    for indices in region.axes
                )
                for region in regions
            )
        )

    def covered_boxes(self, selection: Selection) -> tuple[IntervalBox, ...]:
        """Return the complete scoped fibers for a layout coverage test.

        The proof consumes interval boxes directly. A fragmented intersection
        can cover valid fibers even when its positions would exceed the limit
        on a materialized `IndexSet`.
        """
        if selection.tensor != self.tensor:
            raise ValueError("Selection belongs to another port tensor")
        if not self._native_coordinates(selection):
            return region_boxes(self.select(self.fully_selected_indices(selection)).regions)
        scope = self.region
        (box,) = region_boxes((scope,))
        dim = self.axis.dim
        missing = canonical_boxes((box,), region_boxes(selection.regions))
        fibers = complete_intervals(scope.axes[dim], missing, dim)
        return ((*box[:dim], fibers, *box[dim + 1 :]),) if fibers else ()

    def select(self, indices: IndexSet) -> Selection:
        """Select scoped cross-sections at the given original axis positions.

        Args:
            indices: Original axis positions. Positions outside the port's
                region are ignored.

        Returns:
            A selection limited to the port's region.
        """
        axes = list(self.region.axes)
        axes[self.axis.dim] = axes[self.axis.dim].intersect(indices)
        return Selection(self.tensor, (Region(tuple(axes)),))


@dataclass(frozen=True)
class BlockMap:
    """Map corresponding contiguous blocks with explicit completion policies.

    Attributes:
        source_start: Beginning of the source block range.
        target_start: Beginning of the target block range.
        count: Number of corresponding blocks.
        source_block: Number of source positions per block.
        target_block: Number of target positions per block.
        require_full_source: Whether forward mapping requires a whole source block.
        require_full_target: Whether reverse mapping requires a whole target block.

    Raises:
        ValueError: A start or count is negative, a block size is not positive,
            or one of them is not an integer.
        TypeError: A completion policy is not a boolean.
    """

    source_start: int
    target_start: int
    count: int
    source_block: int = 1
    target_block: int = 1
    require_full_source: bool = False
    require_full_target: bool = False

    def __post_init__(self) -> None:
        for name in ("source_start", "target_start", "count", "source_block", "target_block"):
            value = getattr(self, name)
            minimum = 1 if name.endswith("block") else 0
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Invalid block mapping {name}")
        if not isinstance(self.require_full_source, bool) or not isinstance(
            self.require_full_target, bool
        ):
            raise TypeError("Block completion policies must be boolean")

    def map(self, indices: IndexSet, reverse: bool = False) -> IndexSet:
        """Map selected positions into corresponding blocks.

        Args:
            indices: Original positions on the source side, or on the target side
                if reversed. Positions outside the mapped range are ignored.
            reverse: Whether to map from target to source, using
                `require_full_target` instead of `require_full_source`.

        Returns:
            Complete corresponding blocks on the other side. With a full-block
            policy, partial blocks do not propagate; otherwise, touching a block
            selects its counterpart.
        """
        source_start, target_start = self.source_start, self.target_start
        source_block, target_block = self.source_block, self.target_block
        if reverse:
            source_start, target_start = target_start, source_start
            source_block, target_block = target_block, source_block
        inside = indices.intersect(
            IndexSet.span(source_start, source_start + self.count * source_block)
        )
        require_full = self.require_full_target if reverse else self.require_full_source
        block_intervals = []
        for start, stop in inside.intervals:
            if require_full:
                # Round inward: only completely selected source blocks propagate.
                first_block = (start - source_start + source_block - 1) // source_block
                stop_block = (stop - source_start) // source_block
            else:
                # Round outward: touching any source position selects its block.
                first_block = (start - source_start) // source_block
                stop_block = (stop - 1 - source_start) // source_block + 1
            if first_block < stop_block:
                block_intervals.append((first_block, stop_block))
        blocks = IndexSet(tuple(block_intervals))
        return IndexSet(
            tuple(
                (target_start + start * target_block, target_start + stop * target_block)
                for start, stop in blocks.intervals
            )
        )


@dataclass(frozen=True)
class AxisRelation:
    """Map scoped axis positions in both directions through block mappings.

    Mapping ranges must fit the physical axes. Ports then intersect the mapped
    positions with their scopes, allowing a full-axis map to serve a partition.

    Attributes:
        left: Source-side logical axis and optional scope.
        right: Target-side logical axis and optional scope.
        maps: Block mappings applied in both directions.
        reason: Explanation attached to propagated selections.

    Raises:
        TypeError: An endpoint is not an `AxisPort` or a mapping is not a `BlockMap`.
        ValueError: A mapping range exceeds its endpoint axis.
    """

    left: AxisPort
    right: AxisPort
    maps: tuple[BlockMap, ...]
    reason: str = "axis correspondence"

    def __post_init__(self) -> None:
        if not isinstance(self.left, AxisPort) or not isinstance(self.right, AxisPort):
            raise TypeError("Axis relations require AxisPort endpoints")
        maps = tuple(self.maps)
        for mapping in maps:
            if not isinstance(mapping, BlockMap):
                raise TypeError("Axis relations require BlockMap mappings")
            if (
                mapping.source_start + mapping.count * mapping.source_block
                > self.left.tensor.shape[self.left.axis.dim]
                or mapping.target_start + mapping.count * mapping.target_block
                > self.right.tensor.shape[self.right.axis.dim]
            ):
                raise ValueError("Block mapping exceeds endpoint axis bounds")
        object.__setattr__(self, "maps", maps)

    @classmethod
    def equal(
        cls, left: AxisRef, right: AxisRef, reason: str = "axis correspondence"
    ) -> AxisRelation:
        """Construct an identity mapping between equally sized axes.

        Args:
            left: First axis.
            right: Corresponding axis.
            reason: Explanation attached to propagated selections.

        Returns:
            A bidirectional axis relation.

        Raises:
            ValueError: The axes have different original lengths.
        """
        size = left.tensor.shape[left.dim]
        if size != right.tensor.shape[right.dim]:
            raise ValueError("Equal axes must have equal sizes")
        return cls(AxisPort(left), AxisPort(right), (BlockMap(0, 0, size),), reason)

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor endpoints of this relation."""
        return (self.left.tensor, self.right.tensor)

    def propagate(self, source: Selection) -> tuple[Selection, ...]:
        """Map fully selected scoped cross-sections to the other endpoint.

        If both ports belong to the same tensor, both directions apply.

        Args:
            source: Accumulated selection on one endpoint tensor.

        Returns:
            Nonempty selections implied on the opposite endpoints.

        Raises:
            ValueError: `source` does not belong to either endpoint.
            torch_kirigami.AnalysisLimitError: The mapped positions exceed the
                interval limit.
        """
        if source.tensor not in self.refs:
            raise ValueError("Selection is not a relation endpoint")
        result = []
        for origin, target, reverse in (
            (self.left, self.right, False),
            (self.right, self.left, True),
        ):
            if source.tensor != origin.tensor:
                continue
            indices = origin.fully_selected_indices(source)
            intervals: list[tuple[int, int]] = []
            for relation in self.maps:
                intervals.extend(relation.map(indices, reverse).intervals)
                # Bound temporary storage while amortizing normalization across
                # blocks instead of repeatedly sorting the entire prefix.
                if len(intervals) > config.max_parts:
                    intervals = list(IndexSet(tuple(intervals)).intervals)
            selected = target.select(IndexSet(tuple(intervals)))
            if selected:
                result.append(selected)
        return tuple(result)


@dataclass(frozen=True)
class BroadcastRelation:
    """Map selections through elementwise broadcasting.

    Notes:
        Forward propagation repeats selected regions. Reverse propagation selects
        an original position only after every broadcast copy has been selected.

    Attributes:
        small: Lower-rank or singleton-dimension tensor.
        big: Broadcast result tensor.
        reason: Explanation attached to propagated selections.

    Raises:
        ValueError: The tensor shapes cannot broadcast from small to big.
    """

    small: TensorRef
    big: TensorRef
    reason: str = "broadcast correspondence"

    def __post_init__(self) -> None:
        offset = len(self.big.shape) - len(self.small.shape)
        if offset < 0 or any(
            size not in (1, self.big.shape[offset + dim])
            for dim, size in enumerate(self.small.shape)
        ):
            raise ValueError("Incompatible broadcast shapes")

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor endpoints of this relation."""
        return (self.small, self.big)

    def propagate(self, source: Selection) -> tuple[Selection, ...]:
        """Expand regions forward, or require complete broadcast fibers in reverse.

        Args:
            source: Accumulated selection on `small` or `big`.

        Returns:
            One selection on the opposite endpoint, which may be empty.

        Raises:
            ValueError: `source` does not belong to either endpoint.
            torch_kirigami.AnalysisLimitError: The mapped regions exceed the region limit.
        """
        if source.tensor not in self.refs:
            raise ValueError("Selection is not a relation endpoint")
        offset = len(self.big.shape) - len(self.small.shape)
        if source.tensor == self.small:
            regions = []
            for region in source.regions:
                axes = list(full_region(self.big.shape).axes)
                for dim, size in enumerate(self.small.shape):
                    if size != 1:
                        axes[offset + dim] = region.axes[dim]
                regions.append(Region(tuple(axes)))
            return (Selection(self.big, tuple(regions)),)
        # A source position changes only if its whole broadcast fiber disappears.
        remaining = Selection(self.big, (full_region(self.big.shape),)).subtract(source)
        projected = Selection(
            self.small,
            tuple(
                Region(
                    tuple(
                        IndexSet.span(0, 1) if size == 1 else region.axes[offset + dim]
                        for dim, size in enumerate(self.small.shape)
                    )
                )
                for region in remaining.regions
            ),
        )
        return (Selection(self.small, (full_region(self.small.shape),)).subtract(projected),)


def _shared_reshape_axis_counts(
    left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> tuple[int, int]:
    """Return the counts of matching leading and trailing axes without overlap."""
    prefix = 0
    for left, right in zip(left_shape, right_shape, strict=False):
        if left != right:
            break
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(left_shape), len(right_shape)) - prefix
        and left_shape[-1 - suffix] == right_shape[-1 - suffix]
    ):
        suffix += 1
    return prefix, suffix


@dataclass(frozen=True)
class ReshapeRelation:
    """Map row-major positions between shapes with equal element counts.

    Attributes:
        left: Tensor at one end of the reshape.
        right: Tensor at the other end of the reshape.
        reason: Explanation attached to propagated selections.

    Raises:
        ValueError: The endpoint shapes have different element counts.
    """

    left: TensorRef
    right: TensorRef
    reason: str = "row-major reshape"

    def __post_init__(self) -> None:
        if prod(self.left.shape) != prod(self.right.shape):
            raise ValueError("Reshape endpoints must have equal element counts")

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor endpoints of this relation."""
        return (self.left, self.right)

    def propagate(self, source: Selection) -> tuple[Selection, ...]:
        """Map regions to the other endpoint in row-major order.

        Leading and trailing dimensions shared by both shapes are kept as
        Cartesian axes; only the differing middle dimensions are flattened.

        Args:
            source: Accumulated selection on either endpoint.

        Returns:
            One selection on the opposite endpoint.

        Raises:
            ValueError: source does not belong to either endpoint.
            torch_kirigami.AnalysisLimitError: The mapped region representation exceeds its limit.
        """
        if source.tensor not in self.refs:
            raise ValueError("Selection is not a relation endpoint")
        target = self.right if source.tensor == self.left else self.left
        # Row-major reshapes preserve matching prefix and suffix coordinates.
        # Convert only the middle: splitting a merged batch axis must not expand
        # unchanged patch/feature axes into one interval per batch/token position.
        prefix, suffix = _shared_reshape_axis_counts(self.left.shape, self.right.shape)
        source_stop = len(source.tensor.shape) - suffix
        target_stop = len(target.shape) - suffix
        regions: list[Region] = []
        for region in source.regions:
            offsets = linear_indices(
                Region(region.axes[prefix:source_stop]), source.tensor.shape[prefix:source_stop]
            )
            regions.extend(
                Region(region.axes[:prefix] + mapped.axes + region.axes[source_stop:])
                for mapped in offset_regions(offsets, target.shape[prefix:target_stop])
            )
        return (Selection(target, tuple(regions)),)


@dataclass(frozen=True)
class PermuteRelation:
    """Map tensor regions through an axis permutation.

    Attributes:
        left: Tensor before permutation.
        right: Tensor after permutation.
        dims: Source axis index for each axis of the permuted tensor.
            Construction normalizes negative indices.
        reason: Explanation attached to propagated selections.

    Raises:
        ValueError: `dims` is not a permutation of the source axes, or the
            permuted source shape differs from the shape of `right`.
        IndexError: An axis index is outside the source rank.
    """

    left: TensorRef
    right: TensorRef
    dims: tuple[int, ...]
    reason: str = "dimension permutation"

    def __post_init__(self) -> None:
        dims = tuple(self.left.axis(dim).dim for dim in self.dims)
        if sorted(dims) != list(range(len(self.left.shape))):
            raise ValueError("Permutation must contain each axis exactly once")
        if tuple(self.left.shape[dim] for dim in dims) != self.right.shape:
            raise ValueError("Permutation does not match the output shape")
        object.__setattr__(self, "dims", dims)

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor endpoints of this relation."""
        return (self.left, self.right)

    def propagate(self, source: Selection) -> tuple[Selection, ...]:
        """Permute region axes, using the inverse order for reverse propagation.

        Args:
            source: Accumulated selection on either endpoint.

        Returns:
            One selection on the opposite endpoint.

        Raises:
            ValueError: source does not belong to either endpoint.
        """
        if source.tensor not in self.refs:
            raise ValueError("Selection is not a relation endpoint")
        if source.tensor == self.left:
            target, order = self.right, self.dims
        else:
            target = self.left
            order = tuple(self.dims.index(i) for i in range(len(self.dims)))
        return (
            Selection(
                target, tuple(Region(tuple(r.axes[i] for i in order)) for r in source.regions)
            ),
        )


def _stride_map(indices: IndexSet, start: int, step: int, reverse: bool) -> IndexSet:
    """Translate index sets between a positive-stride slice and its source."""
    if step == 1:
        return indices.shift(-start if reverse else start)
    result = []
    if reverse:
        for lo, hi in indices.intervals:
            a = max(0, (lo - start + step - 1) // step)
            b = max(0, (hi - start + step - 1) // step)
            if a < b:
                result.append((a, b))
    else:
        if len(indices) > config.max_parts:
            raise AnalysisLimitError("Strided selection has too many intervals")
        result.extend((start + i * step, start + i * step + 1) for i in indices)
    return IndexSet(tuple(result))


@dataclass(frozen=True)
class SliceRelation:
    """Map regions through normalized basic indexing.

    Attributes:
        big: Tensor before indexing.
        small: Indexed tensor.
        index: One integer or positive-step slice per source axis, plus None
            entries that insert singleton axes without consuming a source axis.
            Construction normalizes integers to nonnegative positions and
            slices to clamped bounds.
        reason: Explanation attached to propagated selections.

    Raises:
        ValueError: The number of non-None entries differs from the source rank,
            a slice step is not positive, or the result shape differs from the
            shape of `small`.
        IndexError: An integer index is outside its source axis.
        TypeError: An entry is not an integer, slice, or None.

    Notes:
        Rules expand ellipses before constructing this relation.
    """

    big: TensorRef
    small: TensorRef
    index: tuple[int | slice | None, ...]
    reason: str = "static slice"

    def __post_init__(self) -> None:
        index = tuple(self.index)
        if sum(item is not None for item in index) != len(self.big.shape):
            raise ValueError("Basic indexing requires one entry per source axis")
        normalized: list[int | slice | None] = []
        shape: list[int] = []
        sizes = iter(self.big.shape)
        for item in index:
            if item is None:
                normalized.append(None)
                shape.append(1)
                continue
            size = next(sizes)
            if isinstance(item, int) and not isinstance(item, bool):
                if not -size <= item < size:
                    raise IndexError("Integer index exceeds source axis bounds")
                normalized.append(item % size)
            elif isinstance(item, slice):
                start, stop, step = item.indices(size)
                if step <= 0:
                    raise ValueError("Only positive slice steps are supported")
                stop = max(start, stop)
                normalized.append(slice(start, stop, step))
                shape.append(len(range(start, stop, step)))
            else:
                raise TypeError("Basic indices must be integers or slices")
        if tuple(shape) != self.small.shape:
            raise ValueError("Basic index does not match the output shape")
        object.__setattr__(self, "index", tuple(normalized))

    @property
    def refs(self) -> tuple[TensorRef, ...]:
        """Tensor endpoints of this relation."""
        return (self.big, self.small)

    def propagate(self, source: Selection) -> tuple[Selection, ...]:
        """Map regions between the source tensor and the indexed tensor.

        Forward propagation clips each region to the indexed positions and drops
        regions that miss an integer index. Reverse propagation places each
        region back at the indexed source positions.

        Args:
            source: Accumulated selection on `big` or `small`.

        Returns:
            One selection on the opposite endpoint, which may be empty.

        Raises:
            ValueError: `source` does not belong to either endpoint.
            torch_kirigami.AnalysisLimitError: A strided reverse mapping exceeds
                the interval limit.
        """
        if source.tensor not in self.refs:
            raise ValueError("Selection is not a relation endpoint")
        forward = source.tensor == self.big
        regions = []
        for region in source.regions:
            result = []
            big_axis = small_axis = 0
            valid = True
            for item in self.index:
                if item is None:
                    if forward:
                        result.append(IndexSet.span(0, 1))
                    small_axis += 1
                    continue
                axis = big_axis
                big_axis += 1
                if isinstance(item, int):
                    if forward and not region.axes[axis].intersect(IndexSet.of([item])):
                        valid = False
                        break
                    if not forward:
                        result.append(IndexSet.of([item]))
                else:
                    start, stop, step = item.indices(self.big.shape[axis])
                    if step <= 0:
                        raise ValueError("Only positive slice steps are supported")
                    if forward:
                        inside = region.axes[axis].intersect(IndexSet.span(start, stop))
                        mapped = _stride_map(inside, start, step, True)
                    else:
                        mapped = _stride_map(region.axes[small_axis], start, step, False)
                    result.append(mapped)
                    small_axis += 1
            if valid:
                regions.append(Region(tuple(result)))
        return (Selection(self.small if forward else self.big, tuple(regions)),)
