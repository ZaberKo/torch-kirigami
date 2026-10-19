"""Interval and Cartesian shortcuts preserve set algebra and validation."""

from itertools import combinations, product

import pytest

from torch_kirigami import IndexSet, Region, Selection, TensorRef


class DerivedIndexSet(IndexSet):
    """Exercise inherited interval methods without exact-class shortcuts."""


def subsets(size: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate finite axis subsets independently of interval normalization."""
    return tuple(items for count in range(size + 1) for items in combinations(range(size), count))


@pytest.mark.parametrize("left_type", [IndexSet, DerivedIndexSet])
@pytest.mark.parametrize("right_type", [IndexSet, DerivedIndexSet])
def test_set_operations_match_python_sets_and_keep_plain_result_types(
    left_type: type[IndexSet], right_type: type[IndexSet]
) -> None:
    """Compare every small set pair, including inherited implementations."""
    for left_values, right_values in product(subsets(4), repeat=2):
        left, right = left_type.of(left_values), right_type.of(right_values)
        a, b = set(left_values), set(right_values)
        for actual, expected in (
            (left.union(right), a | b),
            (left.intersect(right), a & b),
            (left.subtract(right), a - b),
        ):
            assert type(actual) is IndexSet
            assert set(actual) == expected


@pytest.mark.parametrize("method", ["union", "intersect", "subtract"])
def test_nonempty_set_methods_still_reject_wrong_operand(method: str) -> None:
    """Do not turn an invalid operand into a successful shortcut."""
    with pytest.raises(AttributeError):
        getattr(IndexSet.of([0]), method)(None)


def test_empty_union_still_rejects_wrong_operand() -> None:
    """An empty receiver does not make a malformed union operand valid."""
    with pytest.raises(AttributeError):
        IndexSet().union(None)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (3,), (2, 3), (1, 2, 3), (2, 1, 2, 2)])
def test_compact_shape_matches_independent_complete_fiber_reference(shape: tuple[int, ...]) -> None:
    """Derive compact dimensions from independently enumerated coordinates."""
    ref = TensorRef("rectangle", shape)
    universe = set(product(*(range(size) for size in shape)))
    for chosen in product(*(subsets(size) for size in shape)):
        removed = set(product(*chosen))
        selection = Selection(ref, (Region(tuple(IndexSet.of(axis) for axis in chosen)),))
        fully_removed = [
            {
                index
                for index in range(size)
                if (fiber := {point for point in universe if point[dim] == index})
                and fiber <= removed
            }
            for dim, size in enumerate(shape)
        ]
        covered = {
            point
            for point in universe
            if any(point[dim] in indices for dim, indices in enumerate(fully_removed))
        }
        expected = (
            tuple(size - len(indices) for size, indices in zip(shape, fully_removed, strict=True))
            if covered == removed
            else None
        )
        assert selection.compact_shape() == expected, (shape, chosen)


@pytest.mark.parametrize("indices_type", [IndexSet, DerivedIndexSet])
def test_fragmented_bounds_rank_and_zero_sized_axes_remain_checked(
    indices_type: type[IndexSet],
) -> None:
    """Validate all axis bounds even when another axis makes the region empty."""
    ref = TensorRef("bounds", (5, 0))
    with pytest.raises(IndexError):
        Selection(ref, (Region((indices_type.of([0, 2, 5]), IndexSet())),))
    with pytest.raises(IndexError):
        Selection(ref, (Region((IndexSet(), indices_type.of([0]))),))
    with pytest.raises(ValueError, match="rank"):
        Selection(ref, (Region((indices_type.of([0]),)),))
    assert not Selection(ref, (Region((indices_type.of([0, 2, 4]), IndexSet())),))


def test_bounds_validation_preserves_subclass_subtract() -> None:
    """Retain extension validation callbacks outside the exact-class shortcut."""
    calls = []

    class CheckedSet(IndexSet):
        """Record validation's interval subtraction without changing its result."""

        def subtract(self, other: IndexSet) -> IndexSet:
            """Record the bound and delegate to ordinary set subtraction."""
            calls.append(other.intervals)
            return super().subtract(other)

    Selection(TensorRef("custom", (3,)), (Region((CheckedSet.of([1]),)),))
    assert calls == [((0, 3),)]
