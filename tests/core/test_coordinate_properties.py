"""core / coordinate properties contracts."""

from itertools import combinations, permutations, product

import pytest

from torch_kirigami import IndexSet, Region, Selection, TensorRef
from torch_kirigami.relations import BroadcastRelation, PermuteRelation, ReshapeRelation


def subsets(values):
    values = tuple(values)
    return tuple(frozenset(s) for n in range(len(values) + 1) for s in combinations(values, n))


def actual_points(selection):
    return {
        point
        for region in selection.regions
        for point in product(*(tuple(axis) for axis in region.axes))
    }


def points(shape):
    return tuple(product(*(range(size) for size in shape)))


def selected(ref, coordinates):
    return Selection(
        ref,
        tuple(Region(tuple(IndexSet.of([i]) for i in point)) for point in sorted(coordinates)),
    )


def test_all_small_interval_pairs_against_python_sets():
    cases = [(value, IndexSet.of(sorted(value, reverse=True))) for value in subsets(range(6))]
    for left, a in cases:
        assert tuple(a) == tuple(sorted(left))
        assert len(a) == len(left)
        assert set(a.shift(3)) == {i + 3 for i in left}
        for right, b in cases:
            context = (left, right)
            assert set(a.union(b)) == left | right, context
            assert set(a.intersect(b)) == left & right, context
            assert set(a.subtract(b)) == left - right, context


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (2, 3)])
def test_all_small_selection_pairs_against_coordinate_sets(shape):
    ref = TensorRef("coordinates", shape)
    cases = [(value, selected(ref, value)) for value in subsets(points(shape))]
    for left, a in cases:
        assert actual_points(a) == left
        assert a.count == len(left)
        # Reordering and duplicating region pieces must not affect the represented set.
        assert Selection(ref, (*reversed(a.regions), *a.regions)) == a
        for right, b in cases:
            context = (shape, left, right)
            union, difference = a.union(b), a.subtract(b)
            assert actual_points(union) == left | right, context
            assert union.count == len(left | right), context
            assert actual_points(difference) == left - right, context
            assert difference.count == len(left - right), context
            assert (a == b) == (left == right), context


def test_every_small_removal_has_the_correct_compact_shape_and_scoped_axes():
    ref = TensorRef("scoped", (2, 3))
    universe = set(points(ref.shape))
    scopes = tuple(
        (rows, columns)
        for rows in subsets(range(2))
        for columns in subsets(range(3))
        if rows and columns
    )
    for removed in subsets(universe):
        selection = selected(ref, removed)
        fully_removed = [
            {i for i in range(size) if {p for p in universe if p[dim] == i} <= removed}
            for dim, size in enumerate(ref.shape)
        ]
        covered = {
            p for p in universe if any(p[d] in indices for d, indices in enumerate(fully_removed))
        }
        expected = (
            tuple(
                size - len(indices) for size, indices in zip(ref.shape, fully_removed, strict=True)
            )
            if covered == removed
            else None
        )
        assert selection.compact_shape() == expected, removed
        for rows, columns in scopes:
            scope_points = set(product(rows, columns))
            region = Region((IndexSet.of(rows), IndexSet.of(columns)))
            for dim, indices in enumerate((rows, columns)):
                expected_indices = {
                    i for i in indices if {p for p in scope_points if p[dim] == i} <= removed
                }
                assert set(selection.fully_selected_indices(dim, region)) == expected_indices, (
                    removed,
                    rows,
                    columns,
                    dim,
                )


@pytest.mark.parametrize("source_shape", [(6,), (2, 3), (3, 2), (1, 2, 3), (2, 3, 1)])
@pytest.mark.parametrize("target_shape", [(6,), (2, 3), (3, 2), (1, 2, 3), (2, 3, 1)])
def test_reshape_every_subset_uses_row_major_coordinates(source_shape, target_shape):
    source = TensorRef("source", source_shape)
    target = TensorRef("target", target_shape)
    relation = ReshapeRelation(source, target)
    source_points, target_points = points(source_shape), points(target_shape)
    # Python's Cartesian-product order supplies the row-major oracle directly.
    mapping = dict(zip(source_points, target_points, strict=True))
    for removed in subsets(source_points):
        original = selected(source, removed)
        forward = relation.propagate(original)[0]
        assert actual_points(forward) == {mapping[p] for p in removed}, removed
        assert actual_points(relation.propagate(forward)[0]) == removed, removed


@pytest.mark.parametrize("dims", list(permutations(range(3))))
def test_permutation_every_subset_has_the_correct_inverse(dims):
    source = TensorRef("source", (1, 2, 3))
    target = TensorRef("target", tuple(source.shape[d] for d in dims))
    relation = PermuteRelation(source, target, dims)
    for removed in subsets(points(source.shape)):
        forward = relation.propagate(selected(source, removed))[0]
        assert actual_points(forward) == {tuple(p[d] for d in dims) for p in removed}, removed
        assert actual_points(relation.propagate(forward)[0]) == removed, removed


@pytest.mark.parametrize("small_shape", [(), (1,), (3,), (1, 3), (2, 1), (2, 3)])
def test_broadcast_requires_every_copy_before_reverse_removal(small_shape):
    small, big = TensorRef("small", small_shape), TensorRef("big", (2, 3))
    relation = BroadcastRelation(small, big)
    offset = len(big.shape) - len(small.shape)
    fibers = {
        point: {
            expanded
            for expanded in points(big.shape)
            if all(
                size == 1 or point[d] == expanded[offset + d] for d, size in enumerate(small.shape)
            )
        }
        for point in points(small.shape)
    }
    for removed in subsets(points(big.shape)):
        reverse = relation.propagate(selected(big, removed))[0]
        expected = {p for p, copies in fibers.items() if copies <= removed}
        assert actual_points(reverse) == expected, (small_shape, removed)
    for removed in subsets(points(small.shape)):
        forward = relation.propagate(selected(small, removed))[0]
        expected = {expanded for p in removed for expanded in fibers[p]}
        assert actual_points(forward) == expected, (small_shape, removed)
