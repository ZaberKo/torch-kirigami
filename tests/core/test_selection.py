"""core / selection contracts."""

import itertools
import random

import pytest
import torch

from torch_kirigami import (
    AxisRef,
    IndexSet,
    NonEmpty,
    Region,
    ReshapeRelation,
    Selection,
    TensorRef,
)


def coordinates(selection):
    return {
        p
        for region in selection.regions
        for p in itertools.product(*(tuple(a) for a in region.axes))
    }


def test_selection_equality_is_geometric():
    ref = TensorRef("r", (3, 3))
    a = ref.axis(0).select([0]).union(ref.axis(1).select([0]))
    b = Selection(
        ref,
        (
            Region((IndexSet.span(0, 3), IndexSet.of([0]))),
            Region((IndexSet.of([0]), IndexSet.span(1, 3))),
        ),
    )
    assert a == b
    assert not a.subtract(b) and not b.subtract(a)
    with pytest.raises(TypeError):
        hash(a)


def test_axis_identity_and_empty_selection_validation():
    ref = TensorRef("a", [2, 3], paths=["weight"])
    assert AxisRef(ref, -1) == ref.axis(1)
    assert len({AxisRef(ref, -1), ref.axis(1)}) == 1
    for dim in (2, -3):
        with pytest.raises(IndexError):
            AxisRef(ref, dim)
        with pytest.raises(IndexError):
            Selection(ref).fully_selected_indices(dim)
    for dim in (True, 1.0):
        with pytest.raises(TypeError):
            AxisRef(ref, dim)
    with pytest.raises(IndexError):
        Selection(ref).fully_selected_indices(0, Region((IndexSet.span(0, 3), IndexSet.span(0, 3))))
    assert NonEmpty(TensorRef("empty", (0,)).axis(0)).check({}).severity == "conflict"


def test_empty_selection_preserves_zero_shape():
    ref = TensorRef("empty", (3, 0))
    selection = Selection(ref)
    assert not selection.fully_selected_indices(0)
    assert selection.compact_shape() == (3, 0)


def test_region_algebra_against_independent_sets():
    rng = random.Random(142)
    ref = TensorRef("test", (3, 4, 5))
    universe = set(itertools.product(range(3), range(4), range(5)))

    def sample():
        return Selection(
            ref,
            tuple(
                Region(
                    tuple(IndexSet.of(i for i in range(n) if rng.random() < 0.5) for n in ref.shape)
                )
                for _ in range(4)
            ),
        )

    for _ in range(80):
        a, b = sample(), sample()
        sa, sb = coordinates(a), coordinates(b)
        assert coordinates(a.union(b)) == sa | sb
        assert coordinates(a.subtract(b)) == sa - sb
        assert a.count == len(sa)
        assert a.union(a) == a
        for dim, n in enumerate(ref.shape):
            expected = {i for i in range(n) if {p for p in universe if p[dim] == i} <= sa}
            assert set(a.fully_selected_indices(dim)) == expected


@pytest.mark.parametrize(
    "source_shape,target_shape",
    [
        ((2, 3, 4), (4, 6)),
        ((4, 6), (2, 3, 4)),
        ((24,), (2, 3, 4)),
        ((2, 3, 4), (24,)),
    ],
)
def test_reshape_offsets_match_dense_oracle(source_shape, target_shape):
    source, target = TensorRef("s", source_shape), TensorRef("t", target_shape)
    relation = ReshapeRelation(source, target)
    for dim, n in enumerate(source_shape):
        chosen = source.axis(dim).select(range(0, n, 2))
        actual = coordinates(relation.propagate(chosen)[0])
        mask = torch.zeros(source_shape, dtype=torch.bool)
        for p in coordinates(chosen):
            mask[p] = True
        expected = {tuple(p) for p in mask.reshape(target_shape).nonzero().tolist()}
        assert actual == expected
        assert relation.propagate(relation.propagate(chosen)[0])[0] == chosen


def test_invalid_indices_and_foreign_selections():
    a, b = TensorRef("a", (3,)), TensorRef("b", (3,))
    with pytest.raises(IndexError):
        a.axis(0).select([3])
    with pytest.raises(ValueError):
        a.axis(0).select([-1])
    with pytest.raises(ValueError):
        a.axis(0).select([True])
    with pytest.raises(ValueError):
        a.axis(0).select([0]).union(b.axis(0).select([0]))


def test_large_simple_axis_does_not_materialize_elements():
    ref = TensorRef("large", (10**9, 4096))
    selection = ref.axis(1).select([3, 100])
    assert selection.count == 2 * 10**9
    assert len(selection.regions) == 1
    assert selection.compact_shape() == (10**9, 4094)


def test_interval_subtraction_matches_sets_across_shared_boundaries():
    rng = random.Random(182)
    cases = [
        (set(range(0, 2000, 2)), set(range(1, 2000, 2))),
        (set(range(0, 100, 2)), set(range(10, 90))),
    ]
    cases.extend(
        (set(rng.sample(range(100), 40)), set(rng.sample(range(100), 40))) for _ in range(100)
    )
    for left, right in cases:
        assert set(IndexSet.of(left).subtract(IndexSet.of(right))) == left - right
