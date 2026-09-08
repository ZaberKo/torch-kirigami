import itertools
import random

import pytest
import torch

from torch_kirigami import IndexSet, Region, Selection, TensorRef
from torch_kirigami.relations import ReshapeRelation


def coordinates(selection):
    return {
        p
        for region in selection.regions
        for p in itertools.product(*(tuple(a) for a in region.axes))
    }


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
            assert set(a.project(dim)) == expected


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
