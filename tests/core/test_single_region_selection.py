"""Single-region shortcuts agree with Cartesian coordinate references."""

from itertools import combinations, product

import pytest

from torch_kirigami import IndexSet, Region, Selection, TensorRef


def subsets(size: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate axis subsets independently of the interval representation."""
    return tuple(
        selected for count in range(size + 1) for selected in combinations(range(size), count)
    )


@pytest.mark.parametrize("shape", [(0,), (2, 0), (3,), (2, 3), (1, 2, 3), (3, 2, 1, 2)])
def test_every_cartesian_rectangle_matches_complete_coordinate_fibers(shape):
    ref = TensorRef("rectangle", shape)
    universe = set(product(*(range(size) for size in shape)))
    for axes in product(*(subsets(size) for size in shape)):
        removed = set(product(*axes))
        selection = Selection(ref, (Region(tuple(IndexSet.of(axis) for axis in axes)),))
        for dim in range(-len(shape), len(shape)):
            axis = dim % len(shape)
            expected = {
                index
                for index in range(shape[axis])
                if (fiber := {point for point in universe if point[axis] == index})
                and fiber <= removed
            }
            assert set(selection.fully_selected_indices(dim)) == expected, (shape, axes, dim)


def test_fragmented_indices_and_coalesced_regions_preserve_original_positions():
    ref = TensorRef("fragmented", (9, 7, 2))
    indices = IndexSet.of([0, 2, 5, 8])
    selection = Selection(
        ref,
        (
            Region((indices, IndexSet.span(0, 3), IndexSet.span(0, 2))),
            Region((indices, IndexSet.span(3, 7), IndexSet.span(0, 2))),
        ),
    )
    assert len(selection.regions) == 1
    assert tuple(selection.fully_selected_indices(0)) == (0, 2, 5, 8)
    assert not selection.fully_selected_indices(1)
    assert not selection.fully_selected_indices(2)
    assert selection.compact_shape() == (5, 7, 2)


@pytest.mark.parametrize("empty", [False, True])
def test_explicit_scope_keeps_partial_fibers_and_input_validation(empty):
    ref = TensorRef("scoped", (4, 6))
    region = Region((IndexSet.of([1, 3]), IndexSet.span(0, 3)))
    selection = Selection(ref) if empty else Selection(ref, (region,))
    assert not selection.fully_selected_indices(0)
    assert tuple(selection.fully_selected_indices(0, region)) == (() if empty else (1, 3))
    assert tuple(selection.fully_selected_indices(1, region)) == (() if empty else (0, 1, 2))
    with pytest.raises(IndexError):
        selection.fully_selected_indices(0, Region((IndexSet.span(0, 5), IndexSet.span(0, 6))))
    with pytest.raises(ValueError, match="rank"):
        selection.fully_selected_indices(0, Region((IndexSet.span(0, 4),)))
    for dim in (-3, 2):
        with pytest.raises(IndexError):
            selection.fully_selected_indices(dim)
    for dim in (True, 0.0):
        with pytest.raises(TypeError):
            selection.fully_selected_indices(dim)


def test_scalar_selection_rejects_every_axis():
    ref = TensorRef("scalar", ())
    for selection in (Selection(ref), Selection(ref, (Region(()),))):
        for dim in (-1, 0, 1):
            with pytest.raises(IndexError):
                selection.fully_selected_indices(dim)


def test_index_set_subclasses_return_plain_index_sets():
    class DerivedIndexSet(IndexSet):
        pass

    ref = TensorRef("subclass", (5, 3))
    selection = Selection(
        ref,
        (Region((DerivedIndexSet(((0, 1), (3, 5))), DerivedIndexSet(((0, 3),)))),),
    )
    rows = selection.fully_selected_indices(0)
    columns = selection.fully_selected_indices(1)
    assert type(rows) is type(columns) is IndexSet
    assert rows == IndexSet.of([0, 3, 4])
    assert not columns
