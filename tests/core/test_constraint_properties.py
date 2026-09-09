"""Small-domain constraint and block-map checks with independent set oracles."""

from itertools import combinations, product

import pytest

from torch_kirigami import (
    AxisBarrier,
    AxisPort,
    AxisRelation,
    Balanced,
    Barrier,
    BlockBalance,
    BlockMap,
    Divisible,
    Fixed,
    IndexSet,
    LayoutConstraint,
    NonEmpty,
    Region,
    Selection,
    TensorRef,
)


def subsets(values):
    values = tuple(values)
    return [set(items) for size in range(len(values) + 1) for items in combinations(values, size)]


def test_axis_constraints_match_every_small_removal_without_completing_it():
    ref = TensorRef("channels", (6, 2))
    axis = ref.axis(0)
    constraints = {
        "fixed": Fixed(axis),
        "nonempty": NonEmpty(axis),
        "divisible": Divisible(axis, 2),
        "balanced": Balanced(axis, (IndexSet.span(0, 3), IndexSet.span(3, 6))),
        "barrier": AxisBarrier(axis, "fixed call axis"),
        "opaque": Barrier((ref,), "unknown"),
        "layout": LayoutConstraint(ref),
    }
    for removed in subsets(range(6)):
        selection = axis.select(removed)
        selections = {ref.id: selection}
        counts = [len(set(range(start, start + 3)) - removed) for start in (0, 3)]
        expected = {
            "fixed": bool(removed),
            "nonempty": len(removed) == 6,
            "divisible": (6 - len(removed)) % 2 != 0,
            "balanced": 0 in counts or counts[0] != counts[1],
            "barrier": bool(removed),
            "opaque": bool(removed),
            "layout": False,
        }
        for name, constraint in constraints.items():
            result = constraint.check(selections)
            assert (result is not None) == expected[name], (name, removed)
            if result is not None:
                assert result.complete == (name not in ("barrier", "opaque"))
                assert result.tensors == (ref.id,)
            assert selections == {ref.id: selection}
            assert tuple(selection.fully_selected_indices(0)) == tuple(sorted(removed))


def test_block_balance_matches_surviving_group_counts_exhaustively():
    groups, members = TensorRef("groups", (3,)), TensorRef("members", (6,))
    constraint = BlockBalance(groups.axis(0), members.axis(0), 2)
    for removed_groups, removed_members in product(subsets(range(3)), subsets(range(6))):
        selections = {
            groups.id: groups.axis(0).select(removed_groups),
            members.id: members.axis(0).select(removed_members),
        }
        counts = [
            len({2 * group, 2 * group + 1} - removed_members)
            for group in range(3)
            if group not in removed_groups
        ]
        expected_failure = bool(counts) and (min(counts) == 0 or len(set(counts)) > 1)
        result = constraint.check(selections)
        assert (result is not None) == expected_failure, (removed_groups, removed_members)
        if result is not None:
            assert result.code == "unbalanced_blocks" and result.complete


@pytest.mark.parametrize("source_width,target_width", [(1, 2), (2, 1), (2, 3)])
@pytest.mark.parametrize("full_source,full_target", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("reverse", [False, True])
def test_block_maps_match_explicit_block_sets(
    source_width, target_width, full_source, full_target, reverse
):
    mapping = BlockMap(1, 2, 2, source_width, target_width, full_source, full_target)
    source_blocks = [set(range(1 + b * source_width, 1 + (b + 1) * source_width)) for b in range(2)]
    target_blocks = [set(range(2 + b * target_width, 2 + (b + 1) * target_width)) for b in range(2)]
    if reverse:
        source_blocks, target_blocks = target_blocks, source_blocks
    require_full = full_target if reverse else full_source
    for chosen in subsets(range(max(set.union(*source_blocks)) + 2)):
        expected = set()
        for source, target in zip(source_blocks, target_blocks, strict=True):
            if (source <= chosen) if require_full else bool(source & chosen):
                expected.update(target)
        assert set(mapping.map(IndexSet.of(chosen), reverse)) == expected, chosen


def test_scoped_axis_relation_requires_complete_cross_sections():
    source, target = TensorRef("source", (3, 4)), TensorRef("target", (2, 6))
    left = AxisPort(source.axis(1), Region((IndexSet.of([0, 2]), IndexSet.of([1, 2, 3]))))
    right = AxisPort(target.axis(1), Region((IndexSet.of([1]), IndexSet.of([0, 1, 4, 5]))))
    relation = AxisRelation(left, right, (BlockMap(1, 0, 3, 1, 2, require_full_target=True),))
    mapping = {1: {0, 1}, 2: set(), 3: {4, 5}}
    for chosen in subsets(product((0, 2), (1, 2, 3))):
        selection = Selection(
            source, tuple(Region(tuple(IndexSet.of([i]) for i in p)) for p in chosen)
        )
        complete = {column for column in (1, 2, 3) if {(0, column), (2, column)} <= chosen}
        expected = {(1, column) for original in complete for column in mapping[original]}
        propagated = relation.propagate(selection)
        actual = {
            p for output in propagated for region in output.regions for p in product(*region.axes)
        }
        assert actual == expected, chosen
        assert set(left.fully_selected_indices(selection)) == complete


def test_partial_regions_distinguish_fixed_layout_and_opaque_barriers():
    ref = TensorRef("matrix", (2, 3))
    partial = Selection(ref, (Region((IndexSet.of([0]), IndexSet.of([1]))),))
    state = {ref.id: partial}
    assert LayoutConstraint(ref).check(state).code == "unsupported_layout"
    assert Fixed(ref.axis(1)).check(state).code == "partitioned_constraint"
    assert Divisible(ref.axis(1), 1).check(state).code == "partitioned_constraint"
    assert AxisBarrier(ref.axis(1), "axis").check(state) is None
    assert not Barrier((ref,), "opaque").check(state).complete
