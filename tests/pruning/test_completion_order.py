"""Completion scans preserve stable preference order and stop on accepted prefixes."""

import itertools

import pytest

from torch_kirigami import IndexSet, TensorRef
from torch_kirigami.pruning import Candidate
from torch_kirigami.pruning.planner import _completion_order, _ranked_axis_candidates


@pytest.mark.parametrize("known_only", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_completion_order_matches_exhaustive_set_reference(known_only: bool, reverse: bool) -> None:
    axis = TensorRef("channels", (4,)).axis(0)
    unrelated = TensorRef("unrelated", (1,)).axis(0)
    subsets = tuple(
        frozenset(i for i, present in enumerate(bits) if present)
        for bits in itertools.product((False, True), repeat=4)
    )
    candidates = tuple(
        Candidate(str(i), (axis.select(indices) if indices else unrelated.select([0]),))
        for i, indices in enumerate(subsets)
    )
    ranked = candidates[::-1] if reverse else candidates
    removals = {
        c.key: ({axis: IndexSet.of(indices)} if indices else {})
        for c, indices in zip(candidates, subsets, strict=True)
    }
    by_axis = _ranked_axis_candidates(ranked, removals)
    partition_cases = [(), *((indices,) for indices in subsets)]
    partition_cases.extend([({0, 1}, {1, 2}), ({0, 2}, {1, 3})])
    for before, partitions in itertools.product(subsets, partition_cases):
        # Use ordinary Python sets, independent of the interval implementation.
        preferred, remaining = [], []
        for candidate in ranked:
            delta = subsets[int(candidate.key)] - before
            helps = bool(delta) and (not partitions or any(delta & p for p in partitions))
            (preferred if helps else remaining).append(candidate)
        expected = preferred if known_only else preferred + remaining
        actual = _completion_order(
            ranked,
            removals,
            axis,
            IndexSet.of(before),
            tuple(IndexSet.of(p) for p in partitions),
            contributors=by_axis.get(axis, ()),
            known_only=known_only,
        )
        assert tuple(actual) == tuple(expected)


@pytest.mark.parametrize("known_only", [False, True])
def test_completion_does_not_scan_after_the_first_helpful_candidate(
    known_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    axis = TensorRef("channels", (4,)).axis(0)
    ranked = tuple(Candidate(str(i), (axis.select([i % 4]),)) for i in range(1_000))
    removals = {c.key: {axis: IndexSet.of([i % 4])} for i, c in enumerate(ranked)}
    calls = []
    subtract = IndexSet.subtract

    def counted(self: IndexSet, other: IndexSet) -> IndexSet:
        calls.append(self)
        return subtract(self, other)

    monkeypatch.setattr(IndexSet, "subtract", counted)
    order = _completion_order(
        ranked,
        removals,
        axis,
        IndexSet.of([0]),
        (),
        contributors=_ranked_axis_candidates(ranked, removals)[axis],
        known_only=known_only,
    )
    assert not calls
    assert next(order) == ranked[1]
    assert len(calls) == 2
    assert next(order) == ranked[2]
    assert len(calls) == 3


@pytest.mark.parametrize("known_only", [False, True])
def test_axis_index_skips_unrelated_candidates_without_losing_fallback_order(
    known_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    axis = TensorRef("channels", (4,)).axis(0)
    other = TensorRef("other", (4,)).axis(0)
    ranked = (
        *(Candidate(f"unrelated_{i}", (other.select([0]),)) for i in range(1_000)),
        Candidate("covered", (axis.select([0]),)),
        Candidate("other_partition", (axis.select([2]),)),
        Candidate("helpful", (axis.select([1]), other.select([1]))),
    )
    removals = {
        candidate.key: {
            selection.tensor.axis(0): selection.fully_selected_indices(0)
            for selection in candidate.remove
        }
        for candidate in ranked
    }
    by_axis = _ranked_axis_candidates(ranked, removals)
    assert tuple(by_axis[axis]) == ranked[-3:]
    assert tuple(by_axis[other]) == (*ranked[:-3], ranked[-1])
    calls = []
    subtract = IndexSet.subtract

    def counted(self: IndexSet, before: IndexSet) -> IndexSet:
        calls.append(self)
        return subtract(self, before)

    monkeypatch.setattr(IndexSet, "subtract", counted)
    order = _completion_order(
        ranked,
        removals,
        axis,
        IndexSet.of([0]),
        (IndexSet.of([0, 1]),),
        contributors=by_axis[axis],
        known_only=known_only,
    )
    assert next(order) == ranked[-1]
    assert len(calls) == 3  # No empty-axis subtraction for the unrelated entries.
    assert tuple(order) == (() if known_only else ranked[:-1])


@pytest.mark.parametrize("known_only", [False, True])
def test_axis_without_individual_contributions_retains_joint_only_fallback(
    known_only: bool,
) -> None:
    axis = TensorRef("channels", (4,)).axis(0)
    other = TensorRef("other", (1,)).axis(0)
    ranked = (Candidate("seed", (other.select([0]),)),)
    removals = {"seed": {other: IndexSet.of([0])}}
    by_axis = _ranked_axis_candidates(ranked, removals)
    assert tuple(
        _completion_order(
            ranked,
            removals,
            axis,
            IndexSet(),
            (),
            contributors=by_axis.get(axis, ()),
            known_only=known_only,
        )
    ) == (() if known_only else ranked)
