"""Compressed index membership avoids expanding individual positions."""

import pytest

from torch_kirigami import IndexSet


def test_interval_membership_does_not_expand_elements(monkeypatch):
    indices = IndexSet(((0, 2), (100, 1_000_000_000)))

    def forbidden(_):
        pytest.fail("Membership must inspect intervals, not enumerate positions")

    with monkeypatch.context() as patch:
        patch.setattr(IndexSet, "__iter__", forbidden)
        assert 0 in indices and 1 in indices and 999_999_999 in indices
        assert -1 not in indices and 2 not in indices and 1_000_000_000 not in indices
    assert 1.0 in IndexSet.span(0, 2) and "1" not in IndexSet.span(0, 2)

    class DifferentOrder(int):
        def __lt__(self, other):
            return True

    assert DifferentOrder(1) in IndexSet.span(0, 2)
