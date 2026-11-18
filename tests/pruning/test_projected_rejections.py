"""Projected rejection proofs agree with ordinary propagation and keep their invariants."""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

from torch_kirigami import Balanced, DependencyGraph, Divisible, IndexSet, config
from torch_kirigami.dep_graph.contracts import IRREVERSIBLE_CONSTRAINTS, alignment_conflicts
from torch_kirigami.pruning import Candidate, ChannelRatio, PlanningContext
from torch_kirigami.pruning.candidate_checks import CandidateFilter

ROOT = Path(__file__).resolve().parents[2]


def _irreversible(graph, constraints, seeds):
    """Reference rejection reasons from one ordinary propagation."""
    impact = graph.propagate(remove=seeds, constraints=constraints)
    return {d for d in impact.diagnostics if d.irreversible}


@pytest.mark.parametrize("cache_size", [0, 1, 4096])
def test_projected_rejections_match_propagation_for_changing_accepted_impacts(
    cache_size, monkeypatch
):
    config.candidate_cache_size = cache_size
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("0.weight").axis(0)
    column = graph.parameter("2.weight").axis(1)
    # Balanced reports an irreversible empty partition; Divisible stays repairable.
    constraints = (Balanced(axis, (IndexSet.span(0, 2), IndexSet.span(2, 6))), Divisible(axis, 4))
    check = CandidateFilter(graph, constraints)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(6))
    # The column candidate selects the same component through another member.
    candidates += (Candidate("column", (column.select([1]),), column),)
    accepted = [graph.propagate(remove=(), constraints=constraints)]
    accepted += [
        graph.propagate(remove=(axis.select(positions),), constraints=constraints)
        for positions in ([0], [0, 2], [2, 3, 4])
    ]
    expected = {
        (index, candidate.key): _irreversible(
            graph, constraints, (*impact.requested, *candidate.remove)
        )
        for index, impact in enumerate(accepted)
        for candidate in candidates
    }
    assert any(expected.values()) and not all(expected.values())
    calls = []
    original = graph.propagate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "propagate", counted)
    # Revisit every accepted impact, so later passes reuse the bounded proofs.
    for _ in range(2):
        for index, impact in enumerate(accepted):
            for candidate in candidates:
                actual = check.rejections(candidate, impact)
                assert set(actual) == expected[index, candidate.key]
                assert len(actual) == len(set(actual))
    assert not calls
    assert len(check._member_rejections) <= cache_size
    assert len(check._domains) <= cache_size
    assert len(check._accepted_proofs) <= cache_size


@pytest.mark.parametrize("query", ["rejections", "fixed_rejection"])
def test_zero_cache_size_clears_every_cache_at_the_next_query(query):
    """A new bound applies at the next query, also when that query is projected."""
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("0.weight").axis(0)
    check = CandidateFilter(graph, ())
    projected = Candidate("row", (axis.select([1]),), axis)
    # A request without a declared axis is never projected and fills `_rejections`.
    ordinary = Candidate("raw", (axis.select([2]),))
    accepted = graph.propagate(remove=(axis.select([0]),))
    members = check.index.component(axis)
    assert check.rejections(ordinary) == ()
    assert check.rejections(projected) == ()
    assert check.rejections(projected, accepted) == ()
    assert check.complete_positions(members, graph.propagate(remove=()))
    caches = ("_rejections", "_domains", "_member_rejections", "_complete_positions")
    assert all(getattr(check, name) for name in caches) and check._accepted_proofs
    config.candidate_cache_size = 0
    if query == "rejections":
        assert check.rejections(projected, accepted) == ()
    else:
        assert check.fixed_rejection(axis) is None
    assert not any(getattr(check, name) for name in (*caches, "_accepted_proofs"))


@pytest.mark.parametrize("cache_size", [0, 1])
def test_planning_filter_capacity_covers_one_scan_of_its_candidates(cache_size, monkeypatch):
    """A rescan after an accepted change reuses every proof unless caching is disabled."""
    config.candidate_cache_size = cache_size
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    axis = graph.parameter("0.weight").axis(0)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(6))
    context = PlanningContext(graph, candidates, ChannelRatio(0.5), (axis,), ())
    check = context._candidate_filter
    assert check.capacity == (12 if cache_size else 0)
    accepted = context.impact((axis.select([0]),))
    for candidate in candidates:
        check.rejections(candidate, accepted)
    proofs = []
    original = CandidateFilter._check_members

    def counted(*args, **kwargs):
        proofs.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(CandidateFilter, "_check_members", counted)
    for candidate in candidates:
        assert check.rejections(candidate, accepted) == ()
    assert len(proofs) == (len(candidates) if not cache_size else 0)
    assert len(check._domains) == (len(candidates) if cache_size else 0)


@pytest.mark.parametrize("cache_size", [0, 4096])
def test_cross_axis_accepted_removals_match_propagation(cache_size, monkeypatch):
    """Components sharing a weight keep exact projections after each other's removals."""
    config.candidate_cache_size = cache_size
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 4), nn.Linear(4, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    rows = graph.parameter("0.weight").axis(0)
    hidden = graph.parameter("2.weight").axis(0)
    # The rows of `2.weight` belong to one component and its columns to another.
    constraints = (
        Balanced(hidden, (IndexSet.span(0, 2), IndexSet.span(2, 4))),
        Divisible(rows, 2),
    )
    check = CandidateFilter(graph, constraints)
    columns = graph.parameter("2.weight").axis(1)
    candidates = (
        *(Candidate(f"row{i}", (rows.select([i]),), rows) for i in range(6)),
        *(Candidate(f"hidden{i}", (hidden.select([i]),), hidden) for i in range(4)),
        Candidate("column", (columns.select([1]),), columns),
    )
    accepted = [
        graph.propagate(remove=seeds, constraints=constraints)
        for seeds in (
            (hidden.select([0]),),
            (hidden.select([0, 3]),),
            (rows.select([2]),),
            (rows.select([1, 4]), hidden.select([1])),
            (rows.select(range(5)), hidden.select([2])),
        )
    ]
    expected = {
        (index, candidate.key): _irreversible(
            graph, constraints, (*impact.requested, *candidate.remove)
        )
        for index, impact in enumerate(accepted)
        for candidate in candidates
    }
    components = (check.index.component(rows), check.index.component(hidden))

    def proved(impact, members):
        """Reference complete positions, excluding removal of the last position."""
        root = members[0]
        removed = impact.selection(root.tensor).fully_selected_indices(root.dim)
        surviving = [i for i in range(root.tensor.shape[root.dim]) if i not in removed]
        if len(surviving) <= 1:
            return frozenset()
        return frozenset(
            position
            for position in surviving
            if graph.propagate(
                remove=(*impact.requested, root.select([position])), constraints=constraints
            ).complete
        )

    complete = {
        (index, members): proved(impact, members)
        for index, impact in enumerate(accepted)
        for members in components
    }
    assert sum(map(bool, complete.values())) == len(complete) - 1
    assert any(check.component_state(impact, components[0])[1] for impact in accepted)
    calls = []
    original = graph.propagate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph, "propagate", counted)
    for _ in range(2):
        for index, impact in enumerate(accepted):
            for candidate in candidates:
                calls.clear()
                actual = check.rejections(candidate, impact)
                assert set(actual) == expected[index, candidate.key]
                assert len(actual) == len(set(actual))
                # A whole-axis request can activate other relations, so only it
                # keeps ordinary propagation.
                axis = candidate.axis
                whole = impact.selection(axis.tensor).union(candidate.remove[0])
                assert bool(calls) == (len(whole.fully_selected_indices(axis.dim)) == 6)
            for members in components:
                proved_positions = check.complete_positions(members, impact)
                assert set(proved_positions) == complete[index, members]
                # A subset query proves exactly its share of the positions.
                subset = IndexSet.of(list(complete[index, members])[:1])
                assert set(check.complete_positions(members, impact, subset)) == set(subset)
    assert not calls
    assert len(check._accepted_proofs) <= cache_size


def test_only_listed_constraint_types_construct_irreversible_diagnostics():
    """Rejection projections skip every other built-in constraint type."""
    listed = {
        (c.__module__, c.__qualname__) for c in (*IRREVERSIBLE_CONSTRAINTS, alignment_conflicts)
    }
    # The joint proof runs in both propagation and `CandidateFilter` projections.
    listed.add((alignment_conflicts.__module__, alignment_conflicts.__qualname__))
    found = set()
    for path in (ROOT / "torch_kirigami").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            # Module-level functions and statements count as their own owners.
            owner = getattr(node, "name", "<module>")
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id", getattr(call.func, "attr", None))
                # `irreversible` is the seventh `Diagnostic` field.
                positional = name == "Diagnostic" and len(call.args) >= 7
                if positional or any(
                    keyword.arg == "irreversible"
                    and not (
                        isinstance(keyword.value, ast.Constant) and keyword.value.value is False
                    )
                    for keyword in call.keywords
                ):
                    found.add((module, owner))
    assert found == listed
