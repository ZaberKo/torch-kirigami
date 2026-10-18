"""Irreversible proof metadata is independent of current failure and completeness."""

from dataclasses import replace
from itertools import combinations

import pytest
import torch
from torch import nn

from torch_kirigami import (
    Balanced,
    BlockBalance,
    DependencyGraph,
    Diagnostic,
    Fixed,
    IndexSet,
    NonEmpty,
    TensorRef,
)


def test_irreversible_diagnostic_validates_its_explicit_proof_declaration():
    assert not Diagnostic("current_failure", "Not valid now", "conflict").irreversible
    assert Diagnostic("known_failure", "Cannot recover", "conflict", irreversible=True).irreversible
    for value in (None, 0, 1, "yes"):
        with pytest.raises(TypeError, match="irreversibility"):
            Diagnostic("invalid", "Invalid flag", "conflict", irreversible=value)
    with pytest.raises(ValueError, match="must describe a conflict"):
        Diagnostic("invalid", "Unresolved", irreversible=True)


@pytest.mark.parametrize("kind", ["fixed", "nonempty", "balanced"])
def test_native_irreversible_violations_cannot_be_repaired_by_superset_removals(kind):
    ref = TensorRef("value", (4,))
    axis = ref.axis(0)
    constraint = {
        "fixed": Fixed(axis),
        "nonempty": NonEmpty(axis),
        "balanced": Balanced(axis, (IndexSet.of([0, 1]), IndexSet.of([2, 3]))),
    }[kind]
    initial = {"fixed": (0,), "nonempty": (0, 1, 2, 3), "balanced": (0, 1)}[kind]
    diagnostic = constraint.check({ref.id: axis.select(initial)})
    assert diagnostic.irreversible
    others = tuple(i for i in range(4) if i not in initial)
    for count in range(len(others) + 1):
        for addition in combinations(others, count):
            result = constraint.check({ref.id: axis.select((*initial, *addition))})
            assert result is not None and result.irreversible


def test_block_balance_can_recover_by_removing_the_empty_group():
    groups, members = TensorRef("groups", (2,)), TensorRef("members", (6,))
    constraint = BlockBalance(groups.axis(0), members.axis(0), 3)
    removed = {members.id: members.axis(0).select([0, 1, 2])}
    diagnostic = constraint.check(removed)
    assert diagnostic is not None and not diagnostic.irreversible
    assert constraint.check({**removed, groups.id: groups.axis(0).select([0])}) is None


def test_impact_aggregates_proofs_without_conflating_complete_or_conflicting():
    model = nn.Linear(3, 4).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 3),))
    selection = graph.parameter("weight").axis(0).select([0])
    impact = graph.propagate(remove=(selection,))
    assert impact.complete and not impact.irreversible
    current = Diagnostic("current", "Requires another choice", "conflict")
    assert not replace(impact, diagnostics=(current,), status="conflict").irreversible
    proof = Diagnostic("proof", "Cannot recover", "conflict", complete=False, irreversible=True)
    changed = replace(impact, diagnostics=(proof,), status="conflict")
    assert changed.irreversible and not changed.complete
