"""Shared identity proofs agree with propagation and reject unproved mappings."""

from dataclasses import replace

import torch
from torch import nn

from torch_kirigami import AxisRelation, DependencyGraph
from torch_kirigami.pruning import IdentityAxisIndex
from torch_kirigami.selection import full_region


def test_identity_component_matches_every_single_position(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    root = graph.parameter("0.weight").axis(0)
    index = IdentityAxisIndex(graph.relations)
    axes = index.component(root)
    assert axes is not None
    for position in range(6):
        actual = graph.propagate(remove=(root.select([position]),))
        expected = {axis.tensor.id: axis.select([position]) for axis in axes}
        assert expected == dict(actual.selections)
    assert all(index.component(axis) == axes for axis in axes)


def test_identity_component_rejects_scopes_extensions_and_full_width(execution_device):
    class ExtendedRelation(AxisRelation):
        """A subclass is not covered by the exact built-in proof."""

    model = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    root = graph.parameter("0.weight").axis(0)
    relations = list(graph.relations)
    offset = next(
        i
        for i, relation in enumerate(relations)
        if type(relation) is AxisRelation and root.tensor in relation.refs
    )
    relation = relations[offset]
    relations[offset] = replace(
        relation, left=replace(relation.left, scope=full_region(relation.left.tensor.shape))
    )
    assert IdentityAxisIndex(relations).component(root) is None
    relations[offset] = ExtendedRelation(
        relation.left, relation.right, relation.maps, relation.reason
    )
    assert IdentityAxisIndex(relations).component(root) is None
    singleton = nn.Linear(3, 1)
    small = DependencyGraph.build(singleton, args=(torch.randn(2, 3),))
    assert IdentityAxisIndex(small.relations).component(small.parameter("weight").axis(0)) is None


def test_equivalent_seeds_keep_full_closure_with_nonidentity_relations(execution_device):
    model = nn.Sequential(
        nn.Conv2d(4, 6, 1, groups=2), nn.BatchNorm2d(6), nn.ReLU(), nn.Conv2d(6, 2, 1)
    ).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 4, 2, 2),))
    root = graph.parameter("0.weight").axis(0)
    index = IdentityAxisIndex(graph.relations)
    assert index.component(root) is None
    axes = index.equivalent_axes(root)
    assert len(axes) > 1
    for positions in ([0], [3], [0, 3], list(range(6))):
        reference = graph.propagate(remove=(root.select(positions),))
        for axis in axes:
            actual = graph.propagate(remove=(axis.select(positions),))
            assert dict(actual.selections) == dict(reference.selections)
            assert actual.diagnostics == reference.diagnostics
    assert all(index.equivalent_axes(axis) == axes for axis in axes)
