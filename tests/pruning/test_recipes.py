"""pruning / recipes contracts."""

from dataclasses import replace

import torch
from torch import nn

from torch_kirigami import (
    DependencyGraph,
    IndexSet,
    Region,
    TensorRef,
)
from torch_kirigami.pruning import (
    TensorRecipe,
)
from torch_kirigami.pruning.recipes import same_mapping


def test_partial_parameter_region_is_not_claimed_compact():
    graph = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    weight = graph.parameter("weight")
    impact = graph.propagate(remove=[weight.select([Region((IndexSet.of([1]), IndexSet.of([2])))])])
    assert impact.status == "unresolved"


def test_mapping_requires_identical_source_domains():
    ref = TensorRef("p", (4,), "parameter", ("weight",))
    first = TensorRecipe(ref, (Region((IndexSet.of([0, 1]),)),))
    other = TensorRecipe(ref, (Region((IndexSet.of([2, 3]),)),))
    assert not same_mapping(first, other)
    assert not same_mapping(first, replace(first, tensor=replace(ref, id="other")))
    split = TensorRecipe(ref, (Region((IndexSet.of([0]),)), Region((IndexSet.of([1]),))))
    assert same_mapping(first, split)
