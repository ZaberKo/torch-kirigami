"""Propagation explanations replay to the reported closure without duplicate deltas."""

import torch
from torch import nn

from torch_kirigami import DependencyGraph, Selection


def test_provenance_steps_reconstruct_exact_closure():
    model = nn.Sequential(nn.Linear(4, 6), nn.BatchNorm1d(6), nn.ReLU(), nn.Linear(6, 2)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(3, 4),))
    requests = [
        graph.parameter("0.weight").axis(0).select([1]),
        graph.parameter("3.weight").axis(1).select([4]),
    ]
    impact = graph.propagate(remove=requests)
    assert impact.status == "resolved" and impact.complete and impact.provenance
    known = {ref.id: Selection(ref) for ref in graph.values()}
    for selection in requests:
        known[selection.tensor.id] = known[selection.tensor.id].union(selection)
    for step in impact.provenance:
        assert step.reason and step.source and step.target
        assert not step.source.subtract(known[step.source.tensor.id])
        prior = known[step.target.tensor.id]
        assert step.target.subtract(prior) == step.target
        known[step.target.tensor.id] = prior.union(step.target)
    assert all(selection == impact.selection(selection.tensor) for selection in known.values())
    assert {selection.tensor.paths[0] for selection in impact.buffers} == {
        "1.running_mean",
        "1.running_var",
    }
    assert {selection.tensor.paths[0] for selection in impact.parameters} == {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
        "3.weight",
    }
