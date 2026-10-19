"""Portable decisions retain a complete reference catalog across serialization."""

import io
from dataclasses import replace

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, TensorRef
from torch_kirigami.pruning import Pruner, PruningPlan, load_checkpoint, save_checkpoint


def test_portable_queries_cover_affected_unaffected_and_unknown(execution_device):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    ref = graph.parameter("0.weight")
    impact = graph.propagate(remove=[ref.axis(0).select([1])])
    plan = Pruner(model, graph=graph).plan_remove([ref.axis(0).select([1])])
    assert set(plan.to_dict()) == {"format", "plan"}
    plan = PruningPlan.from_dict(plan.to_dict())
    for tensor in graph.values():
        assert plan.analysis.selection(tensor).regions == impact.selection(tensor).regions
        assert (
            plan.analysis.selection(tensor.portable()).regions == impact.selection(tensor).regions
        )
    unknown = TensorRef("unknown", (4,))
    with pytest.raises(KeyError):
        plan.analysis.selection(unknown)
    with pytest.raises(ValueError):
        plan.analysis.selection(replace(ref, shape=(7, 4)))
    _, result = Pruner(model, graph=graph).apply(plan)
    assert result.coordinate_maps[ref] == result.coordinate_maps[ref.portable()]
    with pytest.raises(KeyError):
        result.coordinate_maps[unknown]
    with pytest.raises(TypeError):
        result.coordinate_maps[ref] = ()
    model(x).sum().backward()

    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    assert set(payload) == {"format", "structure", "managed", "state_dict", "nonpersistent_buffers"}
    assert "version" not in model._kirigami_structure
    stream.seek(0)
    restored = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    load_checkpoint(restored, stream)
    torch.testing.assert_close(restored(x), model(x))
