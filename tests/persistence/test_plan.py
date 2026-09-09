"""persistence / plan contracts."""

import copy
import gc
import io
import weakref
from dataclasses import replace

import pytest
import torch

from tests.support.models import _context, chain
from torch_kirigami import (
    DependencyGraph,
)
from torch_kirigami.pruning import (
    ExecutionError,
    Pruner,
    PruningPlan,
)


def test_plan_direct_construction_is_static_and_bad_decode_is_value_error():
    context, candidate = _context()
    plan = Pruner(context.graph.model, graph=context.graph).plan(
        remove=candidate.remove, preserve_io=False
    )
    notes = ["note"]
    rebuilt = replace(plan, notes=notes, recipes=list(plan.recipes), selected=[])
    notes.append("later")
    assert rebuilt.notes == ("note",) and isinstance(rebuilt.recipes, tuple)
    data = plan.to_dict()
    data["plan"]["fields"]["recipes"]["tuple"][0]["fields"]["concat_dim"] = 10
    with pytest.raises(ValueError, match="TensorRecipe"):
        PruningPlan.from_dict(data)


def test_static_plan_is_pure_repeatable_and_does_not_retain_graph(execution_device):
    model = chain()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    remove = [graph.parameter("0.weight").axis(0).select([1, 4])]
    state = dict(pruner.__dict__)
    rng = torch.get_rng_state().clone()
    first = pruner.plan(remove=remove)
    second = pruner.plan(remove=remove)
    assert first.to_dict() == second.to_dict()
    assert state == pruner.__dict__
    assert torch.equal(rng, torch.get_rng_state())
    original = copy.deepcopy(model)
    reference = weakref.ref(graph)
    del graph, pruner, remove, state
    gc.collect()
    assert reference() is None
    stream = io.BytesIO()
    torch.save(first.to_dict(), stream)
    stream.seek(0)
    restored = PruningPlan.from_dict(torch.load(stream, weights_only=True))
    returned, result = Pruner(model).apply(restored)
    assert returned is model and not hasattr(result, "model")
    Pruner(original).apply(restored)
    torch.testing.assert_close(model(x), original(x))
    with pytest.raises(ExecutionError, match="preconditions"):
        Pruner(model).apply(restored)
