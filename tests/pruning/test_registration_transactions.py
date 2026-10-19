"""integration / transactions contracts."""

import io

import pytest
import torch
from torch import nn
from torch.nn.modules.module import register_module_parameter_registration_hook

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ExecutionError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_registration_replacement_cannot_break_committed_aliases(checkpoint, execution_device):
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    model[0].alias = model[0].weight
    model.cached = [model[0].weight]
    graph = DependencyGraph.build(model, args=(torch.ones(1, 4),))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1])])
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    old, cached = model[0].weight, model.cached

    def clone_on_registration(module, name, parameter):
        return nn.Parameter(parameter.detach().clone(), requires_grad=parameter.requires_grad)

    handle = register_module_parameter_registration_hook(clone_on_registration)
    try:
        with pytest.raises(ExecutionError, match=r"Commit|registration|binding"):
            if checkpoint:
                load_checkpoint(model, stream)
            else:
                pruner.apply(plan)
    finally:
        handle.remove()
    assert model[0].weight is model[0].alias is old
    assert model.cached is cached and cached[0] is old
