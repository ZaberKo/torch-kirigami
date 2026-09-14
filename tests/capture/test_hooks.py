"""capture / hooks contracts."""

import pytest
import torch
from torch import nn

from tests.support.models import CachedWeight, Chain
from tests.support.pruning import build
from torch_kirigami import (
    CaptureError,
    DependencyGraph,
)
from torch_kirigami.pruning import (
    ExecutionError,
    Pruner,
    PruningPlan,
)


def test_forward_hook_is_rejected_before_its_unmodeled_effects():
    model = nn.Linear(4, 6)
    model.register_forward_hook(lambda module, args, output: output.flip(-1))
    with pytest.raises(CaptureError, match="hooks"):
        build(model, torch.randn(2, 4))


@pytest.mark.parametrize("pre", [False, True])
def test_global_forward_hooks_rejected_at_build_and_apply(pre):
    from torch.nn.modules import module as runtime

    model = CachedWeight()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan_remove([graph.parameter("weight").axis(0).select([1])])
    register = (
        runtime.register_module_forward_pre_hook if pre else runtime.register_module_forward_hook
    )
    handle = register(lambda *args: None)
    try:
        with pytest.raises(CaptureError, match="global"):
            DependencyGraph.build(model, args=(torch.randn(2, 4),))
        with pytest.raises(ExecutionError, match="global"):
            Pruner(model).apply(plan)
    finally:
        handle.remove()


@pytest.mark.parametrize("where", ["root", "nested", "leaf"])
@pytest.mark.parametrize("pre", [False, True])
def test_forward_hooks_rejected_before_capture(where, pre):
    model = nn.Sequential(Chain())
    target = {"root": model, "nested": model[0], "leaf": model[0].a}[where]
    calls = []
    if pre:
        target.register_forward_pre_hook(lambda *args: calls.append(True))
    else:
        target.register_forward_hook(lambda *args: calls.append(True))
    with pytest.raises(CaptureError, match="hook"):
        DependencyGraph.build(model, args=(torch.randn(2, 4),))
    assert not calls


@pytest.mark.parametrize("where", ["root", "leaf"])
def test_apply_rechecks_added_hooks(where):
    model = Chain()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    plan = Pruner(model, graph=graph).plan_remove([graph.parameter("a.weight").axis(0).select([1])])
    (model if where == "root" else model.a).register_forward_hook(lambda *args: None)
    old = model.a.weight
    with pytest.raises(ExecutionError, match="hook"):
        Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    assert model.a.weight is old
