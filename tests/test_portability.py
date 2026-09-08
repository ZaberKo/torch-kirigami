import copy
import gc
import io
import subprocess
import sys
import weakref

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ExecutionError,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def chain():
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))


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


def test_prune_wrapper_and_empty_plan():
    model = chain()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    p = Pruner(model, graph=graph)
    plan = p.plan(remove=[])
    assert p.apply(plan)[0] is model
    assert p.apply(plan)[0] is model
    returned, result = p.prune(remove=[graph.parameter("0.weight").axis(0).select([1])])
    assert returned is model and result.plan.analysis.status == "resolved"


class WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)
        self.last = nn.Linear(6, 2)
        self.alias = self.fc
        self.register_buffer("offset", torch.randn(6), persistent=False)

    def forward(self, x):
        return self.last((self.fc(x) + self.offset).relu())


def test_checkpoint_final_structure_after_two_rounds(execution_device):
    model = WithBuffer()
    x = torch.randn(2, 4)
    for _ in range(2):
        graph = DependencyGraph.build(model, args=(x,))
        Pruner(model, graph=graph).prune(remove=[graph.parameter("fc.weight").axis(0).select([1])])
        with torch.no_grad():
            model.fc.weight.add_(0.5)
    model.eval()
    model.last.train()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    with torch.inference_mode():
        restored = load_checkpoint(WithBuffer(), stream)
    assert restored.fc is restored.alias
    assert restored.fc.out_features == 4 and restored.offset.shape == (4,)
    assert not restored.training and restored.last.training
    assert not restored.fc.weight.is_inference()
    torch.testing.assert_close(restored(x), model(x))
    restored(x).sum().backward()
    assert restored.fc.weight.grad is not None
    again = io.BytesIO()
    save_checkpoint(restored, again)


def test_checkpoint_rejects_corrupt_alias_values_before_mutation():
    model = WithBuffer()
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    payload = torch.load(stream, weights_only=True)
    payload["state_dict"]["alias.weight"] = payload["state_dict"]["alias.weight"] + 1
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    target = WithBuffer()
    old = target.fc.weight
    with pytest.raises(ExecutionError, match="Conflicting"):
        load_checkpoint(target, stream)
    assert target.fc.weight is old


@pytest.mark.parametrize("reshape", [False, True])
def test_size_consumers_on_unchanged_tensors_are_checked(reshape):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 3)

        def forward(self, x):
            y = self.fc(x)
            other = x.reshape(y.size(1) - 1, -1) if reshape else x.sum(dim=y.size(1) - 2)
            return y.sum(), other

    model = Model()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 2),))
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("fc.weight").axis(0).select([0])])


def test_channels_last_conv_view_keeps_layout(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 1)

        def forward(self, x):
            y = self.conv(x)
            return y.permute(0, 2, 3, 1).view(-1, y.size(1))

    model = Model()
    x = torch.randn(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    original = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).prune(
        remove=[graph.parameter("conv.weight").axis(0).select([0, 1])], preserve_io=False
    )
    torch.testing.assert_close(model(x), original(x)[:, 2:])


def test_checkpoint_and_plan_load_in_fresh_process(tmp_path):
    model = chain()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan(
        remove=[graph.parameter("0.weight").axis(0).select([1, 4])]
    )
    torch.save(plan.to_dict(), tmp_path / "plan.pt")
    Pruner(model).apply(plan)
    save_checkpoint(model, tmp_path / "checkpoint.pt")
    torch.save(x, tmp_path / "input.pt")
    code = """
import sys, torch
from pathlib import Path
from torch import nn
from torch_kirigami.pruning import Pruner, PruningPlan, load_checkpoint
p = Path(sys.argv[1])
def factory():
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
plan = PruningPlan.from_dict(torch.load(p / 'plan.pt', weights_only=True))
model, result = Pruner(factory()).apply(plan)
assert model[0].out_features == 4
model = load_checkpoint(factory(), p / 'checkpoint.pt', map_location='cpu')
x = torch.load(p / 'input.pt', weights_only=True)
torch.save(model(x).detach(), p / 'output.pt')
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], check=True)
    actual = torch.load(tmp_path / "output.pt", weights_only=True)
    torch.testing.assert_close(actual, model(x))


def test_native_extra_state_roundtrip_and_failure_isolation():
    class WithExtra(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)
            self.state = {"steps": [0]}

        def forward(self, x):
            return self.fc(x)

        def get_extra_state(self):
            return self.state

        def set_extra_state(self, state):
            self.state = state

    model = WithExtra()
    model.state["steps"].append(5)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(WithExtra(), stream)
    assert restored.state == model.state


def test_single_fused_definition_supports_pruning_and_checkpoint(execution_device):
    import runpy

    namespace = runpy.run_path("examples/fused_attention.py")
    namespace["main"]()
