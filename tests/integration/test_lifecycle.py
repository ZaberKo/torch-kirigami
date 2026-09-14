"""integration / lifecycle contracts."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.pruning import build
from torch_kirigami import (
    DependencyGraph,
    StaleGraphError,
)
from torch_kirigami.pruning import (
    ChannelRatio,
    ExecutionError,
    Greedy,
    Magnitude,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def assert_references(model):
    assert model.alias is model.fc
    assert model.offset_alias is model.offset
    assert model.holder.references is model.references
    assert model.references[0]["weight"] is model.fc.weight
    assert model.references[0]["offset"] is model.offset
    assert model.references[1][0] is model.fc
    assert model.references[2] is model.references


class SharedPipeline(nn.Module):
    def __init__(self, persistent=True):
        super().__init__()
        self.fc = nn.Linear(4, 6, dtype=torch.float64)
        self.alias = self.fc
        self.out = nn.Linear(6, 2, dtype=torch.float64)
        self.register_buffer(
            "offset", torch.linspace(-0.3, 0.2, 6, dtype=torch.float64), persistent
        )
        # The same buffer is deliberately exposed through both persistence policies.
        self.register_buffer("offset_alias", self.offset, not persistent)
        self.references = [{"weight": self.fc.weight, "offset": self.offset}, (self.fc,)]
        self.references.append(self.references)
        self.holder = nn.Identity()
        self.holder.references = self.references
        self.fc.eval()

    def forward(self, x):
        cached = self.references[0]
        hidden = F.linear(x, cached["weight"], self.fc.bias) + self.alias(x) + cached["offset"]
        return self.out(hidden.tanh())


def observe(model):
    """Keep independent object/value observations without library snapshot helpers."""
    tensors = dict(
        (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    )
    return {
        "tensors": {
            path: (
                tensor,
                tensor.detach().clone(),
                tensor.grad,
                None if tensor.grad is None else tensor.grad.clone(),
                tensor.stride(),
                tensor.requires_grad,
            )
            for path, tensor in tensors.items()
        },
        "modules": dict(model.named_modules(remove_duplicate=False)),
        "modes": {path: m.training for path, m in model.named_modules()},
        "widths": (model.fc.out_features, model.out.in_features),
        "references": model.references,
        "reference_dict": model.references[0],
        "state_keys": tuple(model.state_dict()),
        "metadata_present": "_kirigami_structure" in vars(model),
        "metadata": copy.deepcopy(getattr(model, "_kirigami_structure", None)),
    }


def assert_unchanged(model, before):
    tensors = dict(
        (
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        )
    )
    assert tensors.keys() == before["tensors"].keys()
    for path, (old, value, grad, grad_value, stride, requires_grad) in before["tensors"].items():
        current = tensors[path]
        assert current is old, path
        torch.testing.assert_close(current, value, rtol=0, atol=0)
        assert current.stride() == stride and current.requires_grad == requires_grad, path
        assert current.grad is grad, path
        if grad is not None:
            torch.testing.assert_close(grad, grad_value, rtol=0, atol=0)
    modules = dict(model.named_modules(remove_duplicate=False))
    assert modules.keys() == before["modules"].keys()
    assert all(modules[p] is m for p, m in before["modules"].items())
    assert {path: m.training for path, m in model.named_modules()} == before["modes"]
    assert (model.fc.out_features, model.out.in_features) == before["widths"]
    assert model.references is before["references"]
    assert model.references[0] is before["reference_dict"]
    assert tuple(model.state_dict()) == before["state_keys"]
    assert ("_kirigami_structure" in vars(model)) == before["metadata_present"]
    assert getattr(model, "_kirigami_structure", None) == before["metadata"]
    assert_references(model)


def compact_reference(original, x, keep):
    """Compute the retained network directly from the original tensor coordinates."""
    hidden = 2 * F.linear(x, original.fc.weight[keep], original.fc.bias[keep])
    hidden = hidden + original.offset[keep]
    return F.linear(hidden.tanh(), original.out.weight[:, keep], original.out.bias)


def portable_plan(model, x, remove):
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("fc.weight").axis(0).select(remove)]
    )
    stream = io.BytesIO()
    torch.save(plan.to_dict(), stream)
    stream.seek(0)
    return graph, PruningPlan.from_dict(torch.load(stream, weights_only=True))


COMMIT_BOUNDARIES = [
    ("fc", "weight"),
    ("fc", "bias"),
    ("out", "weight"),
    ("", "offset"),
    ("", "offset_alias"),
    ("fc", "out_features"),
    ("out", "in_features"),
    ("", "references"),
    ("holder", "references"),
    ("", "_kirigami_structure"),
]


@pytest.mark.parametrize("schedule", ["joint", "sequential"])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_shared_graph_survives_multiround_pruning_and_restore(
    schedule, persistent, inference, execution_device
):
    model = SharedPipeline(persistent)
    original = copy.deepcopy(model)
    x = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64).reshape(2, 4)
    model(x).sum().backward()
    for remove in [[1, 2]] if schedule == "joint" else [[1], [1]]:
        before = observe(model)
        cpu_rng = torch.get_rng_state().clone()
        device_rng = torch.cuda.get_rng_state().clone() if execution_device == "cuda" else None
        graph, plan = portable_plan(model, x, remove)
        assert_unchanged(model, before)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        if device_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), device_rng)
        with torch.inference_mode(inference):
            _, result = Pruner(model).apply(plan)
        assert_references(model)
        for old, new in result.parameter_map.items():
            assert any(new is parameter for parameter in model.parameters())
            assert old is not new and new.grad is None
            assert not new.is_inference()
        with pytest.raises(StaleGraphError):
            graph.validate()
        with pytest.raises(ExecutionError, match="preconditions"):
            Pruner(model).apply(plan)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    with torch.inference_mode(inference):
        restored = load_checkpoint(SharedPipeline(persistent), stream)
    assert_references(restored)
    assert tuple(restored.state_dict()) == tuple(model.state_dict()) == tuple(original.state_dict())
    assert model.fc.out_features == restored.fc.out_features == 4
    assert model.out.in_features == restored.out.in_features == 4
    assert restored.training and not restored.fc.training and restored.out.training
    for actual_model in (model, restored):
        actual_model.zero_grad(set_to_none=True)
        actual_x = x.detach().clone().requires_grad_()
        reference_x = x.detach().clone().requires_grad_()
        actual = actual_model(actual_x)
        expected = compact_reference(original, reference_x, [0, 3, 4, 5])
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual_x.grad, reference_x.grad)
        for parameter in actual_model.parameters():
            assert parameter.grad is not None and not parameter.is_inference()


@pytest.mark.parametrize("entrypoint", ["apply", "checkpoint"])
@pytest.mark.parametrize("previously_pruned", [False, True])
@pytest.mark.parametrize("owner_path,field", COMMIT_BOUNDARIES)
def test_failure_after_each_kind_of_commit_restores_exact_bindings(
    entrypoint, previously_pruned, owner_path, field, monkeypatch, execution_device
):
    model = SharedPipeline()
    x = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64).reshape(2, 4)
    if previously_pruned:
        _, first_plan = portable_plan(model, x, [1])
        Pruner(model).apply(first_plan)
    original = copy.deepcopy(model)
    model(x).sum().backward()
    _, plan = portable_plan(model, x, [1, 2])
    if entrypoint == "checkpoint":
        source = copy.deepcopy(model)
        Pruner(source).apply(plan)
        stream = io.BytesIO()
        save_checkpoint(source, stream)
        stream.seek(0)

    def invoke():
        if entrypoint == "apply":
            return Pruner(model).apply(plan)
        stream.seek(0)
        return load_checkpoint(model, stream)

    before = observe(model)
    owner = model.get_submodule(owner_path)
    original_setattr = nn.Module.__setattr__
    calls = []

    def fail_after_assignment(self, name, value):
        original_setattr(self, name, value)
        if self is owner and name == field:
            calls.append(name)
            raise RuntimeError("injected failure after assignment")

    with monkeypatch.context() as patch:
        patch.setattr(nn.Module, "__setattr__", fail_after_assignment)
        with pytest.raises(ExecutionError, match="Commit failed"):
            invoke()
    assert calls == [field], "The selected commit boundary must actually be exercised"
    assert_unchanged(model, before)
    torch.testing.assert_close(model(x), original(x))
    # The same operation must remain retryable after rollback.
    invoke()
    assert_references(model)
    keep = [i for i in range(original.fc.out_features) if i not in (1, 2)]
    torch.testing.assert_close(model(x), compact_reference(original, x, keep))


def test_rebuild_explicit_second_round_and_all_old_plans_stale(execution_device):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    x = torch.randn(2, 4)
    graph, pruner = build(model, x)
    a = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([1])])
    b = pruner.plan_remove([graph.parameter("0.weight").axis(0).select([2])])
    pruner.apply(a)
    with pytest.raises(ExecutionError):
        pruner.apply(b)
    graph, pruner = build(model, x)
    pruner.apply(
        pruner.plan(
            pruner.discover_candidates(), budget=ChannelRatio(0.2), strategy=Greedy(Magnitude())
        )
    )
    assert model[0].out_features == 6
    assert model(x).shape == (2, 2)
