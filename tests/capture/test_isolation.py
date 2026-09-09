"""capture / isolation contracts."""

import gc
import weakref

import pytest
import torch
from torch import nn

from torch_kirigami import (
    CaptureError,
    DependencyGraph,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_rng_and_buffers_are_restored():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.5)).cuda().train()
    x = torch.randn(3, 4, device="cuda")
    before = torch.cuda.get_rng_state().clone()
    original = model[0].running_mean
    DependencyGraph.build(model, args=(x,))
    assert torch.equal(torch.cuda.get_rng_state(), before)
    assert model[0].running_mean is original
    assert model[0].num_batches_tracked.item() == 0


def test_successful_metadata_execution_releases_intermediate_activations(monkeypatch):
    import gc
    import weakref

    from torch.fx.passes.shape_prop import ShapeProp

    references = []
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    run_node = ShapeProp.run_node

    def observe(self, node):
        output = run_node(self, node)
        if node.op == "call_module":
            references.append(weakref.ref(output))
        return output

    # Observe the interpreter without adding unmodeled hooks to the input model.
    monkeypatch.setattr(ShapeProp, "run_node", observe)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    gc.collect()
    assert references and all(ref() is None for ref in references)
    assert (
        graph.propagate(remove=[graph.parameter("0.weight").axis(0).select([1])]).status
        == "resolved"
    )


def test_input_buffer_rng_and_mode_preserved(execution_device):
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x):
            return self.bn(x.relu_()) + torch.rand_like(x)

    model = Stateful().train()
    x = torch.randn(5, 4)
    saved_x = x.clone()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if execution_device == "cuda" else None
    graph = DependencyGraph.build(model, args=(x,))
    torch.testing.assert_close(x, saved_x)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    assert model.training and model.bn.training
    graph.propagate(remove=[])
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


def test_failure_restores_buffer_and_input(execution_device):
    class Failing(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(4)
            self.fc = nn.Linear(7, 3)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            return self.fc(self.dropout(self.bn(x.relu_())))

    model = Failing()
    x = torch.randn(3, 4)
    original = x.clone()
    buffer = model.bn.running_mean
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if execution_device == "cuda" else None
    with pytest.raises(CaptureError):
        DependencyGraph.build(model, args=(x,))
    torch.testing.assert_close(x, original)
    assert model.bn.running_mean is buffer
    assert model.bn.num_batches_tracked.item() == 0
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


def test_keyword_defaults_variadics_and_containers():
    class Model(nn.Module):
        def forward(self, batch, *extra, scale=2.0, **options):
            return {"result": (batch["x"] + extra[0]) * scale + options["offset"]}

    graph = DependencyGraph.build(
        Model(),
        args=({"x": torch.randn(2, 4)}, torch.randn(2, 4)),
        kwargs={"scale": 3.0, "offset": torch.ones(2, 4)},
    )
    assert not graph.diagnostics
    inputs = [v for v in graph.values() if v.kind == "input"]
    assert graph.propagate(remove=[inputs[0].axis(1).select([1])]).status == "resolved"
    with pytest.raises(CaptureError, match="Invalid forward"):
        DependencyGraph.build(Model())


def test_known_parameter_alias_write_rejected_before_execution():
    class Mutating(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            self.weight.view(-1).zero_()
            return x

    model = Mutating()
    original = model.weight.detach().clone()
    with pytest.raises(CaptureError, match="write"):
        DependencyGraph.build(model, args=(torch.randn(2, 4),))
    torch.testing.assert_close(model.weight, original)


def test_nonleaf_example_fails_instead_of_mutating_original():
    x = torch.randn(2, 4, requires_grad=True) * 2
    with pytest.raises(CaptureError, match="copy"):
        DependencyGraph.build(nn.ReLU(), args=(x,))


def test_buffer_aliases_and_input_alias_layout_are_preserved(execution_device):
    from torch_kirigami.capture import isolated_execution

    model = nn.BatchNorm1d(4)
    model.register_buffer("alias", model.running_mean)
    base = torch.randn(3, 8)
    with isolated_execution(model, (base, base[:, ::2], base), {}) as (args, _, _):
        assert args[0] is args[2]
        assert args[1].stride() == base[:, ::2].stride()
        assert args[0].untyped_storage().data_ptr() == args[1].untyped_storage().data_ptr()
        assert args[0].untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
        assert model.alias is model.running_mean
    assert model.alias is model.running_mean


def test_dependency_graph_releases_temporary_graphmodule_buffers(monkeypatch, execution_device):
    originals = []
    graphmodule_init = torch.fx.GraphModule.__init__

    def record(self, *args, **kwargs):
        graphmodule_init(self, *args, **kwargs)
        originals.append(weakref.ref(self))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.ones(2048))

        def forward(self, x):
            return self.b + x

    model = Model()
    with monkeypatch.context() as patch:
        patch.setattr(torch.fx.GraphModule, "__init__", record)
        graph = DependencyGraph.build(model, args=(torch.zeros(2048),))
    gc.collect()
    assert originals and all(ref() is None for ref in originals)
    assert graph.propagate(remove=[]).status == "resolved"
