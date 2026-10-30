"""Reject unsafe lazy-view copies and verify explicit materialized alternatives."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency
from torch_kirigami.pruning import Pruner


class Projection(nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.proj = nn.Linear(4, 3, device=cache.device, dtype=cache.dtype)
        self.register_buffer("cache", cache)

    def forward(self, x):
        return self.proj(x + self.cache)


@pytest.mark.parametrize("source", ["input", "buffer"])
@pytest.mark.parametrize("entry", ["capture", "complexity", "latency"])
def test_lazy_negative_view_copy_rejection_preserves_state(source, entry, execution_device):
    base = torch.full((4,) if source == "buffer" else (2, 4), 1 + 2j, dtype=torch.complex64)
    lazy = base.conj().imag
    cache = lazy if source == "buffer" else torch.zeros(4, dtype=lazy.dtype)
    model = Projection(cache).train()
    x = lazy if source == "input" else torch.ones(2, 4, dtype=lazy.dtype)
    saved_cache = model.cache
    before = lazy.clone()
    rng = torch.get_rng_state()
    with pytest.raises(CaptureError, match="lazy conjugate/negative view flags"):
        if entry == "capture":
            DependencyGraph.build(model, args=(x,))
        elif entry == "complexity":
            calculate_model_complexity(model, x, device=execution_device)
        else:
            measure_module_latency(model, x, device=execution_device, warmup=0, repetitions=1)
    assert model.cache is saved_cache and model.training and model.proj.training
    assert lazy.is_neg()
    torch.testing.assert_close(lazy, before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_conjugate_input_buffer_aliases_survive_both_measurements(execution_device):
    class Model(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("cache", value.conj())

        def forward(self, x):
            assert self.cache.is_conj()
            assert not x.is_conj()
            assert x.untyped_storage().data_ptr() == self.cache.untyped_storage().data_ptr()
            return x + self.cache

    x = torch.full((2, 4), 1 + 2j, dtype=torch.complex64)
    model = Model(x)
    cache = model.cache
    result = calculate_model_complexity(model, x, device=execution_device)
    assert result.flops == 0 and "aten::add" in result.uncovered_ops["flops"]
    measure_module_latency(model, x, device=execution_device, warmup=0, repetitions=1)
    assert model.cache is cache and cache.is_conj()
    torch.testing.assert_close(model(x), torch.full_like(x, 2))


@pytest.mark.parametrize("kind", ["conjugate", "negative"])
@pytest.mark.parametrize("source", ["input", "buffer"])
def test_materialized_views_support_measurement_capture_and_pruning(kind, source, execution_device):
    base = torch.full((4,) if source == "buffer" else (2, 4), 1 + 2j, dtype=torch.complex64)
    lazy = base.conj() if kind == "conjugate" else base.conj().imag
    value = lazy.resolve_conj().resolve_neg()
    model = Projection(value if source == "buffer" else torch.zeros(4, dtype=value.dtype))
    x = value if source == "input" else torch.ones(2, 4, dtype=value.dtype)
    result = calculate_model_complexity(model, x, device=execution_device)
    assert result.macs == 24
    graph = DependencyGraph.build(model, args=(x,))
    expected = F.linear(
        x + model.cache, model.proj.weight.detach()[[0, 2]], model.proj.bias.detach()[[0, 2]]
    )
    pruner = Pruner(model, graph=graph, preserve_io=False)
    pruner.apply(pruner.plan_remove([graph.parameter("proj.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(x), expected)
