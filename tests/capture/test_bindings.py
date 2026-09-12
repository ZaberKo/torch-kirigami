"""capture / bindings contracts."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    CaptureError,
    DependencyGraph,
)


def test_distinct_parameters_sharing_storage_are_not_merged():
    class SharedStorage(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.randn(4, 4))
            self.b = nn.Parameter(self.a.detach())

        def forward(self, x):
            return F.linear(x, self.a) + F.linear(x, self.b)

    graph = DependencyGraph.build(SharedStorage(), args=(torch.randn(2, 4),))
    assert graph.parameter("a") is not graph.parameter("b")
    impact = graph.propagate(remove=[graph.parameter("a").axis(0).select([1])])
    assert any(d.code == "storage_alias" for d in impact.diagnostics)


@pytest.mark.parametrize("fail", [False, True])
def test_container_buffer_isolation_success_and_failure(fail, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.zeros(1))
            self.cached = [self.counter]
            self.alias = self.cached

        def forward(self, x):
            self.cached[0].add_(1)
            if fail:
                raise RuntimeError("stop after write")
            return x + self.cached[0]

    model = Model()
    before, container = model.counter, model.cached
    # Both explicit failure and a successful trace with an invisible write must
    # restore the original aliases. Invisible writes are not analyzable FX calls.
    with pytest.raises(CaptureError):
        DependencyGraph.build(model, args=(torch.ones(2),))
    assert model.counter is before and before.item() == 0
    assert model.cached is container and model.alias is container and container[0] is before


def test_cached_storage_view_rejected_before_execution():
    model = nn.Module()
    model.register_buffer("counter", torch.zeros(2))
    model.view_cache = [model.counter[:1]]
    with pytest.raises(CaptureError, match="separate view"):
        DependencyGraph.build(model, args=())
    assert not model.counter.any()
