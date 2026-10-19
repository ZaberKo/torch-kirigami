"""Capture equivalence includes bound facts, not generated FX names."""

import copy

import pytest
import torch
from torch import nn

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner


@pytest.mark.parametrize(
    "write", ["resize", "fill", "replace", "data", "data_replace", "data_dtype", "register", "none"]
)
def test_unrecorded_buffer_writes_are_rejected_and_restored(write, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)
            self.register_buffer("scratch", torch.ones(4))
            self.register_buffer("optional", None)

        def forward(self, x):
            if write == "resize":
                self.scratch.resize_(self.a.out_features).fill_(1)
            elif write == "fill":
                self.scratch.fill_(self.a.out_features)
            elif write == "replace":
                self.scratch = torch.ones(self.a.out_features)
            elif write == "data":
                self.scratch.data.fill_(self.a.out_features)
            elif write == "data_replace":
                self.scratch.data = self.scratch.clone()
            elif write == "data_dtype":
                self.scratch.data = self.scratch.double()
            elif write == "register":
                self.register_buffer("extra", torch.ones(4), persistent=False)
            else:
                self.optional = torch.ones(4)
            return self.b(self.a(x)), x + self.scratch

    model = Model()
    buffer = model.scratch
    with pytest.raises(CaptureError, match=r"buffer|write"):
        DependencyGraph.build(model, args=(torch.ones(2, 4),))
    assert model.scratch is buffer and model.scratch.shape == (4,)
    torch.testing.assert_close(model.scratch, torch.ones(4))
    assert model.optional is None and not hasattr(model, "extra")
    assert not model._non_persistent_buffers_set


@pytest.mark.parametrize("dependent", [False, True])
def test_generated_constants_are_compared_by_fact_and_do_not_leak(dependent, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Linear(4, 2)

        def forward(self, x):
            return self.b(self.a(x)) + torch.full(
                (2,), float(self.a.out_features) if dependent else 1.0
            )

    model = Model()
    x = torch.randn(2, 4)
    fields = set(vars(model))
    for _ in range(3):
        graph = DependencyGraph.build(model, args=(x,))
        assert set(vars(model)) == fields
    ref = copy.deepcopy(model)
    if dependent:
        with pytest.raises(PlanningError, match=r"constant|structure"):
            Pruner(model, graph=graph).plan_remove(
                [graph.parameter("a.weight").axis(0).select([1])]
            )
    else:
        kept = [0, 2, 3]
        ref.a = nn.Linear(4, 3)
        ref.b = nn.Linear(3, 2)
        with torch.no_grad():
            ref.a.weight.copy_(model.a.weight[kept])
            ref.a.bias.copy_(model.a.bias[kept])
            ref.b.weight.copy_(model.b.weight[:, kept])
            ref.b.bias.copy_(model.b.bias)
        Pruner(model, graph=graph).apply(
            Pruner(model, graph=graph).plan_remove(
                [graph.parameter("a.weight").axis(0).select([1])]
            )
        )
        torch.testing.assert_close(model(x), ref(x))
    assert not any(k.startswith("_tensor_constant") for k in vars(model))


@pytest.mark.parametrize("inference", [False, True])
def test_trace_failure_restores_constants_buffers_inputs_and_rng(inference, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("buffer", torch.ones(4))

        def forward(self, x):
            y = x + torch.randn(4)
            self.buffer.fill_(1)
            return y

    with torch.inference_mode(inference):
        model = Model()
        x = torch.ones(2, 4)
        original = model.buffer
        names = set(vars(model))
        rng = torch.get_rng_state().clone()
        with pytest.raises(CaptureError, match="buffer write"):
            DependencyGraph.build(model, args=(x,))
        assert set(vars(model)) == names and model.buffer is original
        torch.testing.assert_close(model.buffer, torch.ones(4))
        torch.testing.assert_close(x, torch.ones(2, 4))
        assert torch.equal(torch.get_rng_state(), rng)
