"""Cross-device measurement preserves argument types and model-owned containers."""

from collections import defaultdict, namedtuple

import pytest
import torch
from torch import nn

from torch_kirigami.measurement import calculate_model_complexity, measure_module_latency

Inputs = namedtuple("Inputs", "tensor shape")


@pytest.mark.parametrize("latency", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_measurement_preserves_containers_during_transfer(latency, fail, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(8.0).reshape(2, 4))
            self.register_buffer("count", torch.zeros(()))
            self.cached = [torch.ones(1, 4)]
            self.empty = []

        def forward(self, inputs, *, alias, cached, empty):
            assert type(inputs) is defaultdict
            assert inputs.default_factory is list
            assert inputs is alias
            assert type(inputs["sample"]) is Inputs
            data, shape = inputs["sample"]
            assert type(shape) is torch.Size
            assert data.device == self.weight.device
            assert cached is self.cached and empty is self.empty
            assert data is inputs["again"]
            self.count.add_(1)
            data.add_(1)
            torch.rand((), device=data.device)
            if fail:
                raise RuntimeError("transfer failure")
            return (data.reshape(shape) + cached[0]) @ self.weight.t()

    model = Model()
    tensor = torch.ones(1, 4, device="cpu")
    inputs = defaultdict(list, sample=Inputs(tensor, tensor.shape), again=tensor)
    kwargs = {"alias": inputs, "cached": model.cached, "empty": model.empty}
    original_buffer, cached, empty = model.count, model.cached, model.empty
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if execution_device == "cuda" else None

    def measure():
        if latency:
            return measure_module_latency(
                model, (inputs,), input_kwargs=kwargs, warmup=0, repetitions=1
            )
        return calculate_model_complexity(model, (inputs,), input_kwargs=kwargs)

    if fail:
        with pytest.raises(RuntimeError, match="transfer failure"):
            measure()
    else:
        result = measure()
        if not latency:
            assert result.params == 8
            assert result.macs == 8
    assert model.training
    assert model.count is original_buffer and model.count.item() == 0
    assert model.cached is cached and model.empty is empty
    assert inputs["sample"].tensor is tensor and inputs["again"] is tensor
    torch.testing.assert_close(tensor, torch.ones(1, 4, device="cpu"))
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)


@pytest.mark.parametrize("device", ["cpu", "cpu:0", torch.device("cpu:1")])
def test_measurement_accepts_cpu_device_aliases(device):
    model = nn.Linear(3, 2).to(device)
    x = torch.ones(2, 3, device=device)
    assert calculate_model_complexity(model, x, device=device).macs == 12
    assert measure_module_latency(model, x, device=device, warmup=0, repetitions=1) >= 0
