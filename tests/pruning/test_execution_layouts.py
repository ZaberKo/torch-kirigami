"""Precision, layout and execution contexts must survive lifecycle boundaries."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("channels_last", [False, True])
def test_compact_singleton_strides_match_plan_and_checkpoint(
    rank, width, channels_last, execution_device
):
    conv = getattr(nn, f"Conv{rank}d")
    fmt = (
        torch.contiguous_format
        if not channels_last
        else torch.channels_last
        if rank == 2
        else torch.channels_last_3d
    )

    def make():
        return nn.Sequential(
            conv(width, width, 3, padding=1), nn.ReLU(), conv(width, 2, 3, padding=1)
        ).to(memory_format=fmt)

    model = make()
    original = copy.deepcopy(model)
    x = torch.randn((2, width) + (5,) * rank)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan_remove([graph.parameter("0.weight").axis(0).select([1])])
        .to_dict()
    )
    Pruner(model).apply(plan)
    keep = [i for i in range(width) if i != 1]
    convolution = getattr(F, f"conv{rank}d")
    expected = convolution(
        convolution(x, original[0].weight[keep], original[0].bias[keep], padding=1).relu(),
        original[2].weight[:, keep],
        original[2].bias,
        padding=1,
    )
    torch.testing.assert_close(model(x), expected)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(make(), stream)
    torch.testing.assert_close(restored(x), expected)
    for instance in (model, restored):
        for state in plan.after.tensors:
            owner, _, field = state.paths[0].rpartition(".")
            assert getattr(instance.get_submodule(owner), field).stride() == state.stride
        instance(x).sum().backward()
        assert all(p.grad is not None for p in instance.parameters())


@pytest.mark.parametrize(
    "overload", ["keyword", "dtype_positional", "tensor_positional", "device_positional"]
)
@pytest.mark.parametrize("copy_output", [False, True])
def test_cast_copy_contract_preserves_native_compact_view_behavior(
    overload, copy_output, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 6)

        def forward(self, x):
            y = self.a(x)[:, ::2]
            if overload == "keyword":
                y = y.to(copy=copy_output)
            elif overload == "dtype_positional":
                y = y.to(torch.float32, False, copy_output)
            elif overload == "tensor_positional":
                y = y.to(x, False, copy_output)
            else:
                y = y.to(x.device, torch.float32, False, copy_output)
            return y.view(-1)

    model = Model()
    x = torch.randn(2, 4)
    expected = model(x).detach()
    graph = DependencyGraph.build(model, args=(x,))
    old = tuple(model.parameters())
    if copy_output:
        Pruner(model, graph=graph).apply(
            Pruner(model, graph=graph).plan_remove(
                [graph.parameter("a.weight").axis(0).select([5])]
            )
        )
        torch.testing.assert_close(model(x), expected)
        model(x).sum().backward()
    else:
        with pytest.raises(PlanningError, match="view"):
            Pruner(model, graph=graph).plan_remove(
                [graph.parameter("a.weight").axis(0).select([5])]
            )
        assert all(a is b for a, b in zip(old, model.parameters(), strict=True))
