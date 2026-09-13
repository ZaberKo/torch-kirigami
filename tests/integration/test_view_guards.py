"""Unknown native layouts allow only views justified independently of stride."""

import itertools
from math import prod

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan
from torch_kirigami.pruning.layouts import view_preserves_stride_boundaries


@pytest.mark.parametrize(
    "operator", ["pool", "channel_shuffle", "pixel_shuffle", "batch_norm", "group_norm", "padding"]
)
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("view_kind", ["identity", "singletons", "split"])
@pytest.mark.parametrize("consumer", ["output", "view", "reshape", "contiguous"])
def test_unknown_layout_views_use_shape_proofs_and_retain_uncertainty(
    operator, functional, view_kind, consumer, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 8, 4, 4))
            self.operation = {
                "pool": nn.MaxPool2d(2),
                "channel_shuffle": nn.ChannelShuffle(2),
                "pixel_shuffle": nn.PixelShuffle(2),
                "batch_norm": nn.BatchNorm2d(8, affine=False, track_running_stats=False),
                "group_norm": nn.GroupNorm(2, 8, affine=False),
                "padding": nn.ReflectionPad2d(1),
            }[operator]

        def forward(self):
            if functional:
                if operator == "pool":
                    value = F.max_pool2d(self.weight, 2)
                elif operator == "channel_shuffle":
                    value = torch.channel_shuffle(self.weight, 2)
                elif operator == "batch_norm":
                    value = F.batch_norm(self.weight, None, None, training=True)
                elif operator == "group_norm":
                    value = F.group_norm(self.weight, 2)
                elif operator == "padding":
                    value = F.pad(self.weight, (1, 1, 1, 1), "reflect")
                else:
                    value = F.pixel_shuffle(self.weight, 2)
            else:
                value = self.operation(self.weight)
            if view_kind == "identity":
                value = value.view(value.shape)
            elif view_kind == "singletons":
                value = value.unsqueeze(2)
                value = value.view(1, value.size(0), value.size(1), value.size(3), value.size(4), 1)
            else:
                value = value.view(
                    value.size(0), value.size(1), value.size(2), 2, value.size(3) // 2
                )
            if consumer == "output":
                return value
            if consumer == "reshape":
                return value.reshape(-1)
            if consumer == "contiguous":
                value = value.contiguous()
            return value.view(-1)

    model = Model()
    original = model.weight
    before = original.detach().clone()
    model()  # The unpruned forward, including the final view, is executable.
    graph = DependencyGraph.build(model, args=())
    pruner = Pruner(model, graph=graph)
    removed = {
        "pool": [1, 3],
        "channel_shuffle": [1, 5],
        "pixel_shuffle": [0, 1, 2, 3],
        "batch_norm": [1, 5],
        "group_norm": [1, 5],
        "padding": [1, 5],
    }[operator]
    request = graph.parameter("weight").axis(1).select(removed)
    if consumer == "view":
        with pytest.raises(PlanningError, match="stride cannot be proved") as error:
            pruner.plan(remove=[request], preserve_io=False)
        assert "reshape(...) or contiguous().view(...)" in str(error.value)
        assert "If a copy is acceptable" in str(error.value)
        assert "rebuild the dependency graph" in str(error.value)
        assert model.weight is original
        torch.testing.assert_close(model.weight, before)
        return

    plan = pruner.plan(remove=[request], preserve_io=False)
    assert model.weight is original
    torch.testing.assert_close(model.weight, before)
    Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    if operator == "pool":
        # Explicit windows, with maxima computed only over the kernel positions.
        expected = before[:, [0, 2, 4, 5, 6, 7]].reshape(2, 6, 2, 2, 2, 2).amax((3, 5))
    elif operator == "channel_shuffle":
        # Original channel IDs after removing equal counts from the two groups.
        expected = before[:, [0, 4, 2, 6, 3, 7]]
    elif operator in ("batch_norm", "group_norm", "padding"):
        expected = before[:, [0, 2, 3, 4, 6, 7]]
        if operator == "padding":
            indices = [1, 0, 1, 2, 3, 2]
            expected = expected[:, :, indices][:, :, :, indices]
        else:
            if operator == "group_norm":
                expected = expected.reshape(2, 2, 3, 4, 4)
            dims = (2, 3, 4) if operator == "group_norm" else (0, 2, 3)
            expected = (expected - expected.mean(dims, keepdim=True)) / (
                expected.var(dims, unbiased=False, keepdim=True) + 1e-5
            ).sqrt()
            expected = expected.reshape(2, 6, 4, 4)
    else:
        # Independent depth-to-space coordinate permutation of retained channels.
        expected = before[:, 4:].reshape(2, 1, 2, 2, 4, 4)
        expected = expected.permute(0, 1, 4, 2, 5, 3).reshape(2, 1, 8, 8)
    if consumer != "output":
        expected = expected.reshape(-1)
    elif view_kind == "singletons":
        expected = expected[None, ..., None]
    elif view_kind == "split":
        expected = expected.reshape(*expected.shape[:-1], 2, expected.shape[-1] // 2)
    torch.testing.assert_close(model(), expected)
    model().sum().backward()
    assert model.weight.grad is not None


@pytest.mark.parametrize(
    ("source", "target", "accepted"),
    [
        ((2, 6), (2, 6), True),
        ((1, 2, 1, 6), (2, 6, 1, 1), True),
        ((2, 6), (1, 2, 2, 1, 3), True),
        ((12,), (2, 3, 2), True),
        ((1, 1), (), True),
        ((), (1, 1), True),
        ((2, 6), (4, 3), False),
        ((2, 6), (12,), False),
        ((6,), (2, 2), False),
        ((6,), (-1,), False),
        ((0, 2), (0, 2), False),
    ],
)
def test_view_shape_proof_requires_source_axis_boundaries(source, target, accepted):
    assert view_preserves_stride_boundaries(source, target) is accepted


@pytest.mark.parametrize("source", [(6,), (1, 6), (2, 3), (2, 1, 3), (2, 2, 3), (1, 1)])
def test_view_shape_proofs_hold_for_gapped_overlapping_and_zero_strides(source):
    # Independently exercise actual PyTorch views over every small stride tuple.
    # Including zero/overlapping and non-dense strides rules out a proof that
    # merely assumes all backend outputs are contiguous or channels-last.
    targets = [
        shape
        for rank in range(1, 5)
        for shape in itertools.product((1, 2, 3, 6), repeat=rank)
        if prod(shape) == prod(source) and view_preserves_stride_boundaries(source, shape)
    ]
    assert targets
    for strides in itertools.product(range(4), repeat=len(source)):
        storage_size = 1 + sum(
            (size - 1) * stride for size, stride in zip(source, strides, strict=True)
        )
        value = torch.arange(storage_size).as_strided(source, strides)
        for target in targets:
            output = value.view(target)
            assert output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
            torch.testing.assert_close(output.reshape(-1), value.reshape(-1))


def test_safe_view_shape_does_not_bypass_original_forward_size_requirements(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 8, 4, 4))

        def forward(self):
            value = F.max_pool2d(self.weight, 2)
            return value.view(value.size(0), 8, value.size(2), value.size(3))

    model = Model()
    original = model.weight
    before = original.detach().clone()
    graph = DependencyGraph.build(model, args=())
    with pytest.raises(PlanningError, match=r"shape|size|dimension|forward"):
        Pruner(model, graph=graph).plan(
            remove=[graph.parameter("weight").axis(1).select([1])], preserve_io=False
        )
    assert model.weight is original
    torch.testing.assert_close(model.weight, before)


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("consumer", ["output", "view", "reshape", "contiguous"])
def test_convolution_unknown_layout_allows_axis_split_only(
    rank, transposed, consumer, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            kind = f"Conv{'Transpose' if transposed else ''}{rank}d"
            self.conv = getattr(nn, kind)(4, 8, 1)

        def forward(self, x):
            value = self.conv(x)
            # Split only the channel axis; spatial dimensions remain separate.
            if rank == 1:
                value = value.view(value.size(0), 2, value.size(1) // 2, value.size(2))
            elif rank == 2:
                value = value.view(
                    value.size(0), 2, value.size(1) // 2, value.size(2), value.size(3)
                )
            else:
                value = value.view(
                    value.size(0),
                    2,
                    value.size(1) // 2,
                    value.size(2),
                    value.size(3),
                    value.size(4),
                )
            if consumer == "output":
                return value
            if consumer == "reshape":
                return value.reshape(-1)
            if consumer == "contiguous":
                value = value.contiguous()
            return value.view(-1)

    model = Model()
    x = torch.randn((2, 4) + (4,) * rank)
    before = model(x).detach().reshape((2, 8) + (4,) * rank)
    weight = model.conv.weight
    original = weight.detach().clone()
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    remove = graph.parameter("conv.weight").axis(1 if transposed else 0).select([1, 5])
    if consumer == "view":
        with pytest.raises(PlanningError, match="stride cannot be proved"):
            pruner.plan(remove=[remove], preserve_io=False)
        assert model.conv.weight is weight
        torch.testing.assert_close(weight, original)
        return
    plan = pruner.plan(remove=[remove], preserve_io=False)
    Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    expected = before[:, [0, 2, 3, 4, 6, 7]]
    expected = (
        expected.reshape((2, 2, 3) + (4,) * rank) if consumer == "output" else expected.reshape(-1)
    )
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert model.conv.weight.grad is not None
