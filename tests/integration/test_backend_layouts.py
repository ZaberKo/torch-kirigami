"""Backend-dependent layouts stay unknown; explicit alternatives remain executable."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    Greedy,
    Magnitude,
    PlanningError,
    Pruner,
    PruningPlan,
)


class NormView(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.conv, self.norm = nn.Conv2d(4, 8, 1), nn.GroupNorm(2, 8)
        self.mode = mode

    def forward(self, x):
        y = self.norm(self.conv(x)).permute(0, 2, 3, 1)
        if self.mode == "contiguous":
            y = y.contiguous()
        if self.mode == "reshape":
            return y.reshape(-1, y.size(-1))
        return y.view(-1, y.size(-1))


@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_cpu_channels_last_groupnorm_view(mode):
    model = NormView(mode)
    original = copy.deepcopy(model)
    x = torch.randn(2, 4, 4, 4).to(memory_format=torch.channels_last)
    graph = DependencyGraph.build(model, args=(x,))
    if mode == "view":
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parameters = tuple(model.parameters())
        with pytest.raises(PlanningError, match="stride cannot be proved") as error:
            Pruner(model, graph=graph, preserve_io=False).plan_remove(
                [graph.parameter("conv.weight").axis(0).select([1, 5])]
            )
        assert "If a copy is acceptable" in str(error.value)
        assert "reshape(...) or contiguous().view(...)" in str(error.value)
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        return
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph, preserve_io=False)
        .plan_remove([graph.parameter("conv.weight").axis(0).select([1, 5])])
        .to_dict()
    )
    Pruner(model).apply(plan)
    keep = [0, 2, 3, 4, 6, 7]
    value = F.conv2d(x, original.conv.weight[keep], original.conv.bias[keep])
    # Explicit compact-domain normalization, not a mask-equivalence reference.
    groups = value.reshape(2, 2, 3, 4, 4)
    normalized = (groups - groups.mean((2, 3, 4), keepdim=True)) / (
        groups.var((2, 3, 4), unbiased=False, keepdim=True) + original.norm.eps
    ).sqrt()
    expected = normalized.reshape_as(value) * original.norm.weight[keep][None, :, None, None]
    expected += original.norm.bias[keep][None, :, None, None]
    expected = expected.permute(0, 2, 3, 1).reshape(-1, 6)
    torch.testing.assert_close(model(x), expected)
    model(x).square().sum().backward()


class Conv3dView(nn.Module):
    def __init__(self, dtype, mode="view"):
        super().__init__()
        self.conv = nn.Conv3d(4, 8, 1, dtype=dtype)
        self.mode = mode

    def forward(self, x):
        y = self.conv(x)
        if self.mode == "contiguous":
            y = y.contiguous()
        return y.reshape(y.size(0), -1) if self.mode == "reshape" else y.view(y.size(0), -1)


@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_cpu_float64_conv3d_layout_remains_unknown(mode):
    model = Conv3dView(torch.float64, mode)
    original = copy.deepcopy(model)
    x = torch.randn(2, 4, 4, 4, 4, dtype=torch.float64).to(memory_format=torch.channels_last_3d)
    graph = DependencyGraph.build(model, args=(x,))
    if mode == "view":
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parameters = tuple(model.parameters())
        with pytest.raises(PlanningError, match="stride cannot be proved") as error:
            Pruner(model, graph=graph, preserve_io=False).plan_remove(
                [graph.parameter("conv.weight").axis(0).select([1])]
            )
        assert "If a copy is acceptable" in str(error.value)
        assert "reshape(...) or contiguous().view(...)" in str(error.value)
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        return
    plan = Pruner(model, graph=graph, preserve_io=False).plan_remove(
        [graph.parameter("conv.weight").axis(0).select([1])]
    )
    Pruner(model).apply(plan)
    keep = [0, 2, 3, 4, 5, 6, 7]
    expected = F.conv3d(x, original.conv.weight[keep], original.conv.bias[keep]).reshape(2, -1)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_backend_dependent_conv_layout_requires_explicit_layout_for_view(mode, execution_device):
    # A channel-last input doesn't prove that a compact Conv3d uses that layout.
    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(4, 8, 1)

        def forward(self, x):
            y = self.conv(x)
            if mode == "contiguous":
                y = y.contiguous()
            # Flattening spatial dimensions works for both native C and CL3D;
            # the planner still needs a proven stride for the original view.
            return (
                y.reshape(y.size(0), y.size(1), -1)
                if mode == "reshape"
                else y.view(y.size(0), y.size(1), -1)
            )

    model = Network()
    x = torch.randn(2, 4, 4, 4, 4).to(memory_format=torch.channels_last_3d)
    graph = DependencyGraph.build(model, args=(x,))
    old = tuple(model.parameters())

    def plan():
        return Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("conv.weight").axis(0).select([1])]
        )

    if mode == "view":
        with pytest.raises(PlanningError, match="stride cannot be proved"):
            plan()
        assert all(a is b for a, b in zip(old, model.parameters(), strict=True))
    else:
        original = copy.deepcopy(model)
        Pruner(model).apply(plan())
        torch.testing.assert_close(model(x), original(x)[:, [0, 2, 3, 4, 5, 6, 7]])
        model(x).sum().backward()


@pytest.mark.parametrize("mode", ["view", "reshape"])
@pytest.mark.parametrize("cudnn_enabled", [False, True])
def test_cudnn_settings_do_not_establish_layout(mode, cudnn_enabled, execution_device):

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3)

        def forward(self, x):
            y = self.conv(x).permute(0, 2, 3, 1)
            return y.reshape(-1, y.size(-1)) if mode == "reshape" else y.view(-1, y.size(-1))

    model = Model()
    x = torch.randn(2, 4, 6, 6).to(memory_format=torch.channels_last)
    with torch.backends.cudnn.flags(enabled=True):
        graph = DependencyGraph.build(model, args=(x,))
    old = tuple(model.parameters())
    with torch.backends.cudnn.flags(enabled=cudnn_enabled):
        if mode == "view":
            with pytest.raises(PlanningError, match="stride"):
                Pruner(model, graph=graph, preserve_io=False).plan_remove(
                    [graph.parameter("conv.weight").axis(0).select([1])]
                )
            assert all(a is b for a, b in zip(old, model.parameters(), strict=True))
        else:
            reference = model(x)[:, [0, 2, 3, 4, 5, 6, 7]]
            Pruner(model, graph=graph, preserve_io=False).apply(
                Pruner(model, graph=graph, preserve_io=False).plan_remove(
                    [graph.parameter("conv.weight").axis(0).select([1])]
                )
            )
            torch.testing.assert_close(model(x), reference)
            model(x).sum().backward()


@pytest.mark.parametrize("padding_mode", ["circular", "reflect", "replicate", "zeros"])
@pytest.mark.parametrize("weight_format", ["contiguous", "channels_last"])
@pytest.mark.parametrize("mode", ["reshape", "contiguous"])
def test_padded_convolution_with_explicit_layout(
    padding_mode, weight_format, mode, execution_device
):

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 6, 3, padding=1, padding_mode=padding_mode)
            if weight_format == "channels_last":
                self.conv.to(memory_format=torch.channels_last)

        def forward(self, x):
            y = self.conv(x)
            if mode == "contiguous":
                return y.contiguous().view(y.size(0), -1)
            return y.reshape(y.size(0), -1)

    model = Model()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4, 4, 4).to(memory_format=torch.channels_last)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("conv.weight").axis(0).select([1])]
        )
    )
    # Independent native compact operator with the same public padding contract.
    compact = nn.Conv2d(4, 5, 3, padding=1, padding_mode=padding_mode)
    with torch.no_grad():
        compact.weight.copy_(original.conv.weight[[0, 2, 3, 4, 5]])
        compact.bias.copy_(original.conv.bias[[0, 2, 3, 4, 5]])
    y = compact(x)
    expected = y.reshape(2, -1)
    torch.testing.assert_close(model(x), expected)
    model(x).square().sum().backward()


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate"])
@pytest.mark.parametrize("module_form", [False, True])
@pytest.mark.parametrize("mode", ["reshape", "contiguous"])
def test_explicit_padding_with_explicit_layout(padding_mode, module_form, mode, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 6, 1)
            self.pad = (
                nn.ReflectionPad2d(1) if padding_mode == "reflect" else nn.ReplicationPad2d(1)
            )

        def forward(self, x):
            y = self.conv(x)
            y = self.pad(y) if module_form else F.pad(y, (1, 1, 1, 1), padding_mode)
            if mode == "contiguous":
                return y.contiguous().view(y.size(0), -1)
            return y.reshape(y.size(0), -1)

    model = Model()
    x = torch.randn(2, 4, 4, 4).to(memory_format=torch.channels_last)
    reference = model(x).detach()
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("conv.weight").axis(0).select([1])]
        )
    )
    expected = reference.reshape(2, 6, -1)[:, [0, 2, 3, 4, 5]].reshape(2, -1)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_batchnorm_spatial_transpose_layout_is_unknown(rank, functional, training, mode):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = getattr(nn, f"Conv{rank}d")(4, 8, 1)
            self.bn = getattr(nn, f"BatchNorm{rank}d")(8)

        def forward(self, x):
            y = self.conv(x).transpose(2, 3)
            y = (
                F.batch_norm(
                    y,
                    self.bn.running_mean,
                    self.bn.running_var,
                    self.bn.weight,
                    self.bn.bias,
                    training=self.training,
                )
                if functional
                else self.bn(y)
            )
            if mode == "contiguous":
                y = y.contiguous()
            return y.reshape(y.size(0), -1) if mode == "reshape" else y.view(y.size(0), -1)

    model = Model().train(training)
    original = copy.deepcopy(model)
    x = torch.randn((2, 4) + (4,) * rank)
    assert model(x).shape == (2, 8 * 4**rank)
    # Keep the independent reference at the same post-forward running state.
    original.load_state_dict(model.state_dict())
    graph = DependencyGraph.build(model, args=(x,))
    if mode == "view":
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parameters = tuple(model.parameters())
        with pytest.raises(PlanningError, match="stride cannot be proved") as error:
            Pruner(model, graph=graph, preserve_io=False).plan_remove(
                [graph.parameter("conv.weight").axis(0).select([1])]
            )
        assert "If a copy is acceptable" in str(error.value)
        assert "reshape(...) or contiguous().view(...)" in str(error.value)
        assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        return
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph, preserve_io=False)
        .plan_remove([graph.parameter("conv.weight").axis(0).select([1])])
        .to_dict()
    )
    Pruner(model).apply(plan)
    keep = [0, 2, 3, 4, 5, 6, 7]
    y = getattr(F, f"conv{rank}d")(
        x, original.conv.weight[keep], original.conv.bias[keep]
    ).transpose(2, 3)
    expected = F.batch_norm(
        y,
        original.bn.running_mean[keep].clone(),
        original.bn.running_var[keep].clone(),
        original.bn.weight[keep],
        original.bn.bias[keep],
        training=training,
    ).reshape(2, -1)
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize("pixel", [False, True])
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_shuffle_backend_layout_uncertainty_is_local(pixel, functional, mode):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 2, 2, 16 if pixel else 4).transpose(1, 2))
            self.shuffle = nn.PixelShuffle(2) if pixel else nn.ChannelShuffle(2)

        def forward(self):
            value = self.weight.permute(0, 3, 1, 2)
            if functional:
                value = F.pixel_shuffle(value, 2) if pixel else torch.channel_shuffle(value, 2)
            else:
                value = self.shuffle(value)
            if mode == "contiguous":
                value = value.contiguous()
            return value.reshape(-1) if mode == "reshape" else value.view(-1)

    model = Model()
    original = model.weight
    before = original.detach().clone()
    model()  # The original view is valid; shrinking changes the backend's format choice.
    graph = DependencyGraph.build(model, args=())
    remove = (
        graph.parameter("weight").axis(3 if pixel else 1).select([0, 1, 2, 3] if pixel else [1])
    )
    pruner = Pruner(model, graph=graph)
    if mode == "view":
        with pytest.raises(PlanningError, match=r"layout|stride|view"):
            Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove([remove])
        assert model.weight is original
        torch.testing.assert_close(model.weight, before)
        return
    plan = Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove([remove])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    compact = before[:, :, :, 4:] if pixel else before[:, :1, :, :]
    value = compact.permute(0, 3, 1, 2)
    if pixel:
        # Independent depth-to-space permutation in original channel order.
        expected = value.reshape(2, 3, 2, 2, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape(2, 3, 4, 4)
    else:
        expected = value[:, [0, 2, 1, 3]]
    torch.testing.assert_close(model(), expected.reshape(-1))
    model().sum().backward()


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_pool_meta_layout_does_not_claim_a_backend_stride_proof(functional, mode):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(2, 8, 4), requires_grad=False)
            self.pool = nn.MaxPool1d(2)

        def forward(self):
            value = self.weight.permute(0, 2, 1)
            value = F.max_pool1d(value, 2) if functional else self.pool(value)
            if mode == "contiguous":
                value = value.contiguous()
            return value.reshape(2, -1) if mode == "reshape" else value.view(2, -1)

    model = Model()
    original = model.weight
    before = original.detach().clone()
    model()  # Frozen CPU max_pool1d actually returns contiguous, unlike meta.
    graph = DependencyGraph.build(model, args=())
    pruner = Pruner(model, graph=graph)
    remove = graph.parameter("weight").axis(2).select([1])
    if mode == "view":
        with pytest.raises(PlanningError, match="stride cannot be proved"):
            Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove([remove])
        assert model.weight is original
        torch.testing.assert_close(model.weight, before)
        return
    pruner.apply(Pruner(pruner.model, graph=pruner.graph, preserve_io=False).plan_remove([remove]))
    compact = before[:, :, [0, 2, 3]].permute(0, 2, 1)
    expected = compact.reshape(2, 3, 4, 2).amax(-1).reshape(2, -1)
    torch.testing.assert_close(model(), expected)


@pytest.mark.parametrize("mode", ["view", "reshape", "contiguous"])
def test_unknown_convolution_layout_excludes_only_affected_candidate(mode, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.bad = nn.Conv2d(4, 4, 1)
            self.good = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 2, 1))

        def forward(self, x):
            y = self.bad(x)
            if mode == "contiguous":
                y = y.contiguous()
            y = y.reshape(y.size(0), -1) if mode == "reshape" else y.view(y.size(0), -1)
            return y, self.good(x)

    model = Model()
    original = copy.deepcopy(model)
    x = torch.randn(2, 4, 4, 4)
    graph = DependencyGraph.build(model, args=(x,))
    bad, good = (graph.parameter(path).axis(0) for path in ("bad.weight", "good.0.weight"))
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        CandidateSpace(
            candidates=[
                Candidate("bad", (bad.select([1]),)),
                Candidate("good", (good.select([1]),)),
            ],
            channel_axes=(bad, good),
        ),
        budget=ChannelRatio(0.25),
        strategy=Greedy(Magnitude()),
    )
    assert set(plan.selected) == ({"good"} if mode == "view" else {"bad", "good"})
    if mode == "view":
        assert (
            "reshape(...) or contiguous().view(...)"
            in dict(plan.selection_report.exclusions)["bad"]
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name])
    bad_weight = model.bad.weight
    Pruner(model).apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3]
    expected_bad = (
        original(x)[0]
        if mode == "view"
        else F.conv2d(x, original.bad.weight[keep], original.bad.bias[keep]).reshape(2, -1)
    )
    first, last = original.good
    expected_good = F.conv2d(
        F.conv2d(x, first.weight[keep], first.bias[keep]), last.weight[:, keep], last.bias
    )
    torch.testing.assert_close(model(x), (expected_bad, expected_good))
    if mode == "view":
        assert model.bad.weight is bad_weight
    sum(y.sum() for y in model(x)).backward()
