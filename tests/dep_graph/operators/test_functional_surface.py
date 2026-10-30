"""Public lifecycle checks for missing normalization and spatial entry forms."""

import copy
import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan


@pytest.mark.parametrize("kind", ["batch", "instance", "group"])
@pytest.mark.parametrize("keywords", [False, True])
def test_native_norm_argument_roles_and_compact_reference(kind, keywords, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Conv1d(3, 6, 1), nn.Conv1d(6, 2, 1)
            self.weight = nn.Parameter(torch.linspace(0.4, 1.2, 6))
            self.bias = nn.Parameter(torch.linspace(-0.3, 0.6, 6))
            self.register_buffer("mean", torch.linspace(-0.9, 0.7, 6))
            self.register_buffer("variance", torch.linspace(0.8, 1.9, 6))

        def forward(self, x):
            y = self.pre(x)
            if kind == "batch":
                y = (
                    torch.batch_norm(
                        input=y,
                        weight=self.weight,
                        bias=self.bias,
                        running_mean=self.mean,
                        running_var=self.variance,
                        training=False,
                        momentum=0.1,
                        eps=1e-4,
                        cudnn_enabled=False,
                    )
                    if keywords
                    else torch.batch_norm(
                        y, self.weight, self.bias, self.mean, self.variance, False, 0.1, 1e-4, False
                    )
                )
            elif kind == "instance":
                y = (
                    torch.instance_norm(
                        input=y,
                        weight=self.weight,
                        bias=self.bias,
                        running_mean=self.mean,
                        running_var=self.variance,
                        use_input_stats=False,
                        momentum=0.1,
                        eps=1e-4,
                        cudnn_enabled=False,
                    )
                    if keywords
                    else torch.instance_norm(
                        y, self.weight, self.bias, self.mean, self.variance, False, 0.1, 1e-4, False
                    )
                )
            else:
                y = (
                    torch.group_norm(
                        input=y,
                        num_groups=2,
                        weight=self.weight,
                        bias=self.bias,
                        eps=1e-4,
                        cudnn_enabled=False,
                    )
                    if keywords
                    else torch.group_norm(y, 2, self.weight, self.bias, 1e-4, False)
                )
            return self.head(y)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3, 4)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3, 5]
    y = F.conv1d(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    if kind == "group":
        grouped = y.reshape(2, 2, 2, 4)
        y = (
            (grouped - grouped.mean((2, 3), keepdim=True))
            / (grouped.var((2, 3), unbiased=False, keepdim=True) + 1e-4).sqrt()
        ).reshape(2, 4, 4)
    else:
        y = (y - dense.mean[keep].reshape(1, 4, 1)) / (
            dense.variance[keep].reshape(1, 4, 1) + 1e-4
        ).sqrt()
    y = y * dense.weight[keep].reshape(1, 4, 1) + dense.bias[keep].reshape(1, 4, 1)
    expected = F.conv1d(y, dense.head.weight[:, keep], dense.head.bias)
    torch.testing.assert_close(model(sample), expected)
    model(sample).sum().backward()
    assert model.weight.grad is not None


@pytest.mark.parametrize("form", ["module", "function", "softmax2d"])
@pytest.mark.parametrize("batched", [False, True])
def test_softmin_and_softmax2d_recompute_compact_domain(form, batched, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Conv2d(3, 6, 1), nn.Conv2d(6, 2, 1)
            self.activation = nn.Softmax2d() if form == "softmax2d" else nn.Softmin(dim=-3)

        def forward(self, x):
            y = self.pre(x)
            y = F.softmin(y, dim=-3) if form == "function" else self.activation(y)
            return self.head(y)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn((2, 3, 2, 2) if batched else (3, 2, 2))
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    y = F.conv2d(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    y = y if form == "softmax2d" else -y
    exponents = (y - y.amax(-3, keepdim=True)).exp()
    y = exponents / exponents.sum(-3, keepdim=True)
    torch.testing.assert_close(
        model(sample), F.conv2d(y, dense.head.weight[:, keep], dense.head.bias)
    )
    model(sample).sum().backward()


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("functional", [False, True])
def test_lp_pool_spatial_family_preserves_channels(dim, functional, execution_device):
    convolution = getattr(nn, f"Conv{dim}d")
    pool = getattr(F, f"lp_pool{dim}d")
    average = getattr(F, f"avg_pool{dim}d")
    linear = getattr(F, f"conv{dim}d")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = convolution(3, 6, 1), convolution(6, 2, 1)
            self.pool = getattr(nn, f"LPPool{dim}d")(2, 2)

        def forward(self, x):
            y = self.pre(x)
            return self.head(pool(y, 2, 2) if functional else self.pool(y))

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn((2, 3, *((4,) * dim)))
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    y = linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    y = (average(y.square(), 2) * 2**dim).sqrt()
    torch.testing.assert_close(
        model(sample), linear(y, dense.head.weight[:, keep], dense.head.bias)
    )
    model(sample).sum().backward()


@pytest.mark.parametrize(
    "kind", ["circular1", "circular2", "circular3", "zero1", "zero3", "nearest", "bilinear"]
)
def test_padding_and_upsampling_module_entries(kind, execution_device):
    dim = int(kind[-1]) if kind[-1].isdigit() else 2
    convolution = getattr(nn, f"Conv{dim}d")
    linear = getattr(F, f"conv{dim}d")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = convolution(3, 6, 1), convolution(6, 2, 1)
            if kind.startswith("circular"):
                self.transform = getattr(nn, f"CircularPad{dim}d")(1)
            elif kind.startswith("zero"):
                self.transform = getattr(nn, f"ZeroPad{dim}d")(1)
            elif kind == "nearest":
                self.transform = nn.UpsamplingNearest2d(scale_factor=2)
            else:
                self.transform = nn.UpsamplingBilinear2d(scale_factor=2)

        def forward(self, x):
            return self.head(self.transform(self.pre(x)))

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn((2, 3, *((3,) * dim)))
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    y = linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    if kind.startswith("circular"):
        for axis in range(2, 2 + dim):
            y = torch.cat((y.narrow(axis, y.shape[axis] - 1, 1), y, y.narrow(axis, 0, 1)), axis)
    elif kind.startswith("zero"):
        y = F.pad(y, (1, 1) * dim)
    else:
        y = F.interpolate(
            y,
            scale_factor=2,
            mode="nearest" if kind == "nearest" else "bilinear",
            align_corners=None if kind == "nearest" else True,
        )
    torch.testing.assert_close(
        model(sample), linear(y, dense.head.weight[:, keep], dense.head.bias)
    )
    model(sample).sum().backward()


@pytest.mark.parametrize("kind", ["group_balance", "spatial_coordinates"])
def test_functional_surface_rejects_unsupported_changes_atomically(kind, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Conv1d(3, 6, 1), nn.Conv1d(6, 2, 1)
            self.weight = nn.Parameter(torch.linspace(0.4, 1.2, 6))
            self.bias = nn.Parameter(torch.linspace(-0.3, 0.6, 6))

        def forward(self, x):
            y = self.pre(x)
            if kind == "group_balance":
                y = torch.group_norm(y, 2, self.weight, self.bias, 1e-4, False)
            else:
                y = F.lp_pool1d(y, 2, 2)
            return self.head(y)

    model = Model().eval()
    original = copy.deepcopy(model.state_dict())
    identities = tuple(id(p) for p in model.parameters())
    sample = torch.randn(2, 3, 4)
    graph = DependencyGraph.build(model, args=(sample,))
    selection = graph.parameter("pre.weight").axis(0).select([1])
    if kind == "spatial_coordinates":
        call = next(op for op in graph.operations() if op.node.target is F.lp_pool1d)
        selection = call.inputs[0].axis(2).select([1])
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan_remove([selection])
    assert tuple(id(p) for p in model.parameters()) == identities
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize("kind", ["batch", "instance"])
@pytest.mark.parametrize("form", ["functional", "native", "module"])
@pytest.mark.parametrize("fail", [False, True])
def test_normalization_buffer_statistics_stay_isolated(kind, form, fail, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            constructor = nn.BatchNorm1d if kind == "batch" else nn.InstanceNorm1d
            self.norm = constructor(3, track_running_stats=True)

        def forward(self, x):
            if form == "module":
                y = self.norm(x)
            elif form == "native":
                call = torch.batch_norm if kind == "batch" else torch.instance_norm
                y = call(
                    x,
                    None,
                    None,
                    self.norm.running_mean,
                    self.norm.running_var,
                    True,
                    0.1,
                    1e-4,
                    False,
                )
            elif kind == "batch":
                y = F.batch_norm(x, self.norm.running_mean, self.norm.running_var, training=True)
            else:
                y = F.instance_norm(
                    x, self.norm.running_mean, self.norm.running_var, use_input_stats=True
                )
            return y.view(7) if fail else y

    model = Model().train()
    bindings = tuple(model.buffers())
    original = tuple(buffer.clone() for buffer in bindings)
    sample = torch.ones(2, 3, 4) * 5
    with warnings.catch_warnings(record=True) as notices:
        if fail:
            with pytest.raises(CaptureError):
                DependencyGraph.build(model, args=(sample,))
        else:
            DependencyGraph.build(model, args=(sample,))
    assert not notices
    assert model.training and model.norm.training
    assert all(current is before for current, before in zip(model.buffers(), bindings, strict=True))
    for buffer, before in zip(bindings, original, strict=True):
        torch.testing.assert_close(buffer, before)


@pytest.mark.parametrize("kind", ["batch", "instance"])
@pytest.mark.parametrize(
    "form,view",
    [
        ("functional", False),
        ("functional", True),
        ("native", False),
        ("native", True),
        ("module", False),
    ],
)
def test_normalization_parameter_statistics_warn_or_reject_aliases(
    kind, form, view, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            constructor = nn.BatchNorm1d if kind == "batch" else nn.InstanceNorm1d
            self.norm = constructor(3, track_running_stats=True)
            self.norm.running_mean = nn.Parameter(torch.zeros(3), requires_grad=False)
            self.norm.running_var = nn.Parameter(torch.ones(3), requires_grad=False)
            if kind == "batch":
                self.norm.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

        def forward(self, x):
            if form == "module":
                return self.norm(x)
            mean = self.norm.running_mean.detach() if view else self.norm.running_mean
            if form == "native":
                call = torch.batch_norm if kind == "batch" else torch.instance_norm
                return call(x, None, None, mean, self.norm.running_var, True, 0.1, 1e-4, False)
            if kind == "batch":
                return F.batch_norm(x, mean, self.norm.running_var, training=True)
            return F.instance_norm(x, mean, self.norm.running_var, use_input_stats=True)

    model = Model().train()
    before = model.norm.running_mean.detach().clone()
    original_parameters = tuple(parameter.detach().clone() for parameter in model.parameters())
    sample = torch.ones(2, 3, 4) * 5
    if view:
        with (
            warnings.catch_warnings(record=True) as notices,
            pytest.raises(CaptureError, match="Parameter/alias write"),
        ):
            DependencyGraph.build(model, args=(sample,))
        assert not notices
        for parameter, original in zip(model.parameters(), original_parameters, strict=True):
            torch.testing.assert_close(parameter, original)
    else:
        with pytest.warns(UserWarning, match="parameter value updates"):
            DependencyGraph.build(model, args=(sample,))
        torch.testing.assert_close(model.norm.running_mean, before + 0.5)
        if kind == "batch" and form == "module":
            assert model.norm.num_batches_tracked.item() == 1


@pytest.mark.parametrize("kind", ["batch", "instance"])
def test_normalization_function_defaults_declare_only_actual_updates(kind, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = nn.Parameter(torch.zeros(3), requires_grad=False)
            self.variance = nn.Parameter(torch.ones(3), requires_grad=False)

        def forward(self, x):
            if kind == "batch":
                return F.batch_norm(x, self.mean, self.variance)
            return F.instance_norm(x, self.mean, self.variance)

    model = Model().eval()
    sample = torch.ones(2, 3, 4) * 5
    if kind == "batch":
        with warnings.catch_warnings(record=True) as notices:
            DependencyGraph.build(model, args=(sample,))
        assert not notices
        torch.testing.assert_close(model.mean, torch.zeros_like(model.mean))
    else:
        with pytest.warns(UserWarning, match="parameter value updates"):
            DependencyGraph.build(model, args=(sample,))
        torch.testing.assert_close(model.mean, torch.full_like(model.mean, 0.5))


@pytest.mark.parametrize("keywords", [False, True])
def test_raw_embedding_keeps_weight_first_roles_and_has_no_max_norm(keywords, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(42.0).reshape(7, 6) + 10)
            self.head = nn.Linear(6, 2)

        def forward(self, indices):
            y = (
                torch.embedding(
                    weight=self.weight,
                    indices=indices,
                    padding_idx=2,
                    scale_grad_by_freq=True,
                    sparse=False,
                )
                if keywords
                else torch.embedding(self.weight, indices, 2, True, False)
            )
            return self.head(y)

    model = Model().eval()
    dense = copy.deepcopy(model)
    indices = torch.tensor([[0, 2, 3], [3, 4, 2]])
    with warnings.catch_warnings(record=True) as notices:
        graph = DependencyGraph.build(model, args=(indices,))
    assert not notices and not graph.diagnostics
    torch.testing.assert_close(model.weight, dense.weight)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("weight").axis(1).select([1, 4])]))
    keep = [0, 2, 3, 5]
    expected = F.linear(
        dense.weight[indices][..., keep], dense.head.weight[:, keep], dense.head.bias
    )
    torch.testing.assert_close(model(indices), expected)
    model(indices).sum().backward()
    torch.testing.assert_close(model.weight.grad[2], torch.zeros_like(model.weight.grad[2]))


@pytest.mark.parametrize(
    "entry",
    ["max_pool1d", "max_pool2d", "max_pool3d", "max_pool1d_with_indices", "adaptive_max_pool1d"],
)
def test_raw_pooling_entry_preserves_channel_and_index_coordinates(entry, execution_device):
    dim = next(value for value in (1, 2, 3) if f"{value}d" in entry)
    convolution = getattr(nn, f"Conv{dim}d")
    linear = getattr(F, f"conv{dim}d")
    call = getattr(torch, entry)
    paired = "with_indices" in entry or "adaptive" in entry

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = convolution(3, 6, 1), convolution(6, 2, 1)

        def forward(self, x):
            result = call(self.pre(x), 2)
            return self.head(result[0] if paired else result)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn((2, 3, *((4,) * dim)))
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    selection = graph.parameter("pre.weight").axis(0).select([1, 4])
    impact = graph.propagate(remove=[selection])
    if paired:
        operation = next(op for op in graph.operations() if op.node.target is call)
        assert tuple(impact.selection(operation.outputs[1]).fully_selected_indices(1)) == (1, 4)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([selection]))
    keep = [0, 2, 3, 5]
    y = linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    for axis in range(2, 2 + dim):
        y = y.unfold(axis, 2, 2)
    pooled = y.amax(tuple(range(-dim, 0)))
    torch.testing.assert_close(
        model(sample), linear(pooled, dense.head.weight[:, keep], dense.head.bias)
    )
    model(sample).sum().backward()


@pytest.mark.parametrize("functional", [False, True])
def test_local_response_norm_recomputes_compact_channel_windows(functional, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Conv1d(3, 6, 1), nn.Conv1d(6, 2, 1)
            self.norm = nn.LocalResponseNorm(3, alpha=0.3, beta=0.75, k=1.2)

        def forward(self, x):
            y = self.pre(x)
            y = (
                F.local_response_norm(y, 3, alpha=0.3, beta=0.75, k=1.2)
                if functional
                else self.norm(y)
            )
            return self.head(y)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3, 4)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    y = F.conv1d(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    columns = []
    for channel in range(4):
        energy = y[:, max(0, channel - 1) : min(4, channel + 2)].square().sum(1)
        columns.append(y[:, channel] / (1.2 + 0.3 * energy / 3).pow(0.75))
    normalized = torch.stack(columns, dim=1)
    torch.testing.assert_close(
        model(sample), F.conv1d(normalized, dense.head.weight[:, keep], dense.head.bias)
    )
    model(sample).sum().backward()
