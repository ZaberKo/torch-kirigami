"""Independent coordinate and contraction spellings exercise public compaction."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, PruningPlan


@pytest.mark.parametrize("method", [False, True])
def test_ravel_preserves_flattened_feature_coordinates(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(12, 2)

        def forward(self, x):
            y = self.pre(x)
            return self.head(y.ravel() if method else torch.ravel(y))

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    flat_keep = [row * 6 + column for row in range(2) for column in keep]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]).reshape(-1)
    torch.testing.assert_close(
        model(sample), F.linear(hidden, dense.head.weight[:, flat_keep], dense.head.bias)
    )
    model(sample).sum().backward()


@pytest.mark.parametrize("name", ["ndimension", "nelement"])
def test_shape_aliases_preserve_dimension_sources(name, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            width = (
                y.size(y.ndimension() - 1) if name == "ndimension" else y.nelement() // y.size(0)
            )
            return self.head(y.reshape(y.size(0), width))

    model = Model().eval()
    sample = torch.randn(2, 3)
    dense = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    expected = F.linear(
        F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]),
        dense.head.weight[:, keep],
        dense.head.bias,
    )
    torch.testing.assert_close(model(sample), expected)


@pytest.mark.parametrize("method", [False, True])
def test_split_with_sizes_keeps_static_ports_and_feature_coordinates(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            a, b = (
                y.split_with_sizes(split_sizes=[1, 3], dim=0)
                if method
                else torch.split_with_sizes(y, split_sizes=[1, 3], dim=0)
            )
            return self.head(a), self.head(b)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(4, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    values = F.linear(
        F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]),
        dense.head.weight[:, keep],
        dense.head.bias,
    )
    actual = model(sample)
    torch.testing.assert_close(actual[0], values[:1])
    torch.testing.assert_close(actual[1], values[1:])


@pytest.mark.parametrize("method", [False, True])
def test_split_with_sizes_rejects_stale_partition_boundaries(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 6)

        def forward(self, x):
            y = self.pre(x)
            return y.split_with_sizes([3, 3], 1) if method else torch.split_with_sizes(y, [3, 3], 1)

    model = Model().eval()
    original = model.pre.weight.clone()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("pre.weight").axis(0).select([1])]
        )
    torch.testing.assert_close(model.pre.weight, original)
    assert model.pre.out_features == 6


@pytest.mark.parametrize("method", [False, True])
def test_select_preserves_other_axes_but_protects_index_coordinates(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            return self.head(
                y.select(dim=0, index=-1) if method else torch.select(y, dim=0, index=-1)
            )

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    select = next(op for op in graph.operations() if op.node.target in ("select", torch.select))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    with pytest.raises(PlanningError, match="unsupported_axis"):
        pruner.plan_remove([select.inputs[0].axis(0).select([0])])
    torch.testing.assert_close(model.pre.weight, dense.pre.weight)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])[-1]
    torch.testing.assert_close(
        model(sample), F.linear(hidden, dense.head.weight[:, keep], dense.head.bias)
    )


@pytest.mark.parametrize("name", ["dot", "vdot", "mv", "outer", "ger"])
@pytest.mark.parametrize("method", [False, True])
def test_vector_contraction_and_outer_entries_preserve_distinct_axis_roles(
    name, method, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 6)
            self.right = nn.Parameter(torch.randn(4 if name in ("outer", "ger") else 6))

        def forward(self, x):
            y = self.pre(x)
            key = (
                "vec"
                if name == "mv"
                else "vec2"
                if name in ("outer", "ger")
                else "other"
                if name == "vdot"
                else "tensor"
            )
            result = (
                getattr(y, name)(**{key: self.right})
                if method
                else getattr(torch, name)(y, **{key: self.right})
            )
            return result.sum() if name in ("outer", "ger") else result

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3) if name == "mv" else torch.randn(3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])])
    pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    if name in ("outer", "ger"):
        assert model.right.numel() == 4
        expected = hidden.sum() * dense.right.sum()
    else:
        assert model.right.numel() == 5
        expected = hidden @ dense.right[keep]
    torch.testing.assert_close(model(sample), expected)
    model(sample).sum().backward()


@pytest.mark.parametrize("name", ["all", "any", "nansum", "nanmean"])
@pytest.mark.parametrize("method", [False, True])
def test_reduction_spelling_keeps_unreduced_feature_axis(name, method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            y = y > 0 if name in ("all", "any") else y
            y = getattr(y, name)(dim=1) if method else getattr(torch, name)(y, dim=1)
            return self.head(y.float())

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 4, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    if name in ("all", "any"):
        counts = (hidden > 0).sum(1)
        reduced = counts == 4 if name == "all" else counts > 0
    else:
        reduced = hidden.sum(1) / (4 if name == "nanmean" else 1)
    torch.testing.assert_close(
        model(sample), F.linear(reduced.float(), dense.head.weight[:, keep], dense.head.bias)
    )


@pytest.mark.parametrize("name", ["T", "H", "mT", "mH"])
def test_transpose_properties_keep_complex_original_coordinates(name, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 6, dtype=torch.complex64)

        def forward(self, x):
            return getattr(self.pre(x), name)

    model = Model()
    sample = torch.randn(2, 3, dtype=torch.complex64)
    expected = model.pre(sample)[..., [0, 2, 3, 4, 5]]
    expected = expected.transpose(-1, -2)
    if name in ("H", "mH"):
        expected = expected.conj()
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(sample), expected)
