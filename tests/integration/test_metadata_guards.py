"""Eager metadata guards preserve the read value, rather than freezing all axes."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CaptureError, DependencyGraph
from torch_kirigami.pruning import ExecutionError, PlanningError, Pruner, PruningPlan


def observe(tensor, query):
    if query == "row_size":
        return tensor.size(0)
    if query == "row_size_axis":
        return tensor.size(axis=0)
    if query == "column_stride_axis":
        return tensor.stride(axis=-1)
    if query == "row_size_keyword":
        return tensor.size(dim=-2)
    if query == "column_size":
        return tensor.size(-1)
    if query == "shape":
        return tensor.shape[0]
    if query == "whole_size":
        return tensor.size()
    if query == "length":
        return len(tensor)
    if query == "numel":
        return tensor.numel()
    if query == "nelement":
        return tensor.nelement()
    if query == "stride":
        return tensor.stride()
    if query == "column_stride":
        return tensor.stride(dim=-1)
    if query == "row_stride":
        return tensor.stride(0)
    if query == "contiguous":
        return tensor.is_contiguous(memory_format=torch.contiguous_format)
    if query == "dim":
        return tensor.dim()
    if query == "ndimension":
        return tensor.ndimension()
    if query == "floating":
        return tensor.is_floating_point()
    if query == "complex":
        return tensor.is_complex()
    if query == "itemsize":
        return tensor.element_size()
    if query == "device_index":
        return tensor.get_device()
    if query == "storage_offset":
        return tensor.storage_offset()
    return getattr(tensor, query)


class MetadataBranch(nn.Module):
    def __init__(self, query, strided=False):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(3, 4), nn.Linear(4, 2), nn.Linear(4, 2)
        self.register_buffer("scale", torch.randn(4, 2).t() if strided else torch.randn(2, 4))
        self.query = query
        self.expected = observe(self.scale, query)

    def forward(self, x):
        hidden = self.a(x) * self.scale
        return (
            self.b(hidden) if observe(self.scale, self.query) == self.expected else self.c(hidden)
        )


@pytest.mark.parametrize(
    "query",
    [
        "row_size",
        "row_size_keyword",
        "row_size_axis",
        "column_stride_axis",
        "length",
        "column_stride",
        "contiguous",
        "dim",
        "ndimension",
        "ndim",
        "dtype",
        "device",
        "layout",
        "requires_grad",
        "floating",
        "complex",
        "itemsize",
        "device_index",
    ],
)
def test_unchanged_metadata_read_allows_other_axis_compaction(query, execution_device):
    model = MetadataBranch(query)
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    remove = graph.parameter("a.weight").axis(0).select([1])
    assert graph.propagate(remove=[remove]).status == "resolved"
    before = tuple(model.parameters())
    plan = PruningPlan.from_dict(Pruner(model, graph=graph).plan(remove=[remove]).to_dict())
    assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
    target = copy.deepcopy(model)
    Pruner(target).apply(plan)
    keep = [0, 2, 3]
    hidden = F.linear(x, original.a.weight[keep], original.a.bias[keep]) * original.scale[:, keep]
    expected = F.linear(hidden, original.b.weight[:, keep], original.b.bias)
    torch.testing.assert_close(target(x), expected)
    target(x).sum().backward()
    assert target.a.weight.grad is not None and target.b.weight.grad is not None


@pytest.mark.parametrize(
    "query", ["column_size", "shape", "whole_size", "numel", "nelement", "stride", "row_stride"]
)
def test_changed_metadata_read_rejects_before_mutation(query, execution_device):
    model = MetadataBranch(query)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    original, buffer = tuple(model.parameters()), model.scale
    expected = model(x).detach()
    with pytest.raises(PlanningError, match="metadata"):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    assert model.scale is buffer
    assert all(a is b for a, b in zip(original, model.parameters(), strict=True))
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("query", ["stride", "column_stride", "contiguous"])
def test_compaction_checks_actual_planned_buffer_layout(query, execution_device):
    model = MetadataBranch(query, strided=True)
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    before = tuple(model.parameters()), model.scale
    with pytest.raises(PlanningError, match="metadata"):
        Pruner(model, graph=graph).plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
    assert model.scale is before[1]
    assert all(a is b for a, b in zip(before[0], model.parameters(), strict=True))


@pytest.mark.parametrize("change", ["stride", "dtype", "requires_grad"])
def test_portable_metadata_plan_rejects_incompatible_original_binding(change, execution_device):
    model = MetadataBranch("contiguous")
    x = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(x,))
    plan = PruningPlan.from_dict(
        Pruner(model, graph=graph)
        .plan(remove=[graph.parameter("a.weight").axis(0).select([1])])
        .to_dict()
    )
    target = copy.deepcopy(model)
    target.scale = (
        torch.empty_strided((2, 4), (1, 2))
        if change == "stride"
        else target.scale.double()
        if change == "dtype"
        else target.scale.requires_grad_()
    )
    before = tuple(target.parameters()), target.scale
    with pytest.raises(ExecutionError):
        Pruner(target).apply(plan)
    assert target.scale is before[1]
    assert all(a is b for a, b in zip(before[0], target.parameters(), strict=True))


def test_storage_position_reads_are_not_portable_structural_facts(execution_device):
    model = MetadataBranch("storage_offset")
    before = model.scale
    with pytest.raises(CaptureError, match="data/derived-metadata"):
        DependencyGraph.build(model, args=(torch.ones(2, 3),))
    assert model.scale is before


class GroupedMetadataBranch(nn.Module):
    def __init__(self, query):
        super().__init__()
        self.a = nn.Conv2d(3, 6, 1)
        self.grouped = nn.Conv2d(6, 4, 1, groups=2)
        self.alias = self.grouped
        self.b, self.c = nn.Conv2d(4, 2, 1), nn.Conv2d(4, 2, 1)
        self.independent = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 2))
        self.query = query
        self.expected = self.observation()

    def observation(self):
        # Reading through registration storage is eager even for Parameters.
        # The alias must resolve to the same entity used by the grouped call.
        weight = self.alias._parameters["weight"]
        if self.query == "output_size":
            return weight.size(0)
        if self.query == "local_input_size":
            return weight.size(1)
        if self.query == "output_stride":
            return weight.stride(0)
        if self.query == "kernel_stride":
            return weight.stride(axis=-1)
        return weight.is_contiguous()

    def forward(self, x):
        hidden = self.grouped(self.a(x))
        y = self.b(hidden) if self.observation() == self.expected else self.c(hidden)
        return y, self.independent(x.mean((2, 3)))


@pytest.mark.parametrize(
    ("query", "allowed"),
    [
        ("output_size", True),
        ("local_input_size", False),
        ("output_stride", False),
        ("kernel_stride", True),
        ("contiguous", True),
    ],
)
def test_grouped_metadata_uses_final_partitioned_layout_without_blocking_other_branches(
    query, allowed, execution_device
):
    model = GroupedMetadataBranch(query)
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 2, 2)
    graph = DependencyGraph.build(model, args=(x,))
    before = tuple(model.parameters())
    remove = [graph.parameter("a.weight").axis(0).select([0, 4])]
    if not allowed:
        with pytest.raises(PlanningError, match="metadata"):
            Pruner(model, graph=graph).plan(remove=remove)
        assert all(a is b for a, b in zip(before, model.parameters(), strict=True))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original.state_dict()[name])
        # A rejected grouped component must not lock an independent branch.
        remove = [graph.parameter("independent.0.weight").axis(0).select([1])]
    plan = PruningPlan.from_dict(Pruner(model, graph=graph).plan(remove=remove).to_dict())
    target = copy.deepcopy(model)
    Pruner(target).apply(plan)
    assert target.grouped is target.alias
    if allowed:
        # The two old groups lose different local input columns (0 and 1).
        keep = [1, 2, 3, 5]
        packed = torch.cat(
            (original.grouped.weight[:2, 1:], original.grouped.weight[2:, [0, 2]]), dim=0
        )
        hidden = F.conv2d(x, original.a.weight[keep], original.a.bias[keep])
        expected = original.b(F.conv2d(hidden, packed, original.grouped.bias, groups=2))
        independent = original.independent(x.mean((2, 3)))
        recipe = next(r for r in plan.recipes if r.tensor.paths[0] == "grouped.weight")
        assert len(recipe.segments) == 2
        torch.testing.assert_close(target.grouped.weight, packed)
        assert target.grouped.weight.stride() == packed.stride()
        assert target.observation() == original.observation()
    else:
        expected = original(x)[0]
        keep = [0, 2, 3, 4, 5]
        first, last = original.independent
        hidden = F.linear(x.mean((2, 3)), first.weight[keep], first.bias[keep])
        independent = F.linear(hidden, last.weight[:, keep], last.bias)
    actual = target(x)
    torch.testing.assert_close(actual, (expected, independent))
    sum(value.sum() for value in actual).backward()
    assert target.a.weight.grad is not None
    assert target.independent[0].weight.grad is not None
