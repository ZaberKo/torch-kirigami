"""Parameter writes validate finite values, storage aliases, and rollback."""

import pytest
import torch
from torch import nn

import torch_kirigami.sparsity.operations as operations
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import ParameterGroup
from torch_kirigami.sparsity import (
    set_group_norms_,
    zero_groups_,
)


@pytest.mark.parametrize("failure", ["prepare", "commit"])
def test_sparse_region_writes_restore_all_values_on_failure(failure, execution_device, monkeypatch):
    model = nn.Linear(4, 4).double().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 4, dtype=torch.float64),))
    weight, bias = graph.parameter("weight"), graph.parameter("bias")
    group = ParameterGroup(graph, (weight.axis(0).select([1]), bias.axis(0).select([2])))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    identities = {name: id(p) for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    calls = []
    if failure == "prepare":
        clone = torch.Tensor.clone

        def fail_clone(tensor, *args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("allocation failed")
            return clone(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    else:
        write = operations._write

        def fail_write(tensor, index, values):
            write(tensor, index, values)
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("commit failed after writing")

        monkeypatch.setattr(operations, "_write", fail_write)
    with pytest.raises(RuntimeError, match="failed"):
        zero_groups_((group,))
    for name, p in model.named_parameters():
        assert id(p) == identities[name]
        torch.testing.assert_close(p, before[name])
        torch.testing.assert_close(p.grad, torch.ones_like(p))


@pytest.mark.parametrize("axis", [0, 1])
def test_disjoint_norm_targets_sweep_rows_and_columns(axis, execution_device, monkeypatch):
    model = nn.Linear(30, 30, bias=False).double().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 30, dtype=torch.float64),))
    ref = graph.parameter("weight")
    groups = tuple(ParameterGroup(graph, (ref.axis(axis).select([i]),)) for i in range(30))
    set_group_norms_(groups, (1.0,) * 30)
    torch.testing.assert_close(model.weight.norm(dim=1 - axis), torch.ones(30, dtype=torch.float64))
    row = ParameterGroup(graph, (ref.axis(0).select([0]),))
    column = ParameterGroup(graph, (ref.axis(1).select([0]),))
    before = model.weight.detach().clone()
    with pytest.raises(ValueError, match="disjoint"):
        set_group_norms_((row, column), (1.0, 1.0))
    torch.testing.assert_close(model.weight, before)


def test_sparse_writes_keep_unproved_nonoverlapping_layouts_usable(execution_device):
    model = nn.Linear(2, 3, bias=False).double().eval()
    model.weight = nn.Parameter(torch.arange(12, dtype=torch.float64).as_strided((3, 2), (2, 3)))
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    selected = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([1]),))
    expected = model.weight.detach().clone()
    expected[1].zero_()
    zero_groups_((selected,))
    torch.testing.assert_close(model.weight, expected)
    assert model.weight.stride() == (2, 3)


def test_sparse_writes_do_not_silently_modify_aliased_logical_rows(execution_device):
    model = nn.Linear(2, 3, bias=False).double().eval()
    model.weight = nn.Parameter(torch.ones(1, 2, dtype=torch.float64).expand(3, 2))
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    selected = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([1]),))
    before = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match="single memory location"):
        zero_groups_((selected,))
    torch.testing.assert_close(model.weight, before)


def test_sparse_union_checks_final_dtype_before_any_commit(execution_device, monkeypatch):
    model = nn.Linear(4, 4).half().eval()
    with torch.no_grad():
        model.weight.fill_(1)
        model.bias.fill_(60_000)
    graph = DependencyGraph.build(model, args=(torch.ones(1, 4, dtype=torch.float16),))
    group = ParameterGroup(
        graph,
        (
            graph.parameter("weight").axis(0).select([0]),
            graph.parameter("bias").axis(0).select([0]),
        ),
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    writes = []
    original = operations._write

    def counted(*args):
        writes.append(1)
        return original(*args)

    monkeypatch.setattr(operations, "_write", counted)
    with pytest.raises(ValueError, match="nonfinite"):
        operations.scale_groups_((group,), 2)
    assert writes == []
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_sparse_union_batches_finite_checks(execution_device, monkeypatch):
    model = nn.Linear(4, 16).eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 4),))
    axis = graph.parameter("weight").axis(0)
    groups = tuple(ParameterGroup(graph, (axis.select([i]),)) for i in range(0, 16, 2))
    original = torch.Tensor.__bool__
    calls = []

    def counted(tensor):
        calls.append(1)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted)
    zero_groups_(groups)
    # Input and final-cast checks are batched. Device dispatch can add a constant overhead.
    assert len(calls) <= 4
    torch.testing.assert_close(model.weight[::2], torch.zeros_like(model.weight[::2]))
