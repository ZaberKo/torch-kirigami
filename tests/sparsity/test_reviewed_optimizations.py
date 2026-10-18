"""Cached coordinates and sparse writes preserve live values and transactions."""

import pytest
import torch
from torch import nn

import torch_kirigami.sparsity.operations as operations
from torch_kirigami import DependencyGraph, IndexSet, OperatorRegistry, StaleGraphError
from torch_kirigami.pruning import ChannelRatio, ParameterGroup, PlanningContext, Pruner
from torch_kirigami.sparsity import (
    ChannelGate,
    GateBinding,
    GateMagnitude,
    GroupLasso,
    register_gate_operators,
    set_group_norms_,
    zero_groups_,
)


def test_interval_membership_does_not_expand_elements(monkeypatch):
    indices = IndexSet(((0, 2), (100, 1_000_000_000)))

    def forbidden(_):
        pytest.fail("Membership must inspect intervals, not enumerate positions")

    with monkeypatch.context() as patch:
        patch.setattr(IndexSet, "__iter__", forbidden)
        assert 0 in indices and 1 in indices and 999_999_999 in indices
        assert -1 not in indices and 2 not in indices and 1_000_000_000 not in indices
    assert 1.0 in IndexSet.span(0, 2) and "1" not in IndexSet.span(0, 2)

    class DifferentOrder(int):
        def __lt__(self, other):
            return True

    assert DifferentOrder(1) in IndexSet.span(0, 2)


def test_cached_regularizer_indices_keep_live_gradients_and_freshness(
    execution_device, monkeypatch
):
    model = nn.Linear(2, 3, bias=False).double().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2, dtype=torch.float64),))
    axis = graph.parameter("weight").axis(0)
    groups = tuple(ParameterGroup(graph, (axis.select([i]),)) for i in (0, 2))
    with torch.inference_mode():
        regularizer = GroupLasso(groups)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[3.0, 4.0], [9.0, 9.0], [5.0, 12.0]]))

    def forbidden(*args, **kwargs):
        pytest.fail("Repeated regularization must reuse coordinate tensors")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", forbidden)
        value = regularizer()
        value.backward()
    assert value.item() == 18.0
    torch.testing.assert_close(
        model.weight.grad, model.weight.new_tensor([[0.6, 0.8], [0.0, 0.0], [5 / 13, 12 / 13]])
    )
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        regularizer()
    with torch.no_grad():
        model.weight[0, 0] = 3.0
    model.train()
    with pytest.raises(StaleGraphError):
        regularizer()


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


def test_gate_score_batch_transfers_without_scalar_item_calls(execution_device, monkeypatch):
    model = nn.Sequential(nn.Linear(2, 3), ChannelGate(3, -1), nn.Linear(3, 1)).double().eval()
    with torch.no_grad():
        model[1].weight.copy_(model[1].weight.new_tensor([2.0, -3.0, 5.0]))
    graph = DependencyGraph.build(
        model,
        args=(torch.randn(2, 2, dtype=torch.float64),),
        operators=register_gate_operators(OperatorRegistry.default()),
    )
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.25), space.channel_axes, pruner.constraints
    )
    metric = GateMagnitude((GateBinding(graph, "1"),))

    def forbidden(*args, **kwargs):
        pytest.fail("Gate scoring must transfer the batch instead of individual scalars")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden)
        assert context.score(metric, space.candidates) == (2.0, 3.0, 5.0)


@pytest.mark.parametrize("change", ["dtype", "configuration", "binding"])
def test_cached_indices_do_not_replace_snapshot_validation(change, execution_device):
    model = nn.Linear(2, 3).float().eval()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 2),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    regularizer = GroupLasso((group,))
    old = id(model.weight), model.weight._version
    if change == "dtype":
        model.double()
        assert (id(model.weight), model.weight._version) == old
    elif change == "configuration":
        model.out_features = 99
        assert (id(model.weight), model.weight._version) == old
    else:
        model.weight = nn.Parameter(model.weight.detach().clone())
    with pytest.raises(StaleGraphError):
        regularizer()
