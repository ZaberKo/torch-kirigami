import io

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import (
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    Greedy,
    ParameterGroup,
    PlanningContext,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import (
    ChannelGate,
    GateBinding,
    GateMagnitude,
    ScaleL1,
    register_gate_operators,
    scale_groups_,
    set_group_norms_,
    zero_groups_,
)


class GatedModel(nn.Module):
    def __init__(self, convolution=False):
        super().__init__()
        self.first = nn.Conv2d(2, 4, 1) if convolution else nn.Linear(2, 4)
        self.gate = ChannelGate(4, 1 if convolution else -1)
        self.last = nn.Conv2d(4, 2, 1) if convolution else nn.Linear(4, 2)

    def forward(self, x):
        return self.last(self.gate(self.first(x).relu()))


@pytest.mark.parametrize("convolution", [False, True])
def test_gate_training_pruning_independent_reference_and_checkpoint(convolution, execution_device):
    torch.manual_seed(4)
    model = GatedModel(convolution).to(execution_device)
    x = torch.randn((2, 2, 3, 3) if convolution else (2, 2), device=execution_device)
    operators = register_gate_operators(OperatorRegistry.default())
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    pruner = Pruner(graph.model, graph=graph)
    space = pruner.discover_candidates()
    assert len(space.channel_axes) == 1  # No extra gate budget domain.
    regularizer = ScaleL1(graph, ("gate.weight",))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    (model(x).square().mean() + 0.1 * regularizer()).backward()
    optimizer.step()
    with torch.no_grad():
        model.gate.weight.copy_(model.gate.weight.new_tensor([2.0, 0.0, 0.5, 3.0]))
    model.gate.set_mask([1, 0, 1, 1])
    reference = model(x).detach()
    binding = GateBinding(graph, "gate")
    candidates = binding.candidates(pruner, space.candidates)
    plan = Pruner(model, graph=graph).plan(
        CandidateSpace(candidates=candidates, channel_axes=space.channel_axes),
        budget=ChannelCount((1,), space.channel_axes),
        strategy=Greedy(GateMagnitude((binding,))),
    )
    Pruner(model, graph=graph).apply(plan)
    assert model.gate.size == 3
    torch.testing.assert_close(model.gate.weight, model.gate.weight.new_tensor([2.0, 0.5, 3.0]))
    torch.testing.assert_close(model(x), reference)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(GatedModel(convolution).to(execution_device), stream)
    torch.testing.assert_close(restored(x), reference)
    fresh = DependencyGraph.build(restored, args=(x,), operators=operators)
    optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    optimizer.zero_grad()
    (restored(x).square().mean() + 0.1 * ScaleL1(fresh, ("gate.weight",))()).backward()
    optimizer.step()


def test_gate_rejection_and_mask_update_purity():
    gate = ChannelGate(3, -1, trainable=False)
    assert not gate.weight.requires_grad
    with pytest.raises(ValueError):
        gate(torch.ones(2, 4))
    before = gate.mask.clone()
    for mask in ([1, 0], [1, 0.5, 0], [1, float("nan"), 0]):
        with pytest.raises(ValueError):
            gate.set_mask(mask)
        torch.testing.assert_close(before, gate.mask)


def test_region_union_scaling_norm_targets_and_atomic_rejection(execution_device):
    model = nn.Linear(2, 3, bias=False).to(device=execution_device, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[3, 4], [0, 2], [0, 0]]))
    graph = DependencyGraph.build(
        model, args=(torch.ones(1, 2, device=execution_device, dtype=torch.float64),)
    )
    ref = graph.parameter("weight")
    rows = [ParameterGroup(graph, (ref.axis(0).select([i]),)) for i in range(3)]
    column = ParameterGroup(graph, (ref.axis(1).select([0]),))
    model.weight.grad = torch.ones_like(model.weight)
    scale_groups_((rows[0], column), 0.5)
    torch.testing.assert_close(model.weight, model.weight.new_tensor([[1.5, 2], [0, 2], [0, 0]]))
    torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))
    set_group_norms_(rows[:2], [5, 1])
    torch.testing.assert_close(model.weight, model.weight.new_tensor([[3, 4], [0, 1], [0, 0]]))
    before = model.weight.detach().clone()
    for groups, targets in [
        ((rows[0], column), (1, 1)),
        ((rows[0], rows[2]), (1, 2)),
        ((rows[0], rows[0]), (1, 2)),
    ]:
        with pytest.raises(ValueError):
            set_group_norms_(groups, targets)
        torch.testing.assert_close(model.weight, before)
    zero_groups_((rows[0], rows[0]))
    assert not model.weight[0].any()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones(1, 2, device=execution_device, dtype=torch.float64)).sum().backward()
    optimizer.step()
    assert model.weight[0].ne(0).all()  # Soft zeroing explicitly permits regrowth.


def test_gate_and_scale_parameter_aliases_are_not_counted_twice():
    class Aliases(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 3)
            self.gate = ChannelGate(3, -1)
            self.alias = self.gate
            self.out = nn.Linear(3, 1)

        def forward(self, x):
            return self.out(self.gate(self.hidden(x)))

    model = Aliases()
    with torch.no_grad():
        model.gate.weight.copy_(torch.tensor([-2.0, 0.0, 3.0]))
    graph = DependencyGraph.build(
        model,
        args=(torch.ones(2, 2),),
        operators=register_gate_operators(OperatorRegistry.default()),
    )
    assert ScaleL1(graph, ("gate.weight", "alias.weight"))().item() == 5
    pruner = Pruner(graph.model, graph=graph)
    space = pruner.discover_candidates()
    metric = GateMagnitude((GateBinding(graph, "gate"), GateBinding(graph, "alias")))
    context = PlanningContext(
        graph,
        graph.operations(),
        space.candidates,
        ChannelRatio(0.5),
        space.channel_axes,
        pruner.constraints,
    )
    assert context.score(metric, space.candidates) == (2.0, 0.0, 3.0)


@pytest.mark.parametrize("operation", ["scale", "zero", "norm"])
@pytest.mark.parametrize("include_second", [False, True])
def test_parameter_operations_reject_storage_aliases_atomically(
    operation, include_second, execution_device
):
    class SharedStorage(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], device=execution_device))
            self.b = nn.Parameter(self.a.detach())

        def forward(self, x):
            return x * self.a + x * self.b

    model = SharedStorage()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3, device=execution_device),))
    group = ParameterGroup(graph, (graph.parameter("a").axis(0).select([0]),))
    groups = (
        (group,)
        if not include_second
        else (group, ParameterGroup(graph, (graph.parameter("b").axis(0).select([1]),)))
    )
    before = model.a.detach().clone()
    version = model.a._version
    model.a.grad = torch.ones_like(model.a)
    with pytest.raises(ValueError, match="sharing storage"):
        if operation == "scale":
            scale_groups_(groups, 2)
        elif operation == "zero":
            zero_groups_(groups)
        else:
            set_group_norms_(groups, [2] * len(groups))
    torch.testing.assert_close(model.a, before)
    torch.testing.assert_close(model.b, before)
    torch.testing.assert_close(model.a.grad, torch.ones_like(model.a))
    assert model.a._version == version


@pytest.mark.parametrize(
    "dtype,value", [(torch.float16, 1e-7), (torch.float32, 1e-40), (torch.float64, 1e-320)]
)
def test_norm_projection_of_nonzero_subnormals(dtype, value, execution_device):
    model = nn.Linear(1, 2, bias=False, device=execution_device, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[value], [-value]]))
    graph = DependencyGraph.build(
        model, args=(torch.ones(1, 1, device=execution_device, dtype=dtype),)
    )
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0, 1]),))
    set_group_norms_((group,), (1.0,))
    reference = torch.tensor([[2**-0.5], [-(2**-0.5)]], device=execution_device, dtype=dtype)
    torch.testing.assert_close(model.weight, reference)
    with torch.no_grad():
        model.weight.fill_(2)
    before = model.weight.detach().clone()
    version = model.weight._version
    with pytest.raises(ValueError, match="nonfinite"):
        scale_groups_((group,), float("1.7976931348623157e308"))
    torch.testing.assert_close(model.weight, before)
    assert model.weight._version == version


def test_shared_gate_weights_with_distinct_masks_score_and_prune(execution_device):
    class BranchGates(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 3)
            self.g1, self.g2 = ChannelGate(3, -1), ChannelGate(3, -1)
            self.g2.weight = self.g1.weight
            self.alias = self.g1
            self.out = nn.Linear(3, 1)

        def forward(self, x):
            y = self.hidden(x)
            return self.out(self.g1(y) + self.g2(y))

    model = BranchGates().to(execution_device)
    # Module.to may replace parameters on some PyTorch configurations.
    model.g2.weight = model.g1.weight
    model.g1.set_mask([1, 0, 1])
    model.g2.set_mask([0, 1, 1])
    x = torch.ones(2, 2, device=execution_device)
    operators = register_gate_operators(OperatorRegistry.default())
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    pruner = Pruner(graph.model, graph=graph)
    space = pruner.discover_candidates()
    bindings = [GateBinding(graph, name) for name in ("g1", "g2", "alias")]
    budget = ChannelCount((1,), space.channel_axes)
    for order in (bindings, list(reversed(bindings))):
        context = PlanningContext(
            graph,
            graph.operations(),
            space.candidates,
            budget,
            space.channel_axes,
            pruner.constraints,
        )
        assert context.score(GateMagnitude(order), space.candidates) == (1.0, 1.0, 2.0)
    # An actually inactive shared channel remains a valid physical alternative.
    model.g1.set_mask([0, 0, 1])
    model.g2.set_mask([0, 1, 1])
    reference = model(x).detach()
    pruner = Pruner(model, graph=graph)
    pruner.apply(
        pruner.plan(
            pruner.discover_candidates(), budget=budget, strategy=Greedy(GateMagnitude(bindings))
        )
    )
    assert model.g1.weight is model.g2.weight
    assert model.g1 is model.alias
    torch.testing.assert_close(model(x), reference)
    model(x).sum().backward()


@pytest.mark.parametrize("inplace", [False, True])
def test_gate_allocation_allows_following_relu_and_physical_pruning(inplace, execution_device):
    model = nn.Sequential(
        nn.Linear(4, 6), ChannelGate(6, -1), nn.ReLU(inplace=inplace), nn.Linear(6, 3)
    ).to(execution_device)
    x = torch.randn(2, 4, device=execution_device)
    original_input = x.clone()
    model[1].set_mask([1, 0, 1, 0, 1, 1])
    expected = model(x).detach()
    graph = DependencyGraph.build(
        model, args=(x,), operators=register_gate_operators(OperatorRegistry.default())
    )
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove((graph.parameter("0.weight").axis(0).select([1, 3]),)))
    assert model[0].out_features == model[1].size == model[3].in_features == 4
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(x, original_input)
    model(x).sum().backward()
    assert model[1].weight.grad is not None


def test_gate_allocation_does_not_relax_inplace_multiple_consumer_guard():
    class ForkedGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 6)
            self.gate = ChannelGate(6, -1)
            self.relu = nn.ReLU(inplace=True)
            self.last = nn.Linear(6, 3)

        def forward(self, x):
            y = self.gate(self.first(x))
            branch = y + 1
            return self.last(branch + self.relu(y))

    model = ForkedGate()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(
        model, args=(x,), operators=register_gate_operators(OperatorRegistry.default())
    )
    state = {name: (id(value), value.detach().clone()) for name, value in model.named_parameters()}
    with pytest.raises(PlanningError, match="alias/consumer"):
        Pruner(model, graph=graph).plan_remove(
            (graph.parameter("first.weight").axis(0).select([1]),)
        )
    for name, value in model.named_parameters():
        assert id(value) == state[name][0]
        torch.testing.assert_close(value, state[name][1])
