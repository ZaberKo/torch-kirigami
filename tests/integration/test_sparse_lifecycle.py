import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tests.support.workflow_fixtures import (
    MLP,
    ResidualCNN,
    Transformer,
    build_space,
    restore_training,
    save_training,
    train_steps,
)
from torch_kirigami import DependencyGraph, StaleGraphError
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    Greedy,
    Magnitude,
    ParameterGroup,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
    unique_groups,
)
from torch_kirigami.sparsity import (
    GateBinding,
    GateMagnitude,
    GroupLasso,
    GroupSquaredL2,
    set_group_norms_,
    zero_groups_,
)


def test_candidate_group_alignment_training_and_compaction(execution_device):
    model = nn.Sequential(nn.Linear(2, 3, bias=False), nn.Linear(3, 1, bias=False)).double()
    with torch.no_grad():
        model[0].weight.copy_(model[0].weight.new_tensor([[3, 4], [1, 2], [5, 6]]))
        model[1].weight.copy_(model[1].weight.new_tensor([[1, 2, 3]]))
    reference = copy.deepcopy(model)
    x = torch.randn(2, 2, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    producer = graph.parameter("0.weight").axis(0)
    consumer = graph.parameter("1.weight").axis(1)
    candidates = (
        Candidate("consumer_zero", (consumer.select([0]),), consumer),
        Candidate("producer_one", (producer.select([1]),), producer),
        Candidate("producer_zero", (producer.select([0]),), producer),
    )
    with pytest.raises(PlanningError, match=r"consumer_zero.*no parameter regions"):
        pruner.parameter_groups(candidates, parameter_filter=lambda *_: False)
    graph.validate()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    groups = pruner.parameter_groups(iter(candidates))
    assert tuple(g.key for g in groups) == tuple(c.key for c in candidates)
    assert groups[0].equivalent(groups[2])
    assert unique_groups(groups) == (groups[0], groups[1])

    regularizer = GroupSquaredL2(groups, coefficients=(1, 2, 3))
    first = reference[0].weight[0].square().sum() + reference[1].weight[:, 0].square().sum()
    second = reference[0].weight[1].square().sum() + reference[1].weight[:, 1].square().sum()
    expected_loss = 0.5 * (4 * first + 2 * second)
    loss = regularizer()
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    expected_loss.backward()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    graph.validate()  # Numerical training does not invalidate group bindings.
    keep = [1, 2]
    expected = F.linear(F.linear(x, reference[0].weight[keep]), reference[1].weight[:, keep])
    plan = pruner.plan_remove(candidates[0].remove)
    compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    torch.testing.assert_close(compact(x), expected)
    with pytest.raises(StaleGraphError):
        regularizer()
    compact(x).sum().backward()


def test_separate_groups_and_joint_impact_have_distinct_objectives(execution_device):
    model = nn.Sequential(
        nn.Conv2d(1, 2, 1, bias=False), nn.Flatten(), nn.Linear(4, 2, bias=False)
    ).double()
    with torch.no_grad():
        model[0].weight.fill_(3)
        model[2].weight.zero_()
        model[2].weight[0, 0] = 4
        model[2].weight[1, 1] = 12
    graph = DependencyGraph.build(model, args=(torch.ones(1, 1, 1, 2, dtype=torch.float64),))
    pruner = Pruner(model, graph=graph)
    axis = graph.parameter("2.weight").axis(1)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in (0, 1))
    groups = pruner.parameter_groups(candidates)
    assert all(s.tensor == axis.tensor for g in groups for s in g.selections)
    impact = pruner.impact(candidates)
    assert impact.complete
    joint = ParameterGroup(graph, impact.parameters)
    assert {s.tensor for s in joint.selections} == {axis.tensor, graph.parameter("0.weight")}
    assert GroupLasso(groups)().item() == pytest.approx(16)
    assert GroupLasso((joint,))().item() == pytest.approx(13)


def test_residual_grouped_cnn_sparse_training_and_independent_compact_reference(execution_device):
    model = ResidualCNN().to(execution_device).eval()
    x = torch.randn(2, 3, 5, 5, device=execution_device)
    pruner, space = build_space(model, x)
    regularizer = GroupLasso(pruner.parameter_groups(space.candidates))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    (model(x).square().mean() + 0.01 * regularizer()).backward()
    optimizer.step()
    keep = [1, 2, 3, 5, 6, 7]
    with torch.no_grad():
        stem = F.conv2d(x, model.stem.weight[keep], model.stem.bias[keep])
        stem = F.batch_norm(
            stem,
            model.bn.running_mean[keep],
            model.bn.running_var[keep],
            model.bn.weight[keep],
            model.bn.bias[keep],
            training=False,
            eps=model.bn.eps,
        ).relu()
        blocks = model.block.weight.reshape(2, 4, 4, 3, 3)
        weight = blocks[:, [1, 2, 3]][:, :, [1, 2, 3]].reshape(6, 3, 3, 3)
        y = (stem + F.conv2d(stem, weight, model.block.bias[keep], padding=1, groups=2)).relu()
        reference = F.linear(y.mean((2, 3)), model.out.weight[:, keep], model.out.bias)
    pruner = Pruner(model, graph=pruner.graph)
    plan = pruner.plan_remove((pruner.graph.parameter("stem.weight").axis(0).select([0, 4]),))
    pruner.apply(plan)
    torch.testing.assert_close(model(x), reference)
    with pytest.raises(StaleGraphError):
        regularizer()
    fresh_pruner, fresh = build_space(model, x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    (
        model(x).sum() + 0.01 * GroupLasso(fresh_pruner.parameter_groups(fresh.candidates))()
    ).backward()
    optimizer.step()


def test_transformer_head_gate_compaction_and_checkpoint(execution_device):
    model = Transformer(gated=True).to(execution_device).eval()
    x = torch.randn(2, 3, 8, device=execution_device)
    pruner, space = build_space(model, x)
    with torch.no_grad():
        model.attn.gate.weight.copy_(model.attn.gate.weight.new_tensor([0, 0, 1.5, 0.5]))
        model.ffn_gate.weight.fill_(0.7)
    reference = model(x).detach()
    binding = GateBinding(pruner.graph, "attn.gate")
    counts = tuple(
        a.tensor.shape[a.dim] - (4 if a.tensor.paths[0] == "attn.k.weight" else 0)
        for a in space.channel_axes
    )
    plan = Pruner(model, graph=pruner.graph).plan(
        CandidateSpace(
            candidates=binding.candidates(pruner, space.candidates), channel_axes=space.channel_axes
        ),
        budget=ChannelCount(counts, space.channel_axes),
        strategy=Greedy(GateMagnitude((binding,))),
    )
    Pruner(model, graph=pruner.graph).apply(plan)
    assert model.attn.kv_heads == 1 and model.attn.q_heads == model.attn.gate.size == 2
    torch.testing.assert_close(model(x), reference)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(Transformer(gated=True).to(execution_device), stream)
    torch.testing.assert_close(restored(x), reference)
    restored(x).sum().backward()
    torch.optim.SGD(restored.parameters(), lr=0.01).step()


def test_example_training_checkpoint_restores_optimizer_rng_and_algorithm(tmp_path):
    torch.manual_seed(9)
    model = MLP()
    x = torch.randn(3, 8)
    pruner, _space = build_space(model, x)
    Pruner(model, graph=pruner.graph).apply(
        Pruner(model, graph=pruner.graph).plan_remove(
            (pruner.graph.parameter("hidden.weight").axis(0).select([1, 4]),)
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(x).square().mean().backward()
    optimizer.step()
    save_training(tmp_path, model, optimizer, algorithm={"round": 1}, schedule={"step": 2})
    expected_random = torch.randn(3)
    restored, resumed_optimizer, algorithm, schedule = restore_training(
        tmp_path,
        MLP,
        lambda parameters: torch.optim.AdamW(parameters, lr=0.01),
    )
    torch.testing.assert_close(torch.randn(3), expected_random)
    assert algorithm == {"round": 1} and schedule == {"step": 2}
    for network, opt in ((model, optimizer), (restored, resumed_optimizer)):
        opt.zero_grad()
        network(x).square().mean().backward()
        opt.step()
    for left, right in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("operation", ["zero", "decay"])
def test_soft_example_regrowth_keeps_momentum_and_can_compact(operation, execution_device):
    torch.manual_seed(7)
    model = MLP().to(execution_device)
    x = torch.randn(12, 8, device=execution_device)
    y = torch.arange(12, device=execution_device) % 3
    pruner, space = build_space(model, x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    train_steps(model, x, y, optimizer, 1)
    pruner = Pruner(model, graph=pruner.graph)
    plan = pruner.plan(
        pruner.discover_candidates(), budget=ChannelRatio(0.5), strategy=Greedy(Magnitude())
    )
    selected = [c for c in space.candidates if c.key in plan.selected]
    group = ParameterGroup(pruner.graph, pruner.impact(selected).parameters)
    buffers = {p: state["momentum_buffer"].clone() for p, state in optimizer.state.items()}
    if operation == "zero":
        zero_groups_((group,))
    else:
        set_group_norms_((group,), (0.0,))
    assert GroupLasso((group,))().item() == 0
    for parameter, expected in buffers.items():
        torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], expected)
    train_steps(model, x, y, optimizer, 1)
    assert GroupLasso((group,))().item() > 0
    pruner.apply(
        pruner.plan(
            pruner.discover_candidates(), budget=ChannelRatio(0.5), strategy=Greedy(Magnitude())
        )
    )
    build_space(model, x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_steps(model, x, y, optimizer, 1)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(MLP().to(execution_device), stream)
    torch.testing.assert_close(restored(x), model(x))
