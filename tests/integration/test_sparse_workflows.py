import io

import pytest
import torch
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
from torch_kirigami import StaleGraphError
from torch_kirigami.pruning import (
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    Greedy,
    Magnitude,
    ParameterGroup,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import (
    GateBinding,
    GateMagnitude,
    GroupLasso,
    set_group_norms_,
    zero_groups_,
)


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
    counts = tuple(4 if a.tensor.paths[0] == "attn.k.weight" else 0 for a in space.channel_axes)
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
