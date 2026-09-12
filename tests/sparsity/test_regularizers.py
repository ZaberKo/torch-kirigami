import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, StaleGraphError
from torch_kirigami.pruning import (
    CandidateSpace,
    ChannelRatio,
    Magnitude,
    ParameterGroup,
    PlanningError,
    Pruner,
)
from torch_kirigami.sparsity import GroupLasso, GroupSquaredL2, ScaleL1


def setup(dtype=torch.float64, device="cpu"):
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[3.0, 4.0], [1.0, 2.0], [0.0, 0.0]], device=device))
        model[0].bias.fill_(0)
        model[2].weight.fill_(0)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=dtype, device=device),))
    return model, graph


@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
def test_penalty_gradient_overlap_dedup_and_purity(kind, execution_device):
    model, graph = setup(device=execution_device)
    axis = graph.parameter("0.weight").axis(0)
    row = ParameterGroup(graph, (axis.select([0]), axis.select([0])))
    column = ParameterGroup(graph, (graph.parameter("0.weight").axis(1).select([0]),))
    regularizer = kind((row, row, column), coefficients=(2, 2, 3))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    penalty = regularizer()
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    expected = 2 * 5 + 3 * 10**0.5 if kind is GroupLasso else 25 + 15
    torch.testing.assert_close(penalty, penalty.new_tensor(expected))
    penalty.backward()
    expected_grad = torch.zeros_like(model[0].weight)
    if kind is GroupLasso:
        expected_grad[0] += expected_grad.new_tensor([6 / 5, 8 / 5])
        expected_grad[:, 0] += expected_grad.new_tensor([9 / 10**0.5, 3 / 10**0.5, 0])
    else:
        expected_grad[0] += expected_grad.new_tensor([6, 8])
        expected_grad[:, 0] += expected_grad.new_tensor([9, 3, 0])
    torch.testing.assert_close(model[0].weight.grad, expected_grad)
    model.zero_grad()
    regularizer().backward()  # A second call must create a new graph.
    torch.testing.assert_close(model[0].weight.grad, expected_grad)


@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
def test_public_regularizer_gradcheck_and_zero_subgradient(kind):
    model, graph = setup()
    group = ParameterGroup(graph, (graph.parameter("0.weight").axis(0).select([0, 1]),))
    regularizer = kind((group,))
    assert torch.autograd.gradcheck(lambda _: regularizer(), (model[0].weight,))
    with torch.no_grad():
        model[0].weight.zero_()
    regularizer().backward()
    assert torch.count_nonzero(model[0].weight.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_precision_scale_aliases_and_freshness(dtype):
    model, graph = setup(dtype)
    regularizer = ScaleL1(graph, ("0.bias", "0.bias"))
    with torch.no_grad():
        model[0].bias.copy_(model[0].bias.new_tensor([-2, 0, 3]))
    value = regularizer()
    assert value.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    value.backward()
    torch.testing.assert_close(model[0].bias.grad, model[0].bias.new_tensor([-1, 0, 1]))
    assert value.item() == 5
    Pruner(model, graph=graph).prune(metric=Magnitude(), budget=ChannelRatio(0.34))
    with pytest.raises(StaleGraphError):
        regularizer()


def test_group_discovery_filter_duplicates_and_incomplete_path():
    model, graph = setup()
    space = CandidateSpace(graph)
    assert len(space.axes) == 1 and len(space.candidates) == 3
    groups = space.parameter_groups([space.candidates[0]] * 2)
    assert len(groups) == 1
    assert {s.tensor.paths[0] for s in groups[0].selections} == {"0.weight", "0.bias", "2.weight"}
    with pytest.raises(ValueError, match="nonempty"):
        space.parameter_groups(parameter_filter=lambda *_: False)
    with pytest.raises(ValueError, match="conflicting"):
        GroupLasso(groups * 2, coefficients=(1, 2))
    for coefficients in [(-1,), (float("nan"),), (torch.tensor(1.0),), ()]:
        with pytest.raises(ValueError):
            GroupLasso(groups, coefficients=coefficients)
    model[1] = nn.Sigmoid()
    with pytest.raises(StaleGraphError):
        GroupLasso(groups)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.AdamW])
def test_sparse_loss_accumulation_matches_single_batch(optimizer_cls):
    model, graph = setup()
    other, other_graph = setup()
    other.load_state_dict(model.state_dict())
    reg = GroupLasso(CandidateSpace(graph).parameter_groups())
    other_reg = GroupLasso(CandidateSpace(other_graph).parameter_groups())
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    optimizers = [optimizer_cls(m.parameters(), lr=0.01) for m in (model, other)]
    for optimizer in optimizers:
        optimizer.zero_grad()
    (model(x).square().mean() + 0.1 * reg()).backward()
    for sample in x.split(1):
        ((other(sample).square().mean() + 0.1 * other_reg()) / 2).backward()
    for optimizer in optimizers:
        optimizer.step()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left, right)


def test_autocast_sparse_loss_then_public_prune_and_train(execution_device):
    device = torch.device(execution_device)
    model, graph = setup(torch.float32, device)
    regularizer = GroupLasso(CandidateSpace(graph).parameter_groups())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    x = torch.ones(2, 2, device=device)
    optimizer.zero_grad()
    with torch.autocast(
        device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16
    ):
        loss = model(x).square().mean() + 0.01 * regularizer()
    assert loss.dtype == torch.float32
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    scaler.step(optimizer)
    scaler.update()
    Pruner(model, graph=graph).prune(metric=Magnitude(), budget=ChannelRatio(0.34))
    fresh = DependencyGraph.build(model, args=(x,))
    new_reg = GroupLasso(CandidateSpace(fresh).parameter_groups())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    (model(x).square().mean() + 0.01 * new_reg()).backward()
    optimizer.step()
    assert model[0].out_features == 2


def test_unknown_influence_rejects_groups_but_valid_alternative_can_train():
    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.bad = nn.Linear(2, 3)
            self.good = nn.Linear(2, 3)
            self.out = nn.Linear(3, 1)

        def forward(self, x):
            return torch.special.gammaln(self.bad(x)), self.out(self.good(x))

    model = Branches()
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2),))
    space = CandidateSpace(graph)
    before = {n: t.clone() for n, t in model.state_dict().items()}
    with pytest.raises(PlanningError, match="Incomplete"):
        space.parameter_groups()
    candidates = [c for c in space.candidates if c.axis.tensor.paths[0] == "good.weight"]
    penalty = GroupLasso(space.parameter_groups(candidates))
    penalty().backward()
    assert model.good.weight.grad is not None and model.bad.weight.grad is None
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
