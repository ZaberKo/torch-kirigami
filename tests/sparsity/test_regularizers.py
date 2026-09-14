import math
import random
from decimal import Decimal, localcontext
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph, Region, Selection, StaleGraphError
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
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
    Pruner(model, graph=graph).prune(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.34),
        strategy=Greedy(Magnitude()),
    )
    with pytest.raises(StaleGraphError):
        regularizer()


def test_group_discovery_filter_duplicates_and_incomplete_path():
    model, graph = setup()
    pruner = Pruner(graph.model, graph=graph)
    space = pruner.discover_candidates()
    assert len(space.channel_axes) == 1 and len(space.candidates) == 3
    groups = pruner.parameter_groups([space.candidates[0]] * 2)
    assert len(groups) == 1
    assert {s.tensor.paths[0] for s in groups[0].selections} == {"0.weight", "0.bias", "2.weight"}
    with pytest.raises(ValueError, match="nonempty"):
        pruner.parameter_groups(space.candidates, parameter_filter=lambda *_: False)
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
    reg = GroupLasso(
        Pruner(graph.model, graph=graph).parameter_groups(
            Pruner(graph.model, graph=graph).discover_candidates().candidates
        )
    )
    other_reg = GroupLasso(
        Pruner(other_graph.model, graph=other_graph).parameter_groups(
            Pruner(other_graph.model, graph=other_graph).discover_candidates().candidates
        )
    )
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
    regularizer = GroupLasso(
        Pruner(graph.model, graph=graph).parameter_groups(
            Pruner(graph.model, graph=graph).discover_candidates().candidates
        )
    )
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
    Pruner(model, graph=graph).prune(
        Pruner(model, graph=graph).discover_candidates(),
        budget=ChannelRatio(0.34),
        strategy=Greedy(Magnitude()),
    )
    fresh = DependencyGraph.build(model, args=(x,))
    new_reg = GroupLasso(
        Pruner(fresh.model, graph=fresh).parameter_groups(
            Pruner(fresh.model, graph=fresh).discover_candidates().candidates
        )
    )
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
    pruner = Pruner(graph.model, graph=graph)
    space = pruner.discover_candidates()
    before = {n: t.clone() for n, t in model.state_dict().items()}
    with pytest.raises(PlanningError, match="Incomplete"):
        pruner.parameter_groups(space.candidates)
    candidates = [c for c in space.candidates if c.axis.tensor.paths[0] == "good.weight"]
    penalty = GroupLasso(pruner.parameter_groups(candidates))
    penalty().backward()
    assert model.good.weight.grad is not None and model.bad.weight.grad is None
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coordinates", ["row", "rows", "overlap", "irregular"])
def test_subnormal_group_gradient_matches_normalized_integer_reference(
    dtype, coordinates, execution_device
):
    # Integer directions form an independent reference without squaring tiny values.
    model = nn.Linear(2, 2, bias=False, dtype=dtype)
    unit = 2.0 ** (-149 if dtype == torch.float32 else -1074)
    direction = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(direction * unit)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=dtype),))
    ref = graph.parameter("weight")
    row = ref.axis(0).select([0])
    if coordinates == "row":
        selections, masks = [(row,)], [torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)]
    elif coordinates == "rows":
        selections, masks = [(ref.axis(0).select([0, 1]),)], [torch.ones((2, 2), dtype=torch.bool)]
    elif coordinates == "overlap":
        selections = [(row,), (ref.axis(1).select([0]),)]
        masks = [
            torch.tensor([[1, 1], [0, 0]], dtype=torch.bool),
            torch.tensor([[1, 0], [1, 0]], dtype=torch.bool),
        ]
    else:
        selections = [(ref.axis(0).select([0, 1]).subtract(row.subtract(ref.axis(1).select([1]))),)]
        masks = [torch.tensor([[0, 1], [1, 1]], dtype=torch.bool)]
    groups = tuple(ParameterGroup(graph, s) for s in selections)
    loss = GroupLasso(groups)()
    (gradient,) = torch.autograd.grad(loss, model.weight)
    expected = torch.zeros_like(direction)
    for mask in masks:
        expected[mask] += direction[mask] / direction[mask].norm()
    torch.testing.assert_close(gradient, expected)
    assert torch.isfinite(loss) and loss > 0


@pytest.mark.parametrize("scale,strength", [(1e-38, 1e-8), (1e-30, 1e-15), (1.0, 1e-8)])
@pytest.mark.parametrize("internal", [False, True])
def test_scaled_normal_group_penalties_have_stable_direction_gradients(
    scale, strength, internal, execution_device
):
    model = nn.Linear(2, 2, bias=False)
    direction = torch.tensor([[1.0, 2.0], [3.0, -4.0]])
    with torch.no_grad():
        model.weight.copy_(direction * scale)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0, 1]),))
    loss = GroupLasso((group,), coefficients=(strength if internal else 1.0,))()
    if not internal:
        loss = loss * strength
    (gradient,) = torch.autograd.grad(loss, model.weight)
    torch.testing.assert_close(gradient, direction / direction.norm() * strength, rtol=1e-5, atol=0)


@pytest.mark.parametrize("value", [2e19, -2e19, 2.0**-149])
def test_half_squared_loss_avoids_unnecessary_intermediate_range_loss(value, execution_device):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 1),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    loss = GroupSquaredL2((group,))()
    expected = (model.weight.detach().double().square().sum() / 2).float()
    torch.testing.assert_close(loss, expected)
    (gradient,) = torch.autograd.grad(loss, model.weight)
    torch.testing.assert_close(gradient, model.weight, rtol=0, atol=0)


@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
@pytest.mark.parametrize(
    "value,coefficient",
    [(1e-23, 1e30), (1e-30, 1e60), (1e30, 1e-60), (1e-30, 1e40), (1e30, 1e-50), (1e30, 0.0)],
)
def test_weighted_penalties_preserve_representable_results(
    kind, value, coefficient, execution_device
):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 1),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    loss = kind((group,), coefficients=(coefficient,))()
    reference = model.weight.detach().double().requires_grad_()
    expected = coefficient * (
        reference.square().sum() / 2 if kind is GroupSquaredL2 else reference.abs().sum()
    )
    gradient = torch.autograd.grad(loss, model.weight)[0]
    expected_gradient = torch.autograd.grad(expected, reference)[0].to(model.weight.dtype)
    torch.testing.assert_close(loss, expected.to(loss.dtype), rtol=1e-5, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-5, atol=0)


@pytest.mark.parametrize("coefficient", [1.0, 0.3])
@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
def test_l2_penalties_support_first_derivatives_and_reject_double_backward(
    coefficient, kind, execution_device
):
    model = nn.Linear(2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=torch.float64),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0, 1]),))
    loss = kind((group,), coefficients=(coefficient,))()
    gradient = torch.autograd.grad(loss, model.weight, create_graph=True)[0]
    expected = model.weight.detach() * coefficient / (5 if kind is GroupLasso else 1)
    torch.testing.assert_close(gradient, expected)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), model.weight)


@pytest.mark.parametrize(
    "value,coefficient",
    [
        (1e-200, 1e-200),
        (1e-200, 1e-100),
        (1e-200, 1.0),
        (2.0**-1074, 2.0**-1074),
        (2.0**-1074, 0.0),
        (1e-320, 1e-320),
    ],
)
def test_weighted_lasso_preserves_direction_when_loss_underflows(
    value, coefficient, execution_device
):
    model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(value)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=torch.float64),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    loss = GroupLasso((group,), coefficients=(coefficient,))()
    (gradient,) = torch.autograd.grad(loss, model.weight)
    torch.testing.assert_close(
        gradient, torch.full_like(model.weight, coefficient / 2**0.5), rtol=1e-12, atol=0
    )
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    "value,coefficient,count",
    [
        (1e-200, 1e-124, 100),
        (2.0**-1074, 1e300, 2),
        (1e308, 1e-308, 100),
        (1e308, 2.0**-1074, 2),
        (2.0**-1074, 0.5, 100),
    ],
)
def test_weighted_lasso_combines_scale_without_losing_group_range(
    value, coefficient, count, execution_device
):
    model = nn.Linear(count, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(value)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, count, dtype=torch.float64),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    loss = GroupLasso((group,), coefficients=(coefficient,))()
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(count).sqrt() * Decimal(coefficient) * Decimal(value))
    torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=1e-12, atol=0)
    (gradient,) = torch.autograd.grad(loss, model.weight)
    torch.testing.assert_close(
        gradient, torch.full_like(model.weight, coefficient / count**0.5), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "dtype,value,coefficient",
    [
        (torch.float32, 1e-23, 1.0),
        (torch.float64, 1e-162, 1.0),
        (torch.float64, 1e-170, 1e16),
        (torch.float64, 0.0, 0.3),
    ],
)
@pytest.mark.parametrize("individual_groups", [False, True])
def test_squared_group_reduces_before_tiny_squares_are_rounded(
    dtype, value, coefficient, individual_groups, execution_device
):
    model = nn.Linear(100, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(value)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 100, dtype=dtype),))
    ref = graph.parameter("weight")
    selections = (
        [ref.axis(1).select([i]) for i in range(100)]
        if individual_groups
        else [ref.axis(0).select([0])]
    )
    groups = tuple(ParameterGroup(graph, (selection,)) for selection in selections)
    loss = GroupSquaredL2(groups, coefficients=(coefficient,) * len(groups))()
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(model.weight[0, 0].item()) ** 2 * Decimal(coefficient) * 50)
    torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=1e-5, atol=0)
    (gradient,) = torch.autograd.grad(loss, model.weight)
    torch.testing.assert_close(gradient, model.weight * coefficient, rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    "dtype,big,small",
    [
        (torch.float64, 1e200, 1e-200),
        (torch.float64, 1e160, 1e-160),
        (torch.float64, 1e50, 1e-50),
        (torch.float32, 1e20, 1e-20),
    ],
)
def test_group_gradient_preserves_rows_with_underflowing_relative_magnitude(
    dtype, big, small, execution_device
):
    model = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[big, 0], [small, small]]))
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=dtype),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0, 1]),))
    loss = GroupLasso((group,))()
    (gradient,) = torch.autograd.grad(loss, model.weight, grad_outputs=loss.new_tensor(big))
    torch.testing.assert_close(gradient, model.weight.detach(), rtol=1e-5, atol=0)


@pytest.mark.parametrize(
    "weights,coefficient,upstream",
    [
        ([1.0, 100.0], 2.0**-1074, 1e308),
        ([1e300, 1e-300], 1.0, 1e300),
        ([1e-300, 1e-300], 1e300, 1e-300),
        ([1.0, 100.0], 0.0, 1e308),
    ],
)
@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
def test_weighted_gradients_combine_all_factors_before_rounding(
    weights, coefficient, upstream, kind, execution_device
):
    # Quadratic loss for 1e300 is unrepresentable; the norm case remains finite.
    if kind is GroupSquaredL2 and weights[0] == 1e300:
        coefficient = 1e-300
    model = nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([weights]))
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 2, dtype=torch.float64),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0]),))
    loss = kind((group,), coefficients=(coefficient,))()
    (gradient,) = torch.autograd.grad(loss, model.weight, grad_outputs=loss.new_tensor(upstream))
    with localcontext() as context:
        context.prec = 80
        norm = sum(Decimal(w) ** 2 for w in weights).sqrt() if kind is GroupLasso else Decimal(1)
        expected = [
            float(Decimal(w) * Decimal(coefficient) * Decimal(upstream) / norm) for w in weights
        ]
    torch.testing.assert_close(gradient, model.weight.new_tensor([expected]), rtol=1e-12, atol=0)


@pytest.mark.parametrize("kind", [GroupLasso, GroupSquaredL2])
def test_l2_range_combinations_against_decimal_objective(kind, execution_device):
    rng = random.Random(711)
    exponents = (-1073, -1000, -700, -300, -10, 1, 10, 300, 700, 1000)
    model = nn.Linear(3, 2, bias=False, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 3, dtype=torch.float64),))
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0, 1]),))
    for case in range(60):
        weights = [
            math.ldexp(rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1), rng.choice(exponents))
            for _ in range(6)
        ]
        coefficient = 1.0 if case % 3 == 0 else math.ldexp(1.0, rng.choice(exponents))
        upstream = math.ldexp(1.0, rng.choice(exponents[1:]))
        with torch.no_grad():
            model.weight.copy_(model.weight.new_tensor(weights).reshape(2, 3))
        with localcontext() as context:
            context.prec = 100
            square = sum(Decimal(w) ** 2 for w in weights)
            norm = square.sqrt()
            expected = float(Decimal(coefficient) * (norm if kind is GroupLasso else square / 2))
            derivatives = [
                float(
                    Decimal(w)
                    * Decimal(coefficient)
                    * Decimal(upstream)
                    / (norm if kind is GroupLasso else 1)
                )
                for w in weights
            ]
        regularizer = kind((group,), coefficients=(coefficient,))
        if not math.isfinite(expected):
            with pytest.raises(ValueError, match="finite"):
                regularizer()
            continue
        loss = regularizer()
        (gradient,) = torch.autograd.grad(
            loss, model.weight, grad_outputs=loss.new_tensor(upstream)
        )
        # Permit one float64 subnormal ULP, not a broad absolute tolerance that
        # would hide premature underflow of the final loss or first derivative.
        torch.testing.assert_close(loss, loss.new_tensor(expected), rtol=1e-12, atol=2.0**-1074)
        torch.testing.assert_close(
            gradient,
            model.weight.new_tensor(derivatives).reshape(2, 3),
            rtol=1e-12,
            atol=2.0**-1074,
        )


@pytest.mark.parametrize(
    "kind,scale",
    [
        (GroupLasso, 1.0),
        (GroupSquaredL2, 1.0),
        (GroupLasso, 2.0**-149),
        (GroupSquaredL2, 1e-23),
    ],
)
def test_mixed_group_reductions_share_bindings_and_count_each_region_once(
    kind, scale, execution_device, monkeypatch
):
    class MixedGroups(nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3) * scale
            self.a_regular = nn.Parameter(values.clone())
            self.z_irregular = nn.Parameter(-values.clone())
            self.scalar = nn.Parameter(torch.tensor(2.0 * scale))

        def forward(self, x):
            return x @ self.a_regular + x @ self.z_irregular + self.scalar

    model = MixedGroups()
    graph = DependencyGraph.build(model, args=(torch.zeros(1, 3),))
    regular = graph.parameter("a_regular").axis(0).select([0])
    irregular_ref = graph.parameter("z_irregular")
    irregular = irregular_ref.axis(0).select([0]).subtract(irregular_ref.axis(1).select([1]))
    # The first selection qualifies for batching, but the next makes the whole
    # group irregular. Its regular prefix must not also enter the batch.
    mixed = ParameterGroup(graph, (regular, irregular))
    row = ParameterGroup(graph, (regular,))
    scalar = ParameterGroup(graph, (Selection(graph.parameter("scalar"), (Region(()),)),))
    regularizer = kind((mixed, row, mixed, scalar))
    bindings = Mock(wraps=graph.tensor_bindings)
    monkeypatch.setattr(graph, "tensor_bindings", bindings)

    for step in range(2):
        if step:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.mul_(2)
        reference = {
            name: parameter.detach().double().requires_grad_()
            for name, parameter in model.named_parameters()
        }
        vectors = (
            torch.cat((reference["a_regular"][0], reference["z_irregular"][0, [0, 2]])),
            reference["a_regular"][0],
            reference["scalar"].reshape(1),
        )
        if kind is GroupLasso:
            expected = sum(vector.square().sum().sqrt() for vector in vectors)
        else:
            expected = sum(vector.square().sum() / 2 for vector in vectors)
        expected_gradients = torch.autograd.grad(expected, tuple(reference.values()))
        bindings.reset_mock()
        loss = regularizer()
        assert bindings.call_count == 1
        torch.testing.assert_close(loss, expected.float(), rtol=1e-5, atol=2.0**-149)
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        for actual, expected_gradient in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected_gradient.float(), rtol=1e-5, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())

    # Reusing this call's bindings must not skip value validation on later calls.
    with torch.no_grad():
        model.z_irregular[0, 0] = float("nan")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    bindings.reset_mock()
    with pytest.raises(ValueError, match="finite"):
        regularizer()
    assert bindings.call_count == 1
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], equal_nan=True)
        assert parameter.grad is None
