"""Independent OSSCAR reconstruction, search and physical model lifecycle checks."""

import copy
import itertools
import json
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import imagenet_models
import osscar_pruning as workflow
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.vision_transformer import VisionTransformer

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Granularity,
    ParameterBudget,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def problem(device, *, groups=6, group_size=1, seed=2):
    generator = torch.Generator(device=device).manual_seed(seed)
    features = torch.randn(30, groups * group_size, generator=generator, dtype=torch.float64)
    original = torch.randn(groups * group_size, 3, generator=generator, dtype=torch.float64)
    targets = features @ original + torch.randn(30, 3, generator=generator, dtype=torch.float64)
    damping = 0.03
    augmented_x = torch.cat(
        (features, damping**0.5 * torch.eye(features.shape[1], dtype=torch.float64))
    )
    augmented_y = torch.cat((targets, damping**0.5 * original))
    return augmented_x, augmented_y, augmented_x.T @ augmented_x, augmented_x.T @ augmented_y


def coordinates(retained, group_size):
    return [
        position
        for group in retained
        for position in range(group * group_size, (group + 1) * group_size)
    ]


def reference_fit(features, targets, retained, group_size):
    chosen = features[:, coordinates(retained, group_size)]
    fitted = torch.linalg.lstsq(chosen, targets).solution
    objective = 0.5 * (chosen @ fitted - targets).square().sum()
    return fitted, objective


@pytest.mark.parametrize("group_size", [1, 3])
def test_fixed_support_matches_independent_augmented_lstsq(group_size, execution_device):
    features, targets, hessian, cross = problem(execution_device, group_size=group_size)
    retained = (0, 2, 5)
    fitted, inverse = workflow.solve_support(hessian, cross, retained, group_size)
    expected, _ = reference_fit(features, targets, retained, group_size)
    torch.testing.assert_close(fitted, expected)
    indices = coordinates(retained, group_size)
    torch.testing.assert_close(
        hessian[indices][:, indices] @ inverse, torch.eye(len(indices), dtype=torch.float64)
    )


@pytest.mark.parametrize("group_size", [1, 3])
def test_delete_costs_and_schur_downdates_match_full_refits(group_size, execution_device):
    features, targets, hessian, cross = problem(execution_device, group_size=group_size)
    retained = (0, 1, 2, 4, 5)
    fitted, inverse = workflow.solve_support(hessian, cross, retained, group_size)
    _, before = reference_fit(features, targets, retained, group_size)
    costs = workflow.deletion_costs(fitted, inverse, group_size)
    for position, group in enumerate(retained):
        remaining = tuple(item for item in retained if item != group)
        _, after = reference_fit(features, targets, remaining, group_size)
        torch.testing.assert_close(costs[position], after - before)
    for positions in ((1,), (0, 3)):
        remaining, updated, updated_inverse = workflow.remove_groups(
            retained, fitted, inverse, positions, group_size
        )
        expected_groups = tuple(group for i, group in enumerate(retained) if i not in positions)
        assert remaining == expected_groups
        reference, _ = reference_fit(features, targets, remaining, group_size)
        torch.testing.assert_close(updated, reference)
        indices = coordinates(remaining, group_size)
        torch.testing.assert_close(
            hessian[indices][:, indices] @ updated_inverse,
            torch.eye(len(indices), dtype=torch.float64),
        )


@pytest.mark.parametrize("group_size", [1, 3])
def test_restoration_gains_match_independent_full_refits(group_size, execution_device):
    features, targets, hessian, cross = problem(execution_device, group_size=group_size)
    retained = (1, 4)
    fitted, inverse = workflow.solve_support(hessian, cross, retained, group_size)
    absent, gains = workflow.restoration_gains(
        hessian, cross, retained, fitted, inverse, group_size
    )
    assert absent == (0, 2, 3, 5)
    _, before = reference_fit(features, targets, retained, group_size)
    for index, group in enumerate(absent):
        _, after = reference_fit(features, targets, tuple(sorted((*retained, group))), group_size)
        torch.testing.assert_close(gains[index], before - after)


def test_local_swaps_improve_a_known_deletion_only_solution(execution_device):
    # CPU-generated fixture has a strict improving swap, independent of device RNG.
    generator = torch.Generator(device="cpu").manual_seed(2)
    features = torch.randn(10, 6, generator=generator, dtype=torch.float64, device="cpu").to(
        execution_device
    )
    original = torch.randn(6, 2, generator=generator, dtype=torch.float64, device="cpu").to(
        execution_device
    )
    hessian = features.T @ features + 0.01 * torch.eye(6, dtype=torch.float64)
    cross = hessian @ original
    deletion = workflow.solve_osscar(hessian, cross, 6, 3, prune_batch=1, swap_steps=0)
    swapped = workflow.solve_osscar(hessian, cross, 6, 3, prune_batch=1, swap_steps=5)
    assert deletion.retained == (0, 3, 4)
    assert swapped.retained == (2, 3, 4)
    assert swapped.swap_attempts > 0
    assert len(swapped.objectives) > 1
    assert swapped.objectives[-1] < deletion.objectives[-1] - 0.1
    assert all(after < before for before, after in itertools.pairwise(swapped.objectives))
    reference = torch.linalg.solve(
        hessian[list(swapped.retained)][:, list(swapped.retained)], cross[list(swapped.retained)]
    )
    torch.testing.assert_close(swapped.coefficients, reference)


@pytest.mark.parametrize("keep", [1, 3, 4])
def test_support_does_not_depend_on_first_column_or_signed_weight_sum(keep, execution_device):
    original = torch.tensor(
        [[0.0, 1.0, -1.0], [0.0, 2.0, -2.0], [0.0, 3.0, -3.0], [0.0, 4.0, -4.0]],
        dtype=torch.float64,
    )
    result = workflow.solve_osscar(
        torch.eye(4, dtype=torch.float64), original, 4, keep, prune_batch=2, swap_steps=3
    )
    assert result.retained == tuple(range(4 - keep, 4))
    torch.testing.assert_close(result.coefficients, original[list(result.retained)])
    assert len(result.retained) == keep


def test_noop_and_singular_systems(execution_device):
    _, _, hessian, cross = problem(execution_device)
    result = workflow.solve_osscar(hessian, cross, 6, 6)
    assert result.retained == tuple(range(6))
    torch.testing.assert_close(result.coefficients, torch.linalg.solve(hessian, cross))
    with pytest.raises((ValueError, RuntimeError)):
        workflow.solve_osscar(torch.zeros(3, 3), torch.ones(3, 2), 3, 2)
    with pytest.raises(ValueError):
        workflow.solve_osscar(torch.empty(0, 0), torch.empty(0, 2), 3, 2)
    for keep in (0, 7):
        with pytest.raises(ValueError):
            workflow.solve_osscar(hessian, cross, 6, keep)


@pytest.mark.parametrize("group_size,prune_batch", [(1, 1), (1, 3), (2, 2)])
def test_search_retains_requested_groups_and_refits_their_actual_joint_objective(
    group_size, prune_batch, execution_device
):
    features, targets, hessian, cross = problem(execution_device, group_size=group_size)
    result = workflow.solve_osscar(hessian, cross, 6, 3, prune_batch=prune_batch, swap_steps=4)
    assert len(result.retained) == len(set(result.retained)) == 3
    assert result.retained == tuple(sorted(result.retained))
    reference, residual_error = reference_fit(features, targets, result.retained, group_size)
    torch.testing.assert_close(result.coefficients, reference)
    expected_objective = residual_error - 0.5 * targets.square().sum()
    assert result.objectives[-1] == pytest.approx(expected_objective.item())


def test_equal_deletion_costs_have_deterministic_group_order(execution_device):
    hessian = torch.eye(4, dtype=torch.float64)
    cross = torch.ones(4, 2, dtype=torch.float64)
    outcomes = [
        workflow.solve_osscar(hessian, cross, 4, 2, prune_batch=1, swap_steps=3).retained
        for _ in range(3)
    ]
    assert outcomes == [(2, 3)] * 3


def test_moments_are_sample_weighted_and_damped_toward_original(execution_device):
    features, targets, _, _ = problem(execution_device)
    original = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 5
    moments = workflow.ReconstructionMoments()
    for start, end in ((0, 2), (2, 15), (15, len(features))):
        moments.update(features[start:end], targets[start:end])
    hessian, cross = moments.system(original, 0.1)
    empirical = features.T @ features / len(features)
    ridge = 0.1 * empirical.diagonal().mean()
    torch.testing.assert_close(hessian, empirical + ridge * torch.eye(6))
    torch.testing.assert_close(cross, features.T @ targets / len(features) + ridge * original)


def test_dead_features_become_a_finite_solvable_system(execution_device):
    moments = workflow.ReconstructionMoments()
    moments.update(torch.zeros(8, 4, dtype=torch.float64), torch.zeros(8, 2, dtype=torch.float64))
    original = torch.randn(4, 2, dtype=torch.float64)
    hessian, cross = moments.system(original, 0.01)
    assert torch.isfinite(hessian).all()
    assert (hessian.diagonal() > 0).all()
    torch.testing.assert_close(torch.linalg.solve(hessian, cross), original)


def test_invalid_moment_batch_does_not_corrupt_previous_statistics(execution_device):
    moments = workflow.ReconstructionMoments()
    moments.update(torch.randn(4, 3), torch.randn(4, 2))
    before = moments.gram.clone(), moments.cross.clone(), moments.count
    for features, targets in (
        (torch.empty(0, 3), torch.empty(0, 2)),
        (torch.full((2, 3), float("nan")), torch.ones(2, 2)),
        (torch.ones(2, 4), torch.ones(2, 2)),
    ):
        with pytest.raises(ValueError):
            moments.update(features, targets)
        torch.testing.assert_close(moments.gram, before[0])
        torch.testing.assert_close(moments.cross, before[1])
        assert moments.count == before[2]


@pytest.mark.parametrize("kind", ["linear", "conv", "same", "valid"])
def test_sampled_feature_rows_reconstruct_actual_layer_outputs(kind, execution_device):
    if kind == "linear":
        layer = nn.Linear(4, 3).double()
        inputs = torch.randn(2, 5, 4, dtype=torch.float64)
        expected = layer(inputs).reshape(-1, 3)
    else:
        layer = (
            nn.Conv2d(2, 3, (2, 3), stride=(2, 1), padding=(1, 2), dilation=(1, 2))
            if kind == "conv"
            else nn.Conv2d(2, 3, (2, 3), padding=kind, dilation=(3, 2))
        ).double()
        inputs = torch.randn(2, 2, 7, 8, dtype=torch.float64)
        expected = layer(inputs).movedim(1, -1).reshape(-1, 3)
    rows = torch.tensor([0, 2, len(expected) - 1], dtype=torch.long)
    design = workflow.feature_rows(layer, inputs, rows)
    predicted = design @ layer.weight.flatten(1).T + layer.bias
    torch.testing.assert_close(predicted, expected[rows])


@pytest.mark.parametrize("fail", [False, True])
def test_calibration_uses_dense_targets_and_current_inputs_with_cleanup(fail, execution_device):
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2)).double()
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        model[0].weight.mul_(0.7)
        model[2].weight.add_(0.1)
        model[2].bias.add_(0.3)
    model.train()
    model[0].eval()
    teacher.train()
    teacher[2].eval()
    inputs = torch.randn(6, 3, dtype=torch.float64)
    batches = [(inputs[:2], torch.zeros(2)), (inputs[2:], torch.zeros(4))]
    modes = tuple(module.training for parent in (model, teacher) for module in parent.modules())
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = tuple(parameter.grad for parameter in model.parameters())
    if fail:
        inputs[-1, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            workflow.collect_reconstruction(model, teacher, "2", batches, execution_device, 0, 100)
    else:
        moments = workflow.collect_reconstruction(
            model, teacher, "2", batches, execution_device, 0, 100
        )
        current_inputs = F.gelu(F.linear(inputs, model[0].weight, model[0].bias))
        dense_inputs = F.gelu(F.linear(inputs, teacher[0].weight, teacher[0].bias))
        targets = F.linear(dense_inputs, teacher[2].weight, teacher[2].bias) - model[2].bias
        expected_h = current_inputs.T @ current_inputs / len(inputs)
        expected_g = current_inputs.T @ targets / len(inputs)
        hessian, cross = moments.system(model[2].weight.T, 0.01)
        ridge = expected_h.diagonal().mean() * 0.01
        torch.testing.assert_close(hessian, expected_h + ridge * torch.eye(4))
        torch.testing.assert_close(cross, expected_g + ridge * model[2].weight.T)
    assert (
        tuple(module.training for parent in (model, teacher) for module in parent.modules())
        == modes
    )
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for parent in (model, teacher)
        for module in parent.modules()
    )
    assert all(
        parameter.grad is old for parameter, old in zip(model.parameters(), gradients, strict=True)
    )


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, inputs):
        return inputs + self.conv2(F.relu(self.bn1(self.conv1(inputs))))


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(3)
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, inputs):
        return inputs + self.fc2(F.gelu(self.fc1(self.norm(inputs))))


class KeywordConsumer(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.first = nn.Conv2d(3, 6, 1) if convolution else nn.Linear(3, 6)
        self.last = nn.Conv2d(6, 3, 1) if convolution else nn.Linear(6, 3)

    def forward(self, inputs):
        return self.last(input=F.gelu(self.first(input=inputs)))


@pytest.mark.parametrize("convolution", [False, True])
def test_discovered_keyword_consumer_calibrates_and_physically_reconstructs(
    convolution, execution_device
):
    model = KeywordConsumer(convolution).double().eval()
    teacher = copy.deepcopy(model)
    inputs = torch.randn((3, 3, 4, 4) if convolution else (3, 4, 3), dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(inputs,))
    pairs, _ = workflow.discover_reconstruction_pairs(graph)
    assert pairs == {"first": "last"}
    moments = workflow.collect_reconstruction(
        model, teacher, "last", [(inputs, torch.zeros(3))], execution_device, 0, 100
    )
    hessian, cross = moments.system(model.last.weight.flatten(1).T, 0.01)
    solution = workflow.solve_osscar(hessian, cross, 6, 4)
    workflow.apply_reconstruction(Pruner(model, graph=graph), "first", "last", solution)
    model(inputs).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["cnn", "mlp", "pooled"])
def test_pair_discovery_uses_proven_graph_paths_not_model_families(kind, execution_device):
    if kind == "cnn":
        model, inputs = ResidualBlock().eval(), torch.randn(2, 3, 8, 8)
        expected = {"conv1": "conv2"}
    elif kind == "mlp":
        model, inputs = ResidualMLP().eval(), torch.randn(2, 5, 3)
        expected = {"fc1": "fc2"}
    else:
        model = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 4, 3, padding=1),
        ).eval()
        inputs, expected = torch.randn(2, 3, 8, 8), {"0": "4"}
    graph = DependencyGraph.build(model, args=(inputs,))
    pairs, exclusions = workflow.discover_reconstruction_pairs(graph)
    assert pairs == expected
    assert set(pairs).isdisjoint(exclusions)


@pytest.mark.parametrize("middle", [nn.LayerNorm(6), nn.Softmax(dim=-1), nn.Flatten(0, 1)])
def test_pair_discovery_reports_mixing_and_unproved_reindexing(middle, execution_device):
    model = (
        nn.Sequential(nn.Linear(3, 6), copy.deepcopy(middle), nn.Linear(6, 2))
        .to(execution_device)
        .eval()
    )
    graph = DependencyGraph.build(model, args=(torch.randn(2, 5, 3),))
    pairs, exclusions = workflow.discover_reconstruction_pairs(graph)
    assert not pairs
    assert "channel-independent" in exclusions["0"]


class SharedOrBranched(nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.first = nn.Linear(3, 6)
        self.other = nn.Linear(3, 6)
        self.last = nn.Linear(6, 2)
        if case == "shared_parameter":
            self.other.weight = self.first.weight

    def forward(self, inputs):
        hidden = self.first(inputs)
        if self.case == "branch":
            return self.last(F.gelu(hidden)), hidden
        if self.case == "shared_call":
            return self.last(F.gelu(hidden)) + self.last(F.gelu(self.other(inputs)))
        if self.case == "shared_parameter":
            return self.last(F.gelu(hidden)), self.other(inputs)
        # A functional use is also a real shared-parameter consumer.
        return self.last(F.gelu(hidden)), F.linear(inputs, self.first.weight)


@pytest.mark.parametrize("case", ["branch", "shared_call", "shared_parameter", "functional_weight"])
def test_pair_discovery_excludes_unsafe_branches_and_shared_uses(case, execution_device):
    model = SharedOrBranched(case).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pairs, exclusions = workflow.discover_reconstruction_pairs(graph)
    assert "first" not in pairs
    assert "first" in exclusions


def test_discovery_allows_a_single_call_with_registered_module_alias(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).eval()
    model.register_parameter("weight_alias", model[0].weight)
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),))
    pairs, _ = workflow.discover_reconstruction_pairs(graph)
    assert pairs == {"0": "2"}


@pytest.mark.parametrize("track_running_stats", [False, True])
def test_discovery_accepts_evaluation_batchnorm_on_the_actual_channel_axis(
    track_running_stats, execution_device
):
    model = nn.Sequential(
        nn.Linear(3, 6), nn.BatchNorm1d(6, track_running_stats=track_running_stats), nn.Linear(6, 2)
    ).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(4, 3),))
    pairs, _ = workflow.discover_reconstruction_pairs(graph)
    assert pairs == {"0": "2"}


@pytest.mark.parametrize("track_running_stats", [False, True])
def test_discovery_distinguishes_fixed_batchnorm_from_removed_statistics_axis(
    track_running_stats, execution_device
):
    model = nn.Sequential(
        nn.Linear(3, 6), nn.BatchNorm1d(5, track_running_stats=track_running_stats), nn.Linear(6, 2)
    ).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(4, 5, 3),))
    pairs, exclusions = workflow.discover_reconstruction_pairs(graph)
    if track_running_stats:
        assert pairs == {"0": "2"}
    else:
        assert not pairs and "0" in exclusions


class SharedBatchNormState(nn.Module):
    def __init__(self, channel_state):
        super().__init__()
        self.left = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2))
        self.right = copy.deepcopy(self.left)
        if channel_state:
            self.right[1].running_mean = self.left[1].running_mean
            self.right[1].running_var = self.left[1].running_var
        else:
            self.right[1].num_batches_tracked = self.left[1].num_batches_tracked

    def forward(self, inputs):
        return self.left(inputs) + self.right(inputs)


@pytest.mark.parametrize("channel_state", [False, True])
def test_discovery_distinguishes_shared_channel_buffers_from_scalar_counters(
    channel_state, execution_device
):
    model = SharedBatchNormState(channel_state).eval()
    inputs = torch.randn(4, 3)
    graph = DependencyGraph.build(model, args=(inputs,))
    pairs, exclusions = workflow.discover_reconstruction_pairs(graph)
    if channel_state:
        assert not pairs
        assert "shared" in exclusions["left.0"]
        assert "shared" in exclusions["right.0"]
    else:
        assert pairs == {"left.0": "left.3", "right.0": "right.3"}
        before = model.right[0].weight.detach().clone()
        pruner = Pruner(model, graph=graph)
        pruner.apply(pruner.plan_remove([graph.parameter("left.0.weight").axis(0).select([0])]))
        assert model.left[0].out_features == 3 and model.right[0].out_features == 4
        torch.testing.assert_close(model.right[0].weight, before)
        assert model(inputs).shape == (4, 2)


def test_overlapping_pairs_reconstruct_sequentially_and_restore_checkpoint(
    tmp_path, execution_device
):
    model = (
        nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 4), nn.SiLU(), nn.Linear(4, 2))
        .double()
        .eval()
    )
    teacher = copy.deepcopy(model)
    inputs = torch.randn(5, 3, dtype=torch.float64)
    pairs, _ = workflow.discover_reconstruction_pairs(DependencyGraph.build(model, args=(inputs,)))
    assert pairs == {"0": "2", "2": "4"}
    for producer, consumer in pairs.items():
        moments = workflow.collect_reconstruction(
            model, teacher, consumer, [(inputs, torch.zeros(5))], execution_device, 0, 100
        )
        hessian, cross = moments.system(teacher.get_submodule(consumer).weight.T, 0.01)
        old_width = model.get_submodule(producer).out_features
        solution = workflow.solve_osscar(hessian, cross, old_width, old_width - 2)
        graph = DependencyGraph.build(model, args=(inputs,))
        workflow.apply_reconstruction(Pruner(model, graph=graph), producer, consumer, solution)
    assert model[0].out_features == model[2].in_features == 4
    assert model[2].out_features == model[4].in_features == 2
    assert model[4].out_features == 2
    expected = model(inputs)
    expected.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(teacher, tmp_path / "model.pt", map_location=execution_device)
    torch.testing.assert_close(restored(inputs), expected)


def test_width_allocation_noop_and_unreachable_target_are_read_only(execution_device):
    model = ResidualMLP().eval()
    inputs = torch.randn(2, 5, 3)
    graph = DependencyGraph.build(model, args=(inputs,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path={"fc1": 2}))
    state = {path: tensor.clone() for path, tensor in model.state_dict().items()}
    bindings = tuple(model.parameters())
    pairs = {"fc1": "fc2"}
    assert workflow.allocate_widths(pruner, pairs, ParameterBudget.from_ratio(model, 0), 2) == {
        "fc1": 6
    }
    with pytest.raises(PlanningError, match="unreachable"):
        workflow.allocate_widths(pruner, pairs, ParameterBudget(1), 2)
    assert all(
        parameter is original
        for parameter, original in zip(model.parameters(), bindings, strict=True)
    )
    for path, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[path])
    graph.validate()


@pytest.mark.parametrize("invalid", ["unsorted", "duplicate", "out_of_range", "nonfinite"])
def test_reconstruction_rejects_invalid_solution_before_mutating_model(invalid, execution_device):
    model = ResidualMLP().eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 5, 3),))
    pruner = Pruner(model, graph=graph)
    retained = {
        "unsorted": (2, 1, 3, 4),
        "duplicate": (1, 1, 3, 4),
        "out_of_range": (1, 2, 3, 6),
        "nonfinite": (1, 2, 3, 4),
    }[invalid]
    coefficients = torch.randn(4, 3, dtype=torch.float64)
    if invalid == "nonfinite":
        coefficients[0, 0] = float("inf")
    solution = workflow.ReconstructionSolution(retained, coefficients, (0.0,), 0)
    before = {path: tensor.clone() for path, tensor in model.state_dict().items()}
    bindings = tuple(model.parameters())
    with pytest.raises(ValueError):
        workflow.apply_reconstruction(pruner, "fc1", "fc2", solution)
    assert all(
        parameter is old for parameter, old in zip(model.parameters(), bindings, strict=True)
    )
    for path, value in model.state_dict().items():
        torch.testing.assert_close(value, before[path])
    graph.validate()


@pytest.mark.parametrize("kind", ["cnn", "vit_mlp"])
def test_physical_pruning_reconstruction_and_checkpoint_match_independent_reference(
    kind, tmp_path, execution_device
):
    model = (ResidualBlock() if kind == "cnn" else ResidualMLP()).double().eval()
    original = copy.deepcopy(model)
    producer, consumer = ("conv1", "conv2") if kind == "cnn" else ("fc1", "fc2")
    inputs = torch.randn((4, 3, 5, 5) if kind == "cnn" else (4, 5, 3), dtype=torch.float64)
    moments = workflow.collect_reconstruction(
        model, original, consumer, [(inputs, torch.zeros(4))], execution_device, 0, 1000
    )
    layer = model.get_submodule(consumer)
    original_coefficients = layer.weight.flatten(1).T.detach()
    hessian, cross = moments.system(original_coefficients, 0.01)
    result = workflow.solve_osscar(hessian, cross, 6, 4, prune_batch=2, swap_steps=3)
    graph = DependencyGraph.build(model, args=(inputs,))
    pruner = Pruner(model, graph=graph)
    workflow.apply_reconstruction(pruner, producer, consumer, result)
    if kind == "cnn":
        hidden = F.relu(original.bn1(original.conv1(inputs)))[:, result.retained]
        expected = inputs + F.conv2d(hidden, layer.weight, original.conv2.bias, padding=1)
    else:
        hidden = F.gelu(original.fc1(original.norm(inputs)))[..., result.retained]
        expected = inputs + F.linear(hidden, layer.weight, original.fc2.bias)
    torch.testing.assert_close(model(inputs), expected)
    model(inputs).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(original, tmp_path / "model.pt", map_location=execution_device)
    torch.testing.assert_close(restored(inputs), expected)


def miniature_model(name):
    if name in ("resnet18", "resnet50"):
        block = BasicBlock if name == "resnet18" else Bottleneck
        width = 4 * block.expansion
        model = ResNet(block, [1, 1, 1, 1])
        model.conv1 = nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False)
        model.bn1 = nn.BatchNorm2d(width)
        for stage in range(1, 5):
            setattr(model, f"layer{stage}", nn.Sequential(block(width, 4)))
        model.fc = nn.Linear(width, 1000)
        return model
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=2,
        num_heads=2,
        hidden_dim=8,
        mlp_dim=16,
        num_classes=1000,
    )
    nn.init.normal_(model.heads.head.weight, std=0.05)
    return model


@pytest.mark.parametrize(
    "name,ratio,epochs",
    [("resnet18", 0.001, 1), ("resnet50", 0.001, 1), ("vit_b_32", 0.001, 1), ("resnet18", 0, 0)],
)
def test_entry_sequentially_reconstructs_to_parameter_target_and_saves_refitted_model(
    name, ratio, epochs, monkeypatch, tmp_path, execution_device
):
    weights = imagenet_models.MODELS[name][1]
    requested = []
    built = []
    original_states = []

    def builder(*, weights):
        requested.append(weights)
        model = miniature_model(name)
        built.append(model)
        original_states.append(
            {path: tensor.clone() for path, tensor in model.state_dict().items()}
        )
        return model

    def data(selected_weights, data_dir, **options):
        assert selected_weights is weights and options["need_train"]
        dataset = TensorDataset(
            torch.randn(2, 3, 224, 224, device="cpu"),
            torch.tensor([0, 1], device="cpu"),
        )
        return dataset, dataset, {"test_fixture": True}

    monkeypatch.setitem(imagenet_models.MODELS, name, (builder, weights))
    monkeypatch.setattr(workflow, "load_images", data)
    monkeypatch.setattr(workflow, "measure_model", lambda *args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "osscar_pruning.py",
            "--model",
            name,
            "--device",
            execution_device,
            "--train_batch_size",
            "2",
            "--val_batch_size",
            "2",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--calibration_batches",
            "1",
            "--calibration_rows",
            "32",
            "--granularity",
            "2",
            "--pruning_ratio",
            str(ratio),
            "--finetune_epochs",
            str(epochs),
            "--output",
            str(tmp_path),
        ],
    )
    with torch.device("cpu"):
        workflow.main()
    assert requested == [weights, None]
    report = json.loads((tmp_path / "metrics.json").read_text())
    stages = ["pretrained", "pruned", "finetuned"] if epochs else ["pretrained", "pruned"]
    assert [stage["stage"] for stage in report["stages"]] == stages
    stage = report["stages"][1]
    assert stage["after_params"] <= stage["max_params"] <= stage["before_params"]
    assert stage["target_met"]
    if ratio:
        assert len(report["reconstruction"]) >= 2
        assert all(item["observations"] == 32 for item in report["reconstruction"])
    else:
        assert not report["reconstruction"]
        assert stage["after_params"] == stage["before_params"]
        for path, value in built[0].state_dict().items():
            torch.testing.assert_close(value.cpu(), original_states[0][path], rtol=0, atol=0)
    assert (tmp_path / "model.pt").is_file()
    saved = torch.load(tmp_path / "training.pt", map_location="cpu", weights_only=True)
    assert saved["algorithm"]["method"] == "osscar"
    assert bool(saved["algorithm"]["reconstruction"]) is bool(ratio)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--calibration_rows", "0"),
        ("--calibration_batches", "-1"),
        ("--damping", "nan"),
        ("--prune_batch", "0"),
        ("--swap_steps", "-1"),
    ],
)
def test_cli_rejects_invalid_reconstruction_settings(option, value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["osscar_pruning.py", "--device", "cpu", option, value])
    with pytest.raises(SystemExit) as error:
        workflow.parse_args()
    assert error.value.code == 2


def test_persistent_worker_calibration_replays_image_order_across_consumers(monkeypatch, tmp_path):
    """Observe actual forwards, including the first worker startup's RNG use."""
    weights = imagenet_models.MODELS["resnet18"][1]

    def builder(*, weights):
        return miniature_model("resnet18")

    images = (
        torch.arange(8, dtype=torch.float32)[:, None, None, None].expand(8, 3, 224, 224).clone()
    )
    train = TensorDataset(images, torch.arange(8))
    validation = TensorDataset(images[:2], torch.arange(2))
    monkeypatch.setitem(imagenet_models.MODELS, "resnet18", (builder, weights))
    monkeypatch.setattr(workflow, "load_images", lambda *args, **kwargs: (train, validation, {}))
    monkeypatch.setattr(workflow, "measure_model", lambda *args: {})
    collect = workflow.collect_reconstruction
    observed = []

    def observe_calibration(model, teacher, path, loader, device, max_batches, max_rows):
        assert loader.persistent_workers and loader.num_workers == 1
        order = []

        def observe_images(module, args):
            order.extend(args[0][:, 0, 0, 0].tolist())

        handle = model.register_forward_pre_hook(observe_images)
        try:
            result = collect(model, teacher, path, loader, device, max_batches, max_rows)
        finally:
            handle.remove()
        observed.append(tuple(order))
        return result

    monkeypatch.setattr(workflow, "collect_reconstruction", observe_calibration)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "osscar_pruning.py",
            "--model",
            "resnet18",
            "--device",
            "cpu",
            "--train_batch_size",
            "2",
            "--val_batch_size",
            "2",
            "--train_workers",
            "1",
            "--val_workers",
            "0",
            "--calibration_batches",
            "1",
            "--calibration_rows",
            "16",
            "--granularity",
            "2",
            "--pruning_ratio",
            "0.001",
            "--finetune_epochs",
            "0",
            "--output",
            str(tmp_path),
        ],
    )
    workflow.main()
    assert len(observed) == 4
    assert len(observed[0]) == 2
    assert all(order == observed[0] for order in observed[1:])
