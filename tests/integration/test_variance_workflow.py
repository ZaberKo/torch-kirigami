"""Independent VBP statistics, compensation, model lifecycle and executable workflow."""

import copy
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
import variance_pruning as workflow
from imagenet_models import TraceableViT
from torchvision.models.convnext import CNBlockConfig, ConvNeXt
from torchvision.models.vision_transformer import VisionTransformer

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Granularity,
    Greedy,
    ParameterBudget,
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


@pytest.mark.parametrize("parts", [(7,), (1, 2, 4), (3, 4)])
def test_streaming_moments_match_independent_shifted_batch_reference(parts, execution_device):
    values = torch.arange(7 * 3 * 4, dtype=torch.float64).reshape(7, 3, 4)
    values[2:] += 1000
    stats = workflow.ActivationMoments()
    offset = 0
    for size in parts:
        stats.update(values[offset : offset + size])
        offset += size
    reference = values.reshape(-1, 4)
    assert stats.count == len(reference)
    torch.testing.assert_close(stats.mean, reference.mean(0))
    torch.testing.assert_close(stats.variance(), reference.var(0, correction=1))


def test_moments_reject_invalid_samples_without_replacing_prior_state():
    stats = workflow.ActivationMoments()
    with pytest.raises(ValueError, match="at least two"):
        stats.variance()
    stats.update(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="at least two"):
        stats.variance()
    before = stats.mean.clone(), stats.m2.clone(), stats.count
    for invalid in (torch.empty(0, 2), torch.tensor([[float("nan"), 1.0]])):
        with pytest.raises(ValueError, match="nonempty and finite"):
            stats.update(invalid)
    with pytest.raises(ValueError, match="channel width"):
        stats.update(torch.ones(2, 3))
    torch.testing.assert_close(stats.mean, before[0])
    torch.testing.assert_close(stats.m2, before[1])
    assert stats.count == before[2]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_moments_reduce_represented_values_in_float32(dtype, execution_device):
    values = torch.linspace(-2, 3, 120, device=execution_device).reshape(10, 3, 4).to(dtype)
    values[4:] += 8
    stats = workflow.ActivationMoments()
    for batch in values.split((2, 3, 5)):
        stats.update(batch)
    # Compare with the values actually represented in the source dtype, rather
    # than expecting calibration to recover information lost during quantization.
    reference = values.double().reshape(-1, 4)
    assert stats.mean.dtype == stats.m2.dtype == torch.float64
    torch.testing.assert_close(stats.mean, reference.mean(0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(stats.variance(), reference.var(0), atol=1e-6, rtol=1e-6)


def test_joint_mlp_compensation_matches_sequential_mean_substitution(execution_device):
    model = (
        nn.Sequential(
            nn.Linear(3, 6),
            nn.GELU(),
            nn.Linear(6, 4),
            nn.Linear(4, 5),
            nn.GELU(),
            nn.Linear(5, 2),
        )
        .double()
        .eval()
    )
    original = copy.deepcopy(model)
    inputs = torch.randn(9, 3, dtype=torch.float64)
    pairs = {"0": "2", "3": "5"}
    moments = workflow.collect_moments(
        model, pairs, [(inputs, torch.zeros(9))], execution_device, 0
    )
    graph = DependencyGraph.build(model, args=(inputs,))
    pruner = Pruner(model, graph=graph)
    removed = {"2": [0, 4], "5": [1, 3]}
    plan = pruner.plan_remove(
        [
            graph.parameter(f"{first}.weight").axis(0).select(removed[second])
            for first, second in pairs.items()
        ]
    )
    updates = workflow.prepare_bias_compensation(graph, plan, pairs, moments)
    assert set(updates) == {"2.bias", "5.bias"}
    pruner.apply(plan)
    with torch.no_grad():
        for path, value in updates.items():
            model.get_parameter(path).copy_(value)
        hidden = original[1](original[0](inputs))
        hidden[:, removed["2"]] = moments["2"].mean[removed["2"]]
        hidden = original[4](original[3](original[2](hidden)))
        # Later activations change after the earlier substitution, but the method
        # deliberately retains its initial calibration means for every block.
        hidden[:, removed["5"]] = moments["5"].mean[removed["5"]]
        expected = original[5](hidden)
        torch.testing.assert_close(model(inputs), expected)
    model(inputs).square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


class DenseChain(nn.Module):
    """An arbitrary model with adjacent MLP pairs and a harmless module alias."""

    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(3, 6)
        self.bridge = nn.Linear(6, 5)
        self.finish = nn.Linear(5, 2)
        self.alias = self.expand

    def forward(self, x):
        return self.finish(input=F.silu(self.bridge(input=F.gelu(self.expand(x)))))


def test_generic_discovery_adjacent_pairs_compensate_and_restore(execution_device, tmp_path):
    model = DenseChain().double().eval()
    dense = copy.deepcopy(model)
    inputs = torch.randn(8, 3, dtype=torch.float64)
    graph = DependencyGraph.build(model, args=(inputs,))
    pairs, exclusions = workflow.discover_mlps(graph)
    assert pairs == {"expand": "bridge", "bridge": "finish"}
    assert not exclusions
    moments = workflow.collect_moments(
        model, pairs, [(inputs, torch.zeros(8))], execution_device, 0
    )
    graph.validate()
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove(
        [
            graph.parameter("expand.weight").axis(0).select([0, 4]),
            graph.parameter("bridge.weight").axis(0).select([1, 3]),
        ]
    )
    corrections = workflow.prepare_bias_compensation(graph, plan, pairs, moments)
    assert corrections["bridge.bias"].shape == (3,)
    pruner.apply(plan)
    with torch.no_grad():
        for path, value in corrections.items():
            model.get_parameter(path).copy_(value)
        first = F.gelu(dense.expand(inputs))
        first[:, [0, 4]] = moments["bridge"].mean[[0, 4]]
        second = F.silu(dense.bridge(first))
        second[:, [1, 3]] = moments["finish"].mean[[1, 3]]
        expected = dense.finish(second)
    torch.testing.assert_close(model(inputs), expected)
    assert model.alias is model.expand
    model(inputs).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    save_checkpoint(model, tmp_path / "adjacent.pt")
    restored = load_checkpoint(dense, tmp_path / "adjacent.pt", map_location=execution_device)
    torch.testing.assert_close(restored(inputs), expected)


class DiscoveryCases(nn.Module):
    """One safe MLP and one independently configured unsafe hidden path."""

    def __init__(self, case):
        super().__init__()
        self.case = case
        self.safe = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Dropout(0.2), nn.Linear(6, 2))
        self.first = nn.Linear(3, 6)
        self.second = nn.Linear(6, 2, bias=case != "no_bias")
        self.shared = nn.Linear(3, 6)
        if case == "shared_parameter":
            self.shared.weight = self.first.weight

    def forward(self, x):
        hidden = self.first(x)
        if self.case == "branch":
            hidden = hidden + hidden
        elif self.case == "repeated_call":
            hidden = hidden + self.first(x)
        elif self.case == "shared_parameter":
            hidden = hidden + self.shared(x)
        elif self.case == "permutation":
            hidden = hidden.transpose(0, 1)  # Square sample: same shape is insufficient.
        elif self.case == "functional_dropout":
            hidden = F.dropout(hidden, p=0.2, training=False)
        return self.safe(x) + self.second(F.gelu(hidden))


@pytest.mark.parametrize(
    "case", ["branch", "repeated_call", "shared_parameter", "permutation", "no_bias"]
)
def test_discovery_excludes_unsafe_paths_but_keeps_independent_mlp(case, execution_device):
    model = DiscoveryCases(case).eval()
    inputs = torch.randn(6, 3)
    graph = DependencyGraph.build(model, args=(inputs,))
    pairs, exclusions = workflow.discover_mlps(graph)
    assert pairs == {"safe.0": "safe.3"}
    assert "first" in exclusions
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("safe.0.weight").axis(0).select([0])])
    pruner.apply(plan)
    assert model.safe[0].out_features == 5 and model.first.out_features == 6
    model(inputs).sum().backward()


def test_discovery_accepts_evaluation_functional_dropout(execution_device):
    model = DiscoveryCases("functional_dropout").eval()
    graph = DependencyGraph.build(model, args=(torch.randn(6, 3),))
    pairs, exclusions = workflow.discover_mlps(graph)
    assert pairs == {"safe.0": "safe.3", "first": "second"}
    assert not exclusions


@pytest.mark.parametrize("fail", [False, True])
def test_calibration_observes_consumer_input_and_restores_modes_hooks_gradients(
    fail, execution_device
):
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Dropout(0.8), nn.Linear(4, 2)).double()
    model.train()
    model[0].eval()  # A mixed mode configuration must be restored exactly.
    inputs = torch.randn(5, 3, dtype=torch.float64)
    if fail:
        inputs[-1, 0] = float("nan")
    batches = [(inputs[:2], torch.zeros(2)), (inputs[2:], torch.zeros(3))]
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = tuple(parameter.grad for parameter in model.parameters())
    modes = tuple(module.training for module in model.modules())
    if fail:
        with pytest.raises(ValueError, match="finite"):
            workflow.collect_moments(model, {"0": "3"}, batches, execution_device, 0)
    else:
        result = workflow.collect_moments(model, {"0": "3"}, batches, execution_device, 0)
        reference = F.gelu(F.linear(inputs, model[0].weight, model[0].bias))
        torch.testing.assert_close(result["3"].mean, reference.mean(0))
        torch.testing.assert_close(result["3"].variance(), reference.var(0))
    assert tuple(module.training for module in model.modules()) == modes
    assert all(not module._forward_pre_hooks for module in model.modules())
    assert all(p.grad is g for p, g in zip(model.parameters(), gradients, strict=True))


def test_vbp_compensation_matches_mean_replacement_and_restores_checkpoint(
    tmp_path, execution_device
):
    model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).double().eval()
    with torch.no_grad():
        model[0].weight[0].zero_()
        model[0].bias[0] = 2  # Zero variance but a nonzero contribution to preserve.
    original = copy.deepcopy(model)
    x = torch.randn(7, 3, dtype=torch.float64)
    pairs = {"0": "2"}
    moments = workflow.collect_moments(model, pairs, [(x, torch.zeros(7))], execution_device, 0)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan(
        pruner.discover_candidates(targets=("0",)),
        budget=ParameterBudget(32),
        strategy=Greedy(workflow.ActivationVariance(graph, pairs, moments)),
    )
    removed = list(plan.analysis.selection(graph.parameter("0.weight")).fully_selected_indices(0))
    assert removed == [0]
    before = {path: value.clone() for path, value in model.state_dict().items()}
    updates = workflow.prepare_bias_compensation(graph, plan, pairs, moments)
    for path, value in model.state_dict().items():
        torch.testing.assert_close(value, before[path])
    pruner.apply(plan)
    with torch.no_grad():
        for path, value in updates.items():
            model.get_parameter(path).copy_(value)
    hidden = F.gelu(F.linear(x, original[0].weight, original[0].bias))
    hidden[:, removed] = hidden[:, removed].mean(0)
    expected = F.linear(hidden, original[2].weight, original[2].bias)
    torch.testing.assert_close(model(x), expected)
    # Here the removed activation is constant, so compensation is also exact
    # against the dense model; merely zeroing this channel would be incorrect.
    torch.testing.assert_close(model(x), original(x))
    model(x).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(original, tmp_path / "model.pt", map_location=execution_device)
    torch.testing.assert_close(restored(x), expected)


def test_compensation_rejects_unrelated_output_pruning_before_mutation():
    model = nn.Sequential(nn.Linear(3, 6), nn.GELU(), nn.Linear(6, 2)).double().eval()
    x = torch.randn(4, 3, dtype=torch.float64)
    moments = workflow.collect_moments(model, {"0": "2"}, [(x, torch.zeros(4))], "cpu", 0)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph, preserve_io=False).plan_remove(
        [graph.parameter("2.weight").axis(0).select([0])]
    )
    before = tuple(model.parameters())
    with pytest.raises(ValueError, match="matched hidden-width"):
        workflow.prepare_bias_compensation(graph, plan, {"0": "2"}, moments)
    assert all(p is old for p, old in zip(model.parameters(), before, strict=True))
    graph.validate()


def miniature_model(name):
    if name == "convnext_tiny":
        return ConvNeXt(
            [
                CNBlockConfig(4, 8, 1),
                CNBlockConfig(8, 12, 1),
                CNBlockConfig(12, 16, 1),
                CNBlockConfig(16, None, 1),
            ],
            stochastic_depth_prob=0.1,
            num_classes=1000,
        )
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=1,
        num_heads=2,
        hidden_dim=8,
        mlp_dim=16,
        num_classes=1000,
    )
    with torch.no_grad():
        model.heads.head.weight.normal_(std=0.02)
    return model


@pytest.mark.parametrize("name", ["convnext_tiny", "vit_b_32"])
def test_vbp_real_architecture_calibration_plan_apply_and_compensation(name, execution_device):
    official = miniature_model(name)
    model = (TraceableViT(official) if name.startswith("vit") else official).eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 224, 224)
    graph = DependencyGraph.build(model, args=(x[:1],), operators=workflow.vbp_operators())
    pairs, _ = workflow.discover_mlps(graph)
    assert len(pairs) == (4 if name == "convnext_tiny" else 1)
    moments = workflow.collect_moments(model, pairs, [(x, torch.zeros(2))], execution_device, 1)
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path=dict.fromkeys(pairs, 2)))
    first, second = next(iter(pairs.items()))
    plan = pruner.plan_remove([graph.parameter(f"{first}.weight").axis(0).select([0, 1])])
    updates = workflow.prepare_bias_compensation(graph, plan, pairs, moments)
    pruner.apply(plan)
    with torch.no_grad():
        for path, value in updates.items():
            model.get_parameter(path).copy_(value)

    def replace_with_mean(module, args):
        hidden = args[0].clone()
        hidden[..., :2] = moments[second].mean[:2].to(hidden)
        return (hidden,)

    hook = original.get_submodule(second).register_forward_pre_hook(replace_with_mean)
    try:
        torch.testing.assert_close(model(x), original(x), atol=2e-5, rtol=2e-5)
    finally:
        hook.remove()
    model(x).sum().backward()
    assert model.get_submodule(first).weight.grad is not None


@pytest.mark.parametrize("name", ["convnext_tiny", "vit_b_32"])
def test_variance_entry_runs_calibration_training_and_verified_checkpoint(
    name, monkeypatch, tmp_path, execution_device
):
    weights = imagenet_models.MLP_MODELS[name][1]
    requested = []

    def builder(*, weights):
        requested.append(weights)
        return miniature_model(name)

    def data(selected_weights, data_dir, **options):
        assert selected_weights is weights and options["need_train"]
        x = torch.randn(2, 3, 224, 224, device="cpu")
        dataset = TensorDataset(x, torch.tensor([0, 1], device="cpu"))
        return dataset, dataset, {"test_fixture": True}

    monkeypatch.setitem(imagenet_models.MLP_MODELS, name, (builder, weights))
    monkeypatch.setattr(workflow, "load_images", data)
    # Exact complexity/latency is separately covered; preserve real evaluation,
    # calibration, graph, planner, compensation, fine-tuning and restoration here.
    monkeypatch.setattr(workflow, "measure_model", lambda *args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "variance_pruning.py",
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
            "--granularity",
            "2",
            "--pruning_ratio",
            "0.001",
            "--finetune_epochs",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    # CLI processes keep CPU as the default construction device; each entry
    # explicitly moves its model and batches to the requested execution device.
    # In particular, DataLoader's random sampler uses a CPU generator.
    with torch.device("cpu"):
        workflow.main()
    assert requested == [weights, None]
    result = json.loads((tmp_path / "metrics.json").read_text())
    assert [stage["stage"] for stage in result["stages"]] == ["pretrained", "pruned", "finetuned"]
    assert result["stages"][1]["target_met"]
    assert result["config"]["calibration_observations"]
    assert (tmp_path / "model.pt").is_file() and (tmp_path / "training.pt").is_file()
    training = torch.load(tmp_path / "training.pt", map_location="cpu", weights_only=True)
    assert training["algorithm"] == {"method": "variance_based", "selection": "static"}


@pytest.mark.parametrize(
    "args",
    [
        ["--calibration_batches", "-1"],
        ["--weight_decay", "nan"],
        ["--lr", "0"],
        ["--pruning_ratio", "1"],
        ["--model", "resnet18"],
        ["--granularity", "0"],
    ],
)
def test_variance_cli_rejects_unsupported_or_invalid_configuration(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["variance_pruning.py", "--device", "cpu", *args])
    with pytest.raises(SystemExit) as error:
        workflow.parse_args()
    assert error.value.code == 2
