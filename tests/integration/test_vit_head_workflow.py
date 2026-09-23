"""Independent attention references and head/FFN pruning lifecycle coverage."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Granularity,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import vit_head_pruning as workflow
from imagenet_models import HeadPrunableAttention, HeadPrunableViT
from torchvision.models.vision_transformer import VisionTransformer


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_vit(image_size=8):
    model = VisionTransformer(
        image_size=image_size,
        patch_size=image_size // 2,
        num_layers=2,
        num_heads=3,
        hidden_dim=12,
        mlp_dim=24,
        num_classes=5,
    )
    # Keep the full-size workflow input but avoid a huge protected patch kernel
    # dominating the tiny fixture's parameter budget. The token grid is still 2x2.
    if image_size == 224:
        model.conv_proj = nn.Conv2d(3, 12, kernel_size=4, stride=112)
    # Torchvision initializes the classifier to zero; nonzero weights make
    # comparisons sensitive to changes in the encoder.
    nn.init.normal_(model.heads.head.weight, std=0.1)
    return model


def dense_attention(source, x, removed_heads=()):
    batch, length, width = x.shape
    q, k, v = F.linear(x, source.in_proj_weight, source.in_proj_bias).chunk(3, dim=-1)
    heads = source.num_heads
    depth = width // heads
    q, k, v = (t.reshape(batch, length, heads, depth).transpose(1, 2) for t in (q, k, v))
    attention = (q @ k.transpose(-2, -1) / math.sqrt(depth)).softmax(-1) @ v
    keep = torch.ones(heads, device=x.device, dtype=x.dtype)
    keep[list(removed_heads)] = 0
    attention = attention * keep[None, :, None, None]
    return F.linear(
        attention.transpose(1, 2).reshape(batch, length, width),
        source.out_proj.weight,
        source.out_proj.bias,
    )


@pytest.mark.parametrize("heads", [4, 12])
def test_native_eval_conversion_with_even_head_count(execution_device, heads):
    # Native MHA's eval fast path requires an even head count. Most compact
    # fixtures below deliberately use three heads to test retention of one.
    source = nn.MultiheadAttention(heads * 4, heads, batch_first=True).eval()
    source.out_proj.weight.requires_grad_(False)
    adapted = HeadPrunableAttention(source)
    x = torch.randn(2, 7, heads * 4)
    assert not adapted.training
    assert not adapted.proj.weight.requires_grad and adapted.qkv.weight.requires_grad
    with torch.inference_mode():
        expected = source(x, x, x, need_weights=False)[0]
        torch.testing.assert_close(adapted(x), expected)
        torch.testing.assert_close(adapted(x), dense_attention(source, x))


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("removed_heads", [(1,), (0, 2)])
def test_conversion_and_whole_head_compaction_match_dense_math(
    execution_device,
    bias,
    training,
    removed_heads,
    tmp_path,
):
    torch.manual_seed(91)
    source = nn.MultiheadAttention(12, 3, batch_first=True, bias=bias).double().train(training)
    model = HeadPrunableAttention(source)
    x = torch.randn(2, 5, 12, dtype=torch.double)
    official = source(x, x, x, need_weights=False)[0]
    torch.testing.assert_close(model(x), official)
    torch.testing.assert_close(model(x), dense_attention(source, x))
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("qkv.weight").axis(0)
    pruner = Pruner(model, graph=graph)
    rows = [
        offset * 12 + head * 4 + feature
        for offset in range(3)
        for head in removed_heads
        for feature in range(4)
    ]
    plan = pruner.plan_remove([axis.select(rows)])
    model, _ = pruner.apply(plan)
    remaining = 3 - len(removed_heads)
    assert model.num_heads == remaining and model.head_dim == 4
    assert model.qkv.weight.shape == (remaining * 12, 12)
    assert model.proj.weight.shape == (12, remaining * 4)
    # Reference uses explicit matmul/softmax and zeros the removed head's
    # contribution before the original output projection; no compact weights.
    actual_input = x.detach().clone().requires_grad_()
    reference_input = x.detach().clone().requires_grad_()
    actual, reference = model(actual_input), dense_attention(source, reference_input, removed_heads)
    torch.testing.assert_close(actual, reference)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(actual_input.grad, reference_input.grad)
    assert model.qkv.weight.grad is not None and model.proj.weight.grad is not None
    save_checkpoint(model, tmp_path / "heads.pt")
    restored = load_checkpoint(
        HeadPrunableAttention(source),
        tmp_path / "heads.pt",
        map_location=execution_device,
    )
    torch.testing.assert_close(restored(x), model(x))
    assert restored.num_heads == remaining
    # Shape reads and -1 remain valid for different batches and token lengths.
    another = torch.randn(1, 7, 12, dtype=torch.double)
    torch.testing.assert_close(restored(another), dense_attention(source, another, removed_heads))


@pytest.mark.parametrize("invalid", ["partial_head", "misaligned_head", "all_heads"])
def test_invalid_head_request_fails_without_mutation(execution_device, invalid):
    model = HeadPrunableAttention(nn.MultiheadAttention(12, 3, batch_first=True)).eval()
    x = torch.randn(2, 5, 12)
    graph = DependencyGraph.build(model, args=(x,))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    bindings = tuple(model.parameters())
    # Removing one feature from every head would change the fixed head_dim.
    rows = {
        "partial_head": list(range(0, 36, 4)),
        "misaligned_head": [
            offset * 12 + feature for offset in range(3) for feature in (1, 2, 3, 4)
        ],
        "all_heads": list(range(36)),
    }[invalid]
    with pytest.raises(PlanningError):
        Pruner(model, graph=graph).plan_remove([graph.parameter("qkv.weight").axis(0).select(rows)])
    assert all(a is b for a, b in zip(bindings, model.parameters(), strict=True))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    graph.validate()
    assert model(x).shape == x.shape


def test_attention_dropout_and_mode(execution_device, monkeypatch):
    source = nn.MultiheadAttention(12, 3, dropout=0.4, batch_first=True)
    model = HeadPrunableAttention(source)
    x = torch.randn(2, 5, 12)
    sdpa = F.scaled_dot_product_attention
    probabilities = []

    def record_dropout(*args, **kwargs):
        probabilities.append(kwargs["dropout_p"])
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record_dropout)
    first, second = model(x), model(x)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert not torch.equal(first, second)
    model.eval()
    torch.testing.assert_close(model(x), dense_attention(source, x))
    assert probabilities == [0.4, 0.4, 0.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_first": False},
        {"batch_first": True, "kdim": 8},
        {"batch_first": True, "add_bias_kv": True},
        {"batch_first": True, "add_zero_attn": True},
    ],
)
def test_attention_conversion_rejects_other_semantics(kwargs):
    with pytest.raises(ValueError, match="ViT self-attention"):
        HeadPrunableAttention(nn.MultiheadAttention(12, 3, **kwargs))


def test_joint_vit_head_ffn_pruning_and_restore(execution_device, tmp_path):
    torch.manual_seed(6)
    official = tiny_vit().eval()
    model = HeadPrunableViT(copy.deepcopy(official)).eval()
    x = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(model(x), official(x))
    tokens = torch.randn(2, 5, 12)
    torch.testing.assert_close(model.encoder.layers[0](tokens), official.encoder.layers[0](tokens))
    reference = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(x[:1],))
    pruner = Pruner(model, graph=graph)
    space = workflow.candidate_space(pruner)
    heads = [c for c in space.candidates if ":head:" in c.key]
    assert len(heads) == 6
    assert len(space.candidates) == 2 * (24 + 3)
    selected = [heads[1]]
    ffn = graph.parameter("encoder.layers.encoder_layer_1.mlp.0.weight").axis(0)
    plan = pruner.plan_remove([*selected[0].remove, ffn.select([2, 7, 13])])
    model, _ = pruner.apply(plan)

    def mask_head(_module, args):
        values = args[0].clone()
        values[..., 4:8] = 0
        return (values,)

    def mask_neurons(_module, args):
        values = args[0].clone()
        values[..., [2, 7, 13]] = 0
        return (values,)

    handle1 = reference.encoder.layers[0].self_attention.proj.register_forward_pre_hook(mask_head)
    handle2 = reference.encoder.layers[1].mlp[3].register_forward_pre_hook(mask_neurons)
    try:
        torch.testing.assert_close(model(x), reference(x))
    finally:
        handle1.remove()
        handle2.remove()
    assert model.encoder.layers[0].self_attention.num_heads == 2
    assert model.encoder.layers[1].mlp[0].out_features == 21
    assert model.conv_proj.out_channels == 12
    assert model.encoder.pos_embedding.shape[-1] == 12
    assert model.heads.head.in_features == 12
    save_checkpoint(model, tmp_path / "vit.pt")
    restored = load_checkpoint(
        HeadPrunableViT(tiny_vit()),
        tmp_path / "vit.pt",
        map_location=execution_device,
    ).eval()
    torch.testing.assert_close(restored(x), model(x))
    restored.train()
    F.cross_entropy(restored(x), torch.tensor([1, 3])).backward()
    assert restored.encoder.layers[0].self_attention.qkv.weight.grad is not None


def test_logical_head_metric_and_automatic_plan(execution_device):
    torch.manual_seed(11)
    model = HeadPrunableViT(tiny_vit()).eval()
    # Give one complete head a clear, reproducible low-energy signature.
    attention = model.encoder.layers[0].self_attention
    with torch.no_grad():
        for start in (0, 12, 24):
            attention.qkv.weight[start : start + 4].zero_()
        attention.proj.weight[:, :4].zero_()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3, 8, 8),))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(
            by_path={
                "encoder.layers.encoder_layer_0.mlp.0": 4,
                "encoder.layers.encoder_layer_1.mlp.0": 4,
            }
        ),
    )
    space = workflow.candidate_space(pruner)
    budget = ParameterBudget.from_ratio(model, 0.10)
    context = PlanningContext(
        graph,
        graph.operations(),
        space.candidates,
        budget,
        space.channel_axes,
        pruner.constraints,
    )
    scores = context.score(GroupMagnitude(p=2), space.candidates)
    head_candidates = [(i, c) for i, c in enumerate(space.candidates) if ":head:" in c.key]
    assert scores[head_candidates[0][0]] == 0
    # Independently accumulate propagated parameter-region unions for the exact
    # per-axis singleton normalization, excluding biases as the metric specifies.
    bindings = dict(graph.tensor_bindings())

    def energy(impact):
        value = 0.0
        for selection in impact.parameters:
            if not selection.tensor.paths[0].endswith("weight"):
                continue
            weight = bindings[selection.tensor].detach().double()
            mask = torch.zeros_like(weight, dtype=torch.bool)
            for region in selection.regions:
                coordinates = [
                    torch.tensor(list(indices), dtype=torch.long) for indices in region.axes
                ]
                mask[torch.meshgrid(*coordinates, indexing="ij")] = True
            value += weight[mask].square().sum().item()
        return value

    for index, candidate in (head_candidates[1], (0, space.candidates[0])):
        axis = candidate.axis
        singleton = [
            energy(graph.propagate(remove=[axis.select([i])]))
            for i in range(axis.tensor.shape[axis.dim])
        ]
        count = len(candidate.remove[0].fully_selected_indices(axis.dim))
        expected = energy(pruner.impact([candidate])) / (sum(singleton) / len(singleton)) / count
        assert scores[index] == pytest.approx(expected, rel=1e-5)
    plan = pruner.plan(space, budget=budget, strategy=Greedy(GroupMagnitude(p=2)))
    assert plan.selection_report.target_met
    model, _ = pruner.apply(plan)
    assert sum(p.numel() for p in model.parameters()) <= budget.max_params
    assert model.encoder.layers[0].self_attention.num_heads < 3
    assert all(block.mlp[0].out_features % 4 == 0 for block in model.encoder.layers)
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 5)


@pytest.mark.parametrize("ratio", [0.0, 0.12, 0.99])
def test_full_workflow_train_save_load(execution_device, ratio, monkeypatch, tmp_path):
    torch.manual_seed(19)
    model = HeadPrunableViT(tiny_vit(224)).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    dataset = TensorDataset(
        torch.randn(2, 3, 224, 224, device="cpu"), torch.tensor([1, 2], device="cpu")
    )
    monkeypatch.setattr(
        workflow,
        "make_head_prunable_model",
        lambda _name, pretrained=True: model if pretrained else HeadPrunableViT(tiny_vit(224)),
    )
    monkeypatch.setattr(workflow, "load_images", lambda *args, **kwargs: (dataset, dataset, {}))
    monkeypatch.setattr(workflow, "measure_model", lambda *args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vit_head_pruning",
            "--device",
            execution_device,
            "--pruning_ratio",
            str(ratio),
            "--train_batch_size",
            "2",
            "--val_batch_size",
            "2",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--granularity",
            "4",
            "--finetune_epochs",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    if ratio == 0.99:
        with torch.device("cpu"), pytest.raises(PlanningError):
            workflow.main()
        assert not (tmp_path / "model.pt").exists()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        return
    with torch.device("cpu"):
        workflow.main()
    records = json.loads((tmp_path / "metrics.json").read_text())
    assert [s["stage"] for s in records["stages"]] == ["pretrained", "pruned", "finetuned"]
    assert records["stages"][1]["target_met"]
    assert (tmp_path / "model.pt").exists() and (tmp_path / "training.pt").exists()
    if ratio == 0:
        assert records["stages"][0]["structure"] == records["stages"][1]["structure"]


def test_cli_checkpoint_restores_in_a_different_process(tmp_path):
    # Execute the real entry as __main__, but substitute tiny weights and local
    # tensors. Restoration below runs in this separate importing process.
    source = Path(workflow.__file__).resolve()
    script = """
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]).parent))
import torch
import imagenet_data
import imagenet_models
import model_metrics
from torch.utils.data import TensorDataset
from tests.integration.test_vit_head_workflow import tiny_vit

torch.set_num_threads(1)
torch.manual_seed(23)
output, entry = Path(sys.argv[1]), sys.argv[2]
created = []
def build(name, *, pretrained=True):
    model = imagenet_models.HeadPrunableViT(tiny_vit(224))
    created.append(model)
    return model
imagenet_models.make_head_prunable_model = build
data = TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([1, 2]))
imagenet_data.load_images = lambda *args, **kwargs: (None, data, {})
model_metrics.measure_model = lambda *args: {}
sys.argv = [entry, '--device', 'cpu', '--pruning_ratio', '0.1',
            '--granularity', '4', '--val_workers', '0', '--val_batch_size', '2',
            '--output', str(output)]
runpy.run_path(entry, run_name='__main__')
with torch.no_grad():
    torch.save({'x': data.tensors[0], 'y': created[0](data.tensors[0])}, output / 'reference.pt')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(source)],
        cwd=source.parents[2],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = torch.load(tmp_path / "reference.pt", weights_only=True)
    restored = load_checkpoint(HeadPrunableViT(tiny_vit(224)), tmp_path / "model.pt").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(expected["x"]), expected["y"])
