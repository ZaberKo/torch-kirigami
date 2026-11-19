"""Check CPD interval statistics, planning fallback, and compact training."""

import copy
import json
import sys
from argparse import Namespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import cpd_pruning as workflow
from imagenet_models import TraceableViT
from torchvision.models.resnet import BasicBlock

from tests.support.torchvision_models import tiny_vit, workflow_resnet


def test_interval_scores_match_signed_batch_sums_and_group_normalization(execution_device):
    model = nn.Sequential(BasicBlock(3, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
    reference = copy.deepcopy(model)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    batches = [(torch.randn(n, 3, 4, 5), torch.arange(n) % 2) for n in (2, 3)]
    learning_rate = 0.02
    expected = torch.zeros(3)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)
    for x, labels in batches:
        ref_optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(reference(x), labels).backward()
        conv = reference[0].conv1.weight
        scale = reference[0].bn1.weight
        bias = reference[0].bn1.bias
        conv_score = torch.einsum("cijk,cijk->c", conv.detach(), conv.grad).square()
        norm_score = (scale.detach() * scale.grad).square() + (bias.detach() * bias.grad).square()
        # Three scored members, plus BN scale, BN bias, and the consumer weight.
        expected += (conv_score + norm_score) / (3 * 6 * x.numel())
        ref_optimizer.step()

    options = Namespace(device=execution_device, temperature=1.0, distillation_weight=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scores = workflow.train_interval(model, teacher, optimizer, batches, options)
    torch.testing.assert_close(torch.tensor(scores["0.conv1"]), expected)
    torch.testing.assert_close(model.state_dict(), reference.state_dict())
    assert workflow.pruning_groups(model)["0.conv1"][0] == (model[0].conv1, model[0].bn1)


def test_interval_tries_next_group_and_preserves_momentum(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.sizer = nn.Linear(4, 4)
            self.hidden = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            size = self.sizer(x).shape[-1] // 4
            return x.split(size, -1), self.head(self.hidden(x))

    model = Model()
    x = torch.randn(2, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model(x)[1].square().sum().backward()
    optimizer.step()
    before = dict(model.named_parameters())
    with pytest.raises(workflow.PlanningError, match="split size becomes 0"):
        workflow.prune_interval(model, x, {"sizer": [0.0] * 4})
    assert all(parameter is before[name] for name, parameter in model.named_parameters())
    expected = F.linear(model.hidden(x)[:, 3:], model.head.weight[:, 3:], model.head.bias)
    hidden_momentum = optimizer.state[model.hidden.weight]["momentum_buffer"][3:].clone()
    head_momentum = optimizer.state[model.head.weight]["momentum_buffer"][:, 3:].clone()
    result = workflow.prune_interval(model, x, {"sizer": [0.0] * 4, "hidden": [1.0, 2.0, 3.0, 4.0]})
    assert result.plan.selected == ("hidden",)
    assert any(
        "split size becomes 0" in reason for _, reason in result.plan.selection_report.exclusions
    )
    optimizer = workflow.rebuild_optimizer(model, optimizer, before, result)
    torch.testing.assert_close(model(x)[1], expected)
    torch.testing.assert_close(
        optimizer.state[model.hidden.weight]["momentum_buffer"], hidden_momentum
    )
    torch.testing.assert_close(optimizer.state[model.head.weight]["momentum_buffer"], head_momentum)
    optimizer.zero_grad(set_to_none=True)
    model(x)[1].square().sum().backward()
    optimizer.step()


@pytest.mark.parametrize(
    "model_name,ratio,intervals,recovery_epochs,target_reached",
    [
        ("resnet18", "0.012", 2, 1, True),
        ("resnet18", "0.012", 1, 1, False),
        ("vit_b_16", "0.01", 1, 0, True),
    ],
)
def test_cpd_workflow_reaches_target_and_restores_checkpoint(
    execution_device,
    tmp_path,
    monkeypatch,
    capsys,
    model_name,
    ratio,
    intervals,
    recovery_epochs,
    target_reached,
):
    def make_model(name, **_kwargs):
        return workflow_resnet() if name == "resnet18" else TraceableViT(tiny_vit(image_size=224))

    x = torch.randn(2, 3, 224, 224)
    dataset = TensorDataset(x.cpu(), torch.tensor([1, 2], device="cpu"))
    monkeypatch.setattr(workflow, "make_model", make_model)
    monkeypatch.setattr(workflow, "load_images", lambda *a, **k: (dataset, dataset, {}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cpd_pruning.py",
            "--model",
            model_name,
            "--device",
            execution_device,
            "--pruning_ratio",
            ratio,
            "--pruning_interval_batches",
            "2",
            "--max_intervals",
            str(intervals),
            "--recovery_epochs",
            str(recovery_epochs),
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--latency_warmup",
            "0",
            "--latency_repetitions",
            "1",
            "--output",
            str(tmp_path),
        ],
    )
    with torch.device("cpu"):
        if target_reached:
            workflow.main()
        else:
            with pytest.raises(workflow.PlanningError, match="interval limit reached"):
                workflow.main()
    output = capsys.readouterr().out
    assert output.count("Pruned ") == intervals
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    logged = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert [row for row in logged if "stage" in row] == metrics["stages"]
    assert metrics["config"]["weights"] == str(workflow.MODELS[model_name][1])
    assert metrics["stages"][0]["top1_delta_pp"] == 0.0
    pruned_checkpoint = tmp_path / "pruned.pt"
    if not target_reached:
        assert [row["stage"] for row in metrics["stages"]] == ["pretrained"]
        assert not (tmp_path / "model.pt").exists()
        assert not pruned_checkpoint.exists()
        return
    assert "Checkpoint verified; results:" in output
    stages = ["pretrained", "pruned", "recovered"] if recovery_epochs else ["pretrained", "pruned"]
    assert [row["stage"] for row in metrics["stages"]] == stages
    pretrained, pruned = metrics["stages"][:2]
    assert pruned["#Params"] <= metrics["config"]["max_params"] < pretrained["#Params"]
    assert metrics["stages"][-1]["#Params"] == pruned["#Params"]
    assert pruned_checkpoint.is_file() == bool(recovery_epochs)
    assert not (tmp_path / "training.pt").exists()
    if recovery_epochs:
        intermediate = workflow.load_checkpoint(
            make_model(model_name), pruned_checkpoint, map_location=execution_device
        ).eval()
        with torch.no_grad():
            loss = F.cross_entropy(intermediate(x), torch.tensor([1, 2]))
        assert loss.item() == pytest.approx(pruned["loss"])
    compact = workflow.load_checkpoint(
        make_model(model_name), tmp_path / "model.pt", map_location=execution_device
    )
    if model_name == "resnet18":
        blocks = [block for block in compact.modules() if type(block) is BasicBlock]
        assert sum(32 - block.conv1.out_channels for block in blocks) in (3, 4)
        assert all(block.conv1.out_channels == block.conv2.in_channels for block in blocks)
    else:
        assert sorted(block.mlp[0].out_features for block in compact.encoder.layers) == [1, 24]
