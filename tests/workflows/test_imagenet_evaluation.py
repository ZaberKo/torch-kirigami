"""Sample-weighted accuracy, evaluation progress, and mode restoration."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torchvision", reason="Install examples/workflows/requirements-examples.txt")
pytest.importorskip("datasets", reason="Install examples/workflows/requirements-examples.txt")


import imagenet_data as imagenet


def test_accuracy_uses_all_logits_weights_partial_batches_and_restores_modes():
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 0, 1])
    model = nn.Sequential(nn.Identity(), nn.Dropout(0.9)).train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    result = imagenet.evaluate(
        model, DataLoader(TensorDataset(logits, labels), batch_size=2), "cpu"
    )
    assert result["samples"] == 3
    assert result["top1"] == pytest.approx(100 / 3)
    assert result["top5"] == pytest.approx(200 / 3)
    assert result["loss"] == pytest.approx(F.cross_entropy(logits, labels).item())
    assert [m.training for m in model.modules()] == modes
    with pytest.raises(ValueError, match="empty"):
        imagenet.evaluate(model, [], "cpu")
    assert [m.training for m in model.modules()] == modes


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_progress_counts_partial_batches_and_closes_on_error(
    fail, execution_device, monkeypatch, capsys
):
    model = nn.Sequential(nn.Linear(4, 6), nn.Dropout()).train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    bars = []
    original_tqdm = imagenet.tqdm

    def tracked(*args, **kwargs):
        bar = original_tqdm(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(imagenet, "tqdm", tracked)
    if fail:
        loader = [
            (torch.randn(2, 4), torch.zeros(2, dtype=torch.long)),
            (torch.randn(1, 5), torch.zeros(1, dtype=torch.long)),
        ]
        with pytest.raises(RuntimeError):
            imagenet.evaluate(model, loader, execution_device, description="Broken evaluation")
    else:
        x, labels = torch.randn(5, 4), torch.tensor([0, 1, 2, 3, 4])
        loader = DataLoader(TensorDataset(x, labels), batch_size=2)
        result = imagenet.evaluate(
            model, loader, execution_device, description="Partial evaluation"
        )
        assert result["samples"] == 5
        assert bars[0].n == bars[0].total == 3
        output = capsys.readouterr().err
        assert "100%" in output and "3/3" in output and "images=5" in output
    assert len(bars) == 1 and bars[0].disable  # close() disables further rendering.
    assert [m.training for m in model.modules()] == modes
