"""Untrained gate selection warns before loading model weights."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import gate_pruning


def test_untrained_gate_pruning_warns_before_loading_a_model(monkeypatch):
    options = SimpleNamespace(sparse_epochs=0, pruning_ratio=0.1, seed=7, model="resnet18")
    monkeypatch.setattr(gate_pruning, "parse_args", lambda: options)

    def stop(_):
        raise RuntimeError("stop before loading weights")

    monkeypatch.setattr(gate_pruning, "make_model", stop)
    with (
        pytest.warns(UserWarning, match="without learned importance"),
        pytest.raises(RuntimeError, match="stop before"),
    ):
        gate_pruning.main()
