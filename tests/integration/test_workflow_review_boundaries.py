"""Review regressions for workflow calibration and intentionally untrained gates."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import gate_pruning
import osscar_pruning
import variance_pruning

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Candidate, ChannelRatio, PlanningContext


def test_osscar_targets_keep_the_student_bias_as_fixed_intercept():
    student = nn.Sequential(nn.Linear(2, 2)).double().eval()
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher[0].bias.copy_(torch.tensor([10.0, 20.0]))
        student[0].bias.copy_(torch.tensor([1.0, 2.0]))
    x = torch.eye(2, dtype=torch.float64)
    moments = osscar_pruning.collect_reconstruction(
        student, teacher, "0", [(x, torch.zeros(2))], "cpu", 1, 2
    )
    target = teacher(x).detach() - student[0].bias.detach()
    torch.testing.assert_close(moments.cross, x.T @ target)
    fitted = torch.linalg.solve(moments.gram, moments.cross)
    torch.testing.assert_close(x @ fitted + student[0].bias, teacher(x))


def test_variance_metric_subtracts_accepted_positions_on_the_declared_axis():
    model = nn.Linear(4, 3, bias=False).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    axis = graph.parameter("weight").axis(1)
    context = PlanningContext(graph, (), ChannelRatio(0.25), (axis,), ())
    # The workflow constructor declares axis zero. Exercise score's axis contract
    # independently so future supported candidate domains cannot regress it.
    metric = object.__new__(variance_pruning.ActivationVariance)
    metric.graph, metric.scores = graph, {axis: (1.0, 2.0, 3.0, 4.0)}
    accepted = context.impact((axis.select([0]),))
    candidate = Candidate("two", (axis.select([0, 2]),), axis)
    assert context.score(metric, (candidate,), accepted_impact=accepted) == (3.0,)


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
