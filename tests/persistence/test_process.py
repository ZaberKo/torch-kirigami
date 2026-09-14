"""persistence / process contracts."""

import subprocess
import sys

import torch

from tests.support.models import chain
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Pruner,
    save_checkpoint,
)


def test_checkpoint_and_plan_load_in_fresh_process(tmp_path):
    model = chain()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("0.weight").axis(0).select([1, 4])]
    )
    torch.save(plan.to_dict(), tmp_path / "plan.pt")
    Pruner(model).apply(plan)
    save_checkpoint(model, tmp_path / "checkpoint.pt")
    torch.save(x, tmp_path / "input.pt")
    code = """
import sys, torch
from pathlib import Path
from torch import nn
from torch_kirigami.pruning import Pruner, PruningPlan, load_checkpoint
p = Path(sys.argv[1])
def factory():
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 2))
plan = PruningPlan.from_dict(torch.load(p / 'plan.pt', weights_only=True))
model, result = Pruner(factory()).apply(plan)
assert model[0].out_features == 4
model = load_checkpoint(factory(), p / 'checkpoint.pt', map_location='cpu')
x = torch.load(p / 'input.pt', weights_only=True)
torch.save(model(x).detach(), p / 'output.pt')
"""
    subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)], check=True)
    actual = torch.load(tmp_path / "output.pt", weights_only=True)
    torch.testing.assert_close(actual, model(x))
