"""Measure CPU planning and sparse operations with fixed, local inputs.

Run with `PYTHONPATH=. .venv/bin/python experiments/planning_sparsity_benchmark.py`.
This probe uses random ResNet-18 weights. It does not download weights or data.
Results cover planning and sparse operations, not model inference or training.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median
from typing import Any
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import resnet18

import torch_kirigami.dep_graph.dependency_graph as graph_module
from torch_kirigami import DependencyGraph, IndexSet
from torch_kirigami.pruning import (
    ChannelRatio,
    DynamicGreedy,
    Granularity,
    Greedy,
    GroupMagnitude,
    Magnitude,
    ParameterGroup,
    Pruner,
)
from torch_kirigami.sparsity import GroupLasso, set_group_norms_, zero_groups_


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    results: dict[str, dict[str, Any]] = {}
    model = resnet18(weights=None).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(1, 3, 32, 32),))
    pruner = Pruner(model, graph=graph, granularity=Granularity(default=8))
    paths = tuple(f"layer{s}.{b}.conv1" for s in range(1, 5) for b in range(2))
    space = pruner.discover_candidates(targets=paths)
    original = graph_module.fingerprint
    calls: list[int] = []

    def measured(current: nn.Module) -> tuple[object, ...]:
        calls.append(1)
        return original(current)

    with patch.object(graph_module, "fingerprint", measured):
        for name, strategy in (
            ("greedy", Greedy(Magnitude())),
            ("dynamic", DynamicGreedy(GroupMagnitude())),
        ):
            calls.clear()
            start = time.perf_counter()
            plan = pruner.plan(space, budget=ChannelRatio(0.05, scope="global"), strategy=strategy)
            results[name] = {
                "seconds": time.perf_counter() - start,
                "fingerprints": len(calls),
                "trials": plan.selection_report.trials,
                "selected_digest": hashlib.sha256(repr(plan.selected).encode()).hexdigest(),
                "remaining": plan.selection_report.remaining,
            }
        calls.clear()
        start = time.perf_counter()
        groups = pruner.parameter_groups(space.candidates)
        results["parameter_groups"] = {
            "seconds": time.perf_counter() - start,
            "fingerprints": len(calls),
            "count": len(groups),
            "elements": sum(s.count for group in groups for s in group.selections),
        }
        penalty = GroupLasso(groups)
        penalty()
        calls.clear()
        start = time.perf_counter()
        for _ in range(20):
            value = penalty()
        results["lasso"] = {
            "milliseconds": (time.perf_counter() - start) * 50,
            "fingerprints": len(calls),
            "value": value.item(),
        }

    for count in (100, 200, 400):
        layer = nn.Linear(2, count, bias=False).double().eval()
        local = DependencyGraph.build(layer, args=(torch.ones(1, 2, dtype=torch.float64),))
        axis = local.parameter("weight").axis(0)
        rows = tuple(ParameterGroup(local, (axis.select([i]),)) for i in range(count))
        set_group_norms_(rows, (1.0,) * count)
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            set_group_norms_(rows, (1.0,) * count)
            samples.append(time.perf_counter() - start)
        results[f"norms_{count}"] = {"seconds": median(samples), "samples": 7}

    layer = nn.Linear(2048, 2048, bias=False).eval()
    local = DependencyGraph.build(layer, args=(torch.ones(1, 2048),))
    selected = ParameterGroup(local, (local.parameter("weight").axis(0).select([0]),))
    clones: list[int] = []
    clone = torch.Tensor.clone

    def counted_clone(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        clones.append(tensor.numel())
        return clone(tensor, *args, **kwargs)

    zero_groups_((selected,))
    samples = []
    with patch.object(torch.Tensor, "clone", counted_clone):
        for _ in range(7):
            start = time.perf_counter()
            zero_groups_((selected,))
            samples.append(time.perf_counter() - start)
        results["zero_small_region"] = {
            "seconds": median(samples),
            "cloned_elements": sum(clones) // 7,
            "samples": 7,
        }
    indices = IndexSet.span(0, 1_000_000)
    start = time.perf_counter()
    for _ in range(10):
        assert 999_999 in indices
    results["membership"] = {"seconds": time.perf_counter() - start}
    report = json.dumps(results, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
