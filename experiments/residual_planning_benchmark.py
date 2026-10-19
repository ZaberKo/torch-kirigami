"""Benchmark residual-domain discovery, planning and sparse writes on local weights.

Run with `PYTHONPATH=. .venv/bin/python experiments/residual_planning_benchmark.py
--output result.json`. Use the same script and interpreter against each checkout.
No model weights or datasets are downloaded. Timings include instrumentation.
"""

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Any
from unittest.mock import patch

import torch
from torch import nn

import torch_kirigami.dep_graph.dependency_graph as graph_module
from torch_kirigami import DependencyGraph
from torch_kirigami.dep_graph.solver import DependencySolver
from torch_kirigami.pruning import (
    ChannelRatio,
    DynamicGreedy,
    Granularity,
    Greedy,
    Magnitude,
    ParameterGroup,
    Pruner,
)
from torch_kirigami.sparsity import GroupLasso, zero_groups_


class ResidualBlock(nn.Module):
    """Preserve the trunk width while exposing an independent hidden axis."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(64, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.c2(self.c1(x).relu())).relu()


class ResidualCNN(nn.Module):
    """Expose nine equal trunk domains and eight independent hidden domains."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 64, 1)
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(8)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))


def main() -> None:
    """Record costs and exact decision digests with one CPU thread and a fixed seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ResidualCNN().eval()
    x = torch.randn(1, 3, 8, 8)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_type={nn.Conv2d: 8}))
    propagate = DependencySolver.propagate
    fingerprint = graph_module.fingerprint
    counts: Counter[str] = Counter()
    results: dict[str, Any] = {"torch": torch.__version__, "threads": 1, "seed": 7}

    def counted_propagate(self: DependencySolver, *, remove: Any, constraints: Any = ()) -> Any:
        counts["propagations"] += 1
        return propagate(self, remove=remove, constraints=constraints)

    def counted_fingerprint(current: nn.Module) -> tuple[object, ...]:
        counts["fingerprints"] += 1
        return fingerprint(current)

    def measure(name: str, function: Callable[[], Any]) -> Any:
        counts.clear()
        start = time.perf_counter()
        value = function()
        results[name] = {"seconds": time.perf_counter() - start, **counts}
        print(name, results[name], flush=True)
        return value

    with (
        patch.object(DependencySolver, "propagate", counted_propagate),
        patch.object(graph_module, "fingerprint", counted_fingerprint),
    ):
        space = measure("discover", pruner.discover_candidates)
        filtered = measure("filter_discovered", lambda: pruner.filter_candidates(space))
        assert filtered == space
        results["candidates"] = len(space.candidates)
        plans = []
        for name, strategy in (
            ("greedy", Greedy(Magnitude())),
            ("dynamic", DynamicGreedy(Magnitude())),
        ):
            plan = measure(
                name,
                lambda strategy=strategy: pruner.plan(
                    space, budget=ChannelRatio(0.05, scope="global"), strategy=strategy
                ),
            )
            results[name].update(
                trials=plan.selection_report.trials,
                selected_digest=hashlib.sha256(repr(plan.selected).encode()).hexdigest(),
                remaining=plan.selection_report.remaining,
            )
            plans.append(plan)
        sparse_model = copy.deepcopy(model)
        sparse_graph = DependencyGraph.build(sparse_model, args=(x,))
        sparse_pruner = Pruner(sparse_model, graph=sparse_graph)
        sparse_space = sparse_pruner.discover_candidates()
        groups = measure(
            "parameter_groups", lambda: sparse_pruner.parameter_groups(sparse_space.candidates)
        )
        results["parameter_groups"]["count"] = len(groups)
        groups = groups[:8]
        penalty = GroupLasso(groups)
        union = ParameterGroup(sparse_graph, tuple(s for group in groups for s in group.selections))
        for name, function in (
            ("zero_groups", lambda: zero_groups_(groups)),
            ("zero_union", lambda: zero_groups_((union,))),
            ("group_lasso", penalty),
        ):
            function()
            measure(name, function)
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                function()
                samples.append(time.perf_counter() - start)
            results[name]["median_seconds"] = median(samples)
        pruner.apply(plans[0])
        results["applied_shapes"] = {name: list(p.shape) for name, p in model.named_parameters()}
        output = model(x).detach().numpy().tobytes()
        results["output_digest"] = hashlib.sha256(output).hexdigest()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
