"""A complete pruning round on the original module, with explicit graph rebuilding."""

import argparse

import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import ChannelRatio, Greedy, Magnitude, Pruner, WeightTaylor


def main() -> None:
    """Run two pruning rounds with explicit training and dependency rebuilding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    options = parser.parse_args()
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(options.device)
    x = torch.randn(3, 4, device=options.device)

    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=Greedy(Magnitude(p=2)))
    print(plan.explain())
    returned, _result = pruner.apply(plan)
    assert returned is model and model[0].out_features == 6

    # The caller owns training and gradient collection. Recreate the optimizer
    # after replacing Parameters; no optimizer state is migrated automatically.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(x).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    model(x).square().mean().backward()

    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    plan = pruner.plan(space, budget=ChannelRatio(0.2), strategy=Greedy(WeightTaylor()))
    pruner.apply(plan)
    assert model[0].out_features == 5
    model(x).sum().backward()
    print(f"Original model after two rounds: {model[0].out_features} hidden channels")


if __name__ == "__main__":
    main()
