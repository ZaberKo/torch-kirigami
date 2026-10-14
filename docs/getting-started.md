# Getting started

This guide shows how to inspect, plan, apply, train, and save a small PyTorch model. Use the [ImageNet workflows](../examples/workflows/README.md) for accuracy comparisons.

## Install

Use Python 3.10 or newer and PyTorch 2.6 or newer. Run these commands from the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --torch-backend=auto -e .
uv pip install --group dev
```

The project keeps `torch>=2.6` as its portable runtime dependency. The install command selects the PyTorch backend for your host.

## Inspect and prune a model

The Python blocks in this section form one script. Build the graph after you set the model mode and choose representative inputs.

```python
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Pruner, load_checkpoint, save_checkpoint


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3)).eval()


torch.manual_seed(7)
model = make_model()
inputs = torch.randn(2, 4)
graph = DependencyGraph.build(model, args=(inputs,))
```

The hidden width is six. Rows of the first weight tensor represent hidden features. Columns of the last weight tensor consume those features.

Inspect a removal in the graph's original coordinates. This step does not change the model.

```python
hidden_axis = graph.parameter("0.weight").axis(0)
remove = hidden_axis.select([1, 4])
impact = graph.propagate(remove=[remove])

print(graph.explain(impact))
assert impact.status == "resolved"
consumer = graph.parameter("2.weight")
assert set(impact.selection(consumer).fully_selected_indices(1)) == {1, 4}
assert model[0].out_features == 6
```

The impact includes the first layer's bias and the last layer's matching input columns. `impact.complete` reports whether the graph knows the full influence. A complete impact does not prove that a physical rewrite is valid.

Plan the exact request, then apply its static recipes to the original module.

```python
pruner = Pruner(model, graph=graph)
plan = pruner.plan_remove([remove])
print(plan.explain())

returned_model, result = pruner.apply(plan)
assert returned_model is model
assert model[0].out_features == 4
assert model[2].in_features == 4
assert model(inputs).shape == (2, 3)
```

Planning checks the joint request, structural constraints, and execution requirements. Application prepares replacement tensors before it changes bindings. The module keeps its identity, but each replaced parameter is a new object.

Rebuild the graph and graph-bound groups after a structural change. Recreate the optimizer because it still points to replaced parameters.

## Train and save the compact model

This batch shows that the compact model supports ordinary autograd. It is not a fine-tuning recipe.

```python
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
targets = torch.tensor([0, 2])

optimizer.zero_grad()
loss = nn.functional.cross_entropy(model(inputs), targets)
loss.backward()
optimizer.step()

model.eval()
save_checkpoint(model, "compact.pt")
restored = load_checkpoint(make_model(), "compact.pt", map_location="cpu")
torch.testing.assert_close(restored(inputs), model(inputs))
```

The checkpoint restores final tensor sizes and values into a compatible model skeleton. It does not contain model code, pruning history, or optimizer state. See [persistence](persistence.md) for the checkpoint contract.

## Let a policy choose channels

This is an independent example. It uses a fresh model and graph.

```python
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Pruner
from torch_kirigami.pruning import ChannelRatio, Greedy, Magnitude


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3)).eval()


inputs = torch.randn(2, 4)
automatic_model = make_model()
automatic_graph = DependencyGraph.build(automatic_model, args=(inputs,))
automatic_pruner = Pruner(automatic_model, graph=automatic_graph)
space = automatic_pruner.discover_candidates()
automatic_plan = automatic_pruner.plan(
    space,
    strategy=Greedy(Magnitude(p=2)),
    budget=ChannelRatio(0.25),
)
automatic_pruner.apply(automatic_plan)
assert automatic_model[0].out_features == 4
```

Candidate discovery uses channel axes declared by operator rules. The Linear rule declares output features. It does not declare input features as default candidates. A hidden-feature candidate removes the matching producer row, bias entry, and consumer column through dependency propagation.

`Magnitude` scores the union of affected parameter regions. `ChannelRatio` caps final channel widths. It does not cap parameter count, MACs, or latency. The ratio `0.25` gives a final-width cap of `floor(6 * (1 - 0.25))`, which is four. The dependency closure and structural constraints must produce a valid plan within that cap.

Every automatic plan must meet its final-size target. If the search cannot meet the target, planning raises `PlanningError` and does not return a partial plan. A trial limit does not prove that the target is impossible.

For a whole-model parameter cap, use `ParameterBudget(max_params=...)`. It counts all unique registered Parameters, including frozen and protected tensors. It excludes buffers. An unmet parameter target also raises `PlanningError`.

## Diagnose failures

| Error or result | Meaning | Next step |
| --- | --- | --- |
| `CaptureError` | FX tracing or isolated metadata execution failed. | Check forward control flow and example inputs. |
| Unsupported diagnostic | The rules do not fully describe an affected operation. | Read the operator support reference. |
| `StaleGraphError` | Tracked model structure, mode, configuration, or constants changed. | Rebuild the graph and graph-bound objects. |
| `PlanningError` | The request lacks a verified executable recipe or misses its target. | Read the diagnostic and change the request or operator rule. |
| `ExecutionError` | Apply or load rejected a precondition or state change. | Check model compatibility and the reported binding. |
