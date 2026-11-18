# Getting started

This tutorial prunes a two-layer MLP. It first removes two hidden features by
hand, then trains and saves the compact model, and finally lets a policy choose
what to remove. Everything runs on CPU in a few seconds. For accuracy
experiments on real models, see the
[ImageNet workflows](https://github.com/ZaberKo/torch-kirigami/blob/main/examples/workflows/README.md).

## Install

Torch Kirigami needs Python 3.10 or newer and PyTorch 2.6 or newer. Install it
with uv, which selects the PyTorch build for your machine:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --torch-backend=auto torch-kirigami
```

To work on the library itself, see the [development guide](development/index.md).

## Inspect a removal

The code blocks in this section and the next form one script. Set the model mode
and choose representative inputs before you build the graph. Capture records
module modes and collects metadata for the supplied inputs.

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

The hidden layer has six features. Rows of `0.weight` produce them, and columns
of `2.weight` consume them. Select two rows and ask the graph what else they
affect:

```python
hidden_axis = graph.parameter("0.weight").axis(0)
remove = hidden_axis.select([1, 4])
impact = graph.propagate(remove=[remove])

print(graph.explain(impact))
assert impact.status == "resolved" and impact.complete
consumer = graph.parameter("2.weight")
assert set(impact.selection(consumer).fully_selected_indices(1)) == {1, 4}
assert model[0].out_features == 6
```

The impact contains rows 1 and 4 of the first weight, entries 1 and 4 of its
bias, and columns 1 and 4 of the second weight. The query only reports this
closure. The model still has six hidden features.

## Plan and apply

Turn the same request into a plan, then apply it:

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

`plan_remove()` plans exactly the selections you pass. It checks constraints and
execution requirements, and it raises `PlanningError` instead of changing the
request. `apply()` builds all replacement tensors before it changes the module.
The module object stays the same, but every replaced parameter is a new
`nn.Parameter`.

The graph is now out of date. Build a new one before you plan another change,
and create a new optimizer, because an old optimizer still holds the replaced
parameters.

## Train and save

The compact model is an ordinary module. This single step only shows that
autograd works. It is not a fine-tuning recipe.

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

`load_checkpoint()` takes a freshly constructed *original* model and reshapes it
to the saved compact structure and values. The checkpoint holds no model code, no
pruning history, and no optimizer state. See
[plans and checkpoints](guides/persistence.md) for details.

## Let a policy choose

This independent example asks the library to choose the hidden features to remove.

```python
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import ChannelRatio, Greedy, Magnitude, Pruner

model = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 3)).eval()
inputs = torch.randn(2, 4)
graph = DependencyGraph.build(model, args=(inputs,))
pruner = Pruner(model, graph=graph)

space = pruner.discover_candidates()
plan = pruner.plan(space, strategy=Greedy(Magnitude(p=2)), budget=ChannelRatio(0.25))
pruner.apply(plan)
assert model[0].out_features == 4
```

The three arguments to `plan()` are independent choices:

- `discover_candidates()` creates one candidate for each output feature of each
  `Linear` layer. Candidates for the final layer's outputs are excluded because
  `Pruner` protects model inputs and outputs by default. Each remaining
  candidate removes one hidden feature together with its dependent bias entry
  and consumer column.
- `Magnitude(p=2)` scores a candidate by the L2 norm of all parameter values
  that it would remove.
- `ChannelRatio(0.25)` caps the final hidden width at `floor(6 * 0.75) = 4`.
  It limits channels, not parameters, MACs, or latency. Use
  `ParameterBudget(max_params=...)` to cap the final parameter count instead.

Automatic planning either meets the budget or raises `PlanningError`. It never
returns a partial plan, and a failure leaves the model unchanged.

## When something fails

| Signal | Meaning | What to do |
| --- | --- | --- |
| `CaptureError` from `build()` | FX tracing or metadata execution failed for this call. | Check the forward code and example inputs. See [model support](guides/model-support.md). |
| Incomplete impact (`impact.complete` is false) | An unsupported operation or an analysis limit leaves part of the request's influence unknown. | Read `impact.diagnostics`. For an unsupported operation, choose another request or [add a rule](reference/operator-support.md#write-an-operator-rule). For an analysis limit, consult the [configuration reference](reference/configuration.md). |
| `StaleGraphError` | The model's structure, modes, or tracked constants changed after the graph was built. | Build a new graph and new graph-bound objects. |
| `PlanningError` | The request violates a constraint, has no verified recipe, or misses the budget. | Read the message and diagnostics, then change the request, constraints, or budget. |
| `ExecutionError` | `apply()` or `load_checkpoint()` found an incompatible model or failed to commit. | Check that the model matches the plan or checkpoint. |
