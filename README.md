# torch-kirigami

[![CI](https://github.com/ZaberKo/torch-kirigami/actions/workflows/ci.yml/badge.svg)](https://github.com/ZaberKo/torch-kirigami/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torch-kirigami)](https://pypi.org/project/torch-kirigami/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Structured pruning and dependency analysis for PyTorch models.**

torch-kirigami helps you build smaller neural networks by physically removing
channels, features, and attention heads together with their dependent parameters.
It uses PyTorch FX to analyze model structure, then validates and applies pruning
plans to the original `nn.Module`.

Use it for automatic model pruning, custom pruning algorithms, or inspecting how
a structural change propagates through a model.

[Quick start](#quick-start) · [Documentation](docs/index.md) · [ImageNet examples](examples/workflows/README.md) · [Development](docs/development.md)

## Features

- **Dependency-aware pruning:** propagate removals through connected operators
  and coordinate the required tensor and module-attribute changes.
- **Automatic and manual selection:** use magnitude or Taylor scores, supply
  explicit removals, or implement custom metrics and strategies. Set channel
  limits or a final whole-model parameter budget.
- **Sparse training and iterative pruning:** compose regularizers, channel gates,
  parameter operations, and cumulative budgets with your own training loop.
- **Compact-model checkpoints:** save the pruned structure and weights, then
  restore them using the original model definition.
- **Custom operators:** extend analysis and pruning through explicit operator
  rules, including fused attention blocks.
- **Model measurement:** inspect parameter counts, supported MACs, and measured
  inference latency, including compiled execution.

## Installation

Requires **Python 3.10+** and **PyTorch 2.6+**. PyTorch is the only runtime dependency.

Install from [PyPI](https://pypi.org/project/torch-kirigami/) with uv:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --torch-backend=auto torch-kirigami
```

For an editable installation and development tools, see the
[development guide](docs/development.md). ImageNet examples have additional
dependencies listed in the [workflow guide](examples/workflows/README.md).

## Quick start

Prune a small network to a final parameter budget. This example uses the current
accelerator; set `device = torch.device("cpu")` to run it on a CPU.

```python
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import Greedy, GroupMagnitude, ParameterBudget, Pruner

device = torch.accelerator.current_accelerator()
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).to(device).eval()
x = torch.randn(2, 4, device=device)

graph = DependencyGraph.build(model, args=(x,))
pruner = Pruner(model, graph=graph)
candidates = pruner.discover_candidates()

model, result = pruner.prune(
    candidates,
    budget=ParameterBudget(max_params=51),
    strategy=Greedy(GroupMagnitude(p=2)),
)

assert model[0].out_features == model[2].in_features == 6
assert model(x).shape == (2, 3)
print(result.plan.explain())
```

The hidden width shrinks from 8 to 6, reducing the model from 67 to 51 parameters.
The first layer's weight rows and bias entries are removed together with the
second layer's matching input columns. Model input and output dimensions remain
unchanged.

`prune()` selects and applies a plan in place. You can also inspect a plan before
applying it, or specify exactly which positions to remove. The
[getting-started tutorial](docs/getting-started.md) covers manual pruning,
training after pruning, and checkpoint restoration.

For another pruning round, rebuild the dependency graph. Create a new optimizer
before continuing training, because physical pruning replaces parameters.

## Examples

Start with the small API examples:

| Example | What it demonstrates |
| --- | --- |
| [Dependency analysis](examples/dependency.py) | Inspect the effects of a removal without modifying the model |
| [Two pruning rounds](examples/pruning.py) | Automatic selection, training, and graph rebuilding |
| [Custom operator](examples/custom_rule.py) | Declare structural relations for a custom module |
| [Fused attention](examples/fused_attention.py) | Prune attention groups and restore a compact checkpoint |

The [pretrained ImageNet workflows](examples/workflows/README.md) provide complete
pruning and fine-tuning examples for ResNet, ViT, and ConvNeXt models. They cover
magnitude and Taylor pruning, FPGM, sparse training, iterative pruning, VBP,
Isomorphic Pruning, OSSCAR, and ViT attention-head and FFN pruning. Each workflow
reports validation accuracy, parameter counts, MACs, and latency.

Model choices and pruning scopes vary by method. See
[method selection and adaptations](docs/workflow-methods.md) for the implemented
algorithms and their research references. These examples do not claim to
reproduce published benchmark results.

## Documentation

| Topic | Guide |
| --- | --- |
| First pruning operation | [Getting started](docs/getting-started.md) |
| Supported models and operators | [Model support](docs/model-support.md), [operator coverage](docs/operator-coverage.md) |
| Budgets, scoring, and custom pruning policies | [Pruning](docs/pruning-design.md) |
| Sparse training components | [Sparse training](docs/sparse-training.md) |
| Saving and restoring compact models | [Persistence](docs/persistence.md) |
| Parameters, MACs, and latency | [Measurement](docs/measurement.md) |
| Development, testing, and PyPI releases | [Development guide](docs/development.md) |

The [documentation index](docs/index.md) also links to the architecture and
dependency-graph references.

## Model support

Support depends on the model's operator forms and the dimensions being pruned.
The graph is captured with PyTorch FX; tensor-dependent Python control flow and
unsupported structural transformations are reported explicitly. Consult the
[model support contract](docs/model-support.md) before adapting a new architecture.

Pruning changes the model's computation. Evaluate the compact model on your task
and fine-tune as needed; the library leaves the loss, optimizer, and training loop
under your control.

## Contributing

Bug reports, operator extensions, and pruning workflows are welcome. Include a
minimal model and representative inputs when reporting an issue. Follow the
[development guide](docs/development.md) for environment setup and the
[testing guide](docs/testing.md) for regression tests and public API checks.

## License

[MIT](LICENSE).
