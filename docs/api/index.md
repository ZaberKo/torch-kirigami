# API reference

These pages are generated from the source docstrings. They cover internal modules
as well as the public API. Import public objects from one of three packages:

| Package | Contents |
| --- | --- |
| `torch_kirigami` | Dependency graph, coordinates, relations, constraints, operator rules, errors, and `config`. |
| `torch_kirigami.pruning` | `Pruner`, candidates, budgets, metrics, strategies, plans, and checkpoints. |
| `torch_kirigami.sparsity` | Regularizers, gates, parameter operations, cumulative budgets, and schedules. |

The object index lists both the public import path and the defining module of
each object. The guides explain how the objects work together.

Good starting points:

- [Dependency graph](../apidocs/torch_kirigami/torch_kirigami.dep_graph.dependency_graph.md)
- [Constraints and impacts](../apidocs/torch_kirigami/torch_kirigami.dep_graph.contracts.md)
- [Operator rules](../apidocs/torch_kirigami/torch_kirigami.dep_graph.operation.md)
- [Pruner](../apidocs/torch_kirigami/torch_kirigami.pruning.pruner.md)
- [Metrics](../apidocs/torch_kirigami/torch_kirigami.pruning.metrics.md)
- [Strategies](../apidocs/torch_kirigami/torch_kirigami.pruning.strategies.md)
- [Regularizers](../apidocs/torch_kirigami/torch_kirigami.sparsity.regularizers.md)
- [Configuration](../apidocs/torch_kirigami/torch_kirigami.settings.md)

Measurement functions belong to Torch Prisma. See the
[measurement guide](../guides/measurement.md).

```{toctree}
:hidden:
:maxdepth: 1

../apidocs/index
```

```{toctree}
:hidden:
:glob:

../apidocs/torch_kirigami/*
```
