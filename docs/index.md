# Torch Kirigami

Torch Kirigami analyzes structural dependencies in PyTorch models and uses those
dependencies to plan and apply structured pruning. You control the model,
selection policy, training, and evaluation.

## Start here

1. [Getting started](getting-started.md): inspect a small model, prune it, and restore a checkpoint.
2. [Model support](guides/model-support.md): check capture conditions and the scope of unsupported operations.
3. [Pruning guide](guides/pruning.md): choose candidates, scores, and budgets.

## Explore the library

- [Dependency graph](reference/dependency-graph.md) explains coordinates, relations, constraints, and propagation.
- [Operator support](reference/operator-support.md) describes built-in rules and custom extensions.
- [Sparse training](guides/sparse-training.md) covers losses, gates, and parameter operations.
- [Persistence](guides/persistence.md) covers pruning plans and compact checkpoints.
- [Measurement](guides/measurement.md) explains the Torch Prisma integration.
- [API reference](api/index.md) documents modules from their source docstrings.

The [ImageNet workflow examples](https://github.com/ZaberKo/torch-kirigami/blob/main/examples/workflows/README.md) describe executable experiments and their research scope.
The [architecture](development/architecture.md) and [development guide](development/index.md) explain component ownership and contribution checks.

```{toctree}
:caption: Start here
:maxdepth: 1
:hidden:

getting-started
guides/model-support
```

```{toctree}
:caption: User guides
:maxdepth: 1
:hidden:

guides/pruning
guides/sparse-training
guides/persistence
guides/measurement
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

reference/dependency-graph
reference/operator-support
api/index
```

```{toctree}
:caption: Development
:maxdepth: 1
:hidden:

development/architecture
development/index
development/verification
```
