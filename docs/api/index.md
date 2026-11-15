# API reference

Autodoc2 generates these pages from module source and docstrings.
The reference includes internal modules as well as the public API.
Import public objects from `torch_kirigami`, `torch_kirigami.pruning`, or `torch_kirigami.sparsity`.
The object index includes both public import paths and defining module paths.
The guides explain how to use these objects together.

- [Dependency graph](../apidocs/torch_kirigami/torch_kirigami.dep_graph.dependency_graph.md)
- [Constraints](../apidocs/torch_kirigami/torch_kirigami.dep_graph.contracts.md)
- [Operators](../apidocs/torch_kirigami/torch_kirigami.dep_graph.operation.md)
- [Pruner](../apidocs/torch_kirigami/torch_kirigami.pruning.pruner.md)
- [Metrics](../apidocs/torch_kirigami/torch_kirigami.pruning.metrics.md)
- [Sparse losses](../apidocs/torch_kirigami/torch_kirigami.sparsity.regularizers.md)

Torch Prisma owns the [measurement API](../guides/measurement.md).

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
