# torch-kirigami documentation

torch-kirigami analyzes structural dependencies in PyTorch models.
It uses those dependencies to plan pruning and reduce tensor dimensions.
Training and task evaluation remain under the caller's control.

The diagrams use Mermaid 11, including block diagrams for tensor coordinates.

## Start here

- [Getting started](getting-started.md): build a graph, inspect removals, prune a model, and restore a checkpoint.
- [Architecture](architecture.md): understand the main components and their responsibilities.
- [Model support](model-support.md): check capture requirements and unsupported cases.

## API guides

- [Dependency analysis](dependency-graph.md): tensor coordinates, propagation, constraints, and analysis results.
- [Pruning](pruning.md): candidates, budgets, metrics, strategies, and model changes.
- [Operator support](operator-support.md): supported operator forms and custom rules.
- [Persistence](persistence.md): saved plans, compact checkpoints, and training state.
- [Sparse training](sparse-training.md): penalties, gates, parameter operations, and repeated pruning.
- [Measurement](measurement.md): parameter counts, MACs, and inference latency.

## Workflows and development

[Structural pruning workflows](../examples/workflows/README.md) provide ImageNet experiment entries, research sources, and implementation differences.
[Development](development.md) covers local checks, test selection, and releases.
[Verification](verification.md) explains what each kind of evidence establishes.

[Development notes](dev-notes/index.md) contain historical investigations.
They do not define current API contracts.
