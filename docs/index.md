# Documentation

torch-kirigami analyzes structural dependencies in PyTorch models and uses those dependencies to plan and execute physical pruning. These documents explain both how to use the library and how its implementation enforces its contracts.

## Choose a reading path

| Goal | Start here | Continue with |
| --- | --- | --- |
| Check whether a model can be pruned | [Model support contract](model-support.md) | [Operator support](operator-coverage.md) |
| Run a first pruning operation | [Getting started](getting-started.md) | [Pruning design](pruning-design.md) |
| Understand the whole system | [Architecture overview](architecture.md) | [Dependency graph design](dependency-graph-design.md) |
| Review dependency analysis | [Dependency graph design](dependency-graph-design.md) | [Operator support](operator-coverage.md) |
| Add an operator or a pruning policy | [Operator support](operator-coverage.md) | [Pruning design](pruning-design.md), [Testing](testing.md) |
| Assemble sparse training or iterative pruning | [Sparse training](sparse-training.md) | [ImageNet workflows](../examples/workflows/README.md) |
| Save plans or compact models | [Persistence](persistence.md) | [Pruning design](pruning-design.md) |
| Measure the resulting model | [Measurement](measurement.md) | [Verification guide](testing-coverage.md) |
| Contribute a change | [Development and testing](testing.md) | [Verification guide](testing-coverage.md) |

## Terms used throughout the documentation

| Term | Meaning |
| --- | --- |
| Structural position | A coordinate in a tensor, such as an output channel or an FFN intermediate feature. |
| Selection | A union of tensor regions to remove, expressed in the current graph's original coordinates. |
| Dependency closure | All additional selections implied by the registered structural relations. |
| Constraint | A condition the complete selection must satisfy, such as balanced groups or a nonempty axis. |
| Requirement | A change that execution must handle or reject, such as updating a module attribute. |
| Candidate | A policy-level removal proposal, possibly containing multiple selections. |
| Logical channel axis | An axis used to count deletions without counting dependent representations repeatedly. |
| Parameter group | The affected parameter regions used by a regularizer or parameter operation. |
| Impact | The result of dependency analysis; it is not an executable plan. |
| Pruning plan | Static, validated structure changes that can be applied to a compatible model. |
| Compact model | The original module hierarchy with physically reduced tensor dimensions and updated attributes. |

## Three different correctness questions

1. **Analysis:** What else must change when these positions are removed?
2. **Execution:** Can those changes be expressed and committed to this model?
3. **Model quality:** How do task accuracy, compute, and measured latency change?

The first two are library contracts. The third requires evaluation on the real task and data. A resolved analysis result does not establish numerical equivalence or preserved accuracy.

All source links refer to the current repository. Private classes are documented to support review; their inclusion is not a promise of a stable public API.
