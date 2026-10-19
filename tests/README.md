# Test organization

Place tests with the feature that owns the contract. Public build, plan, apply,
backward, and restoration tests can remain in that feature's directory.
Use `integration/` when a scenario spans several independent feature contracts.

| Directory | Scope |
| --- | --- |
| `architecture/` | Imports, registry contracts, and the contract coverage inventory. |
| `dep_graph/` | Coordinates, constraints, graph construction, propagation, and freshness. |
| `dep_graph/capture/` | FX capture, source bindings, metadata, and execution isolation. |
| `dep_graph/operators/` | Built-in operator rules and their public pruning behavior. |
| `pruning/` | Candidate discovery, metrics, strategies, budgets, recipes, and application. |
| `persistence/` | Portable plans, checkpoints, registration identity, and restoration. |
| `sparsity/` | Regularizers, gates, coordinate reuse, projections, and transactional writes. |
| `measurement/` | Complexity, latency, input transfer, and measurement isolation. |
| `workflows/` | Executable examples, research methods, ImageNet infrastructure, and CLI contracts. |
| `integration/` | Composed networks, multiple pruning rounds, and shared lifecycle assumptions. |
| `support/` | Explicitly imported model factories, numerical references, and rule helpers. |

Name files after behavior. Do not create files named after a review round.
Keep one workflow's algorithm tests together. Separate shared data, evaluation,
CLI, and model-adapter tests from executable workflow tests.
Tests must not import helpers from another test module. Put shared helpers in
`support/` and import them explicitly.

The root `conftest.py` owns device selection and global configuration restoration.
The workflow `conftest.py` limits CPU threads for small models and restores the
previous setting after each test. Optional workflow dependencies belong in
workflow modules, not in the root test configuration.

Use independent numerical references and preserve failure-state assertions.
Keep [contract links](architecture/contract_inventory.json) current when moving tests.
For structural moves, compare parameterized collection before and after the change.
The same test count alone does not establish that every case survived.

```bash
.venv/bin/pytest tests/dep_graph tests/architecture
.venv/bin/pytest tests/pruning tests/persistence tests/sparsity
.venv/bin/pytest tests/measurement tests/workflows tests/integration --require-cuda
```
