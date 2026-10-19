# Verification scope and evidence

Use this page to find representative evidence for a contract. A test proves only the forms and lifecycle that it checks. It does not establish general model compatibility.

## Verification levels

| Level | Question |
| --- | --- |
| Algebra | Do coordinate sets, constraints, and score formulas match independent references? |
| Operator rules | Do declared operators propagate the intended coordinates and effects? |
| Model lifecycle | Do capture, planning, application, backward, and restoration work together? |
| Task evaluation | Does the method achieve the claimed quality on the target data? |

A registered operator-entry test covers a rule. It does not establish complete model lifecycle support. Read the [operator support guide](operator-support.md) for supported forms and conditions.

## Representative test areas

The [test organization guide](../tests/README.md) defines directory ownership. Use the feature directory as the starting point:

- `tests/dep_graph/` covers coordinates, constraints, propagation, and freshness. Its `capture/` and `operators/` directories cover capture and operator contracts.
- `tests/pruning/` and `tests/persistence/` cover selection, budgets, plans, application, and checkpoints.
- `tests/sparsity/` and lifecycle tests cover sparse losses, gates, projections, and iterative budgets.
- `tests/measurement/` covers complexity, latency, and measurement isolation.
- `tests/workflows/` covers executable examples and their shared infrastructure.
- `tests/integration/` covers composed networks and contracts that span independent features.

Inspect the individual test before relying on a claim. Cross-layer changes need public lifecycle coverage. Use an independent numerical reference when checking formulas. For rejected changes, check preserved state and include a valid alternative when one exists.

## Workflow evidence

Pretrained workflow tests use small synthetic datasets and replace weight loading during automated runs. They check requested weight enums, workflow stages, model computation, compact references, backward behavior, and checkpoint restoration. They do not establish ImageNet accuracy or fine-tuning recovery.

Real accuracy claims require the pretrained workflow, the actual dataset, and reported sample counts and settings. Distinguish those results from synthetic workflow tests. Compiler measurements must name their backend. Eager-backend checks and default Inductor measurements provide different evidence.

## Devices and CI

The `execution_device` fixture runs participating tests on CPU and CUDA when a CUDA build and accessible GPU are available. Skipped CUDA tests do not establish GPU validation. The `--require-cuda` option fails when CUDA is unavailable.

CI tests Python 3.10 with PyTorch 2.6 and Python 3.12 with PyTorch 2.14 on CPU. The development-version job also runs Ruff, four examples, and a package build. The compatibility workflow tests intermediate PyTorch releases when manually dispatched. The CUDA workflow needs a configured self-hosted GPU runner. Workflow files describe intended checks. They do not prove a remote run passed.

## Reviewing new contracts

Trace each new invariant to an assertion that can fail independently. Add public build, plan, apply, or save/load coverage when a change crosses layers. A direct rule test alone does not establish lifecycle support.

Keep the [contract inventory](../tests/architecture/contract_inventory.json) current when adding a public export, class, or built-in operator registration. Its guard tests check navigation coverage. They do not establish the quality of each assertion.
