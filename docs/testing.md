# Development and testing

This guide describes how to develop and validate a change. It is not a historical log of local test runs. Use the [verification guide](testing-coverage.md) to locate the tests that support a particular contract.

## Development environment

From the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --torch-backend=auto --group dev -e .
ruff check .
ruff format --check .
python -m pytest
```

The package uses a flat `torch_kirigami/` layout, Python 3.10+, and PyTorch 2.6+.
Runtime dependencies remain limited to PyTorch. The development group supplies
pytest, Ruff, and NumPy. Install packages with `uv pip install` and invoke tools
from the activated environment directly, or use their `.venv/bin/` paths.
Environment synchronization can remove separately installed workflow packages.

The PyTorch backend is selected for the host during installation. CPU
compatibility jobs select CPU wheels explicitly in dedicated test environments;
local development does not force a CPU-only package source.

Optional torchvision/ImageNet workflow tests require the example dependencies. Install them in the root virtual environment as described in the [workflow README](../examples/workflows/README.md). Those tests skip when the optional packages are unavailable.

## Change-to-test workflow

Use Google-style docstrings for public APIs. State accepted inputs, ownership, mutation behavior, return values and failure conditions. Explain non-obvious reasoning in comments rather than repeating the code.

Keep imports explicit and acyclic. Shared records belong below their consumers. Do not conceal a reverse dependency with a local import or `TYPE_CHECKING`. The library must not depend on workflow examples.

## Select an appropriate test scope

| Area changed | Start with | Add when the change crosses layers |
| --- | --- | --- |
| Region algebra or relations | `tests/core/` | Coordinate compositions and compact-network tests |
| FX capture, effects or bindings | `tests/capture/` | Capture equivalence, lifecycle and effect-contract tests |
| Graph propagation or freshness | `tests/graph/` | Shared constraints and planning integration |
| An operator rule | Its file in `tests/operators/` | A model using public build/plan/apply, backward and persistence |
| Ranking, budgets or selection | `tests/pruning/` | Multiround lifecycle and cumulative-budget tests |
| Tensor replacement or checkpointing | `tests/persistence/` | Failure-state, transaction and cross-process tests |
| Sparse penalties, gates or projections | `tests/sparsity/` | Sparse workflows and compact numerical references |
| MACs or latency | `tests/integration/test_measurement.py` | Workflow metric output and compiler behavior |
| Workflow stages or data loading | `tests/integration/test_pretrained_examples.py` | Real dataset evaluation when making accuracy claims |

A typical focused run is:

```bash
python -m pytest tests/core tests/graph tests/architecture
python -m pytest tests/pruning tests/persistence tests/integration/test_lifecycle.py
```

Choose the tests relevant to the actual change. Run the full suite after changes to shared capture, planning or execution contracts.

The ImageNet workflow tests exercise each entry's task-specific CLI and full
pruning/training/checkpoint path on both devices. Launch tests copy only the
entry and its three shared support modules, so workflows cannot depend on each
other. Shared data loading, label alignment and evaluation are tested directly;
the shared ViT adapter is compared with torchvision with and without gates.
Workflow fixtures use small synthetic datasets and random test weights;
passing these checks does not establish ImageNet accuracy.

## Numerical and failure-state references

Good references derive expected results independently of the implementation under test:

- Enumerate small coordinate sets to verify symbolic region operations.
- Compute regularizer formulas directly and use gradcheck at nonzero differentiable points.
- Compare an explicitly masked dense network to a compact network only where equivalence is mathematically justified.
- Use an independently constructed compact normalization domain when removing dimensions changes normalization statistics.
- Check parameter identity, aliases, values, configuration and gradients where a rejected mutation could otherwise leave partial state.
- Include a valid alternative when testing that an unsupported branch or selection is rejected.

A passing direct rule test does not establish capture, lowering or checkpoint support. Cross-layer changes must exercise public APIs.

## CPU, CUDA and compatibility

The `execution_device` fixture parametrizes participating tests over CPU and CUDA. CUDA cases are skipped unless a CUDA PyTorch build and an accessible GPU are available. A CPU run's skipped CUDA cases are not GPU validation.

In an environment already configured with CUDA PyTorch:

```bash
python -m pytest --require-cuda
```

`--require-cuda` fails at session start if CUDA is unavailable. Test setup and tensors for participating cases run on the selected device; installing a CUDA wheel alone is not enough.

| Repository workflow | Purpose |
| --- | --- |
| [CI](../.github/workflows/ci.yml) | Minimum and development CPU pairs, lint, small executable examples and packaging |
| [Compatibility](../.github/workflows/compatibility.yml) | Manually dispatched intermediate PyTorch release matrix |
| [CUDA validation](../.github/workflows/cuda.yml) | Manually dispatched tests on a configured GPU runner |

Workflow definitions describe intended checks, not evidence that a remote run succeeded. Optional ImageNet dependencies are not installed by every CI job; verify the selected environment when reporting workflow coverage.

## Small executable examples and packaging

```bash
python examples/dependency.py
python examples/custom_rule.py
python examples/pruning.py
python examples/fused_attention.py
uv build
```

These small examples demonstrate API and extension mechanics. The [ImageNet workflows](../examples/workflows/README.md) instead use official pretrained models and real held-out data to evaluate task quality.

## Documentation and review

Documentation is plain Markdown with Mermaid diagrams and relative repository links. Keep the [documentation index](index.md) current when adding or moving a page.

A documentation change should check:

1. Local links and anchors resolve.
2. Diagrams parse and use the same terminology as the source.
3. Runnable Python examples use actual public APIs.
4. Partial snippets declare their assumptions.
5. Support claims identify the applicable operator form and validation level.

The machine-readable [contract inventory](../tests/architecture/contract_inventory.json) lives with the architecture tests. Update it when adding a public export, a class or a registered operator entry. Its guard tests verify coverage navigation; they do not establish the quality of every assertion.

When reporting validation, state the command, environment, scope and skipped tests. Separate synthetic workflow checks, actual pretrained-model checks and real ImageNet accuracy measurements. Do not infer one from another.

Preserve unrelated work and reference repositories under `tmp/`. This project updates APIs and persisted schemas directly during initial development; it does not maintain migration or format-version layers.
