# Development and releases

Use Python 3.10 or newer and the flat `torch_kirigami/` package layout.
Keep dependency analysis, pruning policy, and model changes separate.
Use public PyTorch APIs.

## Environment

Run these commands from the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --torch-backend=auto --group dev -e .
```

The installation selects the PyTorch backend for the host.
Install additional packages with `uv pip install`.
Do not use `uv sync` or `uv run` in this environment.
Those commands can remove separately installed workflow dependencies.

Install optional ImageNet packages when a change affects the workflow examples:

```bash
uv pip install --python .venv/bin/python --torch-backend=auto \
  -r examples/workflows/requirements-examples.txt
```

## Code and API contracts

Use the project's Ruff configuration and Google-style docstrings.
Document accepted inputs, ownership, return values, model changes, and failure conditions.
Explain non-obvious reasoning in comments.

Keep imports explicit and acyclic.
Define shared records below the modules that consume them.
Do not conceal reverse dependencies with local imports or `TYPE_CHECKING`.
Keep library code independent of workflow examples.

Update APIs and saved schemas directly during initial development.
Do not add schema versions or migration layers.
Preserve unrelated changes and reference repositories under `tmp/`.

## Tests and checks

Run Ruff and the relevant pytest suites directly:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest tests/dep_graph tests/architecture
.venv/bin/pytest tests/pruning tests/persistence tests/integration/test_lifecycle.py
```

The [test organization guide](../tests/README.md) defines directory ownership and shared fixtures.

Choose suites for the changed contract:

- Region and relation changes need independent coordinate references.
- Capture and operator changes need public graph and pruning tests.
- Budget and selection changes need target, constraint, and failure cases.
- Model changes and checkpoints need round trips and failure-state checks.
- Sparse training needs independent loss and gradient references.
- Workflow changes need public planning, application, training, and restoration checks.

A direct operator test does not establish execution or checkpoint support.
Cross-layer changes must use public APIs on a model.
Include a valid alternative when testing an unsupported request.
Check model state after a rejected operation.

Run the full suite after changes to shared capture, planning, or execution contracts:

```bash
.venv/bin/pytest
```

The small examples check API composition without ImageNet data:

```bash
.venv/bin/python examples/dependency.py
.venv/bin/python examples/custom_rule.py
.venv/bin/python examples/pruning.py
.venv/bin/python examples/fused_attention.py
```

For numerical tests, derive the reference independently of the implementation.
A masked dense model is a valid reference only when masking preserves the relevant calculation.
Removing normalization or attention dimensions can change that calculation.
Use an independent compact reference for those cases.

The [CPU performance probe](../experiments/planning_sparsity_benchmark.py) measures planning and sparse operations without downloaded data.
The [recorded results](../experiments/planning_sparsity_results.json) include baseline timings and selected-key digests.
Compare each strategy against its own baseline. The two rows use different metrics.

The [residual benchmark](../experiments/residual_planning_benchmark.py) also includes repeated domains along a shared residual trunk.
Its [recorded results](../experiments/residual_planning_results.json) compare commit `525ba34` with the optimized implementation.
The model has 64 channels, eight residual blocks, and 1,088 candidates.
Both checkouts use PyTorch 2.14, one CPU thread, seed 7, and the same local random weights.
Both strategies use `Magnitude`, a global channel ratio of 0.05, and Conv2d granularity 8.

| Phase | Before | After |
| --- | ---: | ---: |
| Candidate discovery | 5.488 s | 1.413 s |
| Filter the discovered space again | 5.450 s | 0.00134 s |
| Greedy planning | 11.922 s | 2.107 s |
| DynamicGreedy planning | 64.326 s | 27.564 s |
| Extract all 1,088 parameter groups | 4.277 s | 0.675 s |
| Zero one prepared union, median | 7.058 ms | 4.730 ms |
| Zero multiple groups, median | 8.501 ms | 8.535 ms |

Discovery and group extraction each reduce full propagations from 576 to 64.
Filtering the same discovered space performs one freshness check and no propagation.
Greedy propagations fall from 1,673 to 585. DynamicGreedy propagations fall from 8,033 to 3,873.
Both strategies retain the same selected keys, trial counts, and final widths.
Applying the Greedy plan produces identical parameter shapes and output bytes in both checkouts.

Instrumentation contributes to these host measurements. The results do not guarantee runtime.
Sparse medians use seven subsequent calls. Other phases use one run per checkout.
Multiple-group CPU writes have no measured speedup.
Their combined finite-value tests avoid per-region device synchronization.
Passing an existing union avoids reconstructing that group on every training step.
Regularizers retain their required freshness validation and have no claimed speedup in this comparison.

To repeat the benchmark, use the same script and interpreter with each checkout on `PYTHONPATH`:

```bash
PYTHONPATH=. .venv/bin/python experiments/residual_planning_benchmark.py --output result.json
```

Native proofs permit structural reuse only.
Do not reuse parameter values or custom metric scores across training updates.
Dynamic selection must still rescore after each accepted addition.
Final planning checks remain independent of strategy callbacks.
Custom constraints, incomplete influence, and unsupported operators retain their existing behavior.

### CPU, CUDA, and CI

The device fixture runs participating tests on CPU and CUDA.
CUDA tests skip when a CUDA build or accessible GPU is unavailable.
A skipped CUDA case is not GPU validation.

Require CUDA in a configured GPU environment:

```bash
.venv/bin/pytest --require-cuda
```

The regular CI tests Python 3.10 with PyTorch 2.6 and Python 3.12 with PyTorch 2.14 on CPU.
The development pair also runs Ruff, the small examples, and a package build.
Manual compatibility jobs cover intermediate PyTorch releases.
A separate manual workflow runs tests on a configured GPU runner.

Workflow definitions describe intended checks.
Report remote success only after checking the run for the relevant commit.
Optional workflow dependencies are not present in every CI environment.
The [verification guide](verification.md) explains the limits of each evidence level.

### Documentation checks

Keep formal documentation in English.
Use short sentences and consistent API terms.
Retain conditions and uncertainty when simplifying a contract.

Check documentation changes with these steps:

1. Check local links and heading anchors.
2. Execute complete Python examples.
3. State assumptions for partial code examples.
4. Render retained diagrams.
5. Check diagrams visually for layout and meaning.

Do not turn test counts or successful capture into broad support claims.
Distinguish synthetic tests, pretrained execution checks, and task accuracy measurements.

## Releases

Publishing a GitHub Release starts the release workflow.
Creating or pushing a tag does not start this workflow.
The release tag must match the package version in `pyproject.toml`.
For example, version `0.1.3` requires tag `v0.1.3`.
Use a new version for each uploaded distribution.
PyPI does not permit replacing a distribution with the same filename.

Before creating a release tag:

1. Set the package version to an unpublished value.
2. Commit the release changes.
3. Push the release commit.
4. Check that its CI run passes.
5. Check that the working tree is clean.

Optionally check distributions in a new temporary directory:

```bash
uv pip install --python .venv/bin/python twine
KIRIGAMI_DIST_DIR="$(mktemp -d)"
uv build --out-dir "$KIRIGAMI_DIST_DIR"
.venv/bin/python -m twine check --strict "$KIRIGAMI_DIST_DIR"/*.whl "$KIRIGAMI_DIST_DIR"/*.tar.gz
```

### Trusted Publishing

The workflow uses the GitHub environment `pypi` and PyPI Trusted Publishing.
Configure a GitHub publisher for these values:

- Project: `torch-kirigami`.
- Owner and repository: `ZaberKo/torch-kirigami`.
- Workflow: `publish.yml`.
- Environment: `pypi`.

For the first upload, create a pending publisher in PyPI.
The upload creates the project and activates the publisher.
Later releases use that publisher.
Follow the [PyPI setup instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) for account configuration.

### Create the tag and publish the release

Run these commands after the release commit passes CI:

```bash
git switch main
git pull --ff-only
git status --short
git log -1 --oneline
```

Check the displayed status and commit before creating the tag:

```bash
KIRIGAMI_RELEASE_VERSION="$(uv version --short)"
git tag -a "v${KIRIGAMI_RELEASE_VERSION}" -m "Release v${KIRIGAMI_RELEASE_VERSION}"
git push origin "v${KIRIGAMI_RELEASE_VERSION}"
```

On the GitHub Releases page, create a release for this tag.
Write the title and release notes, select the appropriate prerelease setting, then publish the release.
Saving a draft does not start the workflow.

Alternatively, write the release notes in a local file and publish with GitHub CLI:

```bash
gh release create "v${KIRIGAMI_RELEASE_VERSION}" --verify-tag \
  --title "v${KIRIGAMI_RELEASE_VERSION}" --notes-file release-notes.md
```

Add `--prerelease` for a prerelease.
The workflow uses the [`release: published` event](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release) for both stable releases and prereleases.
It tests the tagged commit, checks the version, builds distributions, and uploads them to PyPI.
It then attaches the same distributions to the published GitHub Release.
It does not generate release notes, read tag messages, or change the release title or prerelease setting.
Editing release notes after publication does not trigger another upload.

Check the workflow result, the PyPI version, and the GitHub Release assets.
If only the `release-assets` job fails, rerun that job.
This retries the asset upload without repeating the successful PyPI upload.

The [workflow guide](../examples/workflows/README.md) covers data and pretrained experiments.
