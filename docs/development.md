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
.venv/bin/pytest tests/core tests/graph tests/architecture
.venv/bin/pytest tests/pruning tests/persistence tests/integration/test_lifecycle.py
```

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

Pushing a tag beginning with `v` starts the release workflow.
The tag must match the package version in `pyproject.toml`.
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

### Tag and check the release

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

Pushing the tag starts the release workflow.
It runs tests, checks the version, builds distributions, and uploads them to PyPI.
It then creates a GitHub Release with the same distributions.
Development and prerelease versions become GitHub prereleases.

Check the workflow result, the PyPI version, and the GitHub Release assets.
If only the GitHub Release job fails, rerun that job.
The job can resume a draft without repeating the PyPI upload.

The [workflow guide](../examples/workflows/README.md) covers data and pretrained experiments.
