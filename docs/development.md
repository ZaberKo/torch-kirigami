# Development and releases

This guide covers the local development loop and publishing a release to PyPI.
For test selection, CUDA validation, and review expectations, see the
[testing guide](testing.md).

## Set up a development environment

Run these commands from the repository root with Python 3.10 or newer available:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --torch-backend=auto --group dev -e .
```

The editable install uses the PyTorch backend appropriate for the host. Add
optional ImageNet workflow dependencies as described in the
[workflow guide](../examples/workflows/README.md). Use `uv pip install` for
additional packages; environment synchronization can remove separately
installed workflow dependencies.

Run the checks relevant to a change before committing it:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
.venv/bin/python examples/dependency.py
.venv/bin/python examples/custom_rule.py
.venv/bin/python examples/pruning.py
.venv/bin/python examples/fused_attention.py
```

The [CI workflow](../.github/workflows/ci.yml) checks Python 3.10 with PyTorch
2.6 and Python 3.12 with PyTorch 2.14 on CPU. It also runs Ruff, the four small
examples, and a package build. A local test run does not replace checking the
result of the pushed commit on GitHub.

## Prepare a release

The package version is `[project].version` in [pyproject.toml](../pyproject.toml).
For every release after the first, change it to a new version, commit the change,
push `main`, and wait for its CI run to pass. PyPI does not allow replacing an
uploaded file with the same name and version.

Before tagging, check that:

1. The release changes are committed and pushed to `main`, and the latest
   [CI run](https://github.com/ZaberKo/torch-kirigami/actions/workflows/ci.yml)
   for that commit passed.
2. `git status --short` has no output, and `main` points to the commit you want
   to release.
3. The version in `pyproject.toml` has not already been published on
   [PyPI](https://pypi.org/project/torch-kirigami/).

Optionally build and inspect the distributions locally. Use a new temporary
directory so older files in the ignored `dist/` directory are not checked by
mistake. The GitHub workflow builds its own distributions from the tagged commit;
it does not upload these local files.

```bash
uv pip install --python .venv/bin/python twine
DIST_DIR="$(mktemp -d)"
uv build --out-dir "$DIST_DIR"
.venv/bin/python -m twine check --strict "$DIST_DIR"/*.whl "$DIST_DIR"/*.tar.gz
```

## Configure Trusted Publishing once

The [publish workflow](../.github/workflows/publish.yml) uses GitHub's `pypi`
environment and PyPI Trusted Publishing. On GitHub, create the `pypi`
environment under **Settings → Environments**. A deployment tag rule of `v*`
allows version tags used by this workflow. No PyPI API token or environment
secret is needed.

For the first release, sign in to PyPI and open **Publishing → Add a new pending
publisher**. Select GitHub Actions and use these values:

| Field | Value |
| --- | --- |
| PyPI project name | `torch-kirigami` |
| GitHub owner | `ZaberKo` |
| Repository | `torch-kirigami` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

After selecting **Add**, confirm the entry appears under **Pending publishers**.
It does not yet create a PyPI project or reserve the name. The first successful
upload creates the project and converts the entry to an active publisher.
Subsequent releases use the same publisher. See the
[PyPI first-project guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for the account-side steps.

## Tag and publish

The release workflow starts when a version tag is pushed. Its tag must equal
`v` followed by `[project].version`; for example, version `0.1.0` uses
`v0.1.0`. A GitHub Release page is optional for this workflow.

From the repository root, after the release commit's CI passes:

```bash
git switch main
git pull --ff-only
git status --short
git log -1 --oneline
VERSION="$(uv version --short)"
echo "Tagging v${VERSION}"
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"
```

Check the status and commit shown by the two inspection commands before
creating the tag. `uv version --short` reads the current version from
`pyproject.toml`, so the tag points to the current `main` commit without copying
a commit hash or editing the command for each release. Pushing it starts the
release workflow and can publish the package to PyPI. Do not reuse a tag or
package version for a different build.

Watch [Publish to PyPI](https://github.com/ZaberKo/torch-kirigami/actions/workflows/publish.yml)
until its `tests`, `build`, and `publish` jobs all pass. The build job checks that
the tag matches the package version and validates the wheel and source archive.
The publish job uses PyPI's short-lived OIDC credentials; it does not need a
password or API token. If it fails, inspect that job's log and the five Trusted
Publisher fields before trying another release.

After success, verify that the new version and both distribution files appear
on the [PyPI project page](https://pypi.org/project/torch-kirigami/). For the
first release, confirm that the publisher has moved from **Pending publishers**
to the project's active publishing configuration.
