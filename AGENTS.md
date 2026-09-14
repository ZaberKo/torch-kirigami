# Development

- Use Python 3.10+, uv for dependencies, and the flat `torch_kirigami/` package layout.
- Follow the Ruff configuration in `pyproject.toml`; use Google-style docstrings. Document public API contracts and explain non-obvious reasoning in comments.
- Use pytest. Add focused regression tests for behavioral changes; keep numerical references independent of the implementation. Cross-layer changes need public build/plan/apply or save/load tests, including valid alternatives and failure-state checks; direct rule tests alone are insufficient.
- Use public, mature PyTorch APIs. Keep dependency analysis separate from pruning policy and model mutation; report unsupported cases explicitly.
- Keep module imports explicit and acyclic. Put shared records/contracts below their consumers; do not hide reverse dependencies with `TYPE_CHECKING` or local imports.
- Manage dependencies with `uv pip install`; do not use `uv sync` or `uv run`, which can remove separately installed workflow dependencies. Run `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, and relevant `.venv/bin/pytest` tests directly. Select the PyTorch backend for the host; do not pin a CPU-only source for local development.
- This project is in initial development: update APIs and persisted schemas directly; do not add format versions, migration layers, or backward-compatibility shims.
- Prefer straightforward control flow and one definition per invariant. Optimize measured bottlenecks without adding general frameworks or weakening correctness checks.
- Edit workflow examples individually by hand; do not generate them with scripts. Keep task-specific CLI, training and pruning steps in each entry; reuse `imagenet_data.py`, `imagenet_models.py` and `model_metrics.py` for infrastructure. Do not introduce a shared workflow runner.
- Keep changes focused, update affected examples/docs, and leave reference repositories under `tmp/` untouched.
