# Development

- Use Python 3.10+, uv for dependencies, and the flat `torch_kirigami/` package layout.
- Follow the Ruff configuration in `pyproject.toml`; use Google-style docstrings. Document public API contracts and explain non-obvious reasoning in comments.
- Use pytest. Add focused regression tests for behavioral changes; keep numerical references independent of the implementation. Cross-layer changes need public build/plan/apply or save/load tests, including valid alternatives and failure-state checks; direct rule tests alone are insufficient.
- Use public, mature PyTorch APIs. Keep dependency analysis separate from pruning policy and model mutation; report unsupported cases explicitly.
- Keep module imports explicit and acyclic. Put shared records/contracts below their consumers; do not hide reverse dependencies with `TYPE_CHECKING` or local imports.
- Run `uv run --locked ruff check .`, `uv run --locked ruff format --check .`, and relevant `uv run --locked pytest` tests. Refresh `uv.lock` when dependencies change.
- This project is in initial development: update APIs and persisted schemas directly; do not add format versions, migration layers, or backward-compatibility shims.
- Keep changes focused, update affected examples/docs, and leave reference repositories under `tmp/` untouched.
