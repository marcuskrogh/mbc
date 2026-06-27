# AGENTS.md

## Cursor Cloud specific instructions

`mbc` is a pure-Python model-based control library (no servers, databases, or
UI). "Running the application" means importing `mbc` and using it from Python;
"end-to-end testing" means running the `pytest` suite. Install/test commands are
documented in `README.md` (`## Installation`, `## Running Tests`).

### Environment

- Dependencies are installed into a virtualenv at `.venv` (created by the
  startup update script). Activate it with `source .venv/bin/activate`, or call
  tools directly via `.venv/bin/python` and `.venv/bin/pytest`. The system
  Python is PEP 668 "externally managed", so do not `pip install` into it.
- `pytest` is not declared in `pyproject.toml`; it is installed into `.venv` by
  the update script for the dev/test workflow.
- The default QP backend is OSQP, installed via the `[osqp]` extra (the bare
  `pip install -e .` in the README omits it; `highspy` is the core fallback).

### Tests

- Run with `pytest tests/` (or `.venv/bin/pytest tests/`).
- Known pre-existing failures (NOT caused by environment setup): 13 tests in
  `tests/test_ocp.py` and `tests/test_input_linear_cost.py` fail at
  `mbc/control/discrete_linear_ocp.py` (`CL @ D` matmul). Their fixtures declare
  `nd=0` but pass a non-empty disturbance vector `D`, a shape mismatch that is
  independent of dependency versions. As of this writing 278 pass / 13 fail / 5
  skip.

### Lint / build

- No linter or CI is configured (no ruff/flake8/black config, no `.github/`).
- Build uses setuptools via `pyproject.toml`; the editable install
  (`pip install -e`) is all that is needed for development.
