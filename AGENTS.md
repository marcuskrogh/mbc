# AGENTS.md

## Cursor Cloud specific instructions

`mbc` is a pure-Python numerical library (model-based control: modelling, state
estimation, MPC/OCP, identification, simulation). There are **no services,
databases, servers, or external infrastructure** — everything runs in-process.
"Development mode" means: install editable (`pip install -e .`), then run the
test suite, the demo scripts, or import `mbc` in Python. See `README.md` for the
public API and quick-start examples, and `pyproject.toml` for dependencies.

The startup update script installs the package editable into the **system**
Python with `--break-system-packages` (PEP 668 is enforced here, so a plain
`pip install` fails without that flag). No virtualenv is used; run tools with
`python3`.

- Tests: `python3 -m pytest tests/`
- Run a demo: demo scripts in `scripts/` call `plt.show()`; run them headless
  with `MPLBACKEND=Agg python3 scripts/<name>.py` (no display in cloud).
- No linter/formatter is configured (no ruff/flake8/black/mypy).

### Known pre-existing test failures (not caused by setup)

On a clean checkout, 280 tests pass and **13 fail** (5 skipped). The failures are
all in `tests/test_ocp.py` and `tests/test_input_linear_cost.py` and stem from a
matmul shape bug in `mbc/control/discrete_linear_ocp.py` (an empty disturbance
operand `D`). They are unrelated to environment setup — do not assume you broke
something if you see them.

### Optional extras

- `pip install -e ".[osqp]"` — alternative QP backend (`osqp`), installed by default in the update script.
- `pip install -e ".[ipopt]"` — `cyipopt` NLP backend; additionally needs the
  system library `coinor-libipopt-dev`. Not required (SciPy NLP backend is the default).
