# AGENTS.md

## Cursor Cloud specific instructions

`mbc` is a pure-Python model-based-control **library** (no runnable app). Dependencies
(`numpy`, `scipy`, `highspy`, plus `pytest`/`matplotlib`/`osqp`) are installed by the
Cursor Cloud update script via an editable install (`pip install -e .`), so `import mbc`
resolves to this working tree. The two sibling repos (`HeatingAssistant`,
`ChargingAssistant`) are also wired to this same editable `mbc`, so edits here are picked
up by them immediately.

- **Tests** (from repo root): `python3 -m pytest tests/`. There is no lint/type-check
  tooling and no CI in this repo.
- `pytest` is not on `PATH`; always invoke it as `python3 -m pytest`.
- **Demo scripts** live in `scripts/`. The `*_visual.py` ones need `matplotlib` and call
  `plt.show()`; with no display, run them headless via `MPLBACKEND=Agg` (then `show()`
  is a no-op — patch it to `savefig` if you want an image). The `*_benchmark.py` scripts
  (e.g. `scripts/qp_formulation_benchmark.py`) print to the terminal and need no display.
- **Known pre-existing failures (not environment-related):** ~13 OCP/QP tests fail on
  `main`. The default solver is `highs`, whose `auto` formulation is `condensed`; the
  condensed builder in `mbc/control/discrete_linear_ocp.py` does `CL @ D` and breaks when
  a model has `nd == 0` but a length-`N` `D` is supplied (shape mismatch). The OSQP
  (`sparse`) path handles the same case fine. Do not "fix" this as part of env setup.
