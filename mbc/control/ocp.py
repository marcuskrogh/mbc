"""
Backward-compatibility shim: re-exports from ``discrete_linear_ocp``.

``OptimalControlProblem`` is an alias for :class:`DiscreteLinearOCP`.
New code should import from :mod:`mbc.control.discrete_linear_ocp` directly.
"""

from .discrete_linear_ocp import DiscreteLinearOCP, _build_D_diff, _shift_warm_start

# Backward-compatible alias
OptimalControlProblem = DiscreteLinearOCP

__all__ = [
    "DiscreteLinearOCP",
    "OptimalControlProblem",
    "_build_D_diff",
    "_shift_warm_start",
]
