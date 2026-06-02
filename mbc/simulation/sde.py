"""
Backward-compatibility shim — use
:mod:`mbc.simulation.continuous_discrete_sde_simulator` directly.
"""

from .continuous_discrete_sde_simulator import (  # noqa: F401
    ContinuousDiscreteSDESimulator as SDESimulator,
    ContinuousDiscreteSDESimulatorParams,
)

__all__ = ["SDESimulator", "ContinuousDiscreteSDESimulatorParams"]
