"""
Backward-compatibility shim — use
:mod:`mbc.simulation.continuous_discrete_sdae_simulator` directly.
"""

from .continuous_discrete_sdae_simulator import (  # noqa: F401
    ContinuousDiscreteSDAESimulator as SDAESimulator,
    ContinuousDiscreteSDAESimulatorParams,
)

__all__ = ["SDAESimulator", "ContinuousDiscreteSDAESimulatorParams"]
