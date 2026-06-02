"""Numerical simulation sub-package for continuous-discrete SDE/SDAE models."""

from ._base import (
    SimulatorParams,
    ContinuousDiscreteSimulator,
    ContinuousDiscreteDAESimulator,
)
from .continuous_discrete_sde_simulator import (
    ContinuousDiscreteSDESimulatorParams,
    ContinuousDiscreteSDESimulator,
)
from .continuous_discrete_sdae_simulator import (
    ContinuousDiscreteSDAESimulatorParams,
    ContinuousDiscreteSDAESimulator,
)

# Backward-compatible aliases
SDESimulator = ContinuousDiscreteSDESimulator
SDAESimulator = ContinuousDiscreteSDAESimulator

__all__ = [
    # Abstract bases
    "SimulatorParams",
    "ContinuousDiscreteSimulator",
    "ContinuousDiscreteDAESimulator",
    # Parameter structures
    "ContinuousDiscreteSDESimulatorParams",
    "ContinuousDiscreteSDAESimulatorParams",
    # Simulators
    "ContinuousDiscreteSDESimulator",
    "ContinuousDiscreteSDAESimulator",
    # Backward-compatible aliases
    "SDESimulator",
    "SDAESimulator",
]
