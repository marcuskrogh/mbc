"""Abstract base classes for all OCP types."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OCP(ABC):
    """
    Abstract base class for all optimal control problems.

    Every concrete OCP must expose the prediction horizon ``N`` and the input
    dimension ``nu`` so that closed-loop controller wrappers can query them
    without knowing the concrete OCP type.
    """

    @property
    @abstractmethod
    def N(self) -> int:
        """Prediction horizon (number of control intervals)."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Input dimension nᵘ."""


class DiscreteLinearOCPBase(OCP):
    """Abstract base for discrete-time linear tracking OCPs.

    Exposes abstract properties for all tuning parameters (Q, R, P, S, rho,
    y_offset, u_min, u_max) with both getters and setters, plus a
    ``solve`` method.
    """

    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """Stage output tracking cost matrix."""

    @Q.setter
    @abstractmethod
    def Q(self, value) -> None: ...

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Stage input cost matrix."""

    @R.setter
    @abstractmethod
    def R(self, value) -> None: ...

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Terminal output tracking cost matrix."""

    @P.setter
    @abstractmethod
    def P(self, value) -> None: ...

    @property
    @abstractmethod
    def S(self) -> np.ndarray | None:
        """Input rate-of-movement cost matrix (None if disabled)."""

    @S.setter
    @abstractmethod
    def S(self, value) -> None: ...

    @property
    @abstractmethod
    def rho(self) -> float:
        """Soft-output penalty weight."""

    @rho.setter
    @abstractmethod
    def rho(self, value) -> None: ...

    @property
    @abstractmethod
    def y_offset(self) -> float:
        """Soft-output band half-width."""

    @y_offset.setter
    @abstractmethod
    def y_offset(self, value) -> None: ...

    @property
    @abstractmethod
    def u_min(self) -> np.ndarray:
        """Lower input bounds."""

    @u_min.setter
    @abstractmethod
    def u_min(self, value) -> None: ...

    @property
    @abstractmethod
    def u_max(self) -> np.ndarray:
        """Upper input bounds."""

    @u_max.setter
    @abstractmethod
    def u_max(self, value) -> None: ...

    @abstractmethod
    def solve(self, x0, D, x_ref, u_prev=None, warm_start=None): ...


class DiscreteLinearisedOCPBase(DiscreteLinearOCPBase):
    """Abstract base for linearised discrete-time OCPs (deviation coordinates)."""
    pass


class ContinuousLinearOCPBase(OCP):
    """Abstract base for continuous-discrete linear OCPs.

    Same abstract properties as DiscreteLinearOCPBase but semantically
    separate (the model is a continuous-discrete system).
    """

    @property
    @abstractmethod
    def Q(self) -> np.ndarray:
        """Stage output tracking cost matrix."""

    @Q.setter
    @abstractmethod
    def Q(self, value) -> None: ...

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Stage input cost matrix."""

    @R.setter
    @abstractmethod
    def R(self, value) -> None: ...

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Terminal output tracking cost matrix."""

    @P.setter
    @abstractmethod
    def P(self, value) -> None: ...

    @property
    @abstractmethod
    def S(self) -> np.ndarray | None:
        """Input rate-of-movement cost matrix (None if disabled)."""

    @S.setter
    @abstractmethod
    def S(self, value) -> None: ...

    @property
    @abstractmethod
    def rho(self) -> float:
        """Soft-output penalty weight."""

    @rho.setter
    @abstractmethod
    def rho(self, value) -> None: ...

    @property
    @abstractmethod
    def y_offset(self) -> float:
        """Soft-output band half-width."""

    @y_offset.setter
    @abstractmethod
    def y_offset(self, value) -> None: ...

    @property
    @abstractmethod
    def u_min(self) -> np.ndarray:
        """Lower input bounds."""

    @u_min.setter
    @abstractmethod
    def u_min(self, value) -> None: ...

    @property
    @abstractmethod
    def u_max(self) -> np.ndarray:
        """Upper input bounds."""

    @u_max.setter
    @abstractmethod
    def u_max(self, value) -> None: ...

    @abstractmethod
    def solve(self, x0, D, x_ref, u_prev=None, warm_start=None): ...


class ContinuousLinearisedOCPBase(ContinuousLinearOCPBase):
    """Abstract base for linearised continuous-discrete OCPs (deviation coordinates)."""
    pass


class ContinuousNonlinearOCPBase(OCP):
    """Abstract base for CD nonlinear OCPs (NLP formulation)."""

    @property
    @abstractmethod
    def Q_z(self) -> np.ndarray | None:
        """Tracking weight matrix."""

    @Q_z.setter
    @abstractmethod
    def Q_z(self, value) -> None: ...

    @property
    @abstractmethod
    def z_ref(self) -> np.ndarray | None:
        """Tracking reference trajectory."""

    @z_ref.setter
    @abstractmethod
    def z_ref(self, value) -> None: ...

    @property
    @abstractmethod
    def R_stage(self) -> np.ndarray | None:
        """Stage input cost matrix."""

    @R_stage.setter
    @abstractmethod
    def R_stage(self, value) -> None: ...

    @property
    @abstractmethod
    def P_terminal(self) -> np.ndarray | None:
        """Terminal tracking cost matrix."""

    @P_terminal.setter
    @abstractmethod
    def P_terminal(self, value) -> None: ...

    @property
    @abstractmethod
    def Q_du(self) -> np.ndarray | None:
        """Input rate-of-movement penalty matrix."""

    @Q_du.setter
    @abstractmethod
    def Q_du(self, value) -> None: ...

    @property
    @abstractmethod
    def u_min(self) -> np.ndarray | None:
        """Lower input bounds (hard)."""

    @u_min.setter
    @abstractmethod
    def u_min(self, value) -> None: ...

    @property
    @abstractmethod
    def u_max(self) -> np.ndarray | None:
        """Upper input bounds (hard)."""

    @u_max.setter
    @abstractmethod
    def u_max(self, value) -> None: ...

    @property
    @abstractmethod
    def du_min(self) -> np.ndarray | None:
        """Lower input rate-of-movement bounds (hard)."""

    @du_min.setter
    @abstractmethod
    def du_min(self, value) -> None: ...

    @property
    @abstractmethod
    def du_max(self) -> np.ndarray | None:
        """Upper input rate-of-movement bounds (hard)."""

    @du_max.setter
    @abstractmethod
    def du_max(self, value) -> None: ...

    @property
    @abstractmethod
    def z_min(self) -> np.ndarray | None:
        """Lower output bounds (soft)."""

    @z_min.setter
    @abstractmethod
    def z_min(self, value) -> None: ...

    @property
    @abstractmethod
    def z_max(self) -> np.ndarray | None:
        """Upper output bounds (soft)."""

    @z_max.setter
    @abstractmethod
    def z_max(self, value) -> None: ...

    @property
    @abstractmethod
    def rho_z_1(self) -> float:
        """L1 penalty weight on output slacks."""

    @rho_z_1.setter
    @abstractmethod
    def rho_z_1(self, value) -> None: ...

    @property
    @abstractmethod
    def rho_z_2(self) -> float:
        """L2 penalty weight on output slacks."""

    @rho_z_2.setter
    @abstractmethod
    def rho_z_2(self, value) -> None: ...

    @abstractmethod
    def solve(self, x0, d_trajectory, u_prev=None, x_prev=None, y_prev=None, p=None, t0=0.0): ...
