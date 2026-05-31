"""
Linearised continuous-discrete tracking OCP (deviation coordinates).

``ContinuousLinearisedOCP`` uses composition to delegate to
:class:`ContinuousLinearOCP` and handles the coordinate shift between
absolute and deviation space.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from ._base import ContinuousLinearisedOCPBase
from .._utils import _any_to_np1d

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteLinearSDE


class ContinuousLinearisedOCP(ContinuousLinearisedOCPBase):
    """
    Linearised continuous-discrete tracking OCP in deviation coordinates.

    Wraps :class:`ContinuousLinearOCP` and handles the coordinate shift so
    that the underlying QP sees deviation variables ``δx = x − x_ss``,
    ``δu = u − u_ss``.

    Parameters
    ----------
    model : ContinuousDiscreteLinearSDE
        Linearised continuous-discrete model with deviation-coordinate info.
    N : int
        Prediction horizon.
    Q, R, P, S, rho, y_offset, solver, solver_options, formulation
        Forwarded to :class:`ContinuousLinearOCP`.
    """

    def __init__(
        self,
        model: "ContinuousDiscreteLinearSDE",
        N: int,
        Q: Any,
        R: Any,
        P: Any | None = None,
        S: Any | None = None,
        rho: float = 1e4,
        y_offset: float = 2.0,
        solver: str = "osqp",
        solver_options: dict | None = None,
        formulation: str = "auto",
    ) -> None:
        from .continuous_linear import ContinuousLinearOCP
        self._cd_model = model
        self._impl = ContinuousLinearOCP(
            model, N, Q, R,
            P=P, S=S, rho=rho, y_offset=y_offset,
            solver=solver, solver_options=solver_options,
            formulation=formulation,
        )

    # ── Abstract property implementations ───────────────────────────────

    @property
    def N(self) -> int:
        return self._impl.N

    @property
    def nu(self) -> int:
        return self._impl.nu

    @property
    def Q(self) -> np.ndarray:
        return self._impl.Q

    @Q.setter
    def Q(self, v) -> None:
        self._impl.Q = v

    @property
    def R(self) -> np.ndarray:
        return self._impl.R

    @R.setter
    def R(self, v) -> None:
        self._impl.R = v

    @property
    def P(self) -> np.ndarray:
        return self._impl.P

    @P.setter
    def P(self, v) -> None:
        self._impl.P = v

    @property
    def S(self) -> np.ndarray | None:
        return self._impl.S

    @S.setter
    def S(self, v) -> None:
        self._impl.S = v

    @property
    def rho(self) -> float:
        return self._impl.rho

    @rho.setter
    def rho(self, v) -> None:
        self._impl.rho = v

    @property
    def y_offset(self) -> float:
        return self._impl.y_offset

    @y_offset.setter
    def y_offset(self, v) -> None:
        self._impl.y_offset = v

    @property
    def u_min(self) -> np.ndarray:
        return self._impl.u_min

    @u_min.setter
    def u_min(self, v) -> None:
        self._impl.u_min = v

    @property
    def u_max(self) -> np.ndarray:
        return self._impl.u_max

    @u_max.setter
    def u_max(self, v) -> None:
        self._impl.u_max = v

    def solve(
        self,
        x0: Any,
        D: Any,
        x_ref: Any,
        u_prev: Any | None = None,
        warm_start: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the linearised CD OCP, returning absolute-coordinate outputs.

        Parameters
        ----------
        x0 : (nx,) array-like
            Current absolute state estimate.
        D : (N · nd,) array-like
            Stacked disturbance forecast.
        x_ref : (nx,) array-like
            Absolute state reference.
        u_prev : (nu,) array-like, optional
            Previously-applied absolute input (used for ROM penalty S).
        warm_start : dict, optional
            Warm-start dict.

        Returns
        -------
        U : (N · nu,) ndarray — optimal absolute input sequence.
        X : (N · nx,) ndarray — predicted absolute state trajectory.
        """
        # The underlying model stores x_ss / u_ss for coordinate shifting.
        x_ss = np.asarray(self._cd_model.x_ss, dtype=float).reshape(-1)
        u_ss = np.asarray(self._cd_model.u_ss, dtype=float).reshape(-1)

        x0_np = _any_to_np1d(x0).reshape(-1)
        x_ref_np = _any_to_np1d(x_ref).reshape(-1)

        # Convert to deviation coordinates.
        x0_dev = x0_np - x_ss
        x_ref_dev = x_ref_np - x_ss
        u_prev_dev = (
            None if u_prev is None
            else _any_to_np1d(u_prev).reshape(-1) - u_ss
        )

        U_dev, X_dev = self._impl.solve(
            x0_dev, D, x_ref_dev,
            u_prev=u_prev_dev, warm_start=warm_start,
        )

        # Convert back to absolute coordinates.
        N = self._impl.N
        nu = self._impl.nu
        nx = self._cd_model.nx
        U_abs = U_dev.reshape(N, nu) + u_ss.reshape(1, -1)
        X_abs = X_dev.reshape(N, nx) + x_ss.reshape(1, -1)
        return U_abs.reshape(-1), X_abs.reshape(-1)
