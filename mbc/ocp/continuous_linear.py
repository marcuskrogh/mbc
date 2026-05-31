"""
Continuous-discrete linear tracking OCP (QP via ZOH discretisation).

``ContinuousLinearOCP`` wraps a :class:`~mbc.models.ContinuousDiscreteLinearSDE`
and delegates to :class:`DiscreteLinearOCP` by adapting the continuous model to
the discrete interface expected by the QP builder.
"""

from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from ._base import ContinuousLinearOCPBase
from .._utils import _any_to_np1d, _any_to_np2d

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteLinearSDE


class _CDModelAdapter:
    """
    Thin adapter that wraps a ``ContinuousDiscreteLinearSDE`` and exposes
    the numpy interface expected by ``DiscreteLinearOCP``.

    ``DiscreteLinearOCP.solve`` accesses:
      - ``model.nx``, ``model.nu``, ``model.nd``  (int)
      - ``model.Cz``        (numpy ndarray, for output prediction)
      - ``model.Ad``        (numpy ndarray, ZOH-discretised state matrix)
      - ``model.Bd``        (numpy ndarray, ZOH-discretised input matrix)
      - ``model.Ed``        (numpy ndarray, ZOH-discretised disturbance matrix)
      - ``model.u_bounds``  (tuple of numpy (nu,) arrays)

    The ZOH-discretised matrices are computed once at construction time.
    """

    def __init__(self, model: "ContinuousDiscreteLinearSDE") -> None:
        self._m = model
        # Compute ZOH-discretised matrices once at construction time.
        from .._utils import _zoh_full
        self._Ad_np, self._Bd_np, self._Ed_np = _zoh_full(
            model.A, model.B, model.E, model.Ts
        )

    @property
    def nx(self) -> int:
        return self._m.nx

    @property
    def nu(self) -> int:
        return self._m.nu

    @property
    def nd(self) -> int:
        return self._m.nd

    @property
    def Cm(self) -> np.ndarray:
        """Measurement output matrix Cm (numpy ndarray)."""
        return self._m.Cm

    @property
    def Cz(self) -> np.ndarray:
        """Output matrix Cz (numpy ndarray)."""
        return self._m.Cz

    @property
    def Ad(self) -> np.ndarray:
        """ZOH-discretised state-transition matrix Ad."""
        return self._Ad_np

    @property
    def Bd(self) -> np.ndarray:
        """ZOH-discretised input matrix Bd."""
        return self._Bd_np

    @property
    def Ed(self) -> np.ndarray:
        """ZOH-discretised disturbance matrix Ed."""
        return self._Ed_np

    @property
    def u_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Input box constraints (u_min, u_max), each a (nu,) ndarray."""
        return self._m.u_bounds


class ContinuousLinearOCP(ContinuousLinearOCPBase):
    """
    Receding-horizon QP for a linear continuous-discrete system.

    Wraps a :class:`~mbc.models.ContinuousDiscreteLinearSDE` and delegates to
    :class:`DiscreteLinearOCP` via :class:`_CDModelAdapter`.

    Parameters
    ----------
    model : ContinuousDiscreteLinearSDE
        Plant model.  Must implement ``nx``, ``nu``, ``nd``, ``A``, ``B``,
        ``E``, ``Cz``, ``Ts``, and ``u_bounds``.
    N : int
        Prediction horizon (number of sampling intervals).
    Q : (ny, ny) array-like
        Stage output tracking cost  ‖y − r‖²_Q.
    R : (nu, nu) array-like
        Stage input cost  ‖u‖²_R.
    P : (ny, ny) array-like, optional
        Terminal output tracking cost.  Default: Q.
    S : (nu, nu) array-like, optional
        Input rate-of-movement cost  ‖Δu‖²_S.  ``None`` → disabled.
    rho : float, optional
        Penalty weight on soft output constraint violation.  Default: 1e4.
    y_offset : float, optional
        Symmetric half-width δ of the soft output constraint band.
        Default: 2.0.
    solver : str or QPSolverBackend, optional
        Convex-QP backend selector.
    solver_options : dict, optional
        Backend-specific options forwarded to the QP solver.
    formulation : {"auto", "condensed", "sparse"}, optional
        QP construction strategy.
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
        solver_options: dict[str, Any] | None = None,
        formulation: str = "auto",
    ) -> None:
        from .discrete_linear import DiscreteLinearOCP
        self._cd_model = model
        self._impl = DiscreteLinearOCP(
            _CDModelAdapter(model),  # type: ignore[arg-type]
            N=N,
            Q=Q,
            R=R,
            P=P,
            S=S,
            rho=rho,
            y_offset=y_offset,
            solver=solver,
            solver_options=solver_options,
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
        Solve the QP starting from state estimate ``x0``.

        Parameters
        ----------
        x0    : (nx,) array-like — current state estimate.
        D     : (N · nd,) array-like — stacked disturbance forecast.
        x_ref : (nx,) array-like — state reference.
        u_prev : (nu,) array-like, optional
            Previously-applied input (used only when S is active).
        warm_start : dict, optional
            ``{"U": (N·nu,), "X": (N·nx,)}`` primal warm-start trajectory.

        Returns
        -------
        U : (N · nu,) ndarray — optimal input sequence.
        X : (N · nx,) ndarray — predicted state trajectory.
        """
        return self._impl.solve(x0, D, x_ref, u_prev=u_prev, warm_start=warm_start)
