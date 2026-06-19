"""
Continuous-discrete linear receding-horizon QP (``StandardLinearContinuousDiscreteOCP``).

A thin, typed wrapper around :class:`StandardLinearDiscreteOCP` for
:class:`~mbc.models.ContinuousDiscreteLinearSDE`.  Solves the
receding-horizon QP via the lifted (batch) formulation using a convex-QP
backend (OSQP by default).  The continuous-discrete model is ZOH-discretised
once at construction time and the resulting discrete matrices are passed to
the parent ``StandardLinearDiscreteOCP``.

Given the continuous-discrete model

    dx = (A_c x + B_c u + E_c d) dt + G dw
    y[k] = C x[k] + v[k]

the controller operates on the ZOH-discretised prediction model

    x[k+1] = A_d x[k] + B_d u[k] + E_d d[k]

where (A_d, B_d, E_d) = ZOH(A_c, B_c, E_c, Ts).
"""

from __future__ import annotations

from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from .discrete_linear_ocp import StandardLinearDiscreteOCP
from .qp_solver import QPSolverBackend

if TYPE_CHECKING:
    from ..models import ContinuousDiscreteLinearSDE


class _CDModelAdapter:
    """
    Thin adapter that wraps a ``ContinuousDiscreteLinearSDE`` and exposes
    the numpy interface expected by ``DiscreteLinearOCP``.

    The ZOH-discretised matrices are computed once at construction time.
    """

    def __init__(self, model: "ContinuousDiscreteLinearSDE") -> None:
        self._m = model
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
        return self._m.Cm

    @property
    def Cz(self) -> np.ndarray:
        return self._m.Cz

    @property
    def Ad(self) -> np.ndarray:
        return self._Ad_np

    @property
    def Bd(self) -> np.ndarray:
        return self._Bd_np

    @property
    def Ed(self) -> np.ndarray:
        return self._Ed_np

    @property
    def u_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._m.u_bounds

    @property
    def x_ref(self) -> np.ndarray:
        return self._m.x_ref


class StandardLinearContinuousDiscreteOCP(StandardLinearDiscreteOCP):
    """
    Receding-horizon QP for a linear continuous-discrete system.

    ZOH-discretises the continuous-discrete model at construction time and
    delegates to :class:`StandardLinearDiscreteOCP` for all QP building and solving.
    The original CD model is stored as ``self._cd_model`` for direct access.

    Parameters
    ----------
    model : ContinuousDiscreteLinearSDE
        Plant model.  Must implement ``nx``, ``nu``, ``nd``, ``A``, ``B``,
        ``E``, ``Cm``, ``Ts``, and ``u_bounds``.
    N : int
        Prediction horizon (number of sampling intervals).
    Q : (nz, nz) array-like
        Stage output tracking cost  ‖z − z_ref‖²_Q.
    R : (nu, nu) array-like
        Stage input cost  ‖u‖²_R.
    P : (nz, nz) array-like, optional
        Terminal output tracking cost.  Default: Q.
    S : (nu, nu) array-like, optional
        Input rate-of-movement cost  ‖Δu‖²_S.  ``None`` → disabled.
    rho : float or (N,) array-like, optional
        Quadratic penalty on the soft-output slack variable ``ε``.  Scalar or
        per-step (N,) array.  Default: 1e4.
    rho_lin : float or (N,) array-like, optional
        Linear penalty on the soft-output slack variable ``ε``.  Scalar or
        per-step (N,) array.  Default: 0.0.
    y_offset : float or (N,) array-like, optional
        Symmetric half-width δ of the soft output constraint band.  Scalar or
        per-step (N,) array.  Default: 2.0.
    solver : str or QPSolverBackend, optional
        Convex-QP backend.  Default: ``"highs"``.
    solver_options : dict, optional
        Forwarded to the QP backend.
    formulation : {"auto", "condensed", "sparse"}, optional
        QP construction strategy.  Default: ``"auto"``.

    Notes
    -----
    ``StandardLinearContinuousDiscreteOCP.solve`` has the same signature as
    ``StandardLinearDiscreteOCP.solve`` — see that class for full documentation.
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
        rho_lin: float = 0.0,
        y_offset: float = 2.0,
        solver: str | QPSolverBackend = "highs",
        solver_options: dict[str, Any] | None = None,
        formulation: str = "auto",
    ) -> None:
        self._cd_model = model

        super().__init__(
            model=_CDModelAdapter(model),  # type: ignore[arg-type]
            N=N,
            Q=Q,
            R=R,
            P=P,
            S=S,
            rho=rho,
            rho_lin=rho_lin,
            y_offset=y_offset,
            solver=solver,
            solver_options=solver_options,
            formulation=formulation,
        )
