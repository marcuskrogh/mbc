"""
Successive-linearisation MPC for nonlinear continuous-discrete models.

This module provides a linear-QP MPC controller that, at each control interval,
linearises a nonlinear continuous-discrete model around an operating point,
ZOH-discretises the local model, and solves a
:class:`~mbc.control.DiscreteLinearOCP` in deviation coordinates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .._utils import _fd_jacobian, _zoh_full, _van_loan
from ..models import DiscreteLinearSDE, ContinuousDiscreteSDE
from ._base import ModelPredictiveController
from .discrete_linear_ocp import StandardLinearDiscreteOCP


class _DeviationDiscreteLinearSDE(DiscreteLinearSDE):
    """Mutable linear-discrete model in deviation coordinates."""

    def __init__(
        self,
        nx: int,
        nu: int,
        nd: int,
        nym: int,
        nz: int,
        u_min_abs: np.ndarray,
        u_max_abs: np.ndarray,
    ) -> None:
        self._nx = nx
        self._nu = nu
        self._nd = nd
        self._nym = nym
        self._nz = nz

        self._u_min_abs = np.asarray(u_min_abs, dtype=float).reshape(nu)
        self._u_max_abs = np.asarray(u_max_abs, dtype=float).reshape(nu)

        self._Ad = np.eye(nx)
        self._Bd = np.zeros((nx, nu))
        self._Ed = np.zeros((nx, nd))
        self._Cm = np.zeros((nym, nx))
        self._Cz = np.zeros((nz, nx))
        self._Qd = np.eye(nx) * 1e-8
        self._Rm = np.eye(nym)

        self._x = np.zeros(nx)
        self._x_ref = np.zeros(nx)

        self.x_ss = np.zeros(nx)
        self.u_ss = np.zeros(nu)
        self.d_ss = np.zeros(nd)

    def update(
        self,
        *,
        Ad: np.ndarray,
        Bd: np.ndarray,
        Ed: np.ndarray,
        Cm: np.ndarray,
        Cz: np.ndarray,
        Qd: np.ndarray,
        Rm: np.ndarray,
        x_ss: np.ndarray,
        u_ss: np.ndarray,
        d_ss: np.ndarray,
        x_ref: np.ndarray,
    ) -> None:
        self._Ad = np.asarray(Ad, dtype=float)
        self._Bd = np.asarray(Bd, dtype=float)
        self._Ed = np.asarray(Ed, dtype=float)
        self._Cm = np.asarray(Cm, dtype=float)
        self._Cz = np.asarray(Cz, dtype=float)
        self._Qd = np.asarray(Qd, dtype=float)
        self._Rm = np.asarray(Rm, dtype=float)

        self.x_ss = np.asarray(x_ss, dtype=float).reshape(self._nx)
        self.u_ss = np.asarray(u_ss, dtype=float).reshape(self._nu)
        self.d_ss = np.asarray(d_ss, dtype=float).reshape(self._nd)

        self._x = np.zeros(self._nx)
        self._x_ref = np.asarray(x_ref, dtype=float).reshape(self._nx)

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._nu

    @property
    def nd(self) -> int:
        return self._nd

    @property
    def Ad(self) -> np.ndarray:
        return self._Ad

    @property
    def Bd(self) -> np.ndarray:
        return self._Bd

    @property
    def Ed(self) -> np.ndarray:
        return self._Ed

    @property
    def Cm(self) -> np.ndarray:
        return self._Cm

    @property
    def Cz(self) -> np.ndarray:
        return self._Cz

    @property
    def Qd(self) -> np.ndarray:
        return self._Qd

    @property
    def Rm(self) -> np.ndarray:
        return self._Rm

    @property
    def x(self) -> list[float]:
        return self._x.tolist()

    @x.setter
    def x(self, val: list[float]) -> None:
        self._x = np.asarray(val, dtype=float).reshape(self._nx)

    @property
    def x_ref(self) -> np.ndarray:
        return self._x_ref.copy()

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Bounds are in deviation coordinates: du = u - u_ss.
        return self._u_min_abs - self.u_ss, self._u_max_abs - self.u_ss

    @property
    def abs_u_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Absolute input bounds (u_min_abs, u_max_abs)."""
        return self._u_min_abs.copy(), self._u_max_abs.copy()


def linearize_cd_model(
    model: ContinuousDiscreteSDE,
    x_ss: np.ndarray,
    u_ss: np.ndarray,
    d_ss: np.ndarray,
    p: np.ndarray,
    t: float,
) -> dict[str, np.ndarray]:
    """Linearise model around operating point (x_ss, u_ss, d_ss, p, t)."""
    x_ss = np.asarray(x_ss, dtype=float)
    u_ss = np.asarray(u_ss, dtype=float)
    d_ss = np.asarray(d_ss, dtype=float)
    p = np.asarray(p, dtype=float)

    A = model.dfdx(x_ss, u_ss, d_ss, p, t)
    B = model.dfdu(x_ss, u_ss, d_ss, p, t)
    E = model.dfdd(x_ss, u_ss, d_ss, p, t)

    Cm = model.dhmdx(x_ss, u_ss, d_ss, p, t)
    Cz = model.dgmdx(x_ss, u_ss, d_ss, p, t)

    Dm = model.dhmdu(x_ss, u_ss, d_ss, p, t)
    Fm = model.dhmdd(x_ss, u_ss, d_ss, p, t)
    Dz = model.dgmdu(x_ss, u_ss, d_ss, p, t)

    if model.nd > 0:
        Fz = _fd_jacobian(lambda d_: model.gm(x_ss, u_ss, d_, p, t), d_ss)
    else:
        Fz = np.zeros((model.nz, 0))

    G = model.sigma(x_ss, u_ss, d_ss, p, t)

    return {
        "A": A,
        "B": B,
        "E": E,
        "Cm": Cm,
        "Cz": Cz,
        "Dm": Dm,
        "Fm": Fm,
        "Dz": Dz,
        "Fz": Fz,
        "G": G,
    }


def discretize_cd_linearization(
    lin: dict[str, np.ndarray],
    dt: float,
) -> dict[str, np.ndarray]:
    """Discretise local linearised model using ZOH and Van Loan."""
    A = lin["A"]
    B = lin["B"]
    E = lin["E"]
    G = lin["G"]

    Ad, Bd, Ed = _zoh_full(A, B, E, dt)

    if G.size == 0:
        Qd = np.zeros((A.shape[0], A.shape[0]))
    else:
        Qd = _van_loan(A, G, np.eye(G.shape[1]), dt)

    return {
        "Ad": Ad,
        "Bd": Bd,
        "Ed": Ed,
        "Qd": Qd,
        "Cm": lin["Cm"],
        "Cz": lin["Cz"],
        "Dm": lin["Dm"],
        "Fm": lin["Fm"],
        "Dz": lin["Dz"],
        "Fz": lin["Fz"],
    }


class LinearisedContinuousMPC(ModelPredictiveController, ABC):
    """
    Abstract MPC with **mixed nonlinear estimation and linearised control**.

    State estimation runs on the full **nonlinear** continuous-discrete plant
    (e.g. :class:`~mbc.estimation.ContinuousDiscreteEKF`).  The OCP linearises
    the same model at each sample, ZOH-discretises the Jacobian, and solves a
    **discrete-time** linear QP in deviation coordinates.

    This split is deliberate: there is no need to replace the nonlinear model
    in the estimator just because the control layer is linearised.
    """

    @property
    @abstractmethod
    def x_ref(self) -> np.ndarray:
        """Absolute state reference used for tracking."""

    @property
    @abstractmethod
    def last_disturbance_deviation_trajectory(self) -> np.ndarray:
        """Most recent disturbance trajectory in deviation coordinates."""

    @abstractmethod
    def compute(
        self,
        y: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and return the optimal closed-loop MPC action."""

    @abstractmethod
    def propagate(
        self,
        y: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the estimator without solving the OCP; return ``(x_hat, P)``.

        The elapsed time since the last call to :meth:`compute` or
        :meth:`propagate` is inferred from ``t``, so odd (off-grid) intervals
        are handled automatically without any manual bookkeeping.
        """


class StandardLinearisedContinuousMPC(LinearisedContinuousMPC):
    """
    Standard mixed nonlinear-estimation / linearised-control MPC.

    **Estimator** — any continuous-discrete state estimator on the nonlinear
    ``ContinuousDiscreteSDE`` plant (typically :class:`~mbc.estimation.ContinuousDiscreteEKF`).

    **Controller** — at each sample, Jacobian-linearise at
    ``(x_ss, u_ss, d_ss)`` (current estimate by default, or
    :meth:`set_linearisation_point` for a thermal equilibrium), ZOH-discretise,
    and solve :class:`~mbc.control.StandardLinearDiscreteOCP` in deviation
    coordinates.

    Horizon-varying references, bounds, weights, disturbances, and linear input
    costs are configured via the profile setters before :meth:`compute`.
    """

    def __init__(
        self,
        model: ContinuousDiscreteSDE,
        estimator: Any,
        N: int,
        Q: Any,
        R: Any,
        dt: float,
        u_min: np.ndarray,
        u_max: np.ndarray,
        x_ref: np.ndarray | None = None,
        P: Any | None = None,
        S: Any | None = None,
        rho: float = 1e4,
        rho_lin: float = 0.0,
        z_offset: float = 2.0,
        du_min: np.ndarray | None = None,
        du_max: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._estimator = estimator
        self._N = int(N)
        self._dt = float(dt)

        self._x_ref_abs = (
            np.zeros(model.nx, dtype=float)
            if x_ref is None
            else np.asarray(x_ref, dtype=float).reshape(model.nx)
        )

        self._lin_model = _DeviationDiscreteLinearSDE(
            nx=model.nx,
            nu=model.nu,
            nd=model.nd,
            nym=model.nym,
            nz=model.nz,
            u_min_abs=np.asarray(u_min, dtype=float),
            u_max_abs=np.asarray(u_max, dtype=float),
        )

        self._ocp = StandardLinearDiscreteOCP(
            model=self._lin_model,
            N=self._N,
            Q=Q,
            R=R,
            P=P,
            S=S,
            du_min=du_min,
            du_max=du_max,
            rho=rho,
            rho_lin=rho_lin,
            z_offset=z_offset,
        )
        self._bind_ocp(self._ocp)

        self._u_prev = np.zeros(model.nu, dtype=float)
        self._d_prev = np.zeros(model.nd, dtype=float)
        self._last_D_dev = np.zeros((self._N, model.nd), dtype=float)
        self._t_last: float | None = None

    @property
    def x_ref(self) -> np.ndarray:
        """Absolute state reference used for tracking."""
        return self._x_ref_abs.copy()

    @x_ref.setter
    def x_ref(self, val: np.ndarray) -> None:
        self._x_ref_abs = np.asarray(val, dtype=float).reshape(self._model.nx)

    @property
    def last_disturbance_deviation_trajectory(self) -> np.ndarray:
        """Most recent disturbance trajectory in deviation coordinates."""
        return self._last_D_dev.copy()

    def _resolve_disturbance_deviation(
        self,
        d_ss: np.ndarray,
    ) -> np.ndarray:
        """Absolute disturbance forecast → deviation coordinates ``Δd = d − d_ss``."""
        prof = self._horizon_profile
        nd = self._model.nd
        if prof.disturbance_profile is not None:
            D_abs = np.asarray(prof.disturbance_profile, dtype=float)
            if D_abs.ndim == 1:
                D_abs = D_abs.reshape(self._N, nd)
            if D_abs.shape != (self._N, nd):
                raise ValueError(
                    f"disturbance_profile must have shape ({self._N}, {nd}); "
                    f"got {D_abs.shape}."
                )
            return D_abs - d_ss.reshape(1, -1)
        return np.zeros((self._N, nd))

    def _deviation_input_bound_profiles(
        self,
        u_ss: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Absolute input-bound profiles expressed in deviation coordinates."""
        prof = self._horizon_profile
        if prof.input_min_profile is None or prof.input_max_profile is None:
            return None, None
        u_min_abs = np.asarray(prof.input_min_profile, dtype=float).reshape(
            self._N, self._model.nu
        )
        u_max_abs = np.asarray(prof.input_max_profile, dtype=float).reshape(
            self._N, self._model.nu
        )
        u_ss_row = u_ss.reshape(1, -1)
        return u_min_abs - u_ss_row, u_max_abs - u_ss_row

    def compute(
        self,
        y: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the optimal closed-loop action.

        **Estimation** uses the nonlinear plant and estimator unchanged.
        **Control** linearises, discretises, and solves the deviation QP.

        Configure horizon profiles (disturbance forecast, references, bounds,
        weights, linear input costs) and optionally
        :meth:`set_linearisation_point` before calling.

        When no disturbance profile is set, disturbances are held at ``d_ss``
        over the horizon (``Δd[k] = 0``).
        """
        y = np.asarray(y, dtype=float).reshape(self._model.nym)
        d_now = np.asarray(d, dtype=float).reshape(self._model.nd)
        p_ = np.array([], dtype=float) if p is None else np.asarray(p, dtype=float)

        x_hat_np, _ = self._estimator.step(y, self._u_prev, self._d_prev, p_, t)
        x_hat = np.asarray(x_hat_np, dtype=float).reshape(self._model.nx)

        lp = self._horizon_profile.linearisation_point
        if lp is not None:
            x_ss = np.asarray(lp.x, dtype=float).reshape(self._model.nx)
            u_ss = np.asarray(lp.u, dtype=float).reshape(self._model.nu)
            d_ss = np.asarray(lp.d, dtype=float).reshape(self._model.nd)
        else:
            x_ss = x_hat.copy()
            u_ss = self._u_prev.copy()
            d_ss = d_now.copy()

        lin = linearize_cd_model(self._model, x_ss, u_ss, d_ss, p_, t)
        disc = discretize_cd_linearization(lin, self._dt)

        x_ref_dev = self._x_ref_abs - x_ss
        self._lin_model.update(
            Ad=disc["Ad"],
            Bd=disc["Bd"],
            Ed=disc["Ed"],
            Cm=disc["Cm"],
            Cz=disc["Cz"],
            Qd=disc["Qd"],
            Rm=np.asarray(self._model.Rm, dtype=float),
            x_ss=x_ss,
            u_ss=u_ss,
            d_ss=d_ss,
            x_ref=x_ref_dev,
        )

        D_dev_np = self._resolve_disturbance_deviation(d_ss)
        self._last_D_dev = D_dev_np.copy()

        x0_dev = x_hat - x_ss
        u_prev_dev = self._u_prev - u_ss

        prof = self._horizon_profile
        saved_u_min, saved_u_max = prof.input_min_profile, prof.input_max_profile
        saved_ue = prof.input_equilibrium
        u_min_dev, u_max_dev = self._deviation_input_bound_profiles(u_ss)
        if u_min_dev is not None:
            prof.input_min_profile = u_min_dev
            prof.input_max_profile = u_max_dev
        prof.input_equilibrium = u_ss
        try:
            U_dev, X_dev = self._ocp.solve(
                x0=x0_dev,
                D=D_dev_np.reshape(-1),
                x_ref=None,
                u_prev=u_prev_dev,
            )
        finally:
            prof.input_min_profile = saved_u_min
            prof.input_max_profile = saved_u_max
            prof.input_equilibrium = saved_ue

        U_dev_np = np.asarray(U_dev, dtype=float).reshape(self._N, self._model.nu)
        X_dev_np = np.asarray(X_dev, dtype=float).reshape(self._N, self._model.nx)

        U_abs = U_dev_np + u_ss.reshape(1, -1)
        X_abs = X_dev_np + x_ss.reshape(1, -1)
        u_abs = U_abs[0].copy()

        # Numerical safety against tiny solver tolerances beyond box constraints.
        u_min_abs, u_max_abs = self._lin_model.abs_u_bounds
        u_abs = np.minimum(np.maximum(u_abs, u_min_abs), u_max_abs)
        U_abs[0] = u_abs

        self._u_prev = u_abs.copy()
        self._d_prev = d_now.copy()
        self._t_last = float(t)

        return u_abs, U_abs, X_abs

    def propagate(
        self,
        y: np.ndarray,
        d: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the estimator without solving the OCP; return ``(x_hat, P)``.

        Use this when the controller is switched off but state tracking must
        continue.  The elapsed time since the last :meth:`compute` or
        :meth:`propagate` call is inferred from ``t``, so off-grid (odd)
        arrival times are handled automatically: the estimator integrates for
        exactly ``t - t_last`` regardless of the nominal sampling period.

        On the very first call (before any :meth:`compute` has been made) the
        nominal sampling period is used, matching the standard start-up
        behaviour.

        Parameters
        ----------
        y : (nym,) array-like    — current measurement.
        d : (nd,) array-like     — current disturbance; updates the stored
            disturbance for the next call.
        p : (np,) array-like, optional — parameter vector.
        t : float                — current time (seconds).

        Returns
        -------
        x_hat : (nx,) filtered state estimate.
        P     : (nx, nx) state error covariance.
        """
        y = np.asarray(y, dtype=float).reshape(self._model.nym)
        d_now = np.asarray(d, dtype=float).reshape(self._model.nd)
        p_ = np.array([], dtype=float) if p is None else np.asarray(p, dtype=float)

        if self._t_last is not None:
            dt = float(t) - self._t_last
            if dt > 0.0:
                self._estimator.predict_for(dt, self._u_prev, self._d_prev, p_, self._t_last)
            x_hat, P = self._estimator.update(y, self._u_prev, d_now, p_)
        else:
            x_hat, P = self._estimator.step(y, self._u_prev, self._d_prev, p_, t)

        self._d_prev = d_now.copy()
        self._t_last = float(t)
        return np.asarray(x_hat, dtype=float), P
