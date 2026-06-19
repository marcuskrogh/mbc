"""
Nonlinear continuous-time MPC for continuous-discrete plants.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ._base import ContinuousOptimalControlProblem, ModelPredictiveController


class NonlinearContinuousMPC(ModelPredictiveController):
    """Abstract MPC for nonlinear CD plant + CD estimator + continuous OCP."""

    @abstractmethod
    def compute(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Compute and return the optimal closed-loop NMPC action."""

    @abstractmethod
    def propagate(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the estimator without solving the OCP; return ``(x_hat, P)``.

        The elapsed time since the last call to :meth:`compute` or
        :meth:`propagate` is inferred from ``t``, so odd (off-grid) intervals
        are handled automatically without any manual bookkeeping.
        """


class StandardNonlinearContinuousMPC(NonlinearContinuousMPC):
    """
    Standard closed-loop NMPC for nonlinear continuous-discrete plants.

    Composes a continuous-discrete state estimator with a continuous OCP.
    """

    def __init__(self, estimator, ocp: ContinuousOptimalControlProblem) -> None:
        super().__init__()
        self._estimator = estimator
        self._ocp = ocp
        self._bind_ocp(ocp)
        self._u_seq_prev: np.ndarray | None = None
        self._x_traj_prev: np.ndarray | None = None
        self._y_traj_prev: np.ndarray | None = None
        self._u_prev: np.ndarray = np.zeros(ocp.nu)
        self._d_prev: np.ndarray = np.zeros(ocp._nd)
        self._t_last: float | None = None

    def compute(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        p_ = np.array([], dtype=float) if p is None else np.asarray(p, dtype=float)
        if d_trajectory is not None:
            self.set_disturbance_profile(np.asarray(d_trajectory, dtype=float))

        prof = self._horizon_profile
        if prof.disturbance_profile is None:
            raise ValueError(
                "Disturbance forecast required: pass d_trajectory to compute() or "
                "call set_disturbance_profile()."
            )
        d_arr = np.asarray(prof.disturbance_profile, dtype=float)
        if d_arr.ndim == 1:
            nd = d_arr.size // self._ocp._nd
            d_trajectory = d_arr.reshape(-1, nd)
        else:
            d_trajectory = d_arr
        d0 = d_trajectory[0]

        x_hat, _ = self._estimator.step(y, self._u_prev, d0, p_, t)

        u_opt, _, info = self._ocp.solve(
            x_hat,
            d_trajectory,
            u_prev=self._u_seq_prev,
            x_prev=self._x_traj_prev,
            y_prev=self._y_traj_prev,
            p=p_,
            t0=t,
        )
        u_k = u_opt[0]

        self._u_seq_prev = u_opt
        self._x_traj_prev = info.get("X")
        self._y_traj_prev = info.get("Y")
        self._u_prev = u_k
        self._d_prev = d0.copy()
        self._t_last = float(t)

        return u_k

    def propagate(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the estimator without solving the OCP; return ``(x_hat, P)``.

        Use this when the controller is switched off but state tracking must
        continue.  The elapsed time since the last :meth:`compute` or
        :meth:`propagate` call is inferred from ``t``, so off-grid (odd)
        arrival times are handled automatically.

        Parameters
        ----------
        y            : (nym,) array-like  — current measurement.
        d_trajectory : array-like, optional — disturbance trajectory;
            updates the stored first-step disturbance if provided.
        p            : (np,) array-like, optional — parameter vector.
        t            : float — current time (seconds).

        Returns
        -------
        x_hat : (nx,) filtered state estimate.
        P     : (nx, nx) state error covariance.
        """
        p_ = np.array([], dtype=float) if p is None else np.asarray(p, dtype=float)
        if d_trajectory is not None:
            self.set_disturbance_profile(np.asarray(d_trajectory, dtype=float))

        prof = self._horizon_profile
        if prof.disturbance_profile is not None:
            d_arr = np.asarray(prof.disturbance_profile, dtype=float)
            if d_arr.ndim == 1:
                d_arr = d_arr.reshape(-1, self._ocp._nd)
            d0 = d_arr[0].copy()
        else:
            d0 = self._d_prev.copy()

        if self._t_last is not None:
            dt = float(t) - self._t_last
            if dt > 0.0:
                self._estimator.predict_for(dt, self._u_prev, d0, p_, self._t_last)
            x_hat, P = self._estimator.update(y, self._u_prev, d0, p_)
        else:
            x_hat, P = self._estimator.step(y, self._u_prev, d0, p_, t)

        self._d_prev = d0.copy()
        self._t_last = float(t)
        return np.asarray(x_hat, dtype=float), P
