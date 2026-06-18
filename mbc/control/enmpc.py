"""
Nonlinear continuous-time MPC for continuous-discrete plants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .continuous_ocp import ContinuousOptimalControlProblem, GeneralContinuousOCP
from .mpc_horizon import HorizonProfileMPC


class NonlinearContinuousMPC(ABC):
    """Abstract MPC for nonlinear CD plant + CD estimator + continuous OCP."""

    @abstractmethod
    def step(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """Execute one closed-loop NMPC step."""


class StandardNonlinearContinuousMPC(HorizonProfileMPC, NonlinearContinuousMPC):
    """
    Standard closed-loop NMPC for nonlinear continuous-discrete plants.

    Composes a continuous-discrete state estimator with a
    :class:`GeneralContinuousOCP` (or subclass).
    """

    def __init__(self, estimator, ocp: ContinuousOptimalControlProblem) -> None:
        super().__init__()
        self._estimator = estimator
        self._ocp = ocp
        self._u_seq_prev: np.ndarray | None = None
        self._x_traj_prev: np.ndarray | None = None
        self._y_traj_prev: np.ndarray | None = None
        self._u_prev: np.ndarray = np.zeros(ocp.nu)

    def step(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        p_ = np.array([], dtype=float) if p is None else np.asarray(p, dtype=float)
        prof = self._horizon_profile
        if d_trajectory is None:
            if prof.disturbance_profile is None:
                raise ValueError(
                    "Provide d_trajectory via step(…) or set_disturbance_profile()."
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

        return u_k
