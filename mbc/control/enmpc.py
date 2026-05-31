"""
CDNMPCController — closed-loop continuous-discrete NMPC controller.

This module exposes:

* :class:`CDNMPCController` — closed-loop receding-horizon controller
  composing any continuous-discrete state estimator with any OCP that
  exposes ``solve(x0, d_trajectory, …) → (u_opt, cost, info)``.
"""

from __future__ import annotations

import numpy as np


# ── Generic CD-NMPC Controller ───────────────────────────────────────────────


class CDNMPCController:
    """
    Closed-loop continuous-discrete NMPC controller (ControlToolbox §EMPC —
    *ENMPC Algorithm*).

    Composes any continuous-discrete state estimator with any OCP that
    exposes ``solve(x0, d_trajectory, …) → (u_opt, cost, info)`` (or the
    legacy two-tuple ``(u_opt, cost)``) into a receding-horizon controller.

    At each measurement time t_k:

      1. **Measure**   y^{m,s}_k  (passed in via :meth:`step`)
      2. **Estimate**  z^c_k = κ(z^c_{k−1}, u_{k−1}, d_{k−1}, y^{m,s}_k, θ^c)
                       (delegated to ``estimator.step``)
      3. **Optimise**  u_k = λ(z^c_k, θ^c)  (delegated to ``ocp.solve``)
      4. **Apply**     return ``u_k`` to the caller, who advances the plant.

    Parameters
    ----------
    estimator : object with ``step(ym, u, d, p, t) → (x_hat, P)`` (or ``(x_hat, y_hat, P)`` for SDAEs)
        Continuous-discrete state estimator.
    ocp : object with ``solve``, ``N``, ``nu``
        Optimal control problem (NLP solver) — typically a
        :class:`~mbc.ocp.ContinuousNonlinearOCP`.
    """

    def __init__(self, estimator, ocp) -> None:
        self._estimator = estimator
        self._ocp = ocp
        self._u_seq_prev: np.ndarray | None = None
        self._x_traj_prev: np.ndarray | None = None
        self._y_traj_prev: np.ndarray | None = None
        self._u_prev: np.ndarray = np.zeros(ocp.nu)

    def step(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Execute one closed-loop ENMPC step.

        Parameters
        ----------
        y : (nym,) ndarray
            Current measurement ``y^{m,s}_k``.
        d_trajectory : (N, nd) ndarray
            Disturbance forecast over the horizon; ``d_trajectory[0] = d_k``.
        p : (nparams,) ndarray or None, optional
            Parameter vector ``θ^c``.  ``None`` → empty vector.
        t : float, optional
            Current time ``t_k``.

        Returns
        -------
        u_k : (nu,) ndarray
            Optimal input ``u_k`` to apply over ``[t_k, t_{k+1}]``.
        """
        p_ = np.array([], dtype=float) if p is None else np.asarray(p, dtype=float)
        d0 = d_trajectory[0]

        # 2. Estimate
        est_out = self._estimator.step(y, self._u_prev, d0, p_, t)
        # Estimator may return (x_hat, P) or (x_hat, y_hat, P).
        x_hat = est_out[0]

        # 3. Optimise
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

        # Cache for next warm-start
        self._u_seq_prev = u_opt
        self._x_traj_prev = info.get("X")
        self._y_traj_prev = info.get("Y")
        self._u_prev = u_k

        # 4. Apply (return to caller)
        return u_k
