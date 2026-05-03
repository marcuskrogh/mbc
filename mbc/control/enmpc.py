"""
Economic Nonlinear MPC (Ph.D. Ch. 9).

Unlike tracking MPC (which minimises a quadratic distance to a setpoint),
Economic NMPC minimises an economic stage cost l_e(x, u, d) that directly
represents an economic criterion such as energy cost, yield, or profit.

The finite-horizon optimal control problem solved at each time step is:

    min_{u_0, …, u_{N-1}}   Σ_{k=0}^{N-1} l_e(x_k, u_k, d_k) + V_f(x_N)

    subject to:
        x_{k+1} = f̄(x_k, u_k, d_k)    (discretised model dynamics)
        u_k ∈ U                          (input constraints)
        x_k ∈ X                          (state constraints, optional)

where ``f̄`` is obtained by numerically integrating the continuous
dynamics over one sampling interval via ``SDESimulator`` (mean dynamics,
no noise).

The NLP is solved by ``scipy.optimize.minimize`` with the SLSQP method
(default).  Warm-starting from the previous solution is supported.

Reference:  Ph.D. thesis, Ch. 9.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..models import ContinuousDiscreteModel


class EconomicNMPC:
    """
    Economic Nonlinear MPC controller (Ph.D. Ch. 9).

    Parameters
    ----------
    model : ContinuousDiscreteModel
        Nonlinear continuous-discrete model used for prediction.
    N : int
        Prediction horizon (number of sampling intervals).
    stage_cost : Callable[[np.ndarray, np.ndarray, np.ndarray], float]
        Economic stage cost ``l_e(x, u, d)`` — a scalar-valued function
        of state, input, and disturbance.
    terminal_cost : Callable[[np.ndarray], float] or None, optional
        Terminal cost ``V_f(x_N)``.  ``None`` means no terminal cost.
    constraints : list of dict or None, optional
        Additional constraints in ``scipy.optimize.minimize`` format
        (``{"type": "ineq"/"eq", "fun": ...}``).  Input and state box
        constraints should be supplied here.
    n_steps : int, optional
        Euler integration sub-steps per sampling interval for the
        prediction model (mean dynamics, no noise).  Default: 10.
    solver : str, optional
        NLP solver passed to ``scipy.optimize.minimize``.  Default: ``"SLSQP"``.
    solver_options : dict or None, optional
        Options forwarded to the solver.  ``None`` uses solver defaults.
    """

    def __init__(
        self,
        model: ContinuousDiscreteModel,
        N: int,
        stage_cost: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
        terminal_cost: Callable[[np.ndarray], float] | None = None,
        constraints: list | None = None,
        n_steps: int = 10,
        solver: str = "SLSQP",
        solver_options: dict | None = None,
    ) -> None:
        self._model = model
        self._N = N
        self._stage_cost = stage_cost
        self._terminal_cost = terminal_cost
        self._constraints = constraints if constraints is not None else []
        self._n_steps = n_steps
        self._solver = solver
        self._solver_options = solver_options
        # Sampling interval: taken from model.dt if available, else 1.0
        self._dt: float = float(getattr(model, "dt", 1.0))

    def _predict_mean(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Integrate mean dynamics over one sampling interval (no noise).

        Uses ``n_steps`` explicit Euler sub-steps of size
        ``h = dt / n_steps``.

        Parameters
        ----------
        x : (nx,) state at the start of the interval.
        u : (nu,) input applied over the interval (ZOH).
        d : (nd,) disturbance applied over the interval (ZOH).
        p : (nparams,) parameter vector.
        t : float — start time of the interval.

        Returns
        -------
        x_next : (nx,) state at the end of the interval.
        """
        h = self._dt / self._n_steps
        x_cur = x.copy()
        t_cur = t
        for _ in range(self._n_steps):
            x_cur = x_cur + self._model.f(x_cur, u, d, p, t_cur) * h
            t_cur += h
        return x_cur

    def solve(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Solve the economic NMPC problem from initial state x0.

        Parameters
        ----------
        x0 : (nx,) ndarray
            Current state estimate (initial condition for the NLP).
        d_trajectory : (N, nd) ndarray
            Predicted disturbance trajectory over the horizon.
        u_prev : (N, nu) ndarray or None, optional
            Previous optimal input sequence used as a warm start.
            Shifted by one step (last element repeated).  ``None``
            initialises from zeros.

        Returns
        -------
        u_opt : (N, nu) ndarray
            Optimal input sequence.
        cost : float
            Optimal economic cost.
        """
        from scipy.optimize import minimize

        N = self._N
        nu = self._model.nu
        p = self._model.params

        # ── Warm start ───────────────────────────────────────────────────
        if u_prev is not None:
            u0 = np.empty_like(u_prev)
            u0[:-1] = u_prev[1:]   # shift forward by one step
            u0[-1] = u_prev[-1]    # repeat last element
        else:
            u0 = np.zeros((N, nu))
        u0_flat = u0.ravel()

        # ── Objective function ───────────────────────────────────────────
        def objective(u_flat: np.ndarray) -> float:
            U = u_flat.reshape(N, nu)
            x = x0.copy()
            t = 0.0
            total = 0.0
            for k in range(N):
                total += float(self._stage_cost(x, U[k], d_trajectory[k]))
                x = self._predict_mean(x, U[k], d_trajectory[k], p, t)
                t += self._dt
            if self._terminal_cost is not None:
                total += float(self._terminal_cost(x))
            return total

        # ── Solve NLP ────────────────────────────────────────────────────
        result = minimize(
            objective,
            u0_flat,
            method=self._solver,
            constraints=self._constraints,
            options=self._solver_options,
        )

        u_opt = result.x.reshape(N, nu)
        cost = float(result.fun)
        return u_opt, cost

    def step(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve and return only the first optimal control action.

        This is the standard receding-horizon call for closed-loop control.

        Parameters
        ----------
        x0 : (nx,) ndarray
            Current state estimate.
        d_trajectory : (N, nd) ndarray
            Predicted disturbance trajectory.
        u_prev : (N, nu) ndarray or None, optional
            Previous optimal sequence for warm-starting.

        Returns
        -------
        u0 : (nu,) ndarray
            First element of the optimal input sequence.
        """
        u_opt, _ = self.solve(x0, d_trajectory, u_prev)
        return u_opt[0]
