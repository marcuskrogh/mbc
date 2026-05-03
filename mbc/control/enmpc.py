"""
Economic Nonlinear Optimal Control Problem and CD-NMPC Controller (Ph.D. Ch. 9).

Unlike tracking MPC (which minimises a quadratic distance to a setpoint),
the economic optimal control problem minimises an economic stage cost
l_e(x, u, d) that directly represents an economic criterion such as energy
cost, yield, or profit.

``EconomicOptimalControlProblem``
    Solves the finite-horizon NLP at each step:

        min_{u_0, …, u_{N-1}}   Σ_{k=0}^{N-1} l_e(x_k, u_k, d_k) + V_f(x_N)
                               + soft constraint penalties on x and output(x, u, d)

        subject to:
            x_{k+1} = f̄(x_k, u_k, d_k)    (discretised model dynamics)
            u_k ∈ U                         (hard input constraints via scipy)

    where ``f̄`` is obtained by explicit Euler integration of the mean
    dynamics over one sampling interval (no noise).  The NLP is solved by
    ``scipy.optimize.minimize`` with the SLSQP method (default).

    Soft constraints on states and controlled outputs are penalised via a
    quadratic term added to the objective:

        penalty = (1/2) w ‖max(0, lb − z)‖² + (1/2) w ‖max(0, z − ub)‖²

``CDNMPCController``
    Closed-loop receding-horizon controller for continuous-discrete (CD)
    models.  Composes a state estimator with any OCP exposing a
    ``step(x0, d_trajectory, u_prev, p, t0)`` interface:

      1. **Estimate**   x̂[k] ← estimator.step(y[k], u[k−1], d[k], p, t_k)
      2. **Optimise**   U*   ← ocp.solve(x̂[k], d_trajectory, u_seq_prev, p)
      3. **Apply**      u[k] = U*[0]

    This controller is general: it works with both economic OCPs
    (``EconomicOptimalControlProblem``) and any tracking OCP that follows
    the same numpy-based interface.

Reference:  Ph.D. thesis, Ch. 9.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..models import ContinuousDiscreteModel


class EconomicOptimalControlProblem:
    """
    Economic nonlinear optimal control problem (Ph.D. Ch. 9).

    Solves the finite-horizon economic NLP from a given initial state.
    The predicted trajectory is computed by explicit Euler integration
    of the mean dynamics (no stochastic noise) over each sampling interval.

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
        Hard input constraints in ``scipy.optimize.minimize`` format
        (``{"type": "ineq"/"eq", "fun": ...}``).  These are enforced as
        hard constraints by the NLP solver.
    state_lb : (nx,) array-like or None, optional
        Soft lower bounds on the state vector at each prediction step.
        Violations are penalised quadratically with weight ``state_weight``.
    state_ub : (nx,) array-like or None, optional
        Soft upper bounds on the state vector at each prediction step.
    state_weight : float, optional
        Penalty weight for soft state constraint violations.  Default: 1.0.
    output_lb : (n_out,) array-like or None, optional
        Soft lower bounds on the controlled output ``model.output(x, u, d, p)``
        at each prediction step.
    output_ub : (n_out,) array-like or None, optional
        Soft upper bounds on the controlled output at each prediction step.
    output_weight : float, optional
        Penalty weight for soft output constraint violations.  Default: 1.0.
    n_steps : int, optional
        Explicit-Euler integration sub-steps per sampling interval.
        Default: 10.
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
        state_lb: np.ndarray | None = None,
        state_ub: np.ndarray | None = None,
        state_weight: float = 1.0,
        output_lb: np.ndarray | None = None,
        output_ub: np.ndarray | None = None,
        output_weight: float = 1.0,
        n_steps: int = 10,
        solver: str = "SLSQP",
        solver_options: dict | None = None,
    ) -> None:
        self._model = model
        self._N = N
        self._stage_cost = stage_cost
        self._terminal_cost = terminal_cost
        self._constraints = constraints if constraints is not None else []
        self._state_lb = np.asarray(state_lb, dtype=float) if state_lb is not None else None
        self._state_ub = np.asarray(state_ub, dtype=float) if state_ub is not None else None
        self._state_weight = float(state_weight)
        self._output_lb = np.asarray(output_lb, dtype=float) if output_lb is not None else None
        self._output_ub = np.asarray(output_ub, dtype=float) if output_ub is not None else None
        self._output_weight = float(output_weight)
        self._n_steps = n_steps
        self._solver = solver
        self._solver_options = solver_options
        # Sampling interval: taken from model.dt if available, else 1.0
        self._dt: float = float(getattr(model, "dt", 1.0))

    @property
    def N(self) -> int:
        """Prediction horizon (number of sampling intervals)."""
        return self._N

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

    def _soft_penalty(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """
        Compute the soft-constraint penalty for state and output violations.

        Violations of the form ``max(0, lb − z)`` and ``max(0, z − ub)``
        are penalised quadratically:

            penalty = (1/2) w ‖max(0, lb − z)‖² + (1/2) w ‖max(0, z − ub)‖²

        Parameters
        ----------
        x : (nx,) current state.
        u : (nu,) current input.
        d : (nd,) current disturbance.
        p : (nparams,) parameter vector.

        Returns
        -------
        float — total soft penalty at this prediction step.
        """
        penalty = 0.0
        # ── State soft constraints ────────────────────────────────────────
        if self._state_lb is not None:
            viol = np.maximum(0.0, self._state_lb - x)
            penalty += 0.5 * self._state_weight * float(np.dot(viol, viol))
        if self._state_ub is not None:
            viol = np.maximum(0.0, x - self._state_ub)
            penalty += 0.5 * self._state_weight * float(np.dot(viol, viol))
        # ── Output soft constraints ───────────────────────────────────────
        if self._output_lb is not None or self._output_ub is not None:
            z = self._model.output(x, u, d, p)
            if self._output_lb is not None:
                viol = np.maximum(0.0, self._output_lb - z)
                penalty += 0.5 * self._output_weight * float(np.dot(viol, viol))
            if self._output_ub is not None:
                viol = np.maximum(0.0, z - self._output_ub)
                penalty += 0.5 * self._output_weight * float(np.dot(viol, viol))
        return penalty

    def solve(
        self,
        x0: np.ndarray,
        d_trajectory: np.ndarray,
        u_prev: np.ndarray | None = None,
        p: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """
        Solve the economic OCP from initial state x0.

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
        p : (nparams,) ndarray or None, optional
            Parameter vector.  ``None`` uses ``model.params``.
        t0 : float, optional
            Start time for the prediction horizon.  Default: 0.

        Returns
        -------
        u_opt : (N, nu) ndarray
            Optimal input sequence.
        cost : float
            Optimal economic cost (including soft-constraint penalties).
        """
        from scipy.optimize import minimize

        N = self._N
        nu = self._model.nu
        p_ = self._model.params if p is None else p

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
            t = t0
            total = 0.0
            for k in range(N):
                total += float(self._stage_cost(x, U[k], d_trajectory[k]))
                total += self._soft_penalty(x, U[k], d_trajectory[k], p_)
                x = self._predict_mean(x, U[k], d_trajectory[k], p_, t)
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
        p: np.ndarray | None = None,
        t0: float = 0.0,
    ) -> np.ndarray:
        """
        Solve and return only the first optimal control action.

        This is the standard receding-horizon call for the OCP.

        Parameters
        ----------
        x0 : (nx,) ndarray
            Current state estimate.
        d_trajectory : (N, nd) ndarray
            Predicted disturbance trajectory.
        u_prev : (N, nu) ndarray or None, optional
            Previous optimal sequence for warm-starting.
        p : (nparams,) ndarray or None, optional
            Parameter vector.  ``None`` uses ``model.params``.
        t0 : float, optional
            Start time.  Default: 0.

        Returns
        -------
        u0 : (nu,) ndarray
            First element of the optimal input sequence.
        """
        u_opt, _ = self.solve(x0, d_trajectory, u_prev, p=p, t0=t0)
        return u_opt[0]


class CDNMPCController:
    """
    Continuous-Discrete Nonlinear MPC Controller (Ph.D. Ch. 9).

    A general receding-horizon controller for continuous-discrete (CD) models.
    Composes a state estimator with any optimal control problem that exposes
    a ``step(x0, d_trajectory, u_prev, p, t0) → u0`` interface.

    At each measurement time t_k the following steps are executed:

      1. **Estimate**  x̂[k] ← ``estimator.step(y[k], u[k−1], d[k], p, t_k)``
      2. **Optimise**  u[k]  ← ``ocp.step(x̂[k], d_trajectory, u_seq_prev, p, t_k)``
      3. **Apply**     u[k]   (receding horizon)

    This controller is agnostic to the type of OCP: it works equally with
    economic OCPs (:class:`EconomicOptimalControlProblem`) and any tracking
    OCP that follows the same numpy-based calling convention.

    The estimator is expected to expose a ``step(y, u, d, p, t)`` method that
    performs a combined predict-and-update and returns ``(x_hat, P)``.
    This matches the interface of :class:`~mbc.estimation.ContinuousDiscreteEKF`
    and other CD estimators in the ``mbc.estimation`` sub-package.

    Parameters
    ----------
    model : ContinuousDiscreteModel
        Plant model.  Used only to retrieve ``nu``, ``nd``, and ``params``.
    estimator : object with ``step(y, u, d, p, t) → (x_hat, P)``
        State estimator.  Typically a
        :class:`~mbc.estimation.ContinuousDiscreteEKF`.
    ocp : object with ``step(x0, d_trajectory, u_prev, p, t0) → u0`` and ``N``
        Optimal control problem.  Typically an
        :class:`EconomicOptimalControlProblem`.
    """

    def __init__(
        self,
        model: ContinuousDiscreteModel,
        estimator,
        ocp,
    ) -> None:
        self._model = model
        self._estimator = estimator
        self._ocp = ocp
        self._u_seq_prev: np.ndarray | None = None
        self._u_prev: np.ndarray = np.zeros(model.nu)

    def step(
        self,
        y: np.ndarray,
        d_trajectory: np.ndarray,
        p: np.ndarray | None = None,
        t: float = 0.0,
    ) -> np.ndarray:
        """
        Execute one CD-NMPC step.

        Parameters
        ----------
        y : (ny,) ndarray
            Current measurement y[k].
        d_trajectory : (N, nd) ndarray
            Predicted disturbance trajectory over the horizon.
            ``d_trajectory[0]`` is the current disturbance d[k].
        p : (nparams,) ndarray or None, optional
            Parameter vector.  ``None`` uses ``model.params``.
        t : float, optional
            Current measurement time t_k.  Default: 0.

        Returns
        -------
        u : (nu,) ndarray
            Optimal input u[k] to apply.
        """
        p_ = self._model.params if p is None else p
        d0 = d_trajectory[0]

        # Step 1: estimate
        x_hat, _ = self._estimator.step(y, self._u_prev, d0, p_, t)

        # Step 2: optimise
        u0 = self._ocp.step(x_hat, d_trajectory, self._u_seq_prev, p=p_, t0=t)

        # Update warm-start storage
        nu = self._model.nu
        N = self._ocp.N
        if self._u_seq_prev is None:
            self._u_seq_prev = np.zeros((N, nu))
        self._u_seq_prev[:-1] = self._u_seq_prev[1:]
        self._u_seq_prev[-1] = u0

        self._u_prev = u0
        return u0


#: Backward-compatible alias.  Use :class:`CDNMPCController` for new code.
EconomicNMPCController = CDNMPCController

