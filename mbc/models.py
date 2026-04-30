"""
Abstract model interfaces for the mbc toolbox.

Linear discrete-time interface
-------------------------------
``LinearDiscreteModel`` — abstract base for linear discrete-time systems:

    x[k+1] = A(d) x[k] + B(d) u[k] + E(d) d[k] + offset,   y[k] = C x[k]

    x ∈ ℝⁿ  state,  u ∈ ℝᵐ  input (ZOH over each dt),
    d ∈ ℝᵖ  disturbance,    y ∈ ℝˡ  output.

    Matrices A, B, E may depend on d (LPV); C is time-invariant.

Continuous-discrete SDE interface (Ph.D. Ch. 5–6)
--------------------------------------------------
``ContinuousDiscreteModel`` — abstract base for continuous-discrete
stochastic systems:

    dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c)
    y_k = h(x_k, d_k) + v_k,                     v_k ~ N(0, R)

``ContinuousDiscreteDAEModel`` — extends the above with an algebraic
constraint:

    0 = l(x, z, u, d, t)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cvxopt import matrix


class LinearDiscreteModel(ABC):
    """
    Abstract interface for a linear discrete-time system:

      x[k+1] = A x[k] + B u[k] + E d[k] + offset,   y[k] = C x[k]

    C is time-invariant.  Subclasses implement ``discretize(d)`` to return
    the ZOH-discretised matrices as cvxopt dense matrices.  The ``d``
    argument is passed as operating-point context: LTI implementations may
    ignore it (A, B, E are constant), while LPV implementations use it to
    schedule the matrices (e.g. when actuator gain depends on an exogenous
    signal such as outdoor temperature).

    Parameter-identification interface
    -----------------------------------
    Subclasses that support system identification should additionally
    implement ``params`` and ``with_params``.  The concrete
    ``discretize_jacobian`` method provides finite-difference Jacobians by
    default; subclasses may override it with an analytic implementation for
    improved efficiency.
    """

    @property
    @abstractmethod
    def n_x(self) -> int:
        """State dimension n."""

    @property
    @abstractmethod
    def n_u(self) -> int:
        """Input dimension m."""

    @property
    @abstractmethod
    def n_d(self) -> int:
        """Disturbance dimension p."""

    @property
    @abstractmethod
    def C(self) -> "matrix":
        """Output matrix C ∈ ℝˡˣⁿ (cvxopt dense)."""

    @property
    @abstractmethod
    def x(self) -> list[float]:
        """Current state x as a plain list of floats."""

    @x.setter
    @abstractmethod
    def x(self, val: list[float]) -> None:
        ...

    @property
    @abstractmethod
    def x_ref(self) -> "matrix":
        """Reference / setpoint x_ref ∈ ℝⁿ (cvxopt column vector)."""

    @property
    @abstractmethod
    def u_bounds(self) -> Tuple["matrix", "matrix"]:
        """Box constraint on inputs (u_min, u_max), each (m, 1) cvxopt."""

    @abstractmethod
    def discretize(self, d: "matrix") -> Tuple["matrix", "matrix", "matrix"]:
        """
        Return ZOH-discretised matrices (A_d, B_d, E_d) as cvxopt dense
        matrices.

        The ``d`` argument is passed as operating-point context so that
        LPV implementations can schedule matrices on the current disturbance
        (e.g. heat-pump COP that varies with outdoor temperature).  Pure
        LTI implementations may ignore ``d`` and always return constant
        matrices.
        """

    # ── Parameter-identification interface (non-abstract, overridable) ────

    def predict_offset(self, d_np: np.ndarray) -> np.ndarray:
        """
        Additive constant term for the one-step prediction:

            x_pred = A x + B u + E d + predict_offset(d)

        The default implementation returns a zero vector.  Subclasses that
        model a known constant disturbance or an estimated bias term
        (e.g. internal heat gains that are not captured in *d*) should
        override this method.

        Parameters
        ----------
        d_np : (p,) ndarray  — current disturbance vector (numpy).

        Returns
        -------
        offset : (n,) ndarray
        """
        return np.zeros(self.n_x)

    @property
    def params(self) -> np.ndarray:
        """
        Current parameter vector *θ* as a flat numpy array.

        The default implementation returns an empty array, indicating that
        this model does not expose identifiable parameters via this
        interface.  Subclasses should override to return the natural
        parameter vector for system identification.
        """
        return np.array([], dtype=float)

    def with_params(self, theta: np.ndarray) -> "LinearDiscreteModel":
        """
        Return a **new** model instance constructed from parameter vector *θ*.

        This is the model factory used by
        ``discretize_jacobian`` for finite-difference perturbation.

        The default implementation raises :class:`NotImplementedError`.
        Subclasses that expose ``params`` should override this method.

        Parameters
        ----------
        theta : (p,) ndarray — parameter vector (same layout as ``params``).

        Returns
        -------
        LinearDiscreteModel
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_params."
        )

    def discretize_jacobian(

        self,
        d: np.ndarray,
        h: float = 1e-5,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Jacobians of the discretised matrices w.r.t. the parameter vector.

        Returns lists ``(dA, dB, dE)`` each of length ``len(self.params)``,
        where ``dA[i]``, ``dB[i]``, ``dE[i]`` are (n×n), (n×m), (n×p)
        numpy arrays representing

            ∂A_d/∂θ_i,  ∂B_d/∂θ_i,  ∂E_d/∂θ_i

        evaluated at the current ``self.params``.

        The default implementation uses forward finite differences via
        ``with_params``.  If ``params`` is empty or ``with_params`` is not
        implemented, empty lists are returned.  Subclasses may override
        with an analytic implementation.

        Parameters
        ----------
        d : (p,) ndarray — disturbance vector (numpy).
        h : float        — finite-difference step size.

        Returns
        -------
        dA : list of (n, n) ndarray
        dB : list of (n, m) ndarray
        dE : list of (n, p) ndarray
        """
        from cvxopt import matrix as cvx_matrix

        theta0 = self.params
        if len(theta0) == 0:
            return [], [], []

        def _to_np(m: "matrix") -> np.ndarray:
            rows, cols = m.size
            return np.array(list(m), dtype=float).reshape(
                (rows, cols), order="F"
            )

        d_cvx = cvx_matrix(d.tolist(), (len(d), 1), tc="d")
        A0, B0, E0 = self.discretize(d_cvx)
        A0_np, B0_np, E0_np = _to_np(A0), _to_np(B0), _to_np(E0)

        dA: List[np.ndarray] = []
        dB: List[np.ndarray] = []
        dE: List[np.ndarray] = []
        for i in range(len(theta0)):
            theta_h = theta0.copy()
            theta_h[i] += h
            try:
                m_h = self.with_params(theta_h)
            except NotImplementedError:
                return [], [], []
            Ah, Bh, Eh = m_h.discretize(d_cvx)
            dA.append((_to_np(Ah) - A0_np) / h)
            dB.append((_to_np(Bh) - B0_np) / h)
            dE.append((_to_np(Eh) - E0_np) / h)
        return dA, dB, dE


# ── Linear Continuous-Discrete Model ─────────────────────────────────────────


class LinearContinuousDiscreteModel(ABC):
    """
    Abstract interface for a linear continuous-discrete stochastic system.

    The continuous-time dynamics follow the Itô SDE

        dx = (A_c x[t] + B_c u[t] + E_c d[t]) dt + G dw[t],
        w[t] ~ N(0, Q_c)

    with zero-order-hold (ZOH) inputs and zero-order-hold disturbances over
    each sampling interval [t_k, t_{k+1}].  Observations are collected at the
    discrete measurement times t_k:

        y[k] = C x[k] + v[k],   v[k] ~ N(0, R)

    The model exposes the ZOH-discretised matrices via ``discretize`` so that
    the existing ``OptimalControlProblem`` (and ``CDOptimalControlProblem``) can
    be used without modification.

    Notation (M.Sc. thesis, Ch. 5)
    --------------------------------
        n   – state dimension              x ∈ ℝⁿ
        m   – input dimension              u ∈ ℝᵐ
        p   – disturbance dimension        d ∈ ℝᵖ
        l   – output dimension             y ∈ ℝˡ
        q   – process noise dimension      w ∈ ℝᵍ
        A_c – continuous state matrix      A_c ∈ ℝⁿˣⁿ
        B_c – continuous input matrix      B_c ∈ ℝⁿˣᵐ
        E_c – continuous disturbance mat.  E_c ∈ ℝⁿˣᵖ
        G   – noise input matrix           G ∈ ℝⁿˣᵍ
        Q_c – continuous process noise     Q_c ∈ ℝᵍˣᵍ
        R   – measurement noise cov.       R ∈ ℝˡˣˡ
        C   – output matrix                C ∈ ℝˡˣⁿ (time-invariant)
        dt  – sampling interval

    ZOH discretisation (``discretize``)
    -------------------------------------
    Computed via the augmented-matrix method (no matrix inverse required):

        A_d = expm(A_c · dt)
        [A_d | B_d | E_d] = expm([[A_c, B_c, E_c], [0, 0, 0], [0, 0, 0]] · dt)[:n, :]

    Discrete process noise (``discretize_noise``)
    -----------------------------------------------
    Computed by the Van Loan (1978) method:

        Q_d = ∫₀^{dt} expm(A_c τ) G Q_c Gᵀ expm(A_c τ)ᵀ dτ
    """

    # ── Abstract dimensions ───────────────────────────────────────────────

    @property
    @abstractmethod
    def n_x(self) -> int:
        """State dimension n."""

    @property
    @abstractmethod
    def n_u(self) -> int:
        """Input dimension m."""

    @property
    @abstractmethod
    def n_d(self) -> int:
        """Disturbance dimension p."""

    # ── Abstract continuous-time matrices (numpy) ─────────────────────────

    @property
    @abstractmethod
    def A_c(self) -> np.ndarray:
        """Continuous state matrix A_c ∈ ℝⁿˣⁿ."""

    @property
    @abstractmethod
    def B_c(self) -> np.ndarray:
        """Continuous input matrix B_c ∈ ℝⁿˣᵐ."""

    @property
    @abstractmethod
    def E_c(self) -> np.ndarray:
        """Continuous disturbance matrix E_c ∈ ℝⁿˣᵖ."""

    @property
    @abstractmethod
    def G(self) -> np.ndarray:
        """Noise input matrix G ∈ ℝⁿˣᵍ."""

    @property
    @abstractmethod
    def Q_c(self) -> np.ndarray:
        """Continuous process-noise covariance Q_c ∈ ℝᵍˣᵍ."""

    # ── Abstract observation matrices (cvxopt, for OCP compatibility) ─────

    @property
    @abstractmethod
    def C(self) -> "matrix":
        """Output matrix C ∈ ℝˡˣⁿ (cvxopt dense)."""

    @property
    @abstractmethod
    def R(self) -> "matrix":
        """Measurement noise covariance R ∈ ℝˡˣˡ (cvxopt dense)."""

    # ── Abstract sampling interval ────────────────────────────────────────

    @property
    @abstractmethod
    def dt(self) -> float:
        """Sampling interval (seconds)."""

    # ── Abstract control-interface properties ────────────────────────────

    @property
    @abstractmethod
    def x(self) -> list[float]:
        """Current state x as a plain list of floats."""

    @x.setter
    @abstractmethod
    def x(self, val: list[float]) -> None: ...

    @property
    @abstractmethod
    def x_ref(self) -> "matrix":
        """Reference / setpoint x_ref ∈ ℝⁿ (cvxopt column vector)."""

    @property
    @abstractmethod
    def u_bounds(self) -> Tuple["matrix", "matrix"]:
        """Box constraint on inputs (u_min, u_max), each (m, 1) cvxopt."""

    # ── Concrete discretisation methods ───────────────────────────────────

    def discretize(self, d: "matrix") -> Tuple["matrix", "matrix", "matrix"]:
        """
        ZOH-discretised matrices (A_d, B_d, E_d) as cvxopt dense matrices.

        Uses the augmented-matrix method so that no matrix inverse is
        required:

            expm([[A_c, B_c, E_c], [0, 0, 0], [0, 0, 0]] · dt)[:n, :]
            = [A_d | B_d | E_d]

        The ``d`` argument is accepted for interface compatibility with
        ``OptimalControlProblem`` (LPV sub-classes may override this method
        to schedule matrices on the current disturbance); the default LTI
        implementation ignores it.

        Parameters
        ----------
        d : (p, 1) cvxopt column  — current disturbance (ignored for LTI).

        Returns
        -------
        A_d : (n, n) cvxopt dense — discrete state-transition matrix.
        B_d : (n, m) cvxopt dense — discrete input matrix.
        E_d : (n, p) cvxopt dense — discrete disturbance matrix.
        """
        from ._utils import _zoh_full, _np_to_cvx

        A_d_np, B_d_np, E_d_np = _zoh_full(self.A_c, self.B_c, self.E_c, self.dt)
        return _np_to_cvx(A_d_np), _np_to_cvx(B_d_np), _np_to_cvx(E_d_np)

    def discretize_noise(self) -> "matrix":
        """
        Exact discrete process-noise covariance Q_d via Van Loan (1978).

        Computes

            Q_d = ∫₀^{dt} expm(A_c τ) G Q_c Gᵀ expm(A_c τ)ᵀ dτ

        using the augmented 2n×2n matrix method.  The result is symmetric
        positive semi-definite by construction.

        Returns
        -------
        Q_d : (n, n) cvxopt dense — discrete process-noise covariance.
        """
        from ._utils import _van_loan, _np_to_cvx

        Q_d_np = _van_loan(self.A_c, self.G, self.Q_c, self.dt)
        return _np_to_cvx(Q_d_np)

    # ── Parameter-identification interface (non-abstract, overridable) ────

    @property
    def params(self) -> np.ndarray:
        """
        Current parameter vector θ as a flat numpy array.

        Default: empty (no identifiable parameters exposed).  Subclasses
        should override to support system identification.
        """
        return np.array([], dtype=float)

    def with_params(self, theta: np.ndarray) -> "LinearContinuousDiscreteModel":
        """
        Return a new model instance from parameter vector θ.

        Default implementation raises :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement with_params."
        )


# ── Continuous-Discrete SDE Model ────────────────────────────────────────────


class ContinuousDiscreteModel(ABC):
    """
    Abstract interface for a continuous-discrete stochastic system (Ph.D. Ch. 5).

    The system is governed by the Itô SDE

        dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c)

    with discrete-time observations

        y_k = h(x_k, d_k) + v_k,   v_k ~ N(0, R)

    Subclasses must implement the drift ``f``, diffusion ``g``, observation
    ``h``, and the noise covariance properties ``Q_c`` and ``R``.
    """

    @abstractmethod
    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Drift function f(x, u, d, t).

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        t : current time.

        Returns
        -------
        (nx,) drift value.
        """

    @abstractmethod
    def g(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Diffusion function g(x, u, d, t).

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        t : current time.

        Returns
        -------
        (nx, nw) diffusion matrix.
        """

    @abstractmethod
    def h(
        self,
        x: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        Observation function h(x_k, d_k).

        Parameters
        ----------
        x : (nx,) state vector at measurement time k.
        d : (nd,) disturbance vector at measurement time k.

        Returns
        -------
        (ny,) predicted observation.
        """

    @property
    @abstractmethod
    def Q_c(self) -> np.ndarray:
        """Continuous-time process noise covariance Q_c ∈ ℝⁿʷˣⁿʷ."""

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Measurement noise covariance R ∈ ℝⁿʸˣⁿʸ."""

    @property
    @abstractmethod
    def nx(self) -> int:
        """State dimension."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Input dimension."""

    @property
    @abstractmethod
    def nd(self) -> int:
        """Disturbance dimension."""

    @property
    @abstractmethod
    def ny(self) -> int:
        """Output dimension."""


# ── Continuous-Discrete SDAE Model ───────────────────────────────────────────


class ContinuousDiscreteDAEModel(ContinuousDiscreteModel):
    """
    Abstract interface for a continuous-discrete stochastic DAE (Ph.D. Ch. 6).

    Extends ``ContinuousDiscreteModel`` with algebraic state z and constraint:

        dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
        0  = l(x, z, u, d, t)
        y_k = h(x_k, z_k, d_k) + v_k

    At each integration step the algebraic constraint is enforced by solving
    ``l = 0`` for z (typically via Newton iteration in the simulator).

    Subclasses must additionally implement ``l`` and ``nz``.  The observation
    function ``h`` should be overridden to accept ``(x, z, d)`` if the
    outputs depend on z; the base signature ``h(x, d)`` is retained for
    compatibility with ``ContinuousDiscreteModel``-typed interfaces.
    """

    @abstractmethod
    def l(
        self,
        x: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Algebraic constraint residual.

        The constraint is satisfied when ``l(x, z, u, d, t) = 0``.

        Parameters
        ----------
        x : (nx,) differential state vector.
        z : (nz,) algebraic state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        t : current time.

        Returns
        -------
        (nz,) residual vector.
        """

    @property
    @abstractmethod
    def nz(self) -> int:
        """Algebraic state dimension."""
