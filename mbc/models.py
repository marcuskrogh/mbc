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

    dx = f(x, u, d, t) dt + sigma(x, u, d, t) dw,   w ~ N(0, Q_c)
    y_k = hm(x_k, u_k, d_k) + v_k,                  v_k ~ N(0, R)

``LinearContinuousDiscreteModel`` — extends ``ContinuousDiscreteModel`` for
linear systems where the drift, diffusion, and observation functions take
the specific forms:

    f(x, u, d, t) = A_c x + B_c u + E_c d
    sigma(x, u, d, t) = G            (constant diffusion)
    hm(x, u, d)    = C_m x          (linear output; u ignored)

``ContinuousDiscreteDAEModel`` — extends ``ContinuousDiscreteModel`` with an
algebraic constraint:

    0 = h(x, y, u, d, t)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cvxopt import matrix


_H_FD: float = 1e-5  # default finite-difference step for Jacobians


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


# ── Continuous-Discrete SDE Model ────────────────────────────────────────────


class ContinuousDiscreteModel(ABC):
    """
    Abstract interface for a continuous-discrete stochastic system (Ph.D. Ch. 5).

    The system is governed by the Itô SDE

        dx = f(x, u, d, t) dt + sigma(x, u, d, t) dw,   w ~ N(0, Q_c)

    with discrete-time observations

        y_k = hm(x_k, u_k, d_k) + v_k,   v_k ~ N(0, R)

    Subclasses must implement the drift ``f``, diffusion ``sigma``,
    measurement ``hm``, and the noise covariance properties ``Q_c`` and ``R``.

    Dimensions
    ----------
        nx  – state dimension              x ∈ ℝⁿˣ
        nu  – input dimension              u ∈ ℝⁿᵘ
        nd  – disturbance dimension        d ∈ ℝⁿᵈ
        nym – measurement dimension        y_m ∈ ℝⁿʸᵐ
        nz  – controlled output dimension  z ∈ ℝⁿᶻ  (defaults to nym)
        nw  – process-noise dimension      w ∈ ℝⁿʷ  (columns of sigma's output)
    """

    @abstractmethod
    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Drift function f(x, u, d, p, t).

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (nx,) drift value.
        """

    @abstractmethod
    def sigma(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Diffusion function sigma(x, u, d, p, t).

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (nx, nw) diffusion matrix.
        """

    @abstractmethod
    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Measurement function hm(x_k, u_k, d_k, p).

        Maps the state to the sensor measurements used by state estimators.

        Parameters
        ----------
        x : (nx,) state vector at measurement time k.
        u : (nu,) input vector at measurement time k.
        d : (nd,) disturbance vector at measurement time k.
        p : (nparams,) parameter vector.

        Returns
        -------
        (nym,) predicted measurement.
        """

    def g(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Controlled output function g(x_k, u_k, d_k, p).

        Maps the state to the controlled output variables used by optimal
        control problems for objectives and output constraints.  The output
        variables need not coincide with the sensor measurements returned by
        ``hm``; for example, one may measure temperature but optimise over
        energy cost (a nonlinear function of state and input).

        The default implementation returns ``hm(x, u, d, p)`` so that
        subclasses without a custom output work without modification.
        Override to decouple outputs from measurements.

        Parameters
        ----------
        x : (nx,) state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.

        Returns
        -------
        (nz,) controlled output vector.
        """
        return self.hm(x, u, d, p)

    @property
    def nz(self) -> int:
        """
        Controlled output dimension.

        Returns the number of outputs produced by ``g()``.  Defaults to
        ``nym`` (same as the measurement dimension).  Override when the
        output dimension differs from the measurement dimension.
        """
        return self.nym

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
    def nym(self) -> int:
        """Measurement output dimension."""

    @property
    @abstractmethod
    def nw(self) -> int:
        """Process-noise / diffusion dimension nw (columns of sigma's output)."""

    # ── Jacobian methods (default: forward finite differences) ────────────

    def dfdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian F = ∂f/∂x evaluated at (x, u, d, p, t)  →  (nx, nx) ndarray.

        Default implementation uses forward finite differences with step
        ``_H_FD``.  Subclasses may override with an analytic Jacobian.
        """
        f0 = self.f(x, u, d, p, t)
        nx = x.shape[0]
        J = np.empty((nx, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.f(x_fwd, u, d, p, t) - f0) / _H_FD
        return J

    def dhmdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian H = ∂hm/∂x evaluated at (x, u, d, p)  →  (nym, nx) ndarray.

        Default implementation uses forward finite differences with step
        ``_H_FD``.  Subclasses may override with an analytic Jacobian.
        """
        h0 = self.hm(x, u, d, p)
        nx = x.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.hm(x_fwd, u, d, p) - h0) / _H_FD
        return J

    def dhmdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂u evaluated at (x, u, d, p)  →  (nym, nu) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        h0 = self.hm(x, u, d, p)
        nu = u.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, u_fwd, d, p) - h0) / _H_FD
        return J

    def dfdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂u evaluated at (x, u, d, p, t)  →  (nx, nu) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        f0 = self.f(x, u, d, p, t)
        nu = u.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.f(x, u_fwd, d, p, t) - f0) / _H_FD
        return J

    def dfdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂d evaluated at (x, u, d, p, t)  →  (nx, nd) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        f0 = self.f(x, u, d, p, t)
        nd = d.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.f(x, u, d_fwd, p, t) - f0) / _H_FD
        return J

    def dhmdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂d evaluated at (x, u, d, p)  →  (nym, nd) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        h0 = self.hm(x, u, d, p)
        nd = d.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, u, d_fwd, p) - h0) / _H_FD
        return J

    def dfdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂p evaluated at (x, u, d, p, t)  →  (nx, nparams) ndarray.

        Default: forward finite differences.  Subclasses may override.
        Returns an empty (nx, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nx = self.nx
        if nparams == 0:
            return np.empty((nx, 0))
        f0 = self.f(x, u, d, p, t)
        J = np.empty((nx, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.f(x, u, d, p_fwd, t) - f0) / _H_FD
        return J

    def dhmdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂p evaluated at (x, u, d, p)  →  (nym, nparams) ndarray.

        Default: forward finite differences.  Subclasses may override.
        Returns an empty (nym, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nym = self.nym
        if nparams == 0:
            return np.empty((nym, 0))
        h0 = self.hm(x, u, d, p)
        J = np.empty((nym, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, u, d, p_fwd) - h0) / _H_FD
        return J

    # ── Output Jacobians (default: forward finite differences) ──────────

    def dgdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂g/∂x evaluated at (x, u, d, p)  →  (nz, nx) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        z0 = self.g(x, u, d, p)
        nx = x.shape[0]
        nz = z0.shape[0]
        J = np.empty((nz, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.g(x_fwd, u, d, p) - z0) / _H_FD
        return J

    def dgdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂g/∂u evaluated at (x, u, d, p)  →  (nz, nu) ndarray.

        Default: forward finite differences.  Subclasses may override.
        """
        z0 = self.g(x, u, d, p)
        nu = u.shape[0]
        nz = z0.shape[0]
        J = np.empty((nz, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.g(x, u_fwd, d, p) - z0) / _H_FD
        return J

    # ── Parameters ────────────────────────────────────────────────────────

    @property
    def params(self) -> np.ndarray:
        """
        Default parameter vector θ as a flat numpy array.

        Default: empty.  Subclasses should override to return the current
        parameter vector, which callers may use as the default ``p``.
        """
        return np.array([], dtype=float)


# ── Linear Continuous-Discrete Model ─────────────────────────────────────────


class LinearContinuousDiscreteModel(ContinuousDiscreteModel):
    """
    Abstract interface for a linear continuous-discrete stochastic system.

    Extends :class:`ContinuousDiscreteModel` with the specific linear forms:

        dx = (A_c x[t] + B_c u[t] + E_c d[t]) dt + G dw[t],
        w[t] ~ N(0, Q_c)

    with zero-order-hold (ZOH) inputs and disturbances over each sampling
    interval [t_k, t_{k+1}].  Observations are collected at the discrete
    measurement times t_k:

        y[k] = C x[k] + v[k],   v[k] ~ N(0, R)

    The model exposes the ZOH-discretised matrices via ``discretize`` so that
    the existing ``OptimalControlProblem`` (and ``CDOptimalControlProblem``) can
    be used without modification.

    Notation (M.Sc. thesis, Ch. 5)
    --------------------------------
        nx  – state dimension              x ∈ ℝⁿˣ
        nu  – input dimension              u ∈ ℝⁿᵘ
        nd  – disturbance dimension        d ∈ ℝⁿᵈ
        ny  – output dimension             y ∈ ℝⁿʸ  (derived: C.shape[0])
        nw  – process-noise dimension      w ∈ ℝⁿʷ  (derived: G.shape[1])
        A_c – continuous state matrix      A_c ∈ ℝⁿˣˣⁿˣ
        B_c – continuous input matrix      B_c ∈ ℝⁿˣˣⁿᵘ
        E_c – continuous disturbance mat.  E_c ∈ ℝⁿˣˣⁿᵈ
        G   – noise input matrix           G ∈ ℝⁿˣˣⁿʷ
        Q_c – continuous process noise     Q_c ∈ ℝⁿʷˣⁿʷ
        R   – measurement noise cov.       R ∈ ℝⁿʸˣⁿʸ
        C   – output matrix                C ∈ ℝⁿʸˣⁿˣ (time-invariant)
        dt  – sampling interval

    Concrete implementations
    ------------------------
    The following abstract methods from :class:`ContinuousDiscreteModel` are
    implemented concretely:

        f(x, u, d, t) = A_c x + B_c u + E_c d
        g(x, u, d, t) = G            (constant diffusion; arguments ignored)
        h(x, d)       = C x          (linear output; d ignored for LTI)
        ny            = C.shape[0]
        nw            = G.shape[1]

    ZOH discretisation (``discretize``)
    -------------------------------------
    Computed via the augmented-matrix method (no matrix inverse required):

        A_d = expm(A_c · dt)
        [A_d | B_d | E_d] = expm([[A_c, B_c, E_c], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]

    Discrete process noise (``discretize_noise``)
    -----------------------------------------------
    Computed by the Van Loan (1978) method:

        Q_d = ∫₀^{dt} expm(A_c τ) G Q_c Gᵀ expm(A_c τ)ᵀ dτ

    Backward-compatible aliases
    ----------------------------
    The deprecated properties ``n_x``, ``n_u``, ``n_d`` are provided as
    concrete aliases mapping to ``nx``, ``nu``, ``nd`` respectively.

    Consumers requiring cvxopt-format matrices should use the alias
    properties ``C_cvx``, ``R_cvx``, ``x_ref_cvx``, and ``u_bounds_cvx``.
    """

    # ── Abstract dimensions (inherited from ContinuousDiscreteModel) ──────
    #   nx, nu, nd are abstract in the parent and must be implemented by
    #   concrete subclasses.  ny and nw are provided as concrete derivations
    #   from C and G below.

    # ── Abstract continuous-time matrices (numpy) ─────────────────────────

    @property
    @abstractmethod
    def A_c(self) -> np.ndarray:
        """Continuous state matrix A_c ∈ ℝⁿˣˣⁿˣ."""

    @property
    @abstractmethod
    def B_c(self) -> np.ndarray:
        """Continuous input matrix B_c ∈ ℝⁿˣˣⁿᵘ."""

    @property
    @abstractmethod
    def E_c(self) -> np.ndarray:
        """Continuous disturbance matrix E_c ∈ ℝⁿˣˣⁿᵈ."""

    @property
    @abstractmethod
    def G(self) -> np.ndarray:
        """Noise input matrix G ∈ ℝⁿˣˣⁿʷ."""

    @property
    @abstractmethod
    def Q_c(self) -> np.ndarray:
        """Continuous process-noise covariance Q_c ∈ ℝⁿʷˣⁿʷ."""

    # ── Abstract observation matrices (numpy) ────────────────────────────

    @property
    @abstractmethod
    def Cm(self) -> np.ndarray:
        """Measurement matrix C_m ∈ ℝⁿʸᵐˣⁿˣ (numpy ndarray)."""

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Measurement noise covariance R ∈ ℝⁿʸˣⁿʸ (numpy ndarray)."""

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
    def x_ref(self) -> np.ndarray:
        """Reference / setpoint x_ref ∈ ℝⁿˣ (numpy 1-D array, length nx)."""

    @property
    @abstractmethod
    def u_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box constraint on inputs (u_min, u_max), each a (nu,) ndarray."""

    # ── Concrete implementations of ContinuousDiscreteModel abstracts ─────

    @property
    def nym(self) -> int:
        """Measurement dimension nym = Cm.shape[0]."""
        return self.Cm.shape[0]

    @property
    def nw(self) -> int:
        """Process-noise dimension nw = G.shape[1]."""
        return self.G.shape[1]

    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Drift f(x, u, d, p, t) = A_c x + B_c u + E_c d  (p ignored for linear)."""
        return self.A_c @ x + self.B_c @ u + self.E_c @ d

    @property
    def Cz(self) -> np.ndarray:
        """Controlled output matrix C_z = C_m (can be overridden)."""
        return self.Cm

    def sigma(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Diffusion sigma(x, u, d, p, t) = G  (constant; arguments ignored)."""
        return self.G

    def g(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Controlled output g(x, u, d, p) = C_z x  (u, d and p ignored for LTI)."""
        return self.Cz @ x

    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Measurement hm(x, u, d, p) = C_m x  (u, d and p ignored for LTI)."""
        return self.Cm @ x

    # ── Analytic Jacobian overrides ───────────────────────────────────────

    def dfdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂x = A_c  (arguments ignored)."""
        return self.A_c.copy()

    def dhmdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂x = C_m  (arguments ignored)."""
        return self.Cm.copy()

    def dhmdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂u = 0  (hm = Cm x does not depend on u)."""
        return np.zeros((self.nym, self.nu))

    def dfdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂u = B_c  (arguments ignored)."""
        return self.B_c.copy()

    def dfdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂d = E_c  (arguments ignored)."""
        return self.E_c.copy()

    def dhmdd(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂d = 0  (hm = Cm x does not depend on d)."""
        return np.zeros((self.nym, self.nd))

    def dfdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Analytic Jacobian ∂f/∂p = 0  (f = A_c x + B_c u + E_c d does not depend on p)."""
        return np.zeros((self.nx, p.shape[0]))

    def dhmdp(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Analytic Jacobian ∂hm/∂p = 0  (hm = Cm x does not depend on p)."""
        return np.zeros((self.nym, p.shape[0]))

    def dgdx(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Analytic Jacobian ∂g/∂x = C_z  (g = C_z x for linear model)."""
        return self.Cz.copy()

    def dgdu(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Analytic Jacobian ∂g/∂u = 0  (g = C_z x does not depend on u)."""
        return np.zeros((self.nz, self.nu))

    # ── Backward-compatible dimension aliases ─────────────────────────────

    @property
    def n_x(self) -> int:
        """Deprecated alias for ``nx``.  Use ``nx`` instead."""
        return self.nx

    @property
    def n_u(self) -> int:
        """Deprecated alias for ``nu``.  Use ``nu`` instead."""
        return self.nu

    @property
    def n_d(self) -> int:
        """Deprecated alias for ``nd``.  Use ``nd`` instead."""
        return self.nd

    # ── cvxopt alias properties (for legacy consumers) ────────────────────

    @property
    def Cm_cvx(self) -> "matrix":
        """Measurement matrix C_m as a cvxopt dense matrix."""
        from ._utils import _np_to_cvx
        return _np_to_cvx(self.Cm)

    @property
    def R_cvx(self) -> "matrix":
        """Measurement noise covariance R as a cvxopt dense matrix."""
        from ._utils import _np_to_cvx
        return _np_to_cvx(self.R)

    @property
    def x_ref_cvx(self) -> "matrix":
        """Reference setpoint x_ref as a cvxopt (nx, 1) column vector."""
        from ._utils import _np_to_cvx
        return _np_to_cvx(np.asarray(self.x_ref, dtype=float).reshape(-1, 1))

    @property
    def u_bounds_cvx(self) -> Tuple["matrix", "matrix"]:
        """Input bounds (u_min, u_max) as cvxopt (nu, 1) column vectors."""
        from ._utils import _np_to_cvx
        lo, hi = self.u_bounds
        return (
            _np_to_cvx(np.asarray(lo, dtype=float).reshape(-1, 1)),
            _np_to_cvx(np.asarray(hi, dtype=float).reshape(-1, 1)),
        )

    # ── Concrete discretisation methods ───────────────────────────────────

    def discretize(self, d: "matrix") -> Tuple["matrix", "matrix", "matrix"]:
        """
        ZOH-discretised matrices (A_d, B_d, E_d) as cvxopt dense matrices.

        Uses the augmented-matrix method so that no matrix inverse is
        required:

            expm([[A_c, B_c, E_c], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]
            = [A_d | B_d | E_d]

        The ``d`` argument is accepted for interface compatibility with
        ``OptimalControlProblem`` (LPV sub-classes may override this method
        to schedule matrices on the current disturbance); the default LTI
        implementation ignores it.

        Parameters
        ----------
        d : (nd, 1) cvxopt column  — current disturbance (ignored for LTI).

        Returns
        -------
        A_d : (nx, nx) cvxopt dense — discrete state-transition matrix.
        B_d : (nx, nu) cvxopt dense — discrete input matrix.
        E_d : (nx, nd) cvxopt dense — discrete disturbance matrix.

        Note
        ----
        Returns cvxopt.matrix for compatibility with ``OptimalControlProblem``.
        """
        from ._utils import _zoh_full, _np_to_cvx

        A_d_np, B_d_np, E_d_np = _zoh_full(self.A_c, self.B_c, self.E_c, self.dt)
        return _np_to_cvx(A_d_np), _np_to_cvx(B_d_np), _np_to_cvx(E_d_np)

    def discretize_noise(self) -> "matrix":
        """
        Exact discrete process-noise covariance Q_d via Van Loan (1978).

        Computes

            Q_d = ∫₀^{dt} expm(A_c τ) G Q_c Gᵀ expm(A_c τ)ᵀ dτ

        using the augmented 2nx×2nx matrix method.  The result is symmetric
        positive semi-definite by construction.

        Returns
        -------
        Q_d : (nx, nx) cvxopt dense — discrete process-noise covariance.

        Note
        ----
        Returns cvxopt.matrix for compatibility with ``OptimalControlProblem``.
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

# ── Continuous-Discrete SDAE Model ───────────────────────────────────────────


class ContinuousDiscreteDAEModel(ContinuousDiscreteModel):
    """
    Abstract interface for a continuous-discrete stochastic DAE (Ph.D. Ch. 6).

    Extends ``ContinuousDiscreteModel`` with algebraic state y and constraint:

        dx = f(x, y, u, d, t) dt + sigma(x, y, u, d, t) dw
        0  = h(x, y, u, d, t)
        y_m[k] = hm(x_k, y_k, u_k, d_k) + v_k

    At each integration step the algebraic constraint is enforced by solving
    ``h = 0`` for y (typically via Newton iteration in the simulator).

    Subclasses must additionally implement ``h`` (constraint) and ``ny``.
    The measurement function ``hm`` may be overridden to accept ``(x, y, d)``
    if the outputs depend on y; the base signature ``hm(x, u, d, p)`` is
    retained for compatibility with ``ContinuousDiscreteModel``-typed interfaces.

    The ``nw`` abstract property is inherited from ``ContinuousDiscreteModel``
    and must be implemented by concrete subclasses.
    """

    @abstractmethod
    def h(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Algebraic constraint residual.

        The constraint is satisfied when ``h(x, y, u, d, p, t) = 0``.

        Parameters
        ----------
        x : (nx,) differential state vector.
        y : (ny,) algebraic state vector.
        u : (nu,) input vector.
        d : (nd,) disturbance vector.
        p : (nparams,) parameter vector.
        t : current time.

        Returns
        -------
        (ny,) residual vector.
        """

    @property
    @abstractmethod
    def ny(self) -> int:
        """Algebraic state dimension."""

    # ── Jacobian methods for DAE systems (default: forward FD) ───────────
    #
    # These override the base-class methods to include the algebraic state y
    # and parameter vector p in their signatures.

    def dfdx(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂x evaluated at (x, y, u, d, p, t)  →  (nx, nx) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, y, u, d, p, t)
        nx = x.shape[0]
        J = np.empty((nx, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.f(x_fwd, y, u, d, p, t) - f0) / _H_FD
        return J

    def dfdy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂y evaluated at (x, y, u, d, p, t)  →  (nx, ny) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, y, u, d, p, t)
        ny = y.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, ny))
        for k in range(ny):
            y_fwd = y.copy()
            y_fwd[k] += _H_FD
            J[:, k] = (self.f(x, y_fwd, u, d, p, t) - f0) / _H_FD
        return J

    def dhmdx(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂x evaluated at (x, y, u, d, p)  →  (nym, nx) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, y, u, d, p)
        nx = x.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.hm(x_fwd, y, u, d, p) - h0) / _H_FD
        return J

    def dhmdy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂y evaluated at (x, y, u, d, p)  →  (nym, ny) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, y, u, d, p)
        ny = y.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, ny))
        for k in range(ny):
            y_fwd = y.copy()
            y_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, y_fwd, u, d, p) - h0) / _H_FD
        return J

    def dhmdu(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂u evaluated at (x, y, u, d, p)  →  (nym, nu) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, y, u, d, p)
        nu = u.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, y, u_fwd, d, p) - h0) / _H_FD
        return J

    def dhdx(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂h/∂x evaluated at (x, y, u, d, p, t)  →  (ny, nx) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.h(x, y, u, d, p, t)
        nx = x.shape[0]
        ny_out = h0.shape[0]
        J = np.empty((ny_out, nx))
        for k in range(nx):
            x_fwd = x.copy()
            x_fwd[k] += _H_FD
            J[:, k] = (self.h(x_fwd, y, u, d, p, t) - h0) / _H_FD
        return J

    def dhdy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂h/∂y evaluated at (x, y, u, d, p, t)  →  (ny, ny) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.h(x, y, u, d, p, t)
        ny_alg = y.shape[0]
        ny_out = h0.shape[0]
        J = np.empty((ny_out, ny_alg))
        for k in range(ny_alg):
            y_fwd = y.copy()
            y_fwd[k] += _H_FD
            J[:, k] = (self.h(x, y_fwd, u, d, p, t) - h0) / _H_FD
        return J

    def dfdu(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂u evaluated at (x, y, u, d, p, t)  →  (nx, nu) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, y, u, d, p, t)
        nu = u.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.f(x, y, u_fwd, d, p, t) - f0) / _H_FD
        return J

    def dfdd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂d evaluated at (x, y, u, d, p, t)  →  (nx, nd) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        f0 = self.f(x, y, u, d, p, t)
        nd = d.shape[0]
        nx = f0.shape[0]
        J = np.empty((nx, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.f(x, y, u, d_fwd, p, t) - f0) / _H_FD
        return J

    def dhmdd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂d evaluated at (x, y, u, d, p)  →  (nym, nd) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.hm(x, y, u, d, p)
        nd = d.shape[0]
        nym = h0.shape[0]
        J = np.empty((nym, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, y, u, d_fwd, p) - h0) / _H_FD
        return J

    def dhdu(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂h/∂u evaluated at (x, y, u, d, p, t)  →  (ny, nu) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.h(x, y, u, d, p, t)
        nu = u.shape[0]
        ny_out = h0.shape[0]
        J = np.empty((ny_out, nu))
        for k in range(nu):
            u_fwd = u.copy()
            u_fwd[k] += _H_FD
            J[:, k] = (self.h(x, y, u_fwd, d, p, t) - h0) / _H_FD
        return J

    def dhdd(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂h/∂d evaluated at (x, y, u, d, p, t)  →  (ny, nd) ndarray.

        Default: forward finite differences.  Override with analytic form.
        """
        h0 = self.h(x, y, u, d, p, t)
        nd = d.shape[0]
        ny_out = h0.shape[0]
        J = np.empty((ny_out, nd))
        for k in range(nd):
            d_fwd = d.copy()
            d_fwd[k] += _H_FD
            J[:, k] = (self.h(x, y, u, d_fwd, p, t) - h0) / _H_FD
        return J

    def dfdp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂f/∂p evaluated at (x, y, u, d, p, t)  →  (nx, nparams) ndarray.

        Default: forward finite differences.  Override with analytic form.
        Returns an empty (nx, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nx = self.nx
        if nparams == 0:
            return np.empty((nx, 0))
        f0 = self.f(x, y, u, d, p, t)
        J = np.empty((nx, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.f(x, y, u, d, p_fwd, t) - f0) / _H_FD
        return J

    def dhmdp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Jacobian ∂hm/∂p evaluated at (x, y, u, d, p)  →  (nym, nparams) ndarray.

        Default: forward finite differences.  Override with analytic form.
        Returns an empty (nym, 0) array when p is empty.
        """
        nparams = p.shape[0]
        nym = self.nym
        if nparams == 0:
            return np.empty((nym, 0))
        h0 = self.hm(x, y, u, d, p)
        J = np.empty((nym, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.hm(x, y, u, d, p_fwd) - h0) / _H_FD
        return J

    def dhdp(
        self,
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Jacobian ∂h/∂p evaluated at (x, y, u, d, p, t)  →  (ny, nparams) ndarray.

        Default: forward finite differences.  Override with analytic form.
        Returns an empty (ny, 0) array when p is empty.
        """
        nparams = p.shape[0]
        ny_out = self.h(x, y, u, d, p, t).shape[0]
        if nparams == 0:
            return np.empty((ny_out, 0))
        h0 = self.h(x, y, u, d, p, t)
        J = np.empty((ny_out, nparams))
        for k in range(nparams):
            p_fwd = p.copy()
            p_fwd[k] += _H_FD
            J[:, k] = (self.h(x, y, u, d, p_fwd, t) - h0) / _H_FD
        return J
