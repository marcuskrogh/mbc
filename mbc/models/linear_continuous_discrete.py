"""
Linear continuous-discrete model interface.

``LinearContinuousDiscreteModel`` — extends ``ContinuousDiscreteModel`` for
linear systems where the drift, diffusion, output, and observation functions
take the specific forms:

    f(x, u, d, p, t)     = A x + B u + E d
    sigma(x, u, d, p, t) = G                         (constant diffusion)
    gm(x, u, d, p, t)    = Cz x + Dz u + Fz d
    hm(x, u, d, p, t)    = Cm x + Dm u + Fm d
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Tuple, TYPE_CHECKING

import numpy as np

from .continuous_discrete import ContinuousDiscreteModel

if TYPE_CHECKING:
    from cvxopt import matrix


class LinearContinuousDiscreteModel(ContinuousDiscreteModel):
    """
    Abstract interface for a linear continuous-discrete stochastic system.

    Extends :class:`ContinuousDiscreteModel` with the specific linear forms:

        dx(t)  = (A x(t) + B u(t) + E d(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
        z(t)   = Cz x(t) + Dz u(t) + Fz d(t)
        ym(tk) = Cm x(tk) + Dm u(tk) + Fm d(tk) + v(tk),   v(tk) ~ N(0, Rm)

    with zero-order-hold (ZOH) inputs and disturbances over each sampling
    interval [t_k, t_{k+1}].

    The model exposes the ZOH-discretised matrices via ``discretize`` and the
    exact discrete process-noise covariance via ``discretize_noise``.

    Dimensions
    ----------
        nx   – state dimension              x ∈ ℝⁿˣ
        nu   – input dimension              u ∈ ℝⁿᵘ
        nd   – disturbance dimension        d ∈ ℝⁿᵈ
        nw   – process-noise dimension      w ∈ ℝⁿʷ  (derived: G.shape[1])
        nz   – output dimension             z ∈ ℝⁿᶻ  (derived: Cz.shape[0])
        nym  – measurement output dimension ym ∈ ℝⁿʸᵐ (derived: Cm.shape[0])
        A    – continuous state matrix      A ∈ ℝⁿˣˣⁿˣ
        B    – continuous input matrix      B ∈ ℝⁿˣˣⁿᵘ
        E    – continuous disturbance mat.  E ∈ ℝⁿˣˣⁿᵈ
        G    – noise input matrix           G ∈ ℝⁿˣˣⁿʷ
        Cm   – measurement output matrix    Cm ∈ ℝⁿʸᵐˣⁿˣ
        Dm   – measurement input D-term     Dm ∈ ℝⁿʸᵐˣⁿᵘ  (default: 0)
        Fm   – measurement disturbance D    Fm ∈ ℝⁿʸᵐˣⁿᵈ  (default: 0)
        Cz   – output matrix                Cz ∈ ℝⁿᶻˣⁿˣ   (default: Cm)
        Dz   – output input D               Dz ∈ ℝⁿᶻˣⁿᵘ   (default: 0)
        Fz   – output disturbance D         Fz ∈ ℝⁿᶻˣⁿᵈ   (default: 0)
        Rm   – measurement noise cov.       Rm ∈ ℝⁿʸᵐˣⁿʸᵐ
        dt   – sampling interval

    ZOH discretisation (``discretize``)
    -------------------------------------
    Computed via the augmented-matrix method (no matrix inverse required):

        [Ad | Bd | Ed] = expm([[A, B, E], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]

    Discrete process noise (``discretize_noise``)
    -----------------------------------------------
    Computed by the Van Loan (1978) method:

        Qd = ∫₀^{dt} expm(A τ) G Gᵀ expm(A τ)ᵀ dτ
    """

    # ── Abstract dimensions (inherited from ContinuousDiscreteModel) ──────
    #   nx, nu, nd are abstract in the parent and must be implemented by
    #   concrete subclasses.  nym and nw are provided as concrete derivations
    #   from Cm and G below.

    # ── Abstract continuous-time matrices (numpy) ─────────────────────────

    @property
    @abstractmethod
    def A(self) -> np.ndarray:
        """Continuous state matrix A ∈ ℝⁿˣˣⁿˣ."""

    @property
    @abstractmethod
    def B(self) -> np.ndarray:
        """Continuous input matrix B ∈ ℝⁿˣˣⁿᵘ."""

    @property
    @abstractmethod
    def E(self) -> np.ndarray:
        """Continuous disturbance matrix E ∈ ℝⁿˣˣⁿᵈ."""

    @property
    @abstractmethod
    def G(self) -> np.ndarray:
        """Noise input matrix G ∈ ℝⁿˣˣⁿʷ."""

    # ── Abstract observation matrices (numpy) ────────────────────────────

    @property
    @abstractmethod
    def Cm(self) -> np.ndarray:
        """Measurement output matrix Cm ∈ ℝⁿʸᵐˣⁿˣ (numpy ndarray)."""

    @property
    @abstractmethod
    def Rm(self) -> np.ndarray:
        """Measurement noise covariance Rm ∈ ℝⁿʸᵐˣⁿʸᵐ (numpy ndarray)."""

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
        """Measurement output dimension nym = Cm.shape[0]."""
        return self.Cm.shape[0]

    @property
    def nw(self) -> int:
        """Process-noise dimension nw = G.shape[1]."""
        return self.G.shape[1]

    @property
    def Cz(self) -> np.ndarray:
        """
        Output matrix Cz ∈ ℝⁿᶻˣⁿˣ.

        Default: Cm (same output set as measurements).
        Subclasses may override for a different output selection.
        """
        return self.Cm

    @property
    def Dz(self) -> np.ndarray:
        """
        Output input feedthrough Dz ∈ ℝⁿᶻˣⁿᵘ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nz, self.nu))

    @property
    def Fz(self) -> np.ndarray:
        """
        Output disturbance feedthrough Fz ∈ ℝⁿᶻˣⁿᵈ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nz, self.nd))

    @property
    def Dm(self) -> np.ndarray:
        """
        Measurement input feedthrough Dm ∈ ℝⁿʸᵐˣⁿᵘ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nym, self.nu))

    @property
    def Fm(self) -> np.ndarray:
        """
        Measurement disturbance feedthrough Fm ∈ ℝⁿʸᵐˣⁿᵈ.

        Default: zeros.  Subclasses may override.
        """
        return np.zeros((self.nym, self.nd))

    @property
    def nz(self) -> int:
        """Output dimension nz = Cz.shape[0]."""
        return self.Cz.shape[0]

    def f(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Drift f(x, u, d, p, t) = A x + B u + E d  (p, t ignored for LTI)."""
        return self.A @ x + self.B @ u + self.E @ d

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

    def gm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Output gm(x, u, d, p, t) = Cz x + Dz u + Fz d."""
        return self.Cz @ x + self.Dz @ u + self.Fz @ d

    def hm(
        self,
        x: np.ndarray,
        u: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        t: float = 0.0,
    ) -> np.ndarray:
        """Measurement hm(x, u, d, p, t) = Cm x + Dm u + Fm d."""
        return self.Cm @ x + self.Dm @ u + self.Fm @ d

    # ── Analytic Jacobian overrides ───────────────────────────────────────

    def dfdx(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂x = A."""
        return self.A.copy()

    def dfdu(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂u = B."""
        return self.B.copy()

    def dfdd(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂d = E."""
        return self.E.copy()

    def dfdp(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂f/∂p = 0  (f = A x + B u + E d does not depend on p)."""
        return np.zeros((self.nx, p.shape[0]))

    def dhmdx(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂x = Cm."""
        return self.Cm.copy()

    def dhmdu(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂u = Dm."""
        return self.Dm.copy()

    def dhmdd(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂d = Fm."""
        return self.Fm.copy()

    def dhmdp(self, x, u, d, p, t=0.0) -> np.ndarray:
        """Analytic ∂hm/∂p = 0  (LTI hm does not depend on p)."""
        return np.zeros((self.nym, p.shape[0]))

    def dgmdx(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂gm/∂x = Cz."""
        return self.Cz.copy()

    def dgmdu(self, x, u, d, p, t) -> np.ndarray:
        """Analytic ∂gm/∂u = Dz."""
        return self.Dz.copy()

    # ── Concrete discretisation methods ───────────────────────────────────

    def discretize(self, d: "matrix") -> Tuple["matrix", "matrix", "matrix"]:
        """
        ZOH-discretised matrices (Ad, Bd, Ed) as cvxopt dense matrices.

        Uses the augmented-matrix method so that no matrix inverse is
        required:

            expm([[A, B, E], [0, 0, 0], [0, 0, 0]] · dt)[:nx, :]
            = [Ad | Bd | Ed]

        The ``d`` argument is accepted for interface compatibility with
        ``OptimalControlProblem`` (LPV sub-classes may override this method
        to schedule matrices on the current disturbance); the default LTI
        implementation ignores it.

        Parameters
        ----------
        d : (nd, 1) cvxopt column  — current disturbance (ignored for LTI).

        Returns
        -------
        Ad : (nx, nx) cvxopt dense — discrete state-transition matrix.
        Bd : (nx, nu) cvxopt dense — discrete input matrix.
        Ed : (nx, nd) cvxopt dense — discrete disturbance matrix.

        Note
        ----
        Returns cvxopt.matrix for compatibility with ``OptimalControlProblem``.
        """
        from .._utils import _zoh_full, _np_to_cvx

        A_d_np, B_d_np, E_d_np = _zoh_full(self.A, self.B, self.E, self.dt)
        return _np_to_cvx(A_d_np), _np_to_cvx(B_d_np), _np_to_cvx(E_d_np)

    def discretize_noise(self) -> "matrix":
        """
        Exact discrete process-noise covariance Qd via Van Loan (1978).

        Computes

            Qd = ∫₀^{dt} expm(A τ) G Gᵀ expm(A τ)ᵀ dτ

        using the augmented 2nx×2nx matrix method.  The result is symmetric
        positive semi-definite by construction.

        Returns
        -------
        Qd : (nx, nx) cvxopt dense — discrete process-noise covariance.

        Note
        ----
        Returns cvxopt.matrix for compatibility with ``OptimalControlProblem``.
        """
        from .._utils import _van_loan, _np_to_cvx

        # dw ~ N(0, I dt), so the noise intensity is G G^T.
        # Computed via the Van Loan (1978) augmented matrix method.
        Q_d_np = _van_loan(self.A, self.G, np.eye(self.nw), self.dt)
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
