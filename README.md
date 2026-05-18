# mbc — Model-Based Control Toolbox

A Python toolbox for linear and nonlinear model-based control, state estimation,
system identification, and realization. The toolbox follows the notation and
algorithms of the author's M.Sc. and Ph.D. theses and is structured to support
both discrete-discrete and continuous-discrete stochastic systems.

All implemented methods are based on the following references:

- **M.Sc. thesis** — Realization, Kalman filtering, discrete-time MPC, and system
  identification.
- **Ph.D. thesis** — Continuous-discrete SDE/SDAE models, nonlinear Kalman filters
  (EKF, UKF, EnKF, PF), DAE estimation, Economic NMPC, and Monte Carlo
  closed-loop simulation.

Matrix types: `numpy.ndarray` is used throughout for all model interfaces and
estimator computations. `cvxopt.matrix` is used only internally by the QP-based
MPC solvers (`KalmanFilter`, `OptimalControlProblem`, `CDKalmanFilter`,
`CDOptimalControlProblem`).

---

## Contents

- [Part I — Discrete-Discrete Systems](#part-i--discrete-discrete-systems)
  - [1.1 Models](#11-models)
  - [1.2 State Estimators](#12-state-estimators)
    - [DelayedObservationFilter (wraps any estimator)](#delayedobservationfilter--mbcestimation)
  - [1.3 Optimal Control Problems](#13-optimal-control-problems)
  - [1.4 MPC Controllers](#14-mpc-controllers)
- [Part II — Continuous-Discrete Systems](#part-ii--continuous-discrete-systems)
  - [2.1 Models](#21-models)
  - [2.2 Simulators](#22-simulators)
  - [2.3 State Estimators](#23-state-estimators)
    - [Delayed-observation update (cross-ref §1.2)](#delayed-observation-update-all-estimator-variants)
  - [2.4 Optimal Control Problems](#24-optimal-control-problems)
  - [2.5 MPC Controllers](#25-mpc-controllers)
- [Part III — System Identification](#part-iii--system-identification)
  - [3.1 Linear Discrete-time Identification](#31-linear-discrete-time-identification)
  - [3.2 Nonlinear Continuous-Discrete Identification](#32-nonlinear-continuous-discrete-identification)
- [Part IV — Realization](#part-iv--realization)
- [Part V — Monte Carlo Simulation](#part-v--monte-carlo-simulation)
- [Installation](#installation)

---

## Part I — Discrete-Discrete Systems

### 1.1 Models

#### `LinearDiscreteModel` — `mbc.models`

Abstract base class for a linear discrete-time stochastic state-space model
— the discrete-time analogue of
:class:`~mbc.models.LinearContinuousDiscreteModel`, using the same
ControlToolbox notation:

```
x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
z[k]   = Cz x[k] + Dz u[k] + Fz d[k]                      (output ``g^m``)
ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)
                                                          (measurement ``h^m``)
```

where `x ∈ ℝⁿˣ` is the state, `u ∈ ℝⁿᵘ` is the control input,
`d ∈ ℝⁿᵈ` is a measured disturbance, `z ∈ ℝⁿᶻ` is the (continuous-output
analogue) ``g^m``, and `ym ∈ ℝⁿʸᵐ` is the discrete noisy measurement
``h^m``.  All system matrices are constant (LTI).

**Abstract interface** — subclasses must implement:

| Member | Type | Description |
|--------|------|-------------|
| `nx` | `int` | State dimension |
| `nu` | `int` | Input dimension |
| `nd` | `int` | Disturbance dimension |
| `Ad` | `(nx, nx) ndarray` | Discrete state-transition matrix |
| `Bd` | `(nx, nu) ndarray` | Discrete input matrix |
| `Ed` | `(nx, nd) ndarray` | Discrete disturbance matrix |
| `Cm` | `(nym, nx) ndarray` | Measurement output matrix |
| `Qd` | `(nw, nw) ndarray` | Discrete process-noise covariance |
| `Rm` | `(nym, nym) ndarray` | Measurement noise covariance |
| `x` | `list[float]` | Current state (read/write) |
| `x_ref` | `(nx,) ndarray` | State setpoint / reference |
| `u_bounds` | `(ndarray, ndarray)` | Input box `(u_min, u_max)`, each `(nu,)` |

**Concrete members** (overridable — sensible defaults are provided):

| Member | Default | Description |
|--------|---------|-------------|
| `Gd` | `I` (identity) | Noise input matrix Gd ∈ ℝⁿˣˣⁿʷ |
| `Cz` | `Cm` | Output matrix Cz ∈ ℝⁿᶻˣⁿˣ (discrete analogue of ``g^m``) |
| `Dz` | zeros | Output feedthrough Dz ∈ ℝⁿᶻˣⁿᵘ |
| `Fz` | zeros | Output disturbance feedthrough Fz ∈ ℝⁿᶻˣⁿᵈ |
| `Dm` | zeros | Measurement input feedthrough Dm ∈ ℝⁿʸᵐˣⁿᵘ |
| `Fm` | zeros | Measurement disturbance feedthrough Fm ∈ ℝⁿʸᵐˣⁿᵈ |
| `nym` | `Cm.shape[0]` | Measurement dimension (derived) |
| `nz` | `Cz.shape[0]` | Output dimension (derived) |
| `nw` | `Gd.shape[1]` | Process-noise dimension (derived) |
| `predict_offset(d)` | zeros | Additive prediction bias `offset(d)` |
| `params` | `array([])` | Flat parameter vector for identification |
| `with_params(theta)` | raises | Construct new model from parameter vector |

**Example**:

```python
import numpy as np
from mbc.models import LinearDiscreteModel

class ThermalRoom(LinearDiscreteModel):
    @property
    def nx(self): return 2
    @property
    def nu(self): return 1
    @property
    def nd(self): return 1
    @property
    def Ad(self): return np.array([[0.95, 0.0], [0.0, 1.0]])
    @property
    def Bd(self): return np.array([[0.03], [0.0]])
    @property
    def Ed(self): return np.array([[0.02], [0.0]])
    @property
    def Cm(self): return np.array([[1.0, 0.0]])      # measure state 0
    @property
    def Qd(self): return np.diag([1e-4, 1e-4])
    @property
    def Rm(self): return np.array([[0.05]])
    @property
    def x(self): return [20.0, 15.0]
    @x.setter
    def x(self, val): ...
    @property
    def x_ref(self): return np.array([21.0, 0.0])
    @property
    def u_bounds(self): return np.array([0.0]), np.array([1.0])

m = ThermalRoom()
print(m.nym)  # 1  (= Cm.shape[0])
print(m.nz)   # 1  (= Cz.shape[0]; Cz defaults to Cm)
```

---

### 1.2 State Estimators

#### `KalmanFilter` — `mbc.estimation`

Discrete-time Kalman filter with Joseph-stabilised covariance update — the
linear specialisation of the continuous-discrete EKF
(:class:`~mbc.estimation.ContinuousDiscreteEKF`) and a direct counterpart of
the M.Sc. thesis Ch. 5 formulation.  The notation matches the
ControlToolbox §SDE / §SDAE state-estimation conventions:

| Symbol | Meaning |
|--------|---------|
| ``x̂_{k\|k-1}`` | predicted state estimate |
| ``x̂_{k\|k}``   | filtered (corrected) state estimate |
| ``P_{k\|k-1}`` | predicted covariance |
| ``P_{k\|k}``   | filtered covariance |
| ``e_k``        | innovation ``ym_k − Cm x̂_{k\|k-1}`` |
| ``R_e``        | innovation covariance ``Cm P_{k\|k-1} Cmᵀ + Rm`` |
| ``K_k``        | Kalman gain |

**Model**: :class:`~mbc.models.LinearDiscreteModel`.  ``Qd``, ``Rm``, and ``Gd``
are read directly from the model — there is no separate constructor knob.

**Time update over ``[t_{k-1}, t_k]``**

```
x̂_{k|k-1} = Ad x̂_{k-1|k-1} + Bd u[k−1] + Ed d[k−1] + offset(d[k−1])
P_{k|k-1} = Ad P_{k-1|k-1} Adᵀ + Gd Qd Gdᵀ
```

Inputs and disturbances are zero-order hold over each sampling interval.

**Measurement update at ``t_k`` (Joseph form)**

```
ŷ^m_{k|k-1} = Cm x̂_{k|k-1}
e_k         = ym_k − ŷ^m_{k|k-1}                    (innovation)
R_e         = Cm P_{k|k-1} Cmᵀ + Rm                  (innovation covariance)
K_k         = P_{k|k-1} Cmᵀ R_e⁻¹                    (Kalman gain)

x̂_{k|k} = x̂_{k|k-1} + K_k e_k
P_{k|k} = (I − K_k Cm) P_{k|k-1} (I − K_k Cm)ᵀ + K_k Rm K_kᵀ        (Joseph)
```

The gain is computed by solving the linear system ``R_e Kᵀ = Cm P_{k|k-1}``
via :func:`numpy.linalg.solve`, exploiting the positive definiteness of
``R_e`` without forming ``R_e⁻¹`` explicitly.  The Joseph stabilising form
preserves symmetry and positive semi-definiteness of ``P_{k|k}`` in
finite-precision arithmetic.

**Missing observations** — the optional ``mask`` argument of
``update(ym, mask)`` and ``step(ym, u, d, mask=mask)`` controls which output
channels are used in the measurement update.  When ``mask[i] = False`` channel
``i`` is excluded; if every entry is ``False`` the update step is skipped
entirely (prediction-only).

**Delayed observations** — wrap any estimator in
:class:`~mbc.estimation.DelayedObservationFilter` and pass an integer
``delay`` array (one entry per output channel) to :meth:`step`.

**Constructor**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearDiscreteModel` | — | Plant model — provides ``Ad``, ``Bd``, ``Ed``, ``Gd``, ``Cm``, ``Qd``, ``Rm``. |
| `x0` | `(nx,) ndarray` | `np.array(model.x)` | Initial state estimate ``x̂_{0\|0}``. |
| `P0` | `(nx, nx) ndarray` | `I_{nx}` | Initial state error covariance ``P_{0\|0}``. |

**Methods** (all signatures match
:class:`~mbc.estimation.ContinuousDiscreteEKF`):

```python
from mbc.estimation import KalmanFilter

kf = KalmanFilter(model, x0=x0, P0=P0)

# Building blocks
x_pred, P_pred = kf.predict(u_prev, d_prev)         # time update
x_hat,  P     = kf.update (ym, mask=None)           # measurement update

# Combined predict + update — same signature as the CD-EKF (p, t are
# accepted for interface compatibility but ignored for LTI plants)
x_hat, P = kf.step(ym, u_prev, d_prev, mask=None)
```

**Public properties**:

| Property | Type | Description |
|----------|------|-------------|
| `x_hat` | `(nx,) ndarray` | Current state estimate (copy) |
| `P`     | `(nx,nx) ndarray` | Current covariance (copy) |
| `last_innovation` | `list[float]` or `None` | Most recent ``e_k`` |

#### `DelayedObservationFilter` — `mbc.estimation`

Transparent wrapper that adds per-channel reporting-delay handling to any
estimator that exposes the unified ``step(ym, u, d, p=None, t=None,
mask=None) → (x_hat, P, …)`` interface — i.e. all estimators in
:mod:`mbc.estimation`, both discrete (KF, CDKF) and continuous-discrete (EKF,
UKF, EnKF, PF, DAE-EKF).  The wrapper itself exposes a single uniform
``step(ym, u, d, p=None, t=None, mask=None, delay=None)`` method, so it can
be substituted into :class:`MPCController`, :class:`CDMPCController`, or
:class:`CDNMPCController` without any change to the controller code.

**Motivation**: some measurement channels have a fixed or variable reporting
delay — a laboratory analyser returns a result ``τ`` sampling steps after the
sample was taken, while on-line sensors have ``τ = 0``.  Passing all channels
together in a single ``step`` call (with their respective delays declared)
allows the filter to apply each measurement at the correct point in time.

**``delay`` argument** — a ``(nym,)`` integer ndarray where ``delay[i]`` is
the number of sampling steps by which output channel ``i`` arrived late.
``delay[i] = 0`` (or ``delay = None``) means a current-step observation: the
wrapper behaves identically to the unwrapped estimator.

**Constructor**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | any supported estimator | — | Wrapped estimator. |
| `lag_max` | `int` | — | Maximum delay (in sampling steps) the internal ring buffer can accommodate. |

**Properties** (``x_hat``, ``P``, ``last_innovation``) are delegated to the
wrapped estimator.

**Algorithm** — at each call to ``step(ym, u, d, p=None, t=None, mask=None,
delay=None)``:

```
1. Partition channels by delay:
       immediate = [i for i where delay[i] == 0 (or delay is None)]
       delayed   = [(i, τ) for i where delay[i] = τ > 0]

2. Immediate update — call the wrapped estimator's step with the immediate-only
   mask, then push {x_hat, P, ym, u, d, p, t, mask} onto the internal buffer.

3. For each (i, τ) in delayed channels (sorted by τ ascending):
   a. Restore the wrapped estimator's (x̂, P) to the posterior at step k − τ
      (read from buffer[-(τ+1)]).
   b. Apply a measurement-only correction for channel i at that prior state.
   c. Replay forward through buffer entries [-(τ) … -1] using the wrapped
      estimator's step(ym, u, d, p, t, mask) to bring the posterior chain
      back up to date.

4. Return the (now corrected) current (x̂, P) to the caller.
```

If ``delay[i] > lag_max`` (or exceeds the current buffer depth), channel
``i`` is dropped with a :class:`RuntimeWarning`.

**Usage with `MPCController`** (linear discrete-time):

```python
from mbc.estimation import KalmanFilter, DelayedObservationFilter
from mbc.control import OptimalControlProblem, MPCController

kf   = KalmanFilter(model, x0=x0, P0=P0)
filt = DelayedObservationFilter(kf, lag_max=10)
ocp  = OptimalControlProblem(model, N=20, Q=Q_z, R=R_u)
ctrl = MPCController(model, estimator=filt, ocp=ocp)

# MPC step (delay handling is internal to the wrapper):
u, U_seq, X_seq = ctrl.step(ym, D)
```

**Usage with `CDNMPCController`** (nonlinear continuous-discrete):

```python
from mbc.estimation import ContinuousDiscreteEKF, DelayedObservationFilter
from mbc.control import EconomicOptimalControlProblem, CDNMPCController

ekf  = ContinuousDiscreteEKF(model, x0, P0, dt=1.0)
filt = DelayedObservationFilter(ekf, lag_max=10)
ocp  = EconomicOptimalControlProblem(model, N=20, dt=1.0, lagrange=cost_fn)
ctrl = CDNMPCController(estimator=filt, ocp=ocp)

u = ctrl.step(ym, d_traj, p, t)                          # no lab result this step
# (delays are typically wired through the wrapper directly when calling step:)
x_hat, P = filt.step(ym, u_prev, d, p, t, delay=np.array([0, 3]))
```

---

### 1.3 Optimal Control Problems

#### `OptimalControlProblem` — `mbc.control`

Finite-horizon quadratic OCP with hard input and soft *output* constraints —
the linear specialisation of the ControlToolbox §EMPC formulation.  When
the plant dynamics are linear and the OCP is restricted to quadratic stage
costs and box / soft-box constraints, the entire NLP reduces to a single
finite-horizon **quadratic program** that the lifted (batch) form solves
directly with ``cvxopt.solvers.qp`` — strictly more efficient than the
direct-simultaneous formulation used by
:class:`~mbc.control.EconomicOptimalControlProblem` for nonlinear plants.

The OCP tracks the **output** ``z[k] = Cz x[k] + Dz u[k] + Fz d[k]`` —
the discrete analogue of the continuous output ``g^m`` from
:class:`~mbc.models.ContinuousDiscreteModel`.  When the plant has
``Cz = Cm`` (the default of :class:`~mbc.models.LinearDiscreteModel`) the
output and the measurement coincide and the OCP tracks the measured channel
directly.

**Cost function** over prediction horizon N:

```
J(U) = Σ_{k=0}^{N-1} [ ‖z[k+1] − z_ref‖²_Q + ‖u[k]‖²_R + ‖Δu[k]‖²_S ]
     + ‖z[N] − z_ref‖²_P
     + ρ Σ_{k=0}^{N-1} ‖ε[k+1]‖²
```

where:
- `z_ref = Cz x_ref` is the output setpoint derived from `model.x_ref`
- `Δu[k] = u[k] − u[k-1]` is the input rate of movement (requires `u_prev`)
- `ε[k] ≥ 0` are slack variables for soft output constraint violations
- `ρ` is the violation penalty weight

**Constraints**:

```
u_min ≤ u[k] ≤ u_max                                (hard input box)
z[k+1] ≥ z_ref − δ − ε[k+1]                         (soft lower output bound)
z[k+1] ≤ z_ref + δ + ε[k+1]                         (soft upper output bound)
ε[k+1] ≥ 0                                          (slack non-negativity)
```

The output bounds are centred at the reference ``z_ref`` with half-width
``δ = y_offset``.  Violations are penalised quadratically via ``ρ ‖ε‖²``,
which guarantees the QP is always feasible.

**Batch (lifted) prediction matrices**

The state trajectory over the horizon is an affine function of the input
sequence ``U = [u[0]; u[1]; …; u[N-1]]`` and the disturbance forecast
``D = [d[0]; d[1]; …; d[N-1]]``:

```
X = Ψ x₀ + Γ U + Λ D

where:
  Ψ ∈ ℝᴺⁿˣˣⁿˣ    with Ψ_{k} = Ad^{k+1}
  Γ ∈ ℝᴺⁿˣˣᴺⁿᵘ   with Γ_{k,j} = Ad^{k-j} Bd   (lower-triangular block structure)
  Λ ∈ ℝᴺⁿˣˣᴺⁿᵈ   with Λ_{k,j} = Ad^{k-j} Ed
```

The output predictions are ``Z = C̄_z X`` with ``C̄_z = blkdiag(Cz, …, Cz)``.
The cost and constraints are expressed entirely in terms of ``U`` and the
slack ``ε``, giving the QP decision variable ``z_qp = [U; ε]``:

```
min_{z_qp}  ½ z_qpᵀ H z_qp + fᵀ z_qp
s.t.        G z_qp ≤ h
```

solved with :func:`cvxopt.solvers.qp`.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearDiscreteModel` | — | Plant model |
| `N` | `int` | — | Prediction horizon |
| `Q` | `(nz, nz) matrix` | — | Stage output tracking cost ``‖z − z_ref‖²_Q`` |
| `R` | `(nu, nu) matrix` | — | Stage input cost ``‖u‖²_R`` |
| `P` | `(nz, nz) matrix` | `Q` | Terminal output tracking cost |
| `S` | `(nu, nu) matrix` or `None` | `None` | Quadratic ROM cost ``‖Δu‖²_S``; `None` disables |
| `rho` | `float` | `1e4` | Soft-constraint slack quadratic penalty weight |
| `y_offset` | `float` | `2.0` | Half-width δ of the symmetric soft-output band |

**Usage**:

```python
from mbc.control import OptimalControlProblem

ocp = OptimalControlProblem(model, N=20, Q=Q_z, R=R_u, P=P_terminal, S=S_rate)

# D is the stacked disturbance forecast [d[0]; d[1]; ...; d[N-1]], shape (N*nd, 1)
U_seq, X_seq = ocp.solve(x0=x_hat, D=D_forecast, x_ref=model.x_ref, u_prev=u_prev)
u_current = U_seq[:model.nu]   # first element of the optimal sequence
```

If the QP is infeasible (solver status ≠ "optimal"), ``solve`` returns
zero inputs and logs a warning.  In practice the slack variables prevent
infeasibility.

---

### 1.4 MPC Controllers

#### `MPCController` — `mbc.control`

Combines a :class:`~mbc.estimation.KalmanFilter` and an
:class:`OptimalControlProblem` into a receding-horizon feedback controller —
the linear-discrete-time specialisation of the ControlToolbox §EMPC ENMPC
algorithm.

**Closed-loop structure**:

```
        ┌────────────────────────────────────────────────────────┐
        │                    MPCController                       │
ym[k] ─┼─► KalmanFilter ── x̂[k|k] ── OCP ── U*[0] ── u[k] ────┼─► Plant
        │   (predict u[k-1], d[k-1];   (lifted batch QP)        │
        │    update with ym[k])                                  │
        │       ▲                                                │
        │    cache (u_prev, d_prev) for next step's predict      │
        └────────────────────────────────────────────────────────┘
```

**Receding-horizon policy** — at each measurement time k:

1. **Measure**:  ``ym[k]``  (passed to ``step``)
2. **Estimate**: ``x̂[k|k] = estimator.step(ym[k], u[k-1], d[k-1])``
3. **Optimise**: ``(U*, X*) = ocp.solve(x̂[k|k], D, model.x_ref, u_prev=u[k-1])``
4. **Apply**:    ``u[k] = U*[0:nu]``
5. **Cache**:    store ``(u[k], d[k])`` as the new ``(u_prev, d_prev)`` for next step

Steps 1–5 are performed by ``step(ym, D)`` which returns
``(u, U_seq, X_seq)``.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `LinearDiscreteModel` | Plant model |
| `estimator` | `KalmanFilter` | State estimator |
| `ocp` | `OptimalControlProblem` | Optimal control problem |

**Usage**:

```python
from mbc.control import MPCController, OptimalControlProblem
from mbc.estimation import KalmanFilter

kf   = KalmanFilter(model)                                # Qd, Rm read from model
ocp  = OptimalControlProblem(model, N=20, Q=Q_z, R=R_u)
ctrl = MPCController(model, estimator=kf, ocp=ocp)

# At each measurement time:
u, U_seq, X_seq = ctrl.step(ym, D)   # D = stacked disturbance forecast (N·nd, 1)
```

---

## Part II — Continuous-Discrete Systems

### 2.1 Models

#### `ContinuousDiscreteModel` — `mbc.models`

Abstract base class for a nonlinear continuous-discrete stochastic SDE system
(ControlToolbox §SDE):

```
dx(t)   = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
z(t)    = gm(x, u, d, p, t)
ym(tk)  = hm(x, u, d, p, tk) + v(tk),                         v(tk) ~ N(0, Rm)
```

where `x ∈ ℝⁿˣ` is the state, `u ∈ ℝⁿᵘ` is the control input, `d ∈ ℝⁿᵈ` is a
measured disturbance, `p` is a parameter vector, `z ∈ ℝⁿᶻ` is the continuous
output (`g^m` in the ControlToolbox notation), `ym ∈ ℝⁿʸᵐ` is the discrete
noisy measurement (`h^m`), and `dw(t) ~ N(0, I dt)` is standard Brownian
motion.  The instantaneous noise covariance is `sigma sigma^T dt`.

All arrays use `numpy.ndarray`.  This ABC is accepted by all nonlinear estimators
(`ContinuousDiscreteEKF`, `ContinuousDiscreteUKF`, `ContinuousDiscreteEnKF`,
`ContinuousDiscreteParticleFilter`) and by `SDESimulator` and `EconomicOptimalControlProblem`.

**Abstract interface** — subclasses must implement:

| Member | Signature / Type | Description |
|--------|-----------------|-------------|
| `f` | `(x, u, d, p, t) → (nx,) ndarray` | Drift function |
| `sigma` | `(x, u, d, p, t) → (nx, nw) ndarray` | Diffusion matrix |
| `hm` | `(x, u, d, p, t) → (nym,) ndarray` | Measurement function `h^m` |
| `gm` | `(x, u, d, p, t) → (nz,) ndarray` | Continuous output function `g^m` |
| `Rm` | `(nym, nym) ndarray` | Measurement noise covariance |
| `nx` | `int` | State dimension |
| `nu` | `int` | Input dimension |
| `nd` | `int` | Disturbance dimension |
| `nym` | `int` | Measurement dimension |
| `nz` | `int` | Output dimension |
| `nw` | `int` | Process-noise / diffusion dimension (columns of `sigma`) |

**Optional analytic Jacobians** — subclasses may override these to avoid
finite-difference computation in the EKF and DAE-EKF:

| Member | Signature | Default |
|--------|-----------|---------|
| `dfdx` | `(x, u, d, p, t) → (nx, nx)` | Forward FD |
| `dfdu` | `(x, u, d, p, t) → (nx, nu)` | Forward FD |
| `dfdd` | `(x, u, d, p, t) → (nx, nd)` | Forward FD |
| `dfdp` | `(x, u, d, p, t) → (nx, np)` | Forward FD |
| `dhmdx` | `(x, u, d, p, t) → (nym, nx)` | Forward FD |
| `dhmdu` | `(x, u, d, p, t) → (nym, nu)` | Forward FD |
| `dhmdd` | `(x, u, d, p, t) → (nym, nd)` | Forward FD |
| `dhmdp` | `(x, u, d, p, t) → (nym, np)` | Forward FD |

**Example** — van de Vusse CSTR (A → B → C, 2A → D):

```python
import numpy as np
from mbc.models import ContinuousDiscreteModel

class VanDeVusseCSTR(ContinuousDiscreteModel):
    """
    Van de Vusse CSTR: states [c_A, c_B] (mol/L), input [D] (dilution rate 1/h).
    No disturbance, no parameters — all kinetics fixed.
    """
    _k1, _k2, _k3, _c_Af = 50.0, 100.0, 10.0, 10.0   # kinetic constants

    @property
    def nx(self): return 2   # c_A, c_B
    @property
    def nu(self): return 1   # dilution rate D = F/V
    @property
    def nd(self): return 0   # no measured disturbance
    @property
    def nw(self): return 2   # noise on both states
    @property
    def nym(self): return 1  # measure c_B only
    @property
    def nz(self): return 1   # output c_B
    @property
    def Rm(self): return np.array([[0.05]])

    def f(self, x, u, d, p, t):
        c_A, c_B = x; D = u[0]
        return np.array([
            (self._c_Af - c_A) * D - self._k1 * c_A - self._k3 * c_A**2,
            -c_B * D + self._k1 * c_A - self._k2 * c_B,
        ])

    def sigma(self, x, u, d, p, t):
        return np.diag([0.1, np.sqrt(0.005)])  # constant diffusion; sigma @ sigma.T = diag([0.01, 0.005])

    def hm(self, x, u, d, p, t):
        return np.array([x[1]])   # measure c_B

    def gm(self, x, u, d, p, t):
        return np.array([x[1]])   # output c_B

m = VanDeVusseCSTR()
print(m.nx, m.nu, m.nd, m.nw, m.nym, m.nz)  # 2  1  0  2  1  1

# Evaluate model functions at a nominal operating point
x0  = np.array([3.0, 1.1])          # [c_A, c_B] mol/L
u0  = np.array([0.5])               # dilution rate 1/h
d0  = np.zeros(m.nd)                # no disturbance
p0  = np.zeros(0)                   # no parameters
print(m.f(x0, u0, d0, p0, t=0.0))  # (2,) drift vector
print(m.hm(x0, u0, d0, p0, t=0.0)) # (1,) predicted measurement
print(m.gm(x0, u0, d0, p0, t=0.0)) # (1,) output
```

#### `LinearContinuousDiscreteModel` — `mbc.models`

Extends `ContinuousDiscreteModel` for linear systems.  The state evolves
continuously according to the Itô SDE

```
dx(t)  = (A x(t) + B u(t) + E d(t)) dt + G dw(t),  dw(t) ~ N(0, I dt)
z(t)   = Cz x(t) + Dz u(t) + Fz d(t)
ym(tk) = Cm x(tk) + Dm u(tk) + Fm d(tk) + v(tk),   v(tk) ~ N(0, Rm)
```

Inputs `u` and disturbances `d` are held constant (zero-order hold) over each
sampling interval `[tk, tk+1]`.

**Notation** (ControlToolbox §SDE; M.Sc. thesis Ch. 5):

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| `nx` | `int` | State dimension |
| `nu` | `int` | Input dimension |
| `nd` | `int` | Disturbance dimension |
| `nym` | `int` | Measurement output dimension (derived: `Cm.shape[0]`) |
| `nz` | `int` | Output dimension (derived: `Cz.shape[0]`) |
| `nw` | `int` | Process-noise dimension (derived: `G.shape[1]`) |
| `A` | `(nx, nx)` | Continuous state matrix |
| `B` | `(nx, nu)` | Continuous input matrix |
| `E` | `(nx, nd)` | Continuous disturbance matrix |
| `G` | `(nx, nw)` | Noise input matrix |
| `Cz` | `(nz, nx)` | Output matrix |
| `Dz` | `(nz, nu)` | Output input feedthrough |
| `Fz` | `(nz, nd)` | Output disturbance feedthrough |
| `Cm` | `(nym, nx)` | Measurement output matrix |
| `Dm` | `(nym, nu)` | Measurement input feedthrough |
| `Fm` | `(nym, nd)` | Measurement disturbance feedthrough |
| `Rm` | `(nym, nym)` | Measurement noise covariance |
| `dt` | `float` | Sampling interval |

**Abstract interface** — subclasses must implement:

| Member | Type | Description |
|--------|------|-------------|
| `nx`, `nu`, `nd` | `int` | Dimensions (inherited abstracts) |
| `A`, `B`, `E`, `G` | `ndarray` | Continuous-time matrices |
| `Cz` | `(nz, nx) ndarray` | Output matrix |
| `Cm` | `(nym, nx) ndarray` | Measurement output matrix |
| `Rm` | `(nym, nym) ndarray` | Measurement noise covariance |
| `dt` | `float` | Sampling interval |
| `x` | `list[float]` | Current state (read/write) |
| `x_ref` | `(nx,) ndarray` | State reference / setpoint |
| `u_bounds` | `(ndarray, ndarray)` | Input box `(u_min, u_max)`, each `(nu,)` |

**Concrete implementations** (provided by the ABC — no override needed):

| Member | Implementation |
|--------|---------------|
| `f(x, u, d, p, t)` | `A @ x + B @ u + E @ d` |
| `sigma(x, u, d, p, t)` | `G` (constant; arguments ignored) |
| `hm(x, u, d, p, t)` | `Cm @ x + Dm @ u + Fm @ d` |
| `gm(x, u, d, p, t)` | `Cz @ x + Dz @ u + Fz @ d` |
| `nym` | `Cm.shape[0]` |
| `nz` | `Cz.shape[0]` |
| `nw` | `G.shape[1]` |
| `Dz` | zeros `(nz, nu)` |
| `Fz` | zeros `(nz, nd)` |
| `Dm` | zeros `(nym, nu)` |
| `Fm` | zeros `(nym, nd)` |

**Concrete utility methods** (provided by the ABC):

*ZOH discretisation* — `discretize(d) → (Ad, Bd, Ed)`:

Uses the augmented-matrix method (no matrix inverse required):

```
expm([[A, B, E],          =  [[Ad, Bd, Ed],
      [0,  0,  0],             [ 0,  I,  0],
      [0,  0,  0]] * dt)       [ 0,  0,  I]]
```

where `expm` is the matrix exponential computed via eigendecomposition.

*Discrete noise covariance* — `discretize_noise() → Qd`:

The exact discrete process-noise covariance via the Van Loan (1978) method:

```
Qd = ∫₀^{dt} expm(A τ) G Gᵀ expm(A τ)ᵀ dτ
```

Computed using the `2nx × 2nx` augmented matrix:

```
M = [[-A,   G Gᵀ],   * dt
     [ 0,   Aᵀ  ]]

expm(M) = [[expm(-A dt),  expm(-A dt) Qd],
           [     0,       expm( A dt)   ]]

⟹  Qd = Ad · expm(M)[:nx, nx:]
```

These utility methods are used for analysis and for initialising discrete filters
from continuous model parameters.  They are **not** called internally by
`CDKalmanFilter`, which integrates the ODEs directly.  They **are** called
internally by `CDOptimalControlProblem`, which uses ZOH-discretised matrices
for the QP.

**Example**:

```python
import numpy as np
from mbc.models import LinearContinuousDiscreteModel

class CSTRLinear(LinearContinuousDiscreteModel):
    @property
    def nx(self): return 2
    @property
    def nu(self): return 1
    @property
    def nd(self): return 1
    @property
    def A(self): return np.array([[-1.0, 0.5], [0.0, -2.0]])
    @property
    def B(self): return np.array([[1.0], [0.0]])
    @property
    def E(self): return np.array([[0.0], [1.0]])
    @property
    def G(self): return np.eye(2)
    @property
    def Cz(self): return np.array([[1.0, 0.0]])    # output: state 0
    @property
    def Cm(self): return np.array([[1.0, 0.0]])    # measurement: state 0
    @property
    def Rm(self): return np.array([[0.05]])
    @property
    def dt(self): return 60.0                      # 1-minute sampling
    @property
    def x(self): return [0.0, 0.0]
    @x.setter
    def x(self, v): ...
    @property
    def x_ref(self): return np.array([1.0, 0.0])
    @property
    def u_bounds(self): return np.array([0.0]), np.array([2.0])

m = CSTRLinear()
print(m.nym)  # 1  (= Cm.shape[0])
print(m.nz)   # 1  (= Cz.shape[0])
print(m.nw)   # 2  (= G.shape[1])
print(isinstance(m, LinearContinuousDiscreteModel))  # True
```

#### `ContinuousDiscreteDAEModel` — `mbc.models`

Extends `ContinuousDiscreteModel` with an algebraic constraint and algebraic
states `y` (ControlToolbox §SDAE):

```
dx(t)   = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
0       = g(x, y, u, d, p, t)
z(t)    = gm(x, y, u, d, p, t)
ym(tk)  = hm(x, y, u, d, p, tk) + v(tk),                            v(tk) ~ N(0, Rm)
```

where `y ∈ ℝⁿʸ` is the algebraic state vector, kept consistent with the
differential state `x` at all times by enforcing `g = 0`.  The `nw` property
is inherited from `ContinuousDiscreteModel` and must be implemented by
concrete subclasses.

**Additional abstract members**:

| Member | Signature | Description |
|--------|-----------|-------------|
| `g` | `(x, y, u, d, p, t) → (ny,)` | Algebraic constraint residual; zero when satisfied |
| `ny` | `int` | Algebraic state dimension |
| `gm` | `(x, y, u, d, p, t) → (nz,)` | Continuous output function `g^m` |

**Optional analytic Jacobians** for the constraint and cross-terms:

| Member | Signature | Default |
|--------|-----------|---------|
| `dfdx` | `(x, y, u, d, p, t) → (nx, nx)` | Forward FD |
| `dfdy` | `(x, y, u, d, p, t) → (nx, ny)` | Forward FD |
| `dgdx` | `(x, y, u, d, p, t) → (ny, nx)` | Forward FD |
| `dgdy` | `(x, y, u, d, p, t) → (ny, ny)` | Forward FD |
| `dgdu` | `(x, y, u, d, p, t) → (ny, nu)` | Forward FD |
| `dgdd` | `(x, y, u, d, p, t) → (ny, nd)` | Forward FD |
| `dgdp` | `(x, y, u, d, p, t) → (ny, np)` | Forward FD |
| `dhmdy` | `(x, y, u, d, p, t) → (nym, ny)` | Forward FD |

Accepted by `SDAESimulator` and `ContinuousDiscreteDAEEKF`.

**Example** — isomerisation reactor with fast equilibrium (A ⇌ B):

The total concentration `C_tot = C_A + C_B` is the differential state.  The
split between A and B is determined by the fast equilibrium `K_eq = C_B / C_A`,
which is enforced as an algebraic constraint.

```python
import numpy as np
from mbc.models import ContinuousDiscreteDAEModel

class IsomerisationReactor(ContinuousDiscreteDAEModel):
    """
    Fast isomerisation A ⇌ B with equilibrium constant K_eq.

    Differential state : x = [C_tot]   total concentration (mol/L)
    Algebraic state    : y = [C_A]     concentration of species A (mol/L)
    Algebraic constraint: K_eq * C_A = C_B = C_tot - C_A
                          → g(x, y) = (K_eq + 1) * C_A - C_tot = 0
    Drift              : dC_tot/dt = F/V * (C_feed - C_tot)
    Input              : u = [F/V]  specific feed rate (1/min)
    Measurement        : ym = C_A  (UV absorber measuring species A)
    """
    _K_eq = 3.0     # equilibrium constant K_eq = C_B / C_A
    _C_feed = 5.0   # feed concentration (mol/L)

    @property
    def nx(self): return 1   # differential: C_tot
    @property
    def ny(self): return 1   # algebraic: C_A
    @property
    def nu(self): return 1   # input: F/V
    @property
    def nd(self): return 0
    @property
    def nw(self): return 1
    @property
    def nym(self): return 1  # measure C_A
    @property
    def nz(self): return 1   # output C_A
    @property
    def Rm(self): return np.array([[0.01]])

    def f(self, x, y, u, d, p, t):
        C_tot = x[0]; FV = u[0]
        return np.array([FV * (self._C_feed - C_tot)])

    def sigma(self, x, y, u, d, p, t):
        return np.array([[0.02]])   # small additive noise on C_tot

    def g(self, x, y, u, d, p, t):
        # Constraint: (K_eq + 1) * C_A - C_tot = 0
        return np.array([(self._K_eq + 1.0) * y[0] - x[0]])

    def gm(self, x, y, u, d, p, t):
        return np.array([y[0]])   # output: C_A

    def hm(self, x, y, u, d, p, t):
        return np.array([y[0]])   # measure C_A

m = IsomerisationReactor()
print(m.nx, m.ny, m.nu, m.nym, m.nz)  # 1  1  1  1  1

# Evaluate at a consistent operating point
C_tot = np.array([4.0])
C_A   = np.array([C_tot[0] / (m._K_eq + 1)])   # = 1.0 mol/L
u0    = np.array([0.2])
d0    = np.zeros(0); p0 = np.zeros(0)

print(m.g(C_tot, C_A, u0, d0, p0, t=0.0))   # [0.0] — constraint satisfied
print(m.f(C_tot, C_A, u0, d0, p0, t=0.0))   # drift dC_tot/dt
print(m.hm(C_tot, C_A, u0, d0, p0, t=0.0))  # predicted measurement C_A
```

---

### 2.2 Simulators

#### `SDESimulator` — `mbc.simulation`

Numerical integrator for `ContinuousDiscreteModel` (ControlToolbox §SDE —
*Numerical Integration*).  Simulates the continuous SDE from `t_k` to
`t_{k+1} = t_k + dt` using `n_steps` equidistant sub-steps of size
`Δt = dt / n_steps`.  Inputs `u` and disturbances `d` are held constant
(zero-order hold) over each measurement interval.  The discrete Wiener
increment is `Δω_n = z_n √Δt` with `z_n ~ N(0, I)`.

**Explicit-Explicit Euler-Maruyama (EE)** — both drift and diffusion
evaluated at the current sub-step:

```
x_{n+1} = x_n + Δt · f(x_n, u_k, d_k, p, t_n) + sigma(x_n, u_k, d_k, p, t_n) · Δω_n
```

This is the standard Euler-Maruyama discretisation.  Use when drift
dynamics are non-stiff.

**Implicit-Explicit (IE) scheme** — drift evaluated at the *next*
sub-step (implicit), diffusion evaluated explicitly:

```
x_{n+1} = x_n + Δt · f(x_{n+1}, u_k, d_k, p, t_{n+1}) + sigma(x_n, u_k, d_k, p, t_n) · Δω_n
```

The implicit equation is solved by Newton's method on the residual

```
R(x_{n+1}) = x_{n+1} − x_n − f(x_{n+1}, …) · Δt − sigma(x_n, …) · Δω_n = 0,

∂R/∂x = I − (∂f/∂x)(x_{n+1}, …) · Δt.
```

The IE scheme is appropriate when the drift dynamics are stiff (large
eigenvalues in `∂f/∂x`).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `dt` | `float` | — | Measurement sampling interval |
| `n_steps` | `int` | `10` | Integration sub-steps per interval |
| `scheme` | `"EE"` or `"IE"` | `"EE"` | Integration scheme |
| `seed` | `int` or `None` | `None` | Random seed for reproducibility |
| `newton_tol` | `float` | `1e-12` | IE drift Newton tolerance (ignored for EE) |
| `newton_max_iter` | `int` | `50` | IE maximum Newton iterations per sub-step |

**Methods**:

```python
from mbc.simulation import SDESimulator

sim = SDESimulator(model, dt=1.0, n_steps=20, scheme="EE", seed=42)

# Simulate one measurement interval [t, t+dt]
x_next = sim.step(x, u, d, p, t)               # returns (nx,) ndarray

# Simulate full horizon of T intervals
# U : (T, nu) ndarray,  D : (T, nd) ndarray,  P : (T, nparams) ndarray
X = sim.simulate(x0, U, D, P, t0=0.0)          # returns (T+1, nx) ndarray
```

#### `SDAESimulator` — `mbc.simulation`

Implicit-explicit Euler-Maruyama integrator for `ContinuousDiscreteDAEModel`
(ControlToolbox §SDAE — *Numerical Integration: Implicit-Explicit Method*).
Drift and the algebraic constraint are evaluated at the *next* sub-step
(implicit); diffusion is explicit.

At each sub-step the combined variable `z_{n+1} = (x_{n+1}, y_{n+1})` is
the root of

```
R(z_{n+1}) = [
    x_{n+1} − x_n − f(x_{n+1}, y_{n+1}, u_k, d_k, p, t_{n+1}) · Δt − sigma(x_n, y_n, …) · Δω_n;
    g(x_{n+1}, y_{n+1}, p)
] = 0
```

found by Newton's method with residual Jacobian

```
∂R/∂z = [
    I − (∂f/∂x) · Δt,    −(∂f/∂y) · Δt;
    ∂g/∂x,                ∂g/∂y
].
```

There is no explicit-explicit variant — the SDAE always requires the
Newton solve.  For consistent initial conditions the user-provided `y0`
must satisfy `g(x0, y0, …) = 0`.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteDAEModel` | — | SDAE model |
| `dt` | `float` | — | Measurement sampling interval |
| `n_steps` | `int` | `10` | Integration sub-steps per interval |
| `newton_tol` | `float` | `1e-10` | Newton solver convergence tolerance |
| `newton_max_iter` | `int` | `50` | Maximum Newton iterations per sub-step |
| `seed` | `int` or `None` | `None` | Random seed for reproducibility |

**Methods**:

```python
from mbc.simulation import SDAESimulator

sim = SDAESimulator(model, dt=1.0, n_steps=20, newton_tol=1e-10)

x_next, y_next = sim.step(x, y, u, d, p, t)      # one measurement interval
X, Y = sim.simulate(x0, y0, U, D, P, t0=0.0)     # (T+1,nx) and (T+1,ny)
```

---

### 2.3 State Estimators

All continuous-discrete estimators share the interface:

- `predict(u, d, t) → (x_pred, P_pred)` — propagate from `t` to `t + dt`
- `update(y, d, mask=None) → (x_hat, P)` — measurement update at time `t`
- `step(y, u, d, t, mask=None) → (x_hat, P)` — combined predict + update

#### Numerical Jacobians (used by all nonlinear estimators)

When the model does not supply analytic Jacobians, all nonlinear estimators
compute them by **forward finite differences** with step size `h_fd = 1e-5`:

```python
# Drift Jacobian  F = ∂f/∂x  (nx × nx)
def fd_jacobian_f(model, x, u, d, p, t, h_fd=1e-5):
    f0 = model.f(x, u, d, p, t)
    J = np.zeros((len(f0), len(x)))
    for k in range(len(x)):
        x_fwd = x.copy(); x_fwd[k] += h_fd
        J[:, k] = (model.f(x_fwd, u, d, p, t) - f0) / h_fd
    return J

# Measurement Jacobian  Hm = ∂hm/∂x  (nym × nx)
def fd_jacobian_hm(model, x, u, d, p, t, h_fd=1e-5):
    hm0 = model.hm(x, u, d, p, t)
    J = np.zeros((len(hm0), len(x)))
    for k in range(len(x)):
        x_fwd = x.copy(); x_fwd[k] += h_fd
        J[:, k] = (model.hm(x_fwd, u, d, p, t) - hm0) / h_fd
    return J
```

For `ContinuousDiscreteDAEModel`, additional Jacobians are needed:

```python
# ∂f/∂y  (nx × ny),  ∂g/∂x  (ny × nx),  ∂g/∂y  (ny × ny),  ∂hm/∂y  (nym × ny)
# Each follows the same forward-FD pattern, perturbing y or x respectively.
```

#### Missing-observation masking (nonlinear estimators)

All nonlinear estimators accept the same `mask` argument as `KalmanFilter`.
When `mask[i] = False`, output `i` is excluded from the measurement update by
constructing sub-matrices restricted to the active rows before calling the
update equations.  Let `active = [i for i, m in enumerate(mask) if m]` and
`na = len(active)`:

```
ym_sub   = ym[active]                      (na,) subset of the measurement vector
Hm_sub   = Hm[active, :]                   (na, nx) rows of the Jacobian (EKF/DAE-EKF)
Rm_sub   = Rm[np.ix_(active, active)]      (na, na) sub-block of the noise matrix
ŷm_sub   = ŷm[active]                      (na,) predicted observation subset
```

These sub-matrices replace their full-size counterparts in the innovation and
Kalman-gain computations.  For the EnKF and PF, only the active rows of
`hm(x, u, d, p, t)` are evaluated and only `Rm_sub` enters the likelihood computation.

#### Delayed-observation update (all estimator variants)

Measurements with a per-channel reporting delay (e.g. laboratory assays) are
handled by `DelayedObservationFilter` — see §1.2 for the full class description.
The wrapper works with all CD estimators listed below by accepting the same
`step`/`update`/`predict` interface and adding a `delay=(ny,) int ndarray`
argument.  The buffer, retrospective correction, and replay logic are fully
encapsulated inside the wrapper; the wrapped CD estimator is called only through
its standard `predict` and `update` methods.

---

#### `CDKalmanFilter` — `mbc.estimation`

Kalman filter for a **linear** continuous-discrete stochastic system —
the linear specialisation of :class:`~mbc.estimation.ContinuousDiscreteEKF`
(no Jacobian linearisation needed).  The continuous-time matrices
``A``, ``B``, ``E`` are integrated directly by forward Euler — no ZOH
or Van Loan pre-discretisation inside the filter.

**Model**: :class:`~mbc.models.LinearContinuousDiscreteModel`.

**Time update over ``[t_{k-1}, t_k]``** — forward-Euler integration of
the state and Lyapunov-type covariance ODEs with sub-step ``h = dt / n_steps``:

```
For n = 0, 1, …, n_steps − 1:

    ẋ̂ = A x̂ + B u + E d                        (state ODE)
    Ṗ = A P + P Aᵀ + G Gᵀ                      (Lyapunov ODE)

    x̂ ← x̂ + h · ẋ̂
    P  ← P  + h · Ṗ

P ← ½(P + Pᵀ)                                  (symmetrise after integration)
```

Inputs ``u`` and disturbances ``d`` are zero-order hold over the just-
completed sampling interval.  ``G Gᵀ`` is pre-computed and cached.

**Measurement update at ``t_k`` (Joseph form)** — identical structure to
:class:`~mbc.estimation.ContinuousDiscreteEKF` (with the constant Jacobian
``Cm`` instead of ``∂hm/∂x``):

```
e_k    = ym_k − Cm x̂_{k|k-1}                  (innovation)
R_e    = Cm P_{k|k-1} Cmᵀ + Rm                  (innovation covariance)
K_k    = P_{k|k-1} Cmᵀ R_e⁻¹                    (Kalman gain)

x̂_{k|k} = x̂_{k|k-1} + K_k e_k
P_{k|k} = (I − K_k Cm) P_{k|k-1} (I − K_k Cm)ᵀ + K_k Rm K_kᵀ        (Joseph)
```

**Missing observations** — see :class:`KalmanFilter` for the identical
masking logic.

**Constructor**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearContinuousDiscreteModel` | — | Linear CD plant |
| `x0` | `(nx,) ndarray` | `np.array(model.x)` | Initial state estimate |
| `P0` | `(nx, nx) ndarray` | `I_{nx}` | Initial state error covariance |
| `n_steps` | `int` | `10` | Forward-Euler sub-steps per sampling interval |

**Methods** (signatures match
:class:`~mbc.estimation.ContinuousDiscreteEKF`):

```python
from mbc.estimation import CDKalmanFilter

kf = CDKalmanFilter(model, x0=x0, P0=P0, n_steps=10)

# Building blocks
x_pred, P_pred = kf.predict(u_prev, d_prev)       # time update (ODE integration)
x_hat,  P     = kf.update (ym, mask=None)         # measurement update (Joseph)

# Combined predict + update (p, t accepted for interface compatibility)
x_hat, P = kf.step(ym, u_prev, d_prev, mask=None)
```

**Public properties**: ``x_hat`` ``(nx,)``, ``P`` ``(nx, nx)``,
``last_innovation`` (list of floats or ``None``).

---

#### `ContinuousDiscreteEKF` — `mbc.estimation` *(ControlToolbox §SDE — CD-EKF)*

Extended Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Applies the
linear Kalman filter update equations to a local linearisation of the nonlinear
state and measurement dynamics; the state distribution is characterised by its
first two moments — mean and covariance — at all times.

**Time update over [t_k, t_{k+1}]** — two propagation schemes available via
the ``scheme`` parameter:

*Explicit Euler* (`scheme="euler"`, default) — integrates the mean ODE and
the Lyapunov-type covariance ODE with explicit Euler:

```
dx̂_k/dt(t) = f(x̂_k, u, d, p, t)                                       (mean ODE)
dP_k/dt(t) = A_k(t) P_k + P_k A_kᵀ(t) + sigma_k(t) sigma_kᵀ(t)         (Lyapunov ODE)

A_k(t)     = ∂f/∂x(x̂_k(t), u, d, p, t)
sigma_k(t) = sigma(x̂_k(t), u, d, p, t)
```

with sub-step `h = dt / n_steps`:

```
For n = 0, 1, …, n_steps − 1:
    A_n     = ∂f/∂x evaluated at (x̂_n, u, d, p, t_n)
    sigma_n = sigma(x̂_n, u, d, p, t_n)
    x̂_{n+1} = x̂_n + h · f(x̂_n, u, d, p, t_n)
    P_{n+1} = P_n + h · (A_n P_n + P_n A_nᵀ + sigma_n sigma_nᵀ)
    P_{n+1} ← ½(P_{n+1} + P_{n+1}ᵀ)                  (symmetrise)
```

*Implicit Euler* (`scheme="implicit-euler"`) — L-stable; suitable for stiff
drift dynamics.  Uses Newton iteration for the mean and the one-step
sensitivity matrix for the covariance:

```
For n = 0, 1, …, n_steps − 1:
    sigma_n = sigma(x̂_n, u, d, p, t_n)                       (diffusion at start)
    Newton solve:  x̂_{n+1} − x̂_n − h · f(x̂_{n+1}, u, d, p, t_{n+1}) = 0
    M       = I − h · ∂f/∂x(x̂_{n+1}, u, d, p, t_{n+1})
    Φ       = M⁻¹                                             (sensitivity matrix)
    τ       = P_n + h · sigma_n sigma_nᵀ
    P_{n+1} = Φ τ Φᵀ
    P_{n+1} ← ½(P_{n+1} + P_{n+1}ᵀ)                  (symmetrise)
```

The Jacobian `A = ∂f/∂x` is provided analytically by the model or computed by
forward finite differences.

**Measurement update** — Joseph form:

```
C_k  = ∂hm/∂x(x̂_{k|k-1}, u, d, p, 0)     (measurement Jacobian, nym × nx)
ŷ^m  = hm(x̂_{k|k-1}, u, d, p, 0)         (predicted measurement)
e_k  = y^m_k − ŷ^m                        (innovation)
R_e  = C_k P_{k|k-1} C_kᵀ + R             (innovation covariance)
K_k  = P_{k|k-1} C_kᵀ R_e⁻¹               (Kalman gain)

x̂_{k|k} = x̂_{k|k-1} + K_k e_k
P_{k|k} = (I − K_k C_k) P_{k|k-1} (I − K_k C_k)ᵀ + K_k R K_kᵀ          (Joseph)
```

The Joseph stabilising form preserves symmetry and positive definiteness of
the posterior covariance in finite-precision arithmetic.

The filter **always uses explicit Euler** for both the mean ODE and the
Lyapunov-type covariance ODE.  No implicit propagation scheme is available.
To simulate stiff SDE dynamics use [`SDESimulator`](#sdesimulator----mbcsimulation)
with `scheme="IE"`; that choice is independent of the filter.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial state covariance |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Integration sub-steps per interval (≥ 1) |
| `scheme` | `str` | `"euler"` | `"euler"` or `"implicit-euler"` |
| `newton_tol` | `float` | `1e-10` | Newton tolerance (implicit-Euler only) |
| `newton_max_iter` | `int` | `50` | Max Newton iterations (implicit-Euler only) |

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteEKF

# Explicit Euler (default)
ekf = ContinuousDiscreteEKF(model, x0, P0, dt=1.0, n_steps=10)

# Implicit Euler — suitable for stiff drift dynamics
ekf = ContinuousDiscreteEKF(model, x0, P0, dt=1.0, n_steps=10, scheme="implicit-euler")

x_hat, P = ekf.step(y, u, d, t, mask=None)   # predict + update
x_hat, P = ekf.predict(u, d, t)              # prediction only
x_hat, P = ekf.update(y, d, mask=None)       # update only
```

---

#### `ContinuousDiscreteUKF` — `mbc.estimation` *(ControlToolbox §SDE — CD-UKF)*

Unscented Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Replaces
the Jacobian linearisation of the EKF with a deterministic sigma-point
approximation (unscented transform).  The time update uses an **augmented**
dimension `n̄ = nx + nω` to handle state uncertainty and process noise
explicitly through two sets of sigma points propagated separately.  No
Jacobian is required.

**Time update — augmented sigma points**.  Tuning parameters
`α ∈ ]0, 1]`, `κ ≥ 0`, `β ≥ 0` (β = 2 optimal for Gaussian):

```
c̄ = α² (n̄ + κ),    λ̄ = α² (n̄ + κ) − n̄

W̄_m^0 = λ̄ / (n̄ + λ̄)
W̄_c^0 = λ̄ / (n̄ + λ̄) + (1 − α² + β)
W̄_m^i = W̄_c^i = 1 / (2 (n̄ + λ̄))    for i = 1, …, 2 n̄
```

Two sigma-point sets are constructed.  Deterministic state set
(`2 nx + 1` points capturing state covariance):

```
χ^(0)        = x̂_{k|k}
χ^(i)        = x̂_{k|k} + √c̄ · L[:, i-1]            for i = 1, …, nx
χ^(nx+i)     = x̂_{k|k} − √c̄ · L[:, i-1]            for i = 1, …, nx
                where  L L^T = P_{k|k}              (Cholesky)
```

Stochastic noise sigma set (`2 nω` points all placed at the mean) with
structured deterministic Wiener increments `Δω = ±√(c̄ · dt) · e_i`,
distributed equally across sub-steps so that the cumulative increment
matches:

```
Per sub-step n:  Δω_n^(2nx+i)        = +√(c̄ · dt) / n_steps · e_i   for i = 1, …, nω
                  Δω_n^(2nx+nω+i)     = −√(c̄ · dt) / n_steps · e_i   for i = 1, …, nω
```

The deterministic set is propagated through the *drift only*

```
χ^(i) ← χ^(i) + h · f(χ^(i), u, d, p, t)              i = 0, …, 2 nx
```

and the stochastic set through the *full SDE* with structured noise:

```
χ^(2nx+i) ← χ^(2nx+i) + h · f(χ^(2nx+i), …) + sigma(χ^(2nx+i), …) · Δω_n^(2nx+i)
```

Predicted mean and covariance are weighted statistics over all `2 n̄ + 1`
sigma points:

```
x̂_{k+1|k} = Σ_i W̄_m^i χ^(i)
P_{k+1|k} = Σ_i W̄_c^i (χ^(i) − x̂_{k+1|k})(…)ᵀ
```

No additive `sigma sigma^T · dt` term is needed — the diffusion is captured
entirely by the propagation of the stochastic sigma points.

**Measurement update — state-only sigma points** (no augmentation, dimension
`nx`):

```
c = α² (nx + κ),    λ = α² (nx + κ) − nx

χ^(0)        = x̂_{k|k-1}
χ^(i)        = x̂_{k|k-1} + √c · L[:, i-1]            i = 1, …, nx
χ^(nx+i)     = x̂_{k|k-1} − √c · L[:, i-1]            i = 1, …, nx

z^{m,(i)}    = hm(χ^(i), u, d, p, 0),   ŷ^m = Σ_i W_m^i z^{m,(i)}
R_zz         = Σ_i W_c^i (z^{m,(i)} − ŷ^m)(…)ᵀ
R_xy         = Σ_i W_c^i (χ^(i) − x̂_{k|k-1})(z^{m,(i)} − ŷ^m)ᵀ
R_e          = R_zz + R                       (innovation covariance)
K_k          = R_xy R_e⁻¹                     (Kalman gain via cross-covariance)

x̂_{k|k} = x̂_{k|k-1} + K_k (y^m_k − ŷ^m)
P_{k|k} = P_{k|k-1} − K_k R_e K_kᵀ
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial state covariance |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Euler sub-steps per interval |
| `alpha` | `float` | `1.0` | Sigma-point spread α ∈ ]0, 1] |
| `beta` | `float` | `2.0` | Distribution parameter (2 = optimal for Gaussian) |
| `kappa` | `float` | `0.0` | Secondary spread parameter κ ≥ 0 |

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteUKF

ukf = ContinuousDiscreteUKF(model, x0, P0, dt=1.0, alpha=1.0, beta=2.0)
x_hat, P = ukf.step(y, u, d, p, t, mask=None)
```

---

#### `ContinuousDiscreteEnKF` — `mbc.estimation` *(ControlToolbox §SDE — CD-EnKF)*

Ensemble Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Maintains an
ensemble of `N_p` particles to approximate the state distribution without
requiring Jacobian computations.  Each particle is propagated independently
through the *full* stochastic dynamics with state-dependent diffusion; the
ensemble mean and Bessel-corrected sample covariance replace the analytical
Gaussian approximation used by the EKF and UKF.

**Initialisation**: draw `N_p` particles from `N(x0, P0)`:

```
X[:,i] ~ N(x0, P0)   for i = 1, …, N_p
```

**Time update** — per-particle Euler-Maruyama with state-dependent diffusion:

```
For each particle i = 1, …, N_p and sub-step n = 0, …, n_steps−1:

    f_i     = f(X_n[:,i], u, d, p, t_n)
    sigma_i = sigma(X_n[:,i], u, d, p, t_n)
    z_i     ~ N(0, I)            (independent per particle and sub-step)
    X_{n+1}[:,i] = X_n[:,i] + h · f_i + sigma_i · z_i · √h
```

After propagation (Bessel-corrected sample statistics):

```
x̂_{k+1|k} = (1/N_p) Σ_i X[:,i]
P_{k+1|k} = (1/(N_p − 1)) (X − x̂ 1ᵀ)(X − x̂ 1ᵀ)ᵀ
```

**Measurement update** — perturbed observations:

```
Predicted measurements:  Z[:,i] = hm(X⁻[:,i], u, d, p, 0)            i = 1, …, N_p
Sample statistics:
    ŷ^m_{k|k-1} = (1/N_p) Σ_i Z[:,i]
    R_zz        = (1/(N_p − 1)) (Z − ŷ^m 1ᵀ)(…)ᵀ
    R_xy        = (1/(N_p − 1)) (X⁻ − x̂_{k|k-1} 1ᵀ)(Z − ŷ^m 1ᵀ)ᵀ
    R_e         = R_zz + R

Single Kalman gain:   K_k = R_xy R_e⁻¹

Perturbed measurements:  y^{m,(i)}_k = y^m_k + v^(i),  v^(i) ~ N(0, R)

Per-particle update:    x̂_{k|k}^(i) = X⁻[:,i] + K_k (y^{m,(i)}_k − Z[:,i])
```

The perturbation prevents ensemble collapse — without it the ensemble
covariance shrinks by the deterministic factor `(I − K C)` and underestimates
the posterior covariance.  Filtered statistics are sample mean and
Bessel-corrected sample covariance over the updated ensemble.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate (ensemble mean) |
| `P0` | `(nx,nx) ndarray` | — | Initial covariance (for drawing ensemble) |
| `dt` | `float` | — | Sampling interval |
| `N` | `int` | `100` | Ensemble size |
| `n_steps` | `int` | `10` | Euler-Maruyama sub-steps per interval |
| `seed` | `int` or `None` | `None` | Random seed |

**Public properties**: `x_hat`, `P`, `ensemble` (full `(nx, N)` particle matrix).

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteEnKF

enkf = ContinuousDiscreteEnKF(model, x0, P0, dt=1.0, N=200, seed=0)
x_hat, P = enkf.step(y, u, d, t, mask=None)
```

---

#### `ContinuousDiscreteParticleFilter` — `mbc.estimation` *(ControlToolbox §SDE — CD-PF)*

Sequential Monte Carlo (particle filter) for a nonlinear `ContinuousDiscreteModel`.
Represents the posterior `p(x[k] | y[1:k])` as a particle set of size `N_p`.
Unlike the EnKF the measurement update makes **no Gaussian assumption**; the
filter is asymptotically exact as `N_p → ∞`.

**Initialisation**: draw `N_p` particles from `N(x0, P0)`:

```
X[:,i] ~ N(x0, P0)    for i = 1, …, N_p
```

**Time update** — identical to CD-EnKF: each particle is propagated
independently through the full SDE via Euler-Maruyama with state-dependent
diffusion and an independent Wiener increment per particle and sub-step.

**Measurement update** — likelihood-weighted **systematic resampling**.  Per
particle:

```
z^{m,(i)} = hm(X⁻[:,i], u, d, p, 0)               (predicted measurement)
e^(i)     = y^m_k − z^{m,(i)}                     (innovation)
w̃^(i)    = (2π)^{-nym/2} |R|^{-1/2} exp(−½ (e^(i))ᵀ R⁻¹ e^(i))    (Gaussian likelihood)
w^(i)     = w̃^(i) / Σ_j w̃^(j)                  (normalised weights)
```

Numerically the log-weights are computed and normalised with the
log-sum-exp trick to avoid underflow.

**Systematic resampling** (per CD-PF spec — every step, no threshold):

```
1. Compute weight CDF:    s^(i) = Σ_{j ≤ i} w^(j),    s^(0) = 0
2. Draw a single q_1 ~ Uniform[0, 1 / N_p)
3. Form equally-spaced resampling points:  q^(i) = (i − 1) / N_p + q_1   for i = 1, …, N_p
4. For each q^(l), select particle i with s^(i-1) < q^(l) ≤ s^(i)
5. Replace particle set with the selected particles
```

Using a single uniform draw (rather than `N_p` independent draws) gives
**systematic** resampling, which has `O(N_p)` cost and minimal variance among
resampling schemes; each particle is replicated at least
`⌊N_p w^(i)⌋` times.

**Filtered statistics** (sample mean and Bessel-corrected sample covariance
over the resampled set):

```
x̂_{k|k} = (1 / N_p) Σ_i x̂_{k|k}^(i)
P_{k|k} = (1 / (N_p − 1)) Σ_i (x̂_{k|k}^(i) − x̂_{k|k})(…)ᵀ
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial covariance |
| `dt` | `float` | — | Sampling interval |
| `N` | `int` | `500` | Number of particles N_p |
| `n_steps` | `int` | `10` | Euler-Maruyama sub-steps per interval |
| `seed` | `int` or `None` | `None` | Random seed |

**Public properties**: `x_hat`, `P`, `particles` `(nx, N_p)`.

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteParticleFilter

pf = ContinuousDiscreteParticleFilter(model, x0, P0, dt=1.0, N=1000, seed=0)
x_hat, P = pf.step(y, u, d, t, mask=None)
```

---

#### `ContinuousDiscreteDAEEKF` — `mbc.estimation` *(ControlToolbox §SDAE — CD-EKF)*

Extended Kalman Filter for `ContinuousDiscreteDAEModel`.  The algebraic
variables `y` are *never* added to the state vector — the implicit function
theorem expresses `y` as an implicit function of `x` and propagates the
resulting sensitivities through the time and measurement updates.  The state
covariance `P` remains `nx × nx`, and `P_y` is recovered as a post-processing
step from the algebraic sensitivity.

**Initialisation**.  The user-supplied `y0` is projected onto the constraint
manifold by solving `g(x0, y0, …) = 0` (Newton).  The initial algebraic
covariance is

```
P_{y,0|0} = Φ_yx P_{0|0} Φ_yxᵀ,    (∂g/∂y) Φ_yx = −∂g/∂x.
```

**Time update — implicit-Euler with sensitivity propagation**.  The
interval `[t_k, t_{k+1}]` is divided into `n_steps` sub-steps of size
`h = dt / n_steps`.  At each sub-step:

1. **Mean sub-step.**  Solve the implicit-Euler residual

   ```
   R(z_{n+1}) = [
       x_{n+1} − x_n − f(x_{n+1}, y_{n+1}, u, d, p, t_{n+1}) h;
       g(x_{n+1}, y_{n+1}, p)
   ] = 0
   ```

   for `z_{n+1} = (x_{n+1}, y_{n+1})` by Newton's method.

2. **Sensitivity sub-step.**  The one-step sensitivities `Φ_xx`, `Φ_yx`
   solve the linear system (same coefficient matrix as the Newton Jacobian):

   ```
   [ I − (∂f/∂x) h    −(∂f/∂y) h ] [ Φ_xx ]   [ I ]
   [   ∂g/∂x              ∂g/∂y  ] [ Φ_yx ] = [ 0 ]
   ```

3. **Covariance sub-step** (left-rectangular rule for the stochastic integral):

   ```
   τ_n      = P_n + sigma(x_n, y_n, …) sigma^T h
   P_{n+1}  = Φ_xx τ_n Φ_xxᵀ
   ```

4. **Algebraic covariance** at the new sub-step (implicit function theorem):

   ```
   P_{y,n+1} = Φ_yx P_{n+1} Φ_yxᵀ,    (∂g/∂y) Φ_yx = −∂g/∂x   at (x_{n+1}, y_{n+1}).
   ```

**Measurement update — total derivative C and Joseph form**.  The matrix
`C_k` is the *total* derivative of `hm` with respect to `x`, accounting for
the indirect dependence through `y(x)`:

```
C_k = ∂hm/∂x + (∂hm/∂y) (∂y/∂x),    (∂g/∂y) (∂y/∂x) = −(∂g/∂x).

R_e = C_k P_{k|k-1} C_kᵀ + R
K_k = P_{k|k-1} C_kᵀ R_e⁻¹
e_k = y^m_k − hm(x̂_{k|k-1}, ŷ_{k|k-1}, p)
x̂_{k|k} = x̂_{k|k-1} + K_k e_k
P_{k|k} = (I − K_k C_k) P_{k|k-1} (I − K_k C_k)ᵀ + K_k R K_kᵀ          (Joseph)
```

The filtered `ŷ_{k|k}` satisfies `g(x̂_{k|k}, ŷ_{k|k}, p) = 0` (Newton); the
filtered `P_{y,k|k}` is recomputed via the implicit function theorem at the
filtered state.

**Required Jacobians** (forward FD by default, may be overridden analytically):
`∂f/∂x`, `∂f/∂y`, `∂g/∂x`, `∂g/∂y`, `∂hm/∂x`, `∂hm/∂y`.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteDAEModel` | — | SDAE model |
| `x0` | `(nx,) ndarray` | — | Initial differential state estimate |
| `y0` | `(ny,) ndarray` | — | Initial algebraic state guess (will be projected onto `g = 0`) |
| `P0` | `(nx,nx) ndarray` | — | Initial covariance (differential states only) |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Implicit-Euler sub-steps per interval |
| `newton_tol` | `float` | `1e-10` | Newton convergence tolerance |
| `newton_max_iter` | `int` | `50` | Max Newton iterations per solve |

**Properties**: `x_hat`, `y_hat`, `P`, `Py`.

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteDAEEKF

ekf = ContinuousDiscreteDAEEKF(model, x0, y0, P0, dt=1.0)
x_hat, y_hat, P = ekf.step(ym, u, d, p, t, mask=None)
x_hat, y_hat, P = ekf.predict(u, d, p, t)
x_hat, y_hat, P = ekf.update(ym, u, d, p, mask=None)
```

---

### 2.4 Optimal Control Problems

The nonlinear OCP for continuous-discrete SDE/SDAE plants follows the
**ControlToolbox §EMPC** direct-simultaneous formulation: implicit Euler
for the differential dynamics, the algebraic constraint `g = 0` enforced
at every sub-step, and right-rectangular discretisation of the Lagrange
integral.  Decision variables are the inputs `{u_k}_{k=0..N−1}` together
with the differential and algebraic states `{x_n, y_n}_{n=0..M}` at every
sub-step (M = N · n_steps) — multiple shooting, not single shooting.  The
NLP is solved through a swappable backend interface:

- default: SciPy backend (`scipy.optimize.minimize`, method `"SLSQP"`)
- optional: IPOPT backend (`solver="ipopt"`, requires `cyipopt`)

You can swap backends without changing OCP construction.
For comparative runtime/iteration baseline, horizon-scaling checks, and
analytical-vs-numerical gradient/Hessian efficiency examples, run
`python scripts/nlp_solver_benchmark.py`.

#### `CDOptimalControlProblem` — `mbc.control`

Receding-horizon QP for a `LinearContinuousDiscreteModel`.  A typed thin wrapper
around `OptimalControlProblem` that accepts a continuous-discrete model.
Internally wraps the model in a `_CDModelAdapter` that computes ZOH-discretised
matrices `(Ad, Bd, Ed)` from the continuous-time model matrices `(A, B, E)` at
construction time and exposes them as numpy arrays for the QP solver.

The cost function, constraints, batch-form prediction matrices, and QP solver
are identical to `OptimalControlProblem` (see §1.3).  ZOH discretisation is
*exact* for a linear ODE under ZOH inputs, so this linear specialisation is
strictly more accurate than the implicit-Euler scheme used in the nonlinear
OCP — and is the recommended choice whenever the plant model is linear.

**Parameters**: identical to `OptimalControlProblem` with `model` of type
`LinearContinuousDiscreteModel`.

**Usage**:

```python
from mbc.control import CDOptimalControlProblem

ocp = CDOptimalControlProblem(model, N=20, Q=Q_y, R=R_u, P=P_terminal)
U_seq, X_seq = ocp.solve(x0=x_hat, D=D_forecast, x_ref=model.x_ref, u_prev=u_prev)
```

---

#### `EconomicOptimalControlProblem` — `mbc.control` *(ControlToolbox §EMPC)*

Economic Optimal Control Problem (EOCP) for **continuous-discrete nonlinear
SDE / SDAE systems**.  This is the unified nonlinear-OCP class — it supports
any convex combination of the spec's objective terms (tracking, ROM, input
economy, general Lagrange/Mayer) with both hard and soft (slacked, exact-
penalty) constraints.

The class accepts both `ContinuousDiscreteModel` (SDE) and
`ContinuousDiscreteDAEModel` (SDAE) plant models.  When the plant is an
SDAE, the algebraic state `y` is a decision variable of the NLP and the
constraint `g(x, y, …) = 0` is enforced at every sub-step.

##### Continuous-time OCP (Bolza form)

```
min_{x, y, u}  φ = ∫_{t_0}^{t_f} l(t, x, y, u, θ) dt + l̂(x(t_f), y(t_f), θ)

s.t.   x(t_0)        = x̂_{0|0}                 (from state estimator)
       dx/dt         = f(x, y, u, d, θ)
       0             = g(x, y, θ)               (SDAE only)
       z(t)          = g^m(x, y, θ)
       c_lb(t)       ≤ c(t, x, y, u, θ) ≤ c_ub(t)
```

with zero-order-hold inputs `u(t) = u_k` on `[t_k, t_{k+1}]`.

##### Discretisation (direct simultaneous)

Each control interval `[t_k, t_{k+1}]` is split into `n_steps` equidistant
sub-steps of size `Δt = T_s / n_steps`.  Decision variables:

```
{{x_{k,n}, y_{k,n}}_{n=0..n_steps},  u_k}_{k=0..N-1}
```

with continuity `x_{k+1, 0} = x_{k, n_steps}` (and likewise for `y`).
Internally these are stored as a single flat sub-step grid `x_n, y_n` for
`n = 0..M`.

**Sub-step dynamics residual** (implicit Euler — same form used by
`SDAESimulator`):

```
D(z_{n+1}, z_n, u_k, d_k, θ) = [
    x_{n+1} − x_n − f(x_{n+1}, y_{n+1}, u_k, d_k, θ) Δt;
    g(x_{n+1}, y_{n+1}, θ)
] = 0
```

**Lagrange discretisation** — right-rectangular rule:

```
Φ_L = Σ_{n=0..M-1} l(t_{n+1}, x_{n+1}, y_{n+1}, u_{k(n)}, θ) Δt
```

##### Objective terms (convex combination)

```
φ_z       = Σ_n  ‖z_{n+1} − z̄_{n+1}‖²_{Q_z}                  Δt    setpoint tracking
φ_{Δu}    = Σ_k  ‖u_k − u_{k−1}‖²_{Q_du}                       T_s   input ROM
φ_{u,eco} = Σ_k  p_{u,eco}^T u_k                                T_s   linear input cost
φ_{lag}   = Σ_n  l(t_{n+1}, x_{n+1}, y_{n+1}, u_{k(n)}, θ)     Δt    user Lagrange
φ_M       =       l̂(x_M, y_M, θ)                                     user Mayer
φ_pq      = Σ_n  [‖p_n‖²_{rho_·_2} + rho_·_1^T p_n + ‖q_n‖²_{rho_·_2} + rho_·_1^T q_n] Δt
```

The L1 (linear) component of `φ_pq` is an **exact penalty**: with a
sufficiently large `rho_·_1`, the soft-constrained optimum coincides with
the hard-constrained solution whenever the latter exists.

##### Hard constraints

```
u_min  ≤ u_k                 ≤ u_max          (input box)
du_min ≤ u_k − u_{k−1}       ≤ du_max         (input rate-of-movement box)
```

##### Soft (slacked) constraints

```
x_min − p_n ≤ x_n ≤ x_max + q_n,    p_n, q_n ≥ 0
z_min − p_n ≤ z_n ≤ z_max + q_n,    p_n, q_n ≥ 0
```

with `z_n = g^m(x_n, y_n, …)`.  The slacks `p_n, q_n` are themselves NLP
decision variables; they are penalised in the objective by the L1 + L2
exact-penalty form `φ_pq` above.

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` or `…DAEModel` | — | Plant model |
| `N` | `int` | — | Prediction horizon (control intervals) |
| `lagrange` | `(t, x, y, u, θ) → float` or `None` | `None` | General stage cost `l` |
| `mayer` | `(x, y, θ) → float` or `None` | `None` | Terminal cost `l̂` |
| `Q_z` | `(nz, nz) ndarray` or `None` | `None` | Tracking weight (requires `z_ref`) |
| `z_ref` | `(nz,)` or `(M+1, nz) ndarray` | `None` | Tracking reference |
| `Q_du` | `(nu, nu) ndarray` or `None` | `None` | Quadratic ROM penalty matrix |
| `p_u_eco` | `(nu,) ndarray` or `None` | `None` | Linear input cost vector |
| `u_min`, `u_max` | `(nu,) ndarray` or `None` | `None` | Hard input box |
| `du_min`, `du_max` | `(nu,) ndarray` or `None` | `None` | Hard input ROM box |
| `x_min`, `x_max` | `(nx,) ndarray` or `None` | `None` | Soft state box |
| `z_min`, `z_max` | `(nz,) ndarray` or `None` | `None` | Soft output box |
| `rho_x_1` | `float` | `0.0` | L1 weight on state slacks (exact penalty) |
| `rho_x_2` | `float` | `1e4` | L2 weight on state slacks |
| `rho_z_1` | `float` | `0.0` | L1 weight on output slacks (exact penalty) |
| `rho_z_2` | `float` | `1e4` | L2 weight on output slacks |
| `n_steps` | `int` | `10` | Implicit-Euler sub-steps per control interval |
| `solver` | `str` or backend object | `"SLSQP"` | `"scipy"` / `"ipopt"` backend key, or SciPy method name (`"SLSQP"`, `"trust-constr"`, …) |
| `solver_options` | `dict` or `None` | `None` | Forwarded to the solver |
| `solver_scaling` | `dict` or `NLPScalingPolicy` or `None` | `None` | Backend-agnostic scaling (`objective_scale`, `variable_scale`, `constraint_scale`) |
| `dt` | `float` or `None` | `model.dt` or `1.0` | Sampling interval `T_s` |

##### Public properties

| Property | Type | Description |
|----------|------|-------------|
| `N` | `int` | Prediction horizon |
| `nu` | `int` | Input dimension |

##### Methods

```python
from mbc.control import EconomicOptimalControlProblem

# Spec signature: lagrange(t, x, y, u, theta) -> float
def lagrange(t, x, y, u, theta):
    return float(u @ u)            # quadratic input cost as a stage term

# Spec signature: mayer(x, y, theta) -> float
def mayer(x, y, theta):
    return 0.1 * float(x @ x)     # terminal state penalty

ocp = EconomicOptimalControlProblem(
    model, N=20, dt=1.0,
    lagrange=lagrange, mayer=mayer,
    Q_z=Q_z, z_ref=z_setpoint,        # tracking
    Q_du=S_rom,                        # ROM
    p_u_eco=cost_per_unit_input,       # input economy
    u_min=u_lo, u_max=u_hi,
    du_min=du_lo, du_max=du_hi,
    x_min=x_lo, x_max=x_hi,
    z_min=z_lo, z_max=z_hi,
    rho_x_1=1e3, rho_x_2=1e4,         # exact-penalty + quadratic on x slacks
    rho_z_1=1e3, rho_z_2=1e4,
    n_steps=10,
)

# Full sequence solve — returns (u_opt, cost, info)
u_opt, cost, info = ocp.solve(
    x0=x_hat, d_trajectory=d_fcast,
    u_prev=u_seq_prev, x_prev=info_prev["X"], y_prev=info_prev["Y"],
    p=p, t0=t,
)
# info["X"]: (M+1, nx) — full state trajectory at every sub-step
# info["Y"]: (M+1, ny) — algebraic-state trajectory (empty for SDE)

# First action only (receding horizon)
u0 = ocp.step(x_hat, d_fcast, u_prev=u_seq_prev, p=p, t0=t)
```

---

#### `CDTrackingOptimalControlProblem` — `mbc.control`

Convenience wrapper around `EconomicOptimalControlProblem` exposing a
quadratic-tracking-friendly constructor (`Q`, `R`, `P`, `S`, `c_u`).
Translates the standard tracking-OCP arguments into the underlying
`EconomicOptimalControlProblem`:

| Tracking-OCP arg | Maps to |
|------------------|---------|
| `Q` | `Q_z` |
| `R` | a Lagrange callable `(t, x, y, u, θ) → u^T R u` |
| `P` | a Mayer callable `(x, y, θ) → (z_M − z_ref)^T P (z_M − z_ref)` |
| `S` | `Q_du` |
| `c_u` | `p_u_eco` |
| `rho_x` | `rho_x_2` |
| `rho_z` | `rho_z_2` |

Because the underlying problem is the same EOCP, both classes are
interchangeable as the `ocp` argument of `CDNMPCController`.

```python
from mbc.control import CDTrackingOptimalControlProblem

ocp = CDTrackingOptimalControlProblem(
    model, N=20, Q=Q_z, R=R_u,
    S=S_rom, c_u=np.zeros(model.nu), z_ref=z_setpoint,
    u_min=u_lo, u_max=u_hi, du_min=du_lo, du_max=du_hi,
    x_min=x_lo, x_max=x_hi, rho_x=1e4,
    z_min=z_lo, z_max=z_hi, rho_z=1e4,
    dt=1.0, n_steps=10,
)
u_opt, cost, info = ocp.solve(x0=x_hat, d_trajectory=D, u_prev=u_seq_prev, p=p, t0=t)
u0                  = ocp.step (x_hat,                  D, u_prev=u_seq_prev, p=p, t0=t)
```

---

### 2.5 MPC Controllers

An MPC controller is **not** an OCP.  The OCP (§2.4) takes a state estimate as
input and returns an optimal input sequence.  The MPC controller closes the loop
by combining an OCP with a state estimator: it receives the raw measurement `ym[k]`,
passes it to the estimator to produce `x̂[k]`, and then solves the OCP from that
estimate.  The distinction matters: an OCP can be tested and tuned in isolation;
the MPC is the closed-loop composition.

#### `CDMPCController` — `mbc.control`

Combines a :class:`~mbc.estimation.CDKalmanFilter` (estimator) and a
:class:`~mbc.control.CDOptimalControlProblem` (OCP) into a closed-loop
receding-horizon controller for a linear continuous-discrete plant — the
linear specialisation of the ControlToolbox §EMPC ENMPC algorithm.

**Receding-horizon policy** — at each measurement time k:

1. **Measure**:  ``ym[k]``  (passed to ``step``)
2. **Estimate**: ``x̂[k|k] = estimator.step(ym[k], u[k-1], d[k-1])``
3. **Optimise**: ``(U*, X*) = ocp.solve(x̂[k|k], D, model.x_ref, u_prev=u[k-1])``
4. **Apply**:    ``u[k] = U*[0:nu]``
5. **Cache**:    store ``(u[k], d[k])`` as the new ``(u_prev, d_prev)`` for next step

**Usage**:

```python
from mbc.estimation import CDKalmanFilter
from mbc.control import CDOptimalControlProblem, CDMPCController

kf   = CDKalmanFilter(model, n_steps=10)
ocp  = CDOptimalControlProblem(model, N=20, Q=Q_z, R=R_u)
ctrl = CDMPCController(model, estimator=kf, ocp=ocp)

u, U_seq, X_seq = ctrl.step(ym, D)   # D = (N·nd, 1) stacked disturbance forecast
```

**Closed-loop structure**:

```
        ┌──────────────────────────────────────────────────────────┐
        │                    CDMPCController                       │
ym[k] ─┼─► CDKalmanFilter ── x̂[k|k] ── CDOptOCP ── U*[0] ── u[k] ┼─► Plant
        │  (continuous ODE              (ZOH-QP, lifted batch)    │
        │   integration on A, B, E)                               │
        │       ▲                                                 │
        │    cache (u_prev, d_prev) for next step                 │
        └──────────────────────────────────────────────────────────┘
```

Note the split: the *estimator* uses the continuous-time matrices ``A``,
``B``, ``E`` directly via ODE integration; the *OCP* obtains ZOH-discretised
matrices ``(Ad, Bd, Ed)`` via the internal ``_CDModelAdapter`` (computed
from ``A``, ``B``, ``E``, ``dt`` at construction time).  Both operate on
the same ``model`` object.

#### `CDNMPCController` — `mbc.control` *(ControlToolbox §EMPC — *ENMPC Algorithm*)*

Closed-loop continuous-discrete NMPC controller implementing the
ControlToolbox §EMPC algorithm verbatim.  Composes **any** continuous-
discrete state estimator with **any** OCP that exposes
`solve(x0, d_trajectory, u_prev, x_prev, y_prev, p, t0) → (u_opt, cost, info)`
into a receding-horizon feedback controller.

Any combination of:

- **Estimators**: `ContinuousDiscreteEKF`, `ContinuousDiscreteUKF`,
  `ContinuousDiscreteEnKF`, `ContinuousDiscreteParticleFilter`,
  `ContinuousDiscreteDAEEKF`, `DelayedObservationFilter` (wrapping any of the above)
- **OCPs**: `EconomicOptimalControlProblem`, `CDTrackingOptimalControlProblem`

can be composed without any changes to the controller code.

##### ENMPC algorithm at time t_k

Following ControlToolbox §EMPC — *ENMPC Algorithm*:

1. **Measure**   `y^{m,s}_k = h^m(z^s_k, θ^s) + v^s_k(θ^s)`  (passed to `step`)
2. **Estimate**  `z^c_k = κ(z^c_{k−1}, u_{k−1}, d_{k−1}, y^{m,s}_k, θ^c)`
                 (delegated to `estimator.step`)
3. **Optimise** `u_k = λ(z^c_k, θ^c)`  (delegated to `ocp.solve`)
4. **Apply**     return `u_k` to the caller, who advances the plant.

The controller maintains warm-start buffers for the input sequence `U`,
the differential-state trajectory `X`, and the algebraic-state trajectory
`Y` (SDAE only); these are passed back to the OCP at the next iteration to
reduce solver work.

**Closed-loop structure**:

```
           ┌────────────────────────────────────────────────────────────────┐
           │                    CDNMPCController                            │
 y^m[k] ──┤── CD Estimator ── x̂[k] ── Nonlinear OCP ── u_opt[0] ── u[k] ──┼── Plant
           │  (EKF/UKF/EnKF/    │       (Economic OCP                       │
           │   PF/DAE-EKF)      │        — implicit-Euler                   │
           │         ▲          │          direct simultaneous)             │
           │    warm-start ←────┘                                           │
           │       (U, X, Y)                                                │
           └────────────────────────────────────────────────────────────────┘
```

**Required interfaces**:

| Component | Required methods / properties |
|-----------|------------------------------|
| `estimator` | `step(ym, u, d, p, t)` → `(x_hat, P)` *(or `(x_hat, y_hat, P)` for SDAEs)* |
| `ocp` | `solve(x0, d_trajectory, u_prev, x_prev, y_prev, p, t0) → (u_opt, cost, info)`, `N`, `nu` |

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `estimator` | CD estimator | Any CD estimator with `step(ym, u, d, p, t)` |
| `ocp` | OCP | Any OCP with the signature above |

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteEKF
from mbc.control import (
    CDTrackingOptimalControlProblem,
    EconomicOptimalControlProblem,
    CDNMPCController,
)

ekf = ContinuousDiscreteEKF(model, x0, P0, dt=1.0)

# Tracking NMPC (convenience wrapper)
ocp = CDTrackingOptimalControlProblem(
    model, N=20, Q=Q_z, R=R_u,
    z_ref=z_setpoint, u_min=u_lo, u_max=u_hi, dt=1.0,
)
ctrl = CDNMPCController(estimator=ekf, ocp=ocp)
u = ctrl.step(ym, d_trajectory, p=None, t=t_k)

# Economic NMPC (same controller, different OCP)
def lagrange(t, x, y, u, theta):
    return profit_per_unit_input * float(u[0]) * 1.0

eocp = EconomicOptimalControlProblem(model, N=20, lagrange=lagrange, dt=1.0)
ctrl_e = CDNMPCController(estimator=ekf, ocp=eocp)
u = ctrl_e.step(ym, d_trajectory, p=None, t=t_k)

# With a delayed-observation wrapper
from mbc.estimation import DelayedObservationFilter
filt = DelayedObservationFilter(ekf, lag_max=10)
ctrl_d = CDNMPCController(estimator=filt, ocp=ocp)
u = ctrl_d.step(ym, d_trajectory, p=None, t=t_k)
```

---

## Part III — System Identification

### 3.1 Linear Discrete-time Identification

### `ped_neg_log_likelihood` — `mbc.identification.likelihood`

Evaluates the **prediction-error decomposition (PED)** Kalman-filter negative
log-likelihood for a linear discrete-time model parameterised by `θ`.

For a linear state-space model with `Cm = I` (full state observation), the
Kalman filter innovations sequence `{ν_k}` is white and Gaussian under the
true model.  The negative log-likelihood is:

```
−log L(θ) = ½ Σ_{k=2}^{T} [ log|S_k| + ν_kᵀ S_k⁻¹ ν_k ]

where:
    x̂_k⁻  = Ad x̂_{k-1} + Bd u_{k-1} + Ed d_{k-1} + offset(d_{k-1})
    P_k⁻   = Ad P_{k-1} Adᵀ + Qd                    (standard Gd=I form)
    ν_k    = ym_k − x̂_k⁻                            (innovation, Cm=I)
    S_k    = P_k⁻ + Rm                               (innovation covariance, Cm=I)
```

followed by the Joseph-form Kalman update for `P_k`.  The state is bootstrapped
from the first measurement.  Returns the sentinel `1e10` on any numerical
failure (invalid θ, non-positive-definite S, etc.).

**Signature**:

```python
from mbc.identification.likelihood import ped_neg_log_likelihood

neg_ll = ped_neg_log_likelihood(
    model_factory,   # callable: θ → model
    theta,           # (p,) ndarray — parameter vector
    history,         # list of {"y": ndarray, "u": ndarray, "d": ndarray}
    Q,               # (nx,nx) ndarray — process noise covariance
    R,               # (nx,nx) ndarray — measurement noise covariance
)
```

**History format**: each entry `{"y": (nx,) ndarray, "u": (nu,) ndarray, "d": (nd,) ndarray}`
records one time step.  Note: the linear PED uses key `"y"` (full-state measurement, `Cm = I`);
the nonlinear CD variant (`cd_ped_neg_log_likelihood`) uses `"ym"` instead.

### `ped_neg_log_likelihood_gradient` — `mbc.identification.likelihood`

Forward finite-difference gradient `∂(−log L)/∂θ` of the PED log-likelihood.
Step size `h = 1e-5` by default.

```python
from mbc.identification.likelihood import ped_neg_log_likelihood_gradient

grad = ped_neg_log_likelihood_gradient(
    model_factory,   # callable: θ → model
    theta,           # (p,) ndarray — parameter vector
    history,         # list of {"y": ndarray, "u": ndarray, "d": ndarray}
    Q,               # (nx,nx) ndarray — process noise covariance
    R,               # (nx,nx) ndarray — measurement noise covariance
    h=1e-5,          # optional: finite-difference step size
)
# grad : (p,) ndarray — gradient of the negative log-likelihood w.r.t. θ
```

### `ParameterEstimator` — `mbc.identification`

Multi-start optimizer that maximises the PED log-likelihood over the model
parameter vector `θ`.  Wraps `ped_neg_log_likelihood` with box constraints,
optional regularisation, and multiple restarts.

**Algorithm**:

For each restart `r = 0, 1, …, n_restarts−1`:

1. Initialise from `θ_r` (restart 0 uses `theta0`; later restarts add Gaussian
   perturbation `N(0, restart_perturbation²)` or use `perturbation_fn`).
2. Minimise the regularised negative log-likelihood:

```
objective(θ) = −log L(θ|Qd, Rm, history) + regularization_fn(θ)
```

using **Nelder–Mead** (gradient-free, default) or **L-BFGS-B** (gradient-based,
requires `scipy`; activated by `use_gradient=True`).

3. Track the best result across all restarts.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_factory` | `θ → model` | — | Model constructor |
| `theta0` | `(p,) ndarray` | — | Initial parameter guess |
| `bounds` | `list[(lo,hi)]` or `None` | — | Per-parameter box constraints |
| `Q` | `(n,n) ndarray` | — | Process noise covariance |
| `R` | `(n,n) ndarray` | — | Measurement noise covariance |
| `regularization_fn` | `θ → float` or `None` | `None` | Optional regularisation penalty |
| `n_restarts` | `int` | `3` | Number of optimisation restarts |
| `restart_perturbation` | `float` | `0.5` | Std of Gaussian perturbation for restarts |
| `use_gradient` | `bool` | `False` | Use L-BFGS-B; falls back to Nelder–Mead if scipy absent |
| `perturbation_fn` | `callable` or `None` | `None` | Custom restart initialiser |

**Usage**:

```python
from mbc.identification import ParameterEstimator

estimator = ParameterEstimator(
    model_factory=my_factory,
    theta0=np.array([0.95, 0.03]),
    bounds=[(0.5, 1.0), (0.0, 0.5)],
    Q=np.eye(2) * 1e-3,
    R=np.eye(2) * 0.1,
    regularization_fn=None,
    n_restarts=5,
    use_gradient=True,
)
result = estimator.estimate(history)
# result.theta_best         : (p,) ndarray — best parameters
# result.neg_log_likelihood : float — objective at theta_best
# result.converged          : bool
# result.message            : str
```

**Log-likelihood inspection**:

```python
ll = estimator.log_likelihood(history, theta=theta_candidate)  # float or None
```

---

### 3.2 Nonlinear Continuous-Discrete Identification

#### `cd_ped_neg_log_likelihood` — `mbc.identification.likelihood`

Evaluates the **prediction-error decomposition (PED)** negative log-likelihood
for a **nonlinear continuous-discrete stochastic system** using the CD-EKF.

The state is propagated between measurements by integrating the nonlinear drift
ODE and the linearised continuous Riccati ODE (forward Euler with `n_steps`
sub-steps).  At each measurement time the innovation and its covariance are
computed and added to the log-likelihood:

```
−log L(θ) = ½ Σ_k [ log|Sₖ| + νₖᵀ Sₖ⁻¹ νₖ ]

where, at step k:
    x̂_k⁻, P_k⁻  — prior from CD-EKF prediction
    H_k = ∂hm/∂x  evaluated at (x̂_k⁻, u_k, d_k, p, t_k)
    νₖ  = ym_k − hm(x̂_k⁻, u_k, d_k, p, t_k)   innovation
    Sₖ  = H_k P_k⁻ H_kᵀ + Rm                   innovation covariance
```

After computing the likelihood contribution the state is updated with the
Joseph-form Kalman correction.

**History format (nonlinear CD)**:

Each entry in *history* is a ``dict`` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"ym"` | `(nym,) ndarray` | Measurement at time `t_k` |
| `"u"` | `(nu,) ndarray` | Input applied during `[t_k, t_{k+1}]` |
| `"d"` | `(nd,) ndarray` | Disturbance during `[t_k, t_{k+1}]` |
| `"t"` | `float` (optional) | Absolute time `t_k`; defaults to `k * dt` |

**Convention**: entry `k` holds `ym_k` together with `u_k`, `d_k` applied
during `[t_k, t_{k+1}]`.  The inputs from entry `k` drive the prediction to
`t_{k+1}`; the inputs from entry `k+1` enter the measurement function at
`t_{k+1}`.

**Signature**:

```python
from mbc.identification.likelihood import cd_ped_neg_log_likelihood

neg_ll = cd_ped_neg_log_likelihood(
    model_factory,   # callable: θ → ContinuousDiscreteModel
    theta,           # (ntheta,) ndarray — parameter vector
    history,         # list of {"ym": ndarray, "u": ndarray, "d": ndarray}
    x0,              # (nx,) ndarray — initial state estimate
    P0,              # (nx, nx) ndarray — initial state covariance
    dt,              # float — sampling interval
    n_steps=10,      # int — Euler sub-steps per interval
)
```

The `model_factory(θ)` callable must return a
:class:`~mbc.models.ContinuousDiscreteModel` whose `params` property holds
the parameter vector `p` passed to `f`, `sigma`, `hm`, `dfdx`, and `dhmdx`
at each filter step.

Returns `1e10` (sentinel) on any numerical failure.

#### `cd_ped_neg_log_likelihood_gradient` — `mbc.identification.likelihood`

Forward finite-difference gradient `∂(−log L)/∂θ` of the CD-EKF PED
log-likelihood.  Step size `h = 1e-5` by default.

```python
from mbc.identification.likelihood import cd_ped_neg_log_likelihood_gradient

grad = cd_ped_neg_log_likelihood_gradient(
    model_factory, theta, history, x0, P0, dt, n_steps=10, h=1e-5
)
# grad : (ntheta,) ndarray
```

#### `CDParameterEstimator` — `mbc.identification`

Multi-start optimizer that maximises the CD-EKF PED log-likelihood over the
model parameter vector `θ`.

**Algorithm**:

For each restart `r = 0, 1, …, n_restarts−1`:

1. Initialise from `θ_r` (restart 0 uses `theta0`; later restarts add
   Gaussian perturbation `N(0, restart_perturbation²)` or use
   `perturbation_fn`).
2. Minimise the regularised negative log-likelihood:

```
objective(θ) = −log L(θ | x0, P0, dt, history) + regularization_fn(θ)
```

using **Nelder–Mead** (gradient-free, default) or **L-BFGS-B** (gradient-based,
requires scipy; activated by `use_gradient=True`).

3. Track the best result across all restarts.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_factory` | `θ → ContinuousDiscreteModel` | — | Model constructor |
| `theta0` | `(ntheta,) ndarray` | — | Initial parameter guess |
| `bounds` | `list[(lo,hi)]` or `None` | — | Per-parameter box constraints |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial state covariance |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Euler sub-steps per sampling interval |
| `regularization_fn` | `θ → float` or `None` | `None` | Optional regularisation penalty |
| `n_restarts` | `int` | `3` | Number of optimisation restarts |
| `restart_perturbation` | `float` | `0.5` | Std of Gaussian perturbation for restarts |
| `use_gradient` | `bool` | `False` | Use L-BFGS-B; falls back to Nelder–Mead if scipy absent |
| `perturbation_fn` | `callable` or `None` | `None` | Custom restart initialiser |

**Usage** (Monod bioreactor example):

```python
import numpy as np
from mbc.identification import CDParameterEstimator
from mbc.models import ContinuousDiscreteModel

# 1. Define a parameterised model: theta = [log(mu_max), log(K_s)]
class MonodModel(ContinuousDiscreteModel):
    _Y = 0.5
    def __init__(self, mu_max, K_s):
        self._mu_max = mu_max; self._K_s = K_s
    @property
    def nx(self): return 2
    @property
    def nu(self): return 1
    @property
    def nd(self): return 1
    @property
    def nym(self): return 1
    @property
    def nz(self): return 1
    @property
    def nw(self): return 2
    @property
    def Rm(self): return np.array([[0.01]])
    def f(self, x, u, d, p, t):
        S, X = x; FV = u[0]; S_in = d[0]
        mu = self._mu_max * max(S, 0.0) / (self._K_s + max(S, 0.0))
        return np.array([-mu*X/self._Y + (S_in-S)*FV, mu*X - X*FV])
    def sigma(self, x, u, d, p, t): return 0.01 * np.eye(2)
    def hm(self, x, u, d, p, t): return np.array([x[1]])
    def gm(self, x, u, d, p, t): return np.array([x[1]])
    @property
    def params(self): return np.log(np.array([self._mu_max, self._K_s]))

def monod_factory(theta):
    return MonodModel(np.exp(theta[0]), np.exp(theta[1]))

# 2. Build measurement history
# history = [{"ym": ym_k, "u": u_k, "d": d_k, "t": t_k}, ...]

# 3. Construct estimator and run
theta0 = np.array([np.log(0.4), np.log(0.3)])   # initial guess
x0    = np.array([5.0, 0.5])
P0    = np.diag([1.0, 0.5])
dt    = 0.1   # h

estimator = CDParameterEstimator(
    model_factory=monod_factory,
    theta0=theta0,
    bounds=[(-3.0, 1.0), (-3.0, 1.0)],
    x0=x0,
    P0=P0,
    dt=dt,
    n_steps=10,
    n_restarts=5,
)
result = estimator.estimate(history)

# result.theta_best         : (ntheta,) ndarray — best parameters in θ-space
# result.neg_log_likelihood : float — objective at theta_best
# result.converged          : bool
print("mu_max =", np.exp(result.theta_best[0]))
print("K_s    =", np.exp(result.theta_best[1]))
```

**Log-likelihood inspection**:

```python
ll = estimator.log_likelihood(history, theta=theta_candidate)  # float or None
```

---

## Part IV — Realization

Algorithms for constructing minimal state-space models from transfer functions or
input-output data (M.Sc. thesis, Ch. 2–4).

### 4.1 SISO Realization

#### `SISORealization` — `mbc.realization` *(partially implemented — M.Sc. Ch. 2–3)*

Constructs a discrete-time SISO state-space model from a rational transfer
function or sampled impulse response data.

**Realized system**:

```
x[k+1] = A x[k] + B u[k]
y[k]   = C x[k] + D u[k]
```

**Deterministic transfer function** `H(z) = B(z)/A(z)`:

```
H(z) = (b_0 z^n + b_1 z^{n-1} + … + b_n) / (z^n + a_1 z^{n-1} + … + a_n)
```

Two canonical forms are supported:

**Observable canonical form** (default):

```
        [−a_1  1  0  …  0]         [b_1 − b_0 a_1]
        [−a_2  0  1  …  0]         [b_2 − b_0 a_2]
  A  =  [ ⋮        ⋱    ], B  =   [      ⋮       ],  C = [1, 0, …, 0],  D = [b_0]
        [−a_n  0  0  …  0]         [b_n − b_0 a_n]
```

The observable form places the denominator coefficients in the first column of
`A` and the numerator residuals `b_i − b_0 a_i` as the `B` column.

**Controllable canonical form**:

```
        [0  0  …  0  −a_n ]         [b_n − b_0 a_n ]
        [1  0  …  0  −a_{n-1}]       [b_{n-1} − b_0 a_{n-1}]
  A  =  [0  1  …  0    ⋮   ], C  =  [       ⋮       ]ᵀ,  B = eₙ,  D = [b_0]
        [   ⋱         ⋮   ]
        [0  …  1  −a_1    ]
```

**Stochastic transfer function — ARMAX extension** (planned):

For a system driven by both a deterministic input and additive coloured noise,
the ARMAX representation is:

```
A(z) y[k] = B(z) u[k] + C(z) e[k],   e[k] ~ N(0, σ²)
```

where:
- `A(z) = z^n + a_1 z^{n-1} + … + a_n` — common autoregressive denominator
- `B(z) = b_0 z^n + b_1 z^{n-1} + … + b_n` — deterministic input numerator
- `C(z) = z^n + c_1 z^{n-1} + … + c_n` — moving-average noise numerator

The ARMAX model is realised as:

```
x[k+1] = A x[k] + B u[k] + G e[k]
y[k]   = C_out x[k] + D u[k] + e[k]
```

The noise input matrix `G` is computed from the `C`-polynomial coefficients in
**exactly the same way** as `B` is computed from the `B`-polynomial coefficients.
Both share the same denominator `A(z)`, so the canonical state-space construction
applies identically.  In the observable canonical form:

```
G = [c_1 − a_1, c_2 − a_2, …, c_n − a_n]ᵀ
```

which are the residuals between the C- and A-polynomial coefficients — the same
formula as for `B` with `b_i` replaced by `c_i` and `b_0 = 1` (since `C(z)` is
monic).  The output noise `e[k]` in `y[k] = C_out x[k] + D u[k] + e[k]`
corresponds to `C(z)` having a leading coefficient of 1.

This structure means `G` can be passed directly as the `noise_matrix` argument to
`KalmanFilter` or `CDKalmanFilter` for noise-separated filtering (M.Sc. Ch. 5.4).

**Transfer-function normalisation**: `den` must be supplied in **monic** form (leading
coefficient 1.0).  If `den[0] ≠ 1`, divide both `num` and `den` by `den[0]` before
constructing the canonical form.  `num` may have fewer coefficients than `den`; it is
zero-padded on the left so that `len(num) == len(den)` and `b_0 = num_padded[0]`.

**From impulse response** — `from_impulse_response(h, dt, n)`:

Constructs a minimal nth-order model whose impulse response best fits the sampled
sequence `h[0], h[1], …, h[T-1]` (sampled at interval `dt`) using the Ho-Kalman
algorithm restricted to the SISO case.

**SISO Hankel matrix construction** (with `q = len(h) // 2`; require `q ≥ n`):

```
        [h[1],  h[2],  …, h[q]  ]
H_blk = [h[2],  h[3],  …, h[q+1]]   ∈ ℝ^{q × q}
        [ ⋮       ⋮          ⋮  ]
        [h[q], h[q+1], …, h[2q-1]]
```

Note: `h[0]` is excluded from the Hankel matrix (`D = h[0]` for the SISO case).
The rank-`n` truncated SVD, observability/controllability factorisation, and
state-matrix extraction follow the same steps as the MIMO Ho-Kalman algorithm
(§4.2) with `ny = nu = 1`.  Require `len(h) ≥ 2n + 1` to form a well-determined
Hankel matrix.

**Usage**:

```python
from mbc.realization import SISORealization
import numpy as np

# Deterministic TF: H(z) = (0.5 z + 0.3) / (z² − 0.9 z + 0.2)
sys = SISORealization.from_transfer_function(
    num=np.array([0.5, 0.3]),
    den=np.array([1.0, -0.9, 0.2]),
    form="observable",
)

# ARMAX: A(z) y = B(z) u + C(z) e
# (planned; noise_num = C-polynomial coefficients after leading 1)
sys = SISORealization.from_transfer_function(
    num=np.array([0.5, 0.3]),
    den=np.array([1.0, -0.9, 0.2]),
    noise_num=np.array([0.8, 0.1]),   # c_1, c_2 (after monic normalisation)
    form="observable",
)

A, B, C, D = sys.A, sys.B, sys.C, sys.D
G = sys.G   # noise input matrix (None if no noise_num provided)
```

### 4.2 MIMO Realization

#### `MIMORealization` — `mbc.realization` *(stub — M.Sc. Ch. 4)*

Constructs a MIMO discrete-time state-space model from its Markov parameters
(impulse-response matrices) using the Ho–Kalman algorithm.

**Markov parameters**:

```
H[0] = D                  (direct feed-through matrix)
H[k] = C A^{k-1} B       for k = 1, 2, …
```

where `H[k] ∈ ℝⁿʸˣⁿᵘ` are the output response matrices at lag `k`.

**Ho–Kalman algorithm**:

1. Form the block Hankel matrix from `H[1], …, H[2q]` where `q ≥ n`:

```
        [H[1],  H[2],  …, H[q]  ]   ∈ ℝ^{q·ny × q·nu}
H_blk = [H[2],  H[3],  …, H[q+1]]
        [  ⋮      ⋮          ⋮  ]
        [H[q], H[q+1], …, H[2q] ]
```

2. Compute rank-`n` truncated SVD: `H_blk ≈ U_n Σ_n V_nᵀ`

3. Factor into observability and controllability matrices:

```
O_n = U_n Σ_n^{1/2}     (q·ny × n observability matrix)
R_n = Σ_n^{1/2} V_nᵀ   (n × q·nu controllability matrix)
```

4. Extract system matrices:

```
C = O_n[0:ny, :]                    (first ny rows of observability matrix)
B = R_n[:, 0:nu]                    (first nu columns of controllability matrix)
A = O_n[0:(q-1)·ny, :]⁺ O_n[ny:, :] (shift-and-recover from observability)
D = H[0]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `H` | `list[(ny,nu) ndarray]` | Markov parameters `H[0], H[1], …`; minimum length `2n+1` |
| `n` | `int` | Desired model order |

**Minimum H length**: the block Hankel matrix uses `q` block rows and `q` block
columns from `H[1], …, H[2q]`, so `len(H) ≥ 2q + 1`.  The minimum useful choice
is `q = n`, giving `len(H) ≥ 2n + 1`.  Use `q > n` (over-determined Hankel) for
noise-robustness; the rank-`n` truncation then discards sensor-noise contributions.
The Hankel block `H_blk ∈ ℝ^{q·ny × q·nu}` must satisfy `q · ny ≥ n` and
`q · nu ≥ n` for the observability and controllability matrices to have rank `n`.

**Usage**:

```python
from mbc.realization import MIMORealization
import numpy as np

# H[0] = D, H[1] = CB, H[2] = CAB, ...
sys = MIMORealization.from_markov_parameters(H_list, n=4)

A, B, C, D = sys.A, sys.B, sys.C, sys.D
# A : (4, 4), B : (4, nu), C : (ny, 4), D : (ny, nu)
```

---

## Part V — Monte Carlo Simulation

### `MonteCarloSimulation` — `mbc.monte_carlo` *(Ph.D. Ch. 12)*

Closed-loop Monte Carlo framework for assessing controller and estimator
performance under stochastic initial conditions and process noise.

**Trial structure** — each of `N_mc` independent trials proceeds as:

1. Draw initial state: `x₀ⁱ ~ N(x0_mean, x0_cov)`
2. Initialise estimator with `x₀ⁱ` (or skip if `estimator=None`)
3. Compute initial control: `u₀ⁱ = controller.step(x₀ⁱ, D[0:N], …)`
4. For each of `T` measurement intervals `k = 0, 1, …, T-1`:
   - **Simulate**: `x_{k+1}^i = simulator.step(x_k^i, u_k^i, D[k], t_k)` (with SDE noise)
   - **Observe**: `ym_k^i = model.hm(x_{k+1}^i, u_k^i, D[k], p, t_{k+1}) + v_k^i`, `v_k^i ~ N(0, Rm)`
   - **Accumulate cost**: `costs[i] += stage_cost(x_k^i, u_k^i, D[k])`
   - **Estimate**: `x̂_{k+1}^i, _ = estimator.step(ym_k^i, u_k^i, D[k], p, t_{k+1})` (if provided)
   - **Control**: `u_{k+1}^i = controller.step(x̂_{k+1}^i, D[k+1:k+1+N], …)`
5. Record `X^i = [x_0^i, …, x_T^i]`, `Y^i = [y_0^i, …, y_{T-1}^i]`, `U^i = [u_0^i, …, u_{T-1}^i]`

**State timeline**: `X[i, k]` is the true state **before** the k-th control action;
`Y[i, k]` is the observation generated from `x_{k+1}` and reported at time `k`
(i.e. the measurement arrives one step after the state that generated it).

When `estimator=None`, the true state `x_k^i` is fed directly to the controller
(perfect state-information baseline).

**Reproducibility**: trial `i` uses random seed `seed + i`, ensuring independent
but deterministic realisations.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Plant model (for measurement function `hm`) |
| `simulator` | `SDESimulator` or `SDAESimulator` | — | Plant dynamics integrator |
| `controller` | `object` with `.step()` | — | Feedback controller |
| `estimator` | `object` with `.step()` or `None` | `None` | State estimator; `None` = perfect state info |
| `stage_cost` | `(x, u, d) → float` or `None` | `None` | Cost accumulated per step (used for `costs` field); `None` leaves `costs` as zeros |
| `N_mc` | `int` | `100` | Number of Monte Carlo trials |
| `seed` | `int` or `None` | `None` | Base random seed (trial i uses seed+i) |

**Methods**:

```python
from mbc.monte_carlo import MonteCarloSimulation

mc = MonteCarloSimulation(
    model=plant, simulator=sim,
    controller=ctrl, estimator=ekf,
    stage_cost=lambda x, u, d: float(x @ Q_x @ x + u @ R_u @ u),
    N_mc=500, seed=42,
)

result = mc.run(
    x0_mean=np.zeros(nx),
    x0_cov=np.eye(nx) * 0.1,
    D=D_trajectory,     # (T, nd) shared disturbance trajectory
    T=100,              # simulation horizon
)
```

### `MonteCarloResult` — `mbc.monte_carlo`

Dataclass returned by `MonteCarloSimulation.run`:

| Field | Shape | Description |
|-------|-------|-------------|
| `X` | `(N_mc, T+1, nx)` | State trajectories; `X[i, 0] = x₀ⁱ` |
| `Y` | `(N_mc, T, ny)` | Noisy output trajectories |
| `U` | `(N_mc, T, nu)` | Applied input trajectories |
| `costs` | `(N_mc,)` | Cumulative stage cost per trial |

**Analysis example**:

```python
import numpy as np

print(f"Mean cost:    {result.costs.mean():.3f}")
print(f"Std cost:     {result.costs.std():.3f}")
print(f"95th pctile:  {np.percentile(result.costs, 95):.3f}")

# State trajectory statistics
X_mean = result.X.mean(axis=0)    # (T+1, nx) — mean trajectory
X_std  = result.X.std(axis=0)     # (T+1, nx) — trajectory std
```

---

## Installation

```bash
pip install -e .
```

**Core dependencies**: `numpy`, `cvxopt`, `scipy`.

**Optional dependencies**:
- `cyipopt` (`pip install -e ".[ipopt]"`) for IPOPT NLP backend support in nonlinear MPC/OCP.

## Running Tests

```bash
pytest tests/
```
