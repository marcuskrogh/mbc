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

Abstract base class for a linear discrete-time stochastic state-space model:

```
x[k+1] = Ad x[k] + Bd u[k] + Ed d[k] + Gd w[k],   w[k] ~ N(0, Qd)
z[k]   = Cz x[k] + Dz u[k] + Fz d[k]
ym[k]  = Cm x[k] + Dm u[k] + Fm d[k] + v[k],       v[k] ~ N(0, Rm)
```

where `x ∈ ℝⁿˣ` is the state, `u ∈ ℝⁿᵘ` is the control input, `d ∈ ℝⁿᵈ` is a
measured disturbance, `z ∈ ℝⁿᶻ` is the controlled output, and `ym ∈ ℝⁿʸᵐ` is
the measurement.  All system matrices are constant (LTI).

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
| `Qd` | `(nx, nx) ndarray` | Discrete process-noise covariance |
| `Rm` | `(nym, nym) ndarray` | Measurement noise covariance |
| `x` | `list[float]` | Current state (read/write) |
| `x_ref` | `(nx,) ndarray` | State setpoint / reference |
| `u_bounds` | `(ndarray, ndarray)` | Input box `(u_min, u_max)`, each `(nu,)` |

**Concrete members** (overridable — sensible defaults are provided):

| Member | Default | Description |
|--------|---------|-------------|
| `Gd` | `I` (identity) | Noise input matrix Gd ∈ ℝⁿˣˣⁿˣ |
| `Cz` | `Cm` | Controlled output matrix Cz ∈ ℝⁿᶻˣⁿˣ |
| `Dz` | zeros | Controlled output feedthrough Dz ∈ ℝⁿᶻˣⁿᵘ |
| `Fz` | zeros | Controlled output disturbance feedthrough Fz ∈ ℝⁿᶻˣⁿᵈ |
| `Dm` | zeros | Measurement input feedthrough Dm ∈ ℝⁿʸᵐˣⁿᵘ |
| `Fm` | zeros | Measurement disturbance feedthrough Fm ∈ ℝⁿʸᵐˣⁿᵈ |
| `nym` | `Cm.shape[0]` | Measurement dimension (derived) |
| `nz` | `Cz.shape[0]` | Controlled output dimension (derived) |
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

Standard discrete-time Kalman filter with Joseph-stabilised covariance update,
noise-input-matrix support, and missing-observation handling (M.Sc. thesis, Ch. 5).

**Model**: `LinearDiscreteModel` (LTI or LPV).

**Algorithm** — at each measurement time k, given `ym[k]` and `d[k]`:

*Prediction* (time update):

```
x̂⁻[k]  = Ad x̂[k-1] + Bd u[k-1] + Ed d[k-1] + offset(d[k-1])

P⁻[k]  = Ad P[k-1] Adᵀ + Qd             (standard form, Gd = I)
       or Ad P[k-1] Adᵀ + Gd Qd Gdᵀ     (noise-separated form, Gd provided)
```

*Filtering* (measurement update, Joseph stabilised form):

```
ν[k]  = ym[k] − Cm x̂⁻[k]               innovation
S[k]  = Cm P⁻[k] Cmᵀ + Rm              innovation covariance
K[k]  = P⁻[k] Cmᵀ S[k]⁻¹              Kalman gain
x̂[k]  = x̂⁻[k] + K[k] ν[k]             corrected state

IKCm  = I − K[k] Cm
P[k]  = IKCm P⁻[k] IKCmᵀ + K[k] Rm K[k]ᵀ   Joseph form (PSD-preserving)
```

The gain `K` is computed by solving the linear system `S[k] Kᵀ = (P⁻ Cᵀ)ᵀ`
via Cholesky factorisation (`cvxopt.lapack.posv`), which exploits the
positive-definiteness of `S` and avoids forming `S⁻¹` explicitly.

The **Joseph form** `P = IKCm P⁻ IKCmᵀ + K Rm Kᵀ` guarantees that the
posterior covariance remains symmetric positive semi-definite in finite-precision
arithmetic, unlike the conventional form `P = (I−KCm) P⁻` which can accumulate
numerical skew.

**Bootstrap**: on the first call to `update`, before any prediction has been
run, the state is initialised from the measurement via the Moore–Penrose
pseudoinverse: `x̂ = Cm⁺ ym`.  This is computed via `numpy.linalg.lstsq`,
which returns the minimum-norm least-squares solution and handles both the
underdetermined case (`nym < nx`, common when only a subset of states is
measured) and the overdetermined case (`nym ≥ nx`).  For `Cm = I` (full state
observation) this reduces to `x̂ = ym`.

**Missing observations** (M.Sc. Ch. 5.5): the optional `mask` argument of
`update(ym, d, mask)` controls which output channels are used in the measurement
update.  When `mask[i] = False`, output `i` is excluded.  If all entries are
`False` the measurement update is skipped entirely (prediction-only step), which
is the correct treatment for time steps where no sensor reading is available.
The filter continues to propagate the covariance forward using only the
prediction step.

When only a subset of outputs are available, the filter constructs sub-matrices
`Cm_sub`, `Rm_sub`, and `ym_sub` restricted to the active rows before calling
`filter(ym_sub, x_pred, P_pred, Cm_sub)`.  The reduced `Rm_sub` is the
sub-block of `Rm` corresponding to the active output pairs.

**Delayed observations**: measurements with a known reporting delay (e.g.
laboratory assays) are handled by `DelayedObservationFilter` (§1.2), which wraps
any estimator and adds a `delay` argument to `update` / `step`.

**Noise separation** (M.Sc. Ch. 5.4): the standard prediction covariance step
`P⁻ = A P Aᵀ + Q` assumes `Q` is the full state-noise covariance (i.e. the
process noise enters every state channel equally).  When the process noise acts
on only a subset of states via a noise input matrix `G ∈ ℝⁿˣᵍ`, the correct
prediction is `P⁻ = A P Aᵀ + G Q Gᵀ` where `Q ∈ ℝᵍˣᵍ` is the smaller
noise covariance.  Pass `noise_matrix=G` to `__init__` to activate this form.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearDiscreteModel` | — | Plant model |
| `Q` | `(nx,nx) matrix` | `0.01·I` | Process noise covariance (Qd override) |
| `R` | `(nym,nym) matrix` | `0.1·I` | Measurement noise covariance (Rm override) |
| `P0` | `(nx,nx) matrix` | `I` | Initial state error covariance |
| `noise_matrix` | `(nx,g) matrix` or `None` | `None` | Noise input matrix Gd; `None` uses Gd=I |

**Usage**:

```python
from mbc.estimation import KalmanFilter

kf = KalmanFilter(model, Q=Q, R=R, P0=P0, noise_matrix=G)

# At each time step:
x_hat = kf.update(ym, d, mask=None)  # (nx,1) state estimate
kf.record_action(u)                   # store u[k] for next prediction
```

**Public properties**:

| Property | Type | Description |
|----------|------|-------------|
| `x_hat` | `(nx,) ndarray` | Current state estimate x̂[k] (copy) |
| `P` | `(nx,nx) ndarray` | Current covariance P[k] (copy) |
| `last_innovation` | `list[float]` or `None` | Most recent innovation ν = ym − Cm x̂⁻ |

#### `DelayedObservationFilter` — `mbc.estimation`

A transparent wrapper that adds per-channel reporting-delay handling to **any**
state estimator.  Its call interface is identical to the wrapped estimator plus
one optional `delay` argument; it can therefore be substituted into
`MPCController`, `CDMPCController`, or `CDNMPCController` without any change to
the controller code.

**Motivation**: some measurement channels have a fixed or variable reporting
delay — a laboratory analyser returns a result `τ` sampling steps after the
sample was taken, while on-line sensors have `τ = 0`.  Passing all channels
together in a single `update` call (with their respective delays declared)
allows the filter to apply each measurement at the correct point in time.

**Interface** (discrete-time):

```python
x_hat = filt.update(y, d, mask=None, delay=None)   # returns (n,1)
filt.record_action(u)
```

**Interface** (continuous-discrete, wrapping a CD estimator):

```python
x_hat, P = filt.step(y, u, d, t, mask=None, delay=None)
x_hat, P = filt.predict(u, d, t)
x_hat, P = filt.update(y, d, mask=None, delay=None)
```

**`delay` argument**: a `(ny,)` integer `ndarray` where `delay[i]` is the number
of sampling steps by which output channel `i` arrived late.  `delay[i] = 0`
means a current-step observation (no delay).  `None` is equivalent to all zeros
— the filter behaves identically to the unwrapped estimator.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | any estimator | — | Wrapped estimator (KF, EKF, UKF, EnKF, PF, DAE-EKF, …) |
| `lag_max` | `int` | — | Maximum delay in steps that the buffer can accommodate |

**Properties** (`x_hat`, `P`, `last_innovation`): delegated to the wrapped
estimator — the caller sees no difference from a plain estimator.

**Internal algorithm** — at each call to `update(y, d, mask, delay)`:

```
Partition channels by delay:
    immediate = [i for i where delay[i] == 0 (or delay is None)]
    delayed   = [(i, tau) for i where delay[i] = tau > 0]

1. Standard update for immediate channels:
      Apply wrapped_estimator.update(y[immediate], d, mask=immediate_mask)
      Append (x̂, P, u, d, y, mask, t) to the internal buffer (deque, maxlen=lag_max)

2. For each (i, tau) in delayed channels (sorted by tau ascending):
      a. Retrieve buffer entry at position -(tau+1):
             x̂_0, P_0 = buffer[-(tau+1)]["x_hat"], buffer[-(tau+1)]["P"]
      b. Temporarily restore the wrapped estimator to (x̂_0, P_0)
      c. Apply update for channel i only (mask = single-channel, y_i = y[i]):
             x̂_upd, P_upd = wrapped_estimator.update(y_i, d, mask=channel_i_mask)
      d. Re-propagate forward through the buffer from -(tau) to -1:
             for each entry j (oldest delayed entry to most recent):
                 (x̂, P) = wrapped_estimator.predict(entry["u"], entry["d"], ...)
                 if entry had active channels:
                     (x̂, P) = wrapped_estimator.update(entry["y"], entry["d"],
                                                         entry["mask"])
      e. Set wrapped estimator state to the re-propagated (x̂, P)
      f. Update the buffer entries [-(tau) .. -1] with the re-computed posteriors

3. Restore wrapped estimator to the final (x̂, P)
```

If `delay[i] > lag_max`, channel `i` is silently dropped (the required prior has
been evicted from the buffer) and a warning is issued.

**Usage with `MPCController`** (discrete):

```python
from mbc.estimation import KalmanFilter, DelayedObservationFilter
from mbc.control import OptimalControlProblem, MPCController

kf   = KalmanFilter(model, Q=Q, R=R)
filt = DelayedObservationFilter(kf, lag_max=10)   # up to 10-step lab delay

ocp  = OptimalControlProblem(model, N=20, Q=Q_y, R=R_u)
ctrl = MPCController(model, estimator=filt, ocp=ocp)

# Normal on-line sensor (delay=0) + lab result with 5-step delay:
delay = np.array([0, 0, 5])     # 3 output channels; channel 2 is the lab assay
u, U_seq, X_seq = ctrl.step(y, D)   # MPC calls filt.update(y, d) internally
# — when a lab result is ready, set y[2] to the assay value and pass delay;
#   outside the MPC loop, call filt.update(y, d, delay=delay) directly before ctrl.step
```

**Usage with `CDNMPCController`** (continuous-discrete):

```python
from mbc.estimation import ContinuousDiscreteEKF, DelayedObservationFilter
from mbc.control import EconomicOptimalControlProblem, CDNMPCController

ekf  = ContinuousDiscreteEKF(model, x0, P0, dt=1.0)
filt = DelayedObservationFilter(ekf, lag_max=10)

ocp  = EconomicOptimalControlProblem(model, N=20, dt=1.0, lagrange=cost_fn)
ctrl = CDNMPCController(estimator=filt, ocp=ocp)

u = ctrl.step(ym, d, p, t)                          # no lab result this step
u = ctrl.step(ym, d, p, t, delay=np.array([0, 3]))  # channel 1 delayed by 3 steps
```

---

### 1.3 Optimal Control Problems

#### `OptimalControlProblem` — `mbc.control`

Finite-horizon quadratic OCP with hard input and soft output constraints,
solved by a condensed (batch/lifted) QP at each step (M.Sc. thesis, Ch. 6).

**Cost function** over prediction horizon N:

```
J(U) = Σ_{k=0}^{N-1} [ ‖ym[k+1] − r‖²_Q + ‖u[k]‖²_R + ‖Δu[k]‖²_S ]
     + ‖ym[N] − r‖²_P
     + ρ Σ_{k=0}^{N-1} ‖ε[k+1]‖²
```

where:
- `r = Cm x_ref` is the measurement setpoint derived from `model.x_ref`
- `Δu[k] = u[k] − u[k-1]` is the input rate of movement (requires `u_prev`)
- `ε[k] ≥ 0` are slack variables for soft output constraint violations
- `ρ` is the violation penalty weight

**Constraints**:

```
u_min ≤ u[k] ≤ u_max                                (hard input box)
ym[k+1] ≥ Cm x_ref − δ − ε[k+1]                    (soft lower output bound)
ym[k+1] ≤ Cm x_ref + δ + ε[k+1]                    (soft upper output bound)
ε[k+1] ≥ 0                                          (slack non-negativity)
```

The output bounds are centred at the reference `Cm x_ref` with half-width
`δ = y_offset`.  Violations are penalised quadratically via `ρ ‖ε‖²` rather
than infeasible hard constraints, which guarantees the QP is always feasible.

**Batch (lifted) prediction matrices**:

The state trajectory over the horizon is expressed as an affine function of the
input sequence `U = [u[0]; u[1]; …; u[N-1]]` and the disturbance forecast
`D = [d[0]; d[1]; …; d[N-1]]`:

```
X = Ψ x₀ + Γ U + Λ D

where:
  Ψ ∈ ℝᴺⁿˣˣⁿˣ    with Ψ_{k} = Ad^{k+1}
  Γ ∈ ℝᴺⁿˣˣᴺⁿᵘ   with Γ_{k,j} = Ad^{k-j} Bd   (lower-triangular block structure)
  Λ ∈ ℝᴺⁿˣˣᴺⁿᵈ   with Λ_{k,j} = Ad^{k-j} Ed
```

The output predictions are `Ym = C̄m X` where `C̄m = blkdiag(Cm, …, Cm)`.  The cost
and constraints are expressed entirely in terms of `U` and the slack `ε`,
giving the QP decision variable `z = [U; ε]`:

```
min_z  ½ zᵀ H z + fᵀ z
s.t.   G z ≤ h
```

Solved with `cvxopt.solvers.qp`.  The Hessian `H` is block-diagonal:
`H = blkdiag(Cm Γᵀ Q̄ Cm Γ + R̄ + D_diff^T S̄ D_diff, ρ I_{Nl})`.

The model matrices `Ad`, `Bd`, `Ed` are accessed directly from `model.Ad`,
`model.Bd`, `model.Ed` at each `solve` call.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearDiscreteModel` | — | Plant model |
| `N` | `int` | — | Prediction horizon |
| `Q` | `(nym,nym) matrix` | — | Stage output tracking cost |
| `R` | `(nu,nu) matrix` | — | Stage input cost |
| `P` | `(nym,nym) matrix` | `Q` | Terminal output tracking cost |
| `S` | `(nu,nu) matrix` or `None` | `None` | Rate-of-movement cost; `None` disables |
| `rho` | `float` | `1e4` | Soft constraint violation penalty |
| `y_offset` | `float` | `2.0` | Half-width δ of soft output band |

**Usage**:

```python
from mbc.control import OptimalControlProblem

ocp = OptimalControlProblem(model, N=20, Q=Q_y, R=R_u, P=P_terminal, S=S_rate)

# D is the stacked disturbance forecast [d[0]; d[1]; ...; d[N-1]], shape (N*nd, 1)
U_seq, X_seq = ocp.solve(x0=x_hat, D=D_forecast, x_ref=model.x_ref, u_prev=u_prev)
u_current = U_seq[:model.nu]   # first element of the optimal sequence
```

If the QP is infeasible (solver status ≠ "optimal"), `solve` returns zeros and
logs a warning.  In practice the slack variables prevent infeasibility.

---

### 1.4 MPC Controllers

#### `MPCController` — `mbc.control`

Combines a `KalmanFilter` and an `OptimalControlProblem` into a receding-horizon
feedback controller.

**Closed-loop structure**:

```
          ┌──────────────────────────────────────────────┐
          │              MPCController                   │
  y[k] ──┤──► KalmanFilter ──x̂[k]──► OCP ──U*──►u[k]──┼──► Plant ──► y[k+1]
          │    (predict+filter)         (solve QP)       │
          │         ▲                                    │
          │    record_action(u[k])                       │
          └──────────────────────────────────────────────┘
```

**Receding-horizon policy** — at each measurement time k:

1. **Estimate**: `x̂[k] ← estimator.update(ym[k], d[k])`
2. **Optimise**: `(U*, X*) ← ocp.solve(x̂[k], D, model.x_ref, u_prev)`
3. **Apply**: `u[k] = U*[0:nu]` (first element of the optimal sequence)
4. **Record**: `estimator.record_action(u[k])` (stores `u[k]` for next prediction)

Steps 1–4 are performed by `step(y, D)` which returns `(u, U_seq, X_seq)`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `LinearDiscreteModel` | Plant model |
| `estimator` | `KalmanFilter` | State estimator |
| `ocp` | `OptimalControlProblem` | Optimal control problem |

**Usage**:

```python
from mbc.control import MPCController
from mbc.estimation import KalmanFilter

kf  = KalmanFilter(model, Q=Q, R=R)
ocp = OptimalControlProblem(model, N=20, Q=Q_y, R=R_u)
ctrl = MPCController(model, estimator=kf, ocp=ocp)

# At each time step:
u, U_seq, X_seq = ctrl.step(y, D)   # D = stacked disturbance forecast (N*p, 1)
```

---

## Part II — Continuous-Discrete Systems

### 2.1 Models

#### `ContinuousDiscreteModel` — `mbc.models`

Abstract base class for a nonlinear continuous-discrete stochastic SDE system
(Ph.D. thesis, Ch. 5):

```
dx(t)   = f(x, u, d, p, t) dt + sigma(x, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
z(t)    = g(x, u, d, p, t)
ym(tk)  = hm(x, u, d, p, tk) + v(tk),                         v(tk) ~ N(0, Rm)
```

where `x ∈ ℝⁿˣ` is the state, `u ∈ ℝⁿᵘ` is the control input, `d ∈ ℝⁿᵈ` is a
measured disturbance, `p` is a parameter vector, `z ∈ ℝⁿᶻ` is the controlled
output, `ym ∈ ℝⁿʸᵐ` is the measurement, and `dw(t) ~ N(0, I dt)` is standard
Brownian motion.  The instantaneous noise covariance is `sigma sigma^T dt`.

All arrays use `numpy.ndarray`.  This ABC is accepted by all nonlinear estimators
(`ContinuousDiscreteEKF`, `ContinuousDiscreteUKF`, `ContinuousDiscreteEnKF`,
`ContinuousDiscreteParticleFilter`) and by `SDESimulator` and `EconomicOptimalControlProblem`.

**Abstract interface** — subclasses must implement:

| Member | Signature / Type | Description |
|--------|-----------------|-------------|
| `f` | `(x, u, d, p, t) → (nx,) ndarray` | Drift function |
| `sigma` | `(x, u, d, p, t) → (nx, nw) ndarray` | Diffusion matrix |
| `hm` | `(x, u, d, p, t) → (nym,) ndarray` | Measurement function |
| `g` | `(x, u, d, p, t) → (nz,) ndarray` | Controlled output function |
| `Rm` | `(nym, nym) ndarray` | Measurement noise covariance |
| `nx` | `int` | State dimension |
| `nu` | `int` | Input dimension |
| `nd` | `int` | Disturbance dimension |
| `nym` | `int` | Measurement dimension |
| `nz` | `int` | Controlled output dimension |
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

**Notation** (M.Sc. thesis, Ch. 5; Ph.D. thesis, Ch. 5):

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| `nx` | `int` | State dimension |
| `nu` | `int` | Input dimension |
| `nd` | `int` | Disturbance dimension |
| `nym` | `int` | Measurement output dimension (derived: `Cm.shape[0]`) |
| `nz` | `int` | Controlled output dimension (derived: `Cz.shape[0]`) |
| `nw` | `int` | Process-noise dimension (derived: `G.shape[1]`) |
| `A` | `(nx, nx)` | Continuous state matrix |
| `B` | `(nx, nu)` | Continuous input matrix |
| `E` | `(nx, nd)` | Continuous disturbance matrix |
| `G` | `(nx, nw)` | Noise input matrix |
| `Cz` | `(nz, nx)` | Controlled output matrix |
| `Dz` | `(nz, nu)` | Controlled output input feedthrough |
| `Fz` | `(nz, nd)` | Controlled output disturbance feedthrough |
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
| `Cz` | `(nz, nx) ndarray` | Controlled output matrix |
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
| `g(x, u, d, p, t)` | `Cz @ x + Dz @ u + Fz @ d` |
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
    def Cz(self): return np.array([[1.0, 0.0]])    # controlled output: state 0
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
states `y` (Ph.D. thesis, Ch. 6):

```
dx(t)   = f(x, y, u, d, p, t) dt + sigma(x, y, u, d, p, t) dw(t),  dw(t) ~ N(0, I dt)
0       = h(x, y, u, d, p, t)
z(t)    = g(x, y, u, d, p, t)
ym(tk)  = hm(x, y, u, d, p, tk) + v(tk),                            v(tk) ~ N(0, Rm)
```

where `y ∈ ℝⁿʸ` is the algebraic state vector, kept consistent with the
differential state `x` at all times by enforcing `h = 0`.  The `nw` property
is inherited from `ContinuousDiscreteModel` and must be implemented by
concrete subclasses.

**Additional abstract members**:

| Member | Signature | Description |
|--------|-----------|-------------|
| `h` | `(x, y, u, d, p, t) → (ny,)` | Algebraic constraint residual; zero when satisfied |
| `ny` | `int` | Algebraic state dimension |

**Optional analytic Jacobians** for the constraint and cross-terms:

| Member | Signature | Default |
|--------|-----------|---------|
| `dfdx` | `(x, y, u, d, p, t) → (nx, nx)` | Forward FD |
| `dfdy` | `(x, y, u, d, p, t) → (nx, ny)` | Forward FD |
| `dhdx` | `(x, y, u, d, p, t) → (ny, nx)` | Forward FD |
| `dhdy` | `(x, y, u, d, p, t) → (ny, ny)` | Forward FD |
| `dhdu` | `(x, y, u, d, p, t) → (ny, nu)` | Forward FD |
| `dhdd` | `(x, y, u, d, p, t) → (ny, nd)` | Forward FD |
| `dhdp` | `(x, y, u, d, p, t) → (ny, np)` | Forward FD |
| `dhmdy` | `(x, y, u, d, p, t) → (nym, ny)` | Forward FD |

Accepted by `SDAESimulator` and `ContinuousDiscreteDAEEKF`.

---

### 2.2 Simulators

#### `SDESimulator` — `mbc.simulation`

Euler-Maruyama numerical integrator for `ContinuousDiscreteModel` (Ph.D.
thesis, Ch. 5).  Simulates the continuous SDE from `t_k` to `t_{k+1} = t_k + dt`
using `n_steps` sub-steps of size `h = dt / n_steps`.  Inputs `u` and
disturbances `d` are held constant (ZOH) over each measurement interval.

**Explicit-Explicit (EE) scheme** (default):

Both drift and diffusion are evaluated at the beginning of each sub-step:

```
x_{j+1} = x_j + h · f(x_j, u, d, p, t_j) + √h · sigma(x_j, u, d, p, t_j) · w_j

where  w_j ~ N(0, I)  and  t_j = t + j·h
```

This is the standard Euler-Maruyama discretisation.  It is first-order strong
and half-order weak convergent.  Appropriate for non-stiff systems.

**Implicit-Explicit (IE) scheme**:

The drift is evaluated implicitly at `t_{j+1}` while the diffusion remains explicit:

```
x_{j+1} = x_j + h · f(x_{j+1}, u, d, p, t_{j+1}) + √h · sigma(x_j, u, d, p, t_j) · w_j
```

The implicit drift term is resolved by **fixed-point iteration**:

```
noise_term = √h · sigma(x_j, u, d, p, t_j) · w_j    (fixed at start of sub-step)

x^(0) = x_j
x^(ℓ+1) = x_j + h · f(x^(ℓ), u, d, p, t_{j+1}) + noise_term

Converged when  ‖x^(ℓ+1) − x^(ℓ)‖₂ < fp_tol,  or after fp_max_iter iterations.
```

The noise term `√h · sigma(x_j, u, d, p, t_j) · w_j` is fixed during the inner iteration
since it depends on the beginning-of-step state.  The IE scheme has better
stability properties for stiff drift terms and damps spurious oscillations that
the EE scheme can introduce when `h` is not sufficiently small relative to the
system's fastest time constant.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `dt` | `float` | — | Measurement sampling interval |
| `n_steps` | `int` | `10` | Euler-Maruyama sub-steps per interval |
| `scheme` | `"EE"` or `"IE"` | `"EE"` | Integration scheme |
| `seed` | `int` or `None` | `None` | Random seed for reproducibility |
| `fp_tol` | `float` | `1e-8` | IE fixed-point convergence tolerance (ignored for EE) |
| `fp_max_iter` | `int` | `50` | IE maximum fixed-point iterations per sub-step |

**Methods**:

```python
from mbc.simulation import SDESimulator

sim = SDESimulator(model, dt=1.0, n_steps=20, scheme="EE", seed=42)

# Simulate one measurement interval [t, t+dt]
x_next = sim.step(x, u, d, t)                  # returns (nx,) ndarray

# Simulate full horizon of T intervals
# U : (T, nu) ndarray,  D : (T, nd) ndarray
X = sim.simulate(x0, U, D, t0=0.0)             # returns (T+1, nx) ndarray
```

#### `SDAESimulator` — `mbc.simulation`

Euler-Maruyama integrator for `ContinuousDiscreteDAEModel` (Ph.D. thesis, Ch. 6).
Extends `SDESimulator` to maintain the algebraic constraint `h(x, y, u, d, p, t) = 0`
at every sub-step by interleaving Newton iteration for `y` with the Euler step
for `x`.

**Newton iteration for `y`** (used at every sub-step after the Euler update on `x`):

```
y^(0) = y_j   (previous algebraic state)

For ℓ = 0, 1, …, newton_max_iter − 1:

    J_hy ≈ ∂h/∂y  (finite-difference Jacobian, ny × ny)

    where  J_hy[i, k] = (h(x, y + h_fd · e_k, u, d, p, t)[i] − h(x, y, u, d, p, t)[i]) / h_fd
           h_fd = 1e-5,  e_k = k-th standard basis vector ∈ ℝⁿʸ

    y^(ℓ+1) = y^(ℓ) − J_hy⁻¹ h(x, y^(ℓ), u, d, p, t)    (Newton step)

    Converged when  ‖h(x, y^(ℓ+1), u, d, p, t)‖₂ < newton_tol
```

Analytic Jacobians can be provided by implementing `dhdy` on the model
subclass, replacing the finite-difference computation.

**Explicit-Explicit (EE) scheme** (default):

At each sub-step `j`:

```
1. Euler drift update on x (explicit):
      x_trial = x_j + h · f(x_j, y_j, u, d, p, t_j)

2. Add diffusion noise (explicit):
      x_{j+1} = x_trial + √h · sigma(x_j, y_j, u, d, p, t_j) · w_j

3. Newton solve for y at the new x (project onto constraint manifold):
      y_{j+1} = Newton( h(x_{j+1}, y, u, d, p, t_{j+1}) = 0,  y_init = y_j )
```

Drift and diffusion are evaluated at `(x_j, y_j)` (explicitly), then `y` is
projected back onto `h = 0` at the updated `x_{j+1}`.

**Implicit-Explicit (IE) scheme**:

The drift is solved implicitly in `x`, with `y` updated at each inner iteration
to maintain consistency:

```
1. Inner Newton loop on (x, y):
      Outer iterate: x^(ℓ) with y^(ℓ) = Newton(h(x^(ℓ), y, ...) = 0)
      x^(ℓ+1) = x_j + h · f(x^(ℓ), y^(ℓ), u, d, p, t_{j+1})
               + √h · sigma(x_j, y_j, u, d, p, t_j) · w_j   (noise fixed)
      Repeat until ‖x^(ℓ+1) − x^(ℓ)‖ < tol.

2. Final algebraic solve:
      y_{j+1} = Newton( h(x_{j+1}, y, u, d, p, t_{j+1}) = 0,  y_init = y^(last) )
```

At each inner iteration, `y` is updated by Newton so that `h(x^(ℓ), y, ...) = 0`
holds, coupling `x` and `y` implicitly.  This is necessary for index-1 DAEs with
stiff algebraic coupling, where the EE scheme would violate the constraint to
first order.

**Parameters** (extends `SDESimulator`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteDAEModel` | — | SDAE model |
| `newton_tol` | `float` | `1e-10` | Newton solver convergence tolerance |
| `newton_max_iter` | `int` | `50` | Maximum Newton iterations per sub-step |

**Methods**:

```python
from mbc.simulation import SDAESimulator

sim = SDAESimulator(model, dt=1.0, n_steps=20, scheme="EE", newton_tol=1e-10)

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
# ∂f/∂y  (nx × ny),  ∂h/∂x  (ny × nx),  ∂h/∂y  (ny × ny),  ∂hm/∂y  (nym × ny)
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

Kalman filter for a **linear** continuous-discrete stochastic system, implemented
by directly integrating the continuous-time state ODE and matrix Riccati ODE
(Ph.D. thesis, §7.3, specialised to the linear case).  The system matrices
`A`, `B`, `E` are used directly — no ZOH or Van Loan pre-discretisation
is applied inside the filter.

**Model**: `LinearContinuousDiscreteModel`.

**Prediction** — forward-Euler integration of the state and Riccati ODEs over
`[t_{k-1}, t_k]` using `n_steps` sub-steps of size `h = dt / n_steps`:

```
For j = 0, 1, …, n_steps − 1:

    ẋ̂ = A x̂ + B u + E d                        (state ODE, §7.3a)
    Ṗ = A P + P Aᵀ + G Gᵀ                      (Riccati ODE, §7.3b)

    x̂ ← x̂ + h · ẋ̂
    P  ← P  + h · Ṗ

P ← ½(P + Pᵀ)                                  (symmetrise after integration)
```

Inputs `u` and disturbances `d` are the values applied over the previous
sampling interval (zero-order hold).  `G Gᵀ` is pre-computed and cached.

**Filtering** — Joseph-stabilised measurement update (§7.8–7.11):

```
e[k]  = ym[k] − Cm x̂[k|k-1]                   innovation
R_e   = Cm P[k|k-1] Cmᵀ + Rm                  innovation covariance
K     = P[k|k-1] Cmᵀ R_e⁻¹                    Kalman gain
x̂[k]  = x̂[k|k-1] + K e[k]                    corrected state

IKCm  = I − K Cm
P[k]  = IKCm P[k|k-1] IKCmᵀ + K Rm Kᵀ        Joseph form (PSD-preserving)
```

The gain is computed by solving `R_e Kᵀ = Cm P⁻ᵀ` via `cvxopt.lapack.posv`
(Cholesky on the symmetric positive-definite `R_e`).

**Bootstrap**: on the first call to `update`, the state is initialised from the
measurement via the minimum-norm pseudoinverse.  The system `Cm Cmᵀ α = ym` is
solved for `α`, and `x̂ = Cmᵀ α` gives the minimum-norm solution
`x̂ = Cmᵀ (Cm Cmᵀ)⁻¹ ym`.  This reduces to `x̂ = ym` when `Cm = I`.  This form is
correct for full-row-rank `Cm` (the usual case where `nym ≤ nx`) and avoids the
rank deficiency that arises in the normal-equation form when `Cm` has more columns
than rows.

**Missing observations** (M.Sc. Ch. 5.5): see `KalmanFilter` for the identical
masking logic — active-output sub-matrices `Cm_sub`, `Rm_sub`, `ym_sub` are formed
and passed to `filter(...)`.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearContinuousDiscreteModel` | — | Linear CD model |
| `P0` | `(n,n) matrix` or `None` | `I` | Initial state error covariance |
| `n_steps` | `int` | `10` | Forward-Euler sub-steps per sampling interval |

**Usage**:

```python
from mbc.estimation import CDKalmanFilter

kf = CDKalmanFilter(model, P0=None, n_steps=10)

# At each measurement time:
x_hat = kf.update(y, d, mask=None)   # (nx,) ndarray state estimate
kf.record_action(u)                   # store u[k] for next prediction

# Inspect filter state:
kf.x_hat             # (nx,) ndarray state estimate
kf.P                 # (nx,nx) ndarray covariance
kf.last_innovation   # list[float] or None
```

**Internal representation**: state `x̂` and covariance `P` are maintained as
`numpy.ndarray` throughout.

---

#### `ContinuousDiscreteEKF` — `mbc.estimation` *(Ph.D. Ch. 7.1)*

Extended Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Extends the
linear CDKalmanFilter by replacing `A` with the Jacobian
`F(t) = ∂f/∂x|_{x̂(t)}` evaluated along the estimated trajectory.

**Prediction** — forward-Euler integration of the **nonlinear** state ODE and
linearised Riccati ODE:

```
For j = 0, 1, …, n_steps − 1:

    F_j = ∂f/∂x evaluated at (x̂_j, u, d, p, t_j)    (Jacobian, nx × nx)
    G_j = sigma(x̂_j, u, d, p, t_j)                   (diffusion matrix, nx × nw)

    x̂_{j+1} = x̂_j + h · f(x̂_j, u, d, p, t_j)       (nonlinear drift)

    Ṗ_j = F_j P_j + P_j F_jᵀ + G_j G_jᵀ             (linearised Riccati ODE)
    P_{j+1} = P_j + h · Ṗ_j
```

The Jacobian `F = ∂f/∂x` can be provided analytically by the model or computed
by forward finite differences.

**Filtering** — standard EKF linearised measurement update:

```
Hm  = ∂hm/∂x evaluated at (x̂⁻, u, d, p, t)     (measurement Jacobian, nym × nx)
ŷm⁻ = hm(x̂⁻, u, d, p, t)                        (predicted measurement)
e   = ym − ŷm⁻                                   (innovation)
R_e = Hm P⁻ Hmᵀ + Rm                            (innovation covariance)
K   = P⁻ Hmᵀ R_e⁻¹                              (Kalman gain)
x̂   = x̂⁻ + K e                                  (corrected state)
IKHm = I − K Hm
P   = IKHm P⁻ IKHmᵀ + K Rm Kᵀ                   (Joseph form)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial state covariance |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Euler sub-steps per interval |

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteEKF

ekf = ContinuousDiscreteEKF(model, x0, P0, dt=1.0, n_steps=10)

x_hat, P = ekf.step(y, u, d, t, mask=None)   # predict + update
x_hat, P = ekf.predict(u, d, t)              # prediction only
x_hat, P = ekf.update(y, d, mask=None)       # update only
```

---

#### `ContinuousDiscreteUKF` — `mbc.estimation` *(Ph.D. Ch. 7.2)*

Unscented Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Replaces the
Jacobian linearisation of the EKF with a deterministic sigma-point approximation
(unscented transform), which captures the mean and covariance of a nonlinear
transformation to third order for Gaussian distributions.

**Sigma points** — Van der Merwe scaled sigma-point scheme (`2 nx + 1` points):

```
λ = α² (nx + κ) − nx

χ_0    = x̂                           (mean sigma point)
χ_i    = x̂ + L[:, i-1]              for i = 1, …, nx
χ_{nx+i} = x̂ − L[:, i-1]           for i = 1, …, nx

where L = cholesky((nx + λ) · P, lower=True)   ∈ ℝ^{nx × nx}
      L[:, i] denotes the i-th column of L
```

`L` is the **lower** Cholesky factor (i.e. `L @ L.T = (nx + λ) P`), computed via
`scipy.linalg.cholesky((nx + λ) * P, lower=True)`.  If `P` is numerically
near-singular, add a small jitter `ε · I` (e.g. `ε = 1e-8`) before factoring:

```python
try:
    L = scipy.linalg.cholesky((nx + lam) * P, lower=True)
except np.linalg.LinAlgError:
    L = scipy.linalg.cholesky((nx + lam) * P + 1e-8 * np.eye(nx), lower=True)
```

**Weights** for mean and covariance reconstruction:

```
W_m^0 = λ / (nx + λ)
W_c^0 = λ / (nx + λ) + (1 − α² + β)
W_m^i = W_c^i = 1 / (2(nx + λ))   for i = 1, …, 2nx
```

Typical values: `α = 1e-3` (spread), `β = 2` (optimal for Gaussian), `κ = 0`.

**Prediction** — propagate each sigma point through the drift ODE and reconstruct:

```
For each sigma point χ_i (i = 0, …, 2nx):
    For sub-step j = 0, …, n_steps − 1:
        χ_i ← χ_i + h · f(χ_i, u, d, t + j·h)      (pure drift, no noise)

Predicted mean:       x̂⁻ = Σ_i W_m^i χ̃_i
Predicted covariance: P⁻  = Σ_i W_c^i (χ̃_i − x̂⁻)(χ̃_i − x̂⁻)ᵀ + Q_d
```

The discrete diffusion term `Qd` is accumulated across all `n_steps` sub-steps
using the propagated **mean** sigma point `χ_0` (i.e. the mean trajectory) to
evaluate the diffusion matrix:

```
Qd = 0
For sub-step j = 0, …, n_steps − 1:
    x̂_j  = current mean sigma point χ_0 at sub-step j
    G_j   = sigma(x̂_j, u, d, p, t + j·h)         (nx × nw diffusion matrix)
    Qd   += h · G_j · G_jᵀ
```

This evaluates the stochastic integral `∫₀^dt sigma sigma^T dt` with a left-point
Euler rule along the mean trajectory.  Using the mean sigma point (rather than
all `2nx+1` sigma points) for diffusion accumulation is the standard CD-UKF
approximation and avoids `O(nx)` extra model evaluations per sub-step.

**Filtering** — unscented measurement update:

```
Propagate predicted sigma points through the measurement function:
    γ_i = hm(χ̃_i, u, d, p, t)   for i = 0, …, 2nx

Predicted measurement:    ŷm⁻ = Σ_i W_m^i γ_i
Innovation covariance:    S_yy = Σ_i W_c^i (γ_i − ŷm⁻)(γ_i − ŷm⁻)ᵀ + Rm
Cross-covariance:         S_xy = Σ_i W_c^i (χ̃_i − x̂⁻)(γ_i − ŷm⁻)ᵀ

Kalman gain:  K = S_xy S_yy⁻¹
Corrected state:     x̂ = x̂⁻ + K (ym − ŷm⁻)
Corrected covariance: P = P⁻ − K S_yy Kᵀ
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial state covariance |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Euler sub-steps per interval |
| `alpha` | `float` | `1e-3` | Sigma-point spread |
| `beta` | `float` | `2.0` | Distribution parameter (2 = optimal for Gaussian) |
| `kappa` | `float` | `0.0` | Secondary spread parameter |

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteUKF

ukf = ContinuousDiscreteUKF(model, x0, P0, dt=1.0, alpha=1e-3, beta=2.0)
x_hat, P = ukf.step(y, u, d, t, mask=None)
```

---

#### `ContinuousDiscreteEnKF` — `mbc.estimation` *(Ph.D. Ch. 7.3)*

Ensemble Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Maintains an
ensemble of `N` particles to approximate the state distribution without requiring
Jacobian computations.  The ensemble mean and sample covariance replace the
analytical Gaussian approximation used by the EKF and UKF.

**Initialisation**: draw `N` particles from `N(x0, P0)`:

```
X[:,i] ~ N(x0, P0)   for i = 1, …, N
```

**Prediction** — propagate each particle independently via Euler-Maruyama (EE):

```
For each particle i = 1, …, N and sub-step j = 0, …, n_steps−1:

    X_{j+1}[:,i] = X_j[:,i] + h · f(X_j[:,i], u, d, p, t_j)
                 + √h · sigma(X_j[:,i], u, d, p, t_j) · w_{j,i}

where  w_{j,i} ~ N(0, I)  independently per particle and sub-step.
```

After propagation:

```
x̂⁻ = (1/N) Σ_i X[:,i]                           (ensemble mean)
A  = X − x̂⁻ 1ᵀ                                   (anomaly matrix, nx × N)
P⁻ = (1/(N−1)) A Aᵀ                               (ensemble sample covariance)
```

**Filtering** — perturbed-observations ensemble update:

```
Perturb observations:  ym_i = ym + v_i,   v_i ~ N(0, Rm),   i = 1, …, N
                       (freshly drawn at each update call, independent of prediction noise)

Predicted obs for each particle:
    Ŷ[:,i] = hm(X⁻[:,i], u, d, p, t)

Innovation matrix:  E = Ym − Ŷ   where Ym[:,i] = ym_i

Cross-covariance:   P_xy = (1/(N−1)) A_x Ŷ_anom^T
Innovation cov:     P_yy = (1/(N−1)) Ŷ_anom Ŷ_anom^T + Rm

where A_x = X⁻ − x̂⁻ 1ᵀ  and  Ŷ_anom = Ŷ − (mean of Ŷ) 1ᵀ

Kalman gain:  K = P_xy P_yy⁻¹

Update each particle:  X[:,i] ← X⁻[:,i] + K (ym_i − Ŷ[:,i])

x̂ = (1/N) Σ_i X[:,i]                             (updated ensemble mean)
P  = (1/(N−1)) (X − x̂ 1ᵀ)(X − x̂ 1ᵀ)ᵀ           (updated sample covariance)
```

The perturbed-observations scheme ensures that the ensemble covariance satisfies
the correct Kalman update equation in expectation over the random realisations.

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

#### `ContinuousDiscreteParticleFilter` — `mbc.estimation` *(Ph.D. Ch. 7.4)*

Sequential Monte Carlo (particle filter) for a nonlinear `ContinuousDiscreteModel`.
Represents the posterior `p(x[k] | y[1:k])` as a weighted particle cloud.  Unlike
the EnKF, no Gaussian approximation is made; the filter is asymptotically exact
as `N → ∞`.

**Initialisation**: draw `N` particles from `N(x0, P0)` with equal weights:

```
X[:,i] ~ N(x0, P0),   w_i = 1/N   for i = 1, …, N
```

**Prediction** — identical to `ContinuousDiscreteEnKF`: each particle is
propagated independently through the nonlinear SDE via Euler-Maruyama.  Weights
are unchanged during prediction (the transition density acts as the proposal).

**Filtering** — importance weight update (numerically stable log-domain):

```
For each particle i = 1, …, N:

    ŷm_i = hm(X⁻[:,i], u, d, p, t)                (predicted measurement)
    r_i = ym − ŷm_i                                (residual)
    log_ℓ_i = −½ rᵢᵀ Rm⁻¹ r_i − ½ log|2π Rm|    (log Gaussian likelihood)
    log_w_i ← log_w_i + log_ℓ_i                   (log-weight update)
```

To avoid underflow, normalise weights using the **log-sum-exp** trick:

```
log_w_max = max_i(log_w_i)
log_Z     = log_w_max + log( Σ_i exp(log_w_i − log_w_max) )   (log normaliser)
w_i       = exp(log_w_i − log_Z)                               (normalised weights)
```

```
Effective sample size:  N_eff = 1 / Σ_i w_i²      (w_i already normalised)
```

**Initialisation**: set `log_w_i = −log(N)` (i.e. uniform weights in log domain).

When `mask` is provided, only active output channels enter the log-likelihood:
`r_i = ym_sub − ŷm_i[active]`, `Rm_sub = Rm[np.ix_(active, active)]`.

**Systematic resampling** — triggered when `N_eff < resample_threshold · N`:

```
1. Compute cumulative weight CDF: C[0] = 0, C[i] = C[i-1] + w_i
2. Draw u_0 ~ Uniform(0, 1/N)
3. For each new particle j = 1, …, N:
      t_j = u_0 + (j−1)/N
      select particle i such that C[i−1] < t_j ≤ C[i]
4. Replace ensemble with selected particles; reset w_i = 1/N
```

Systematic resampling has `O(N)` cost and minimal variance among resampling
schemes, making it the standard choice for particle filters.

**Weighted statistics**:

```
x̂ = Σ_i w_i X[:,i]                               (weighted mean)
P  = Σ_i w_i (X[:,i] − x̂)(X[:,i] − x̂)ᵀ          (weighted covariance)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `x0` | `(nx,) ndarray` | — | Initial state estimate |
| `P0` | `(nx,nx) ndarray` | — | Initial covariance |
| `dt` | `float` | — | Sampling interval |
| `N` | `int` | `500` | Number of particles |
| `n_steps` | `int` | `10` | Euler-Maruyama sub-steps per interval |
| `resample_threshold` | `float` | `0.5` | `N_eff / N` threshold for resampling |
| `seed` | `int` or `None` | `None` | Random seed |

**Public properties**: `x_hat`, `P`, `particles` `(nx, N)`, `weights` `(N,)`.

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteParticleFilter

pf = ContinuousDiscreteParticleFilter(model, x0, P0, dt=1.0, N=1000, seed=0)
x_hat, P = pf.step(y, u, d, t, mask=None)
```

---

#### `ContinuousDiscreteDAEEKF` — `mbc.estimation` *(stub — Ph.D. Ch. 8)*

Extended Kalman Filter for `ContinuousDiscreteDAEModel`.  Extends the CD-EKF to
handle systems where the state is partially constrained by `h(x, y, u, d, p, t) = 0`.
The covariance propagation uses an **effective Jacobian** that accounts for the
implicit dependence of `y` on `x` via the constraint.

**Effective Jacobian** (implicit function theorem):

If `h(x, y, ...) = 0` is satisfied, differentiating with respect to `x` gives:

```
∂h/∂x + (∂h/∂y)(∂y/∂x) = 0   ⟹   ∂y/∂x = −(∂h/∂y)⁻¹ (∂h/∂x)
```

The effective drift Jacobian for the Riccati ODE is:

```
F_eff = ∂f/∂x + (∂f/∂y)(∂y/∂x)
      = ∂f/∂x − (∂f/∂y)(∂h/∂y)⁻¹ (∂h/∂x)
```

**Four Jacobians required** (all computed by forward FD with `h_fd = 1e-5` if not
supplied analytically):

| Jacobian | Shape | Formula |
|----------|-------|---------|
| `∂f/∂x` | `(nx, nx)` | perturb `x` in `f(x, y, u, d, p, t)` |
| `∂f/∂y` | `(nx, ny)` | perturb `y` in `f(x, y, u, d, p, t)` |
| `∂h/∂x` | `(ny, nx)` | perturb `x` in `h(x, y, u, d, p, t)` |
| `∂h/∂y` | `(ny, ny)` | perturb `y` in `h(x, y, u, d, p, t)` |

If `hm` depends on `y`, also compute `∂hm/∂y` (shape `nym × ny`) and use the
extended observation Jacobian `Hm_eff = ∂hm/∂x − (∂hm/∂y)(∂h/∂y)⁻¹ (∂h/∂x)`.

`(∂h/∂y)⁻¹` is solved via `np.linalg.solve(J_hy, rhs)` rather than explicit
inversion, following the same pattern as the Newton solver in `SDAESimulator`.

**Prediction** — interleaved Euler/Newton:

```
For j = 0, 1, …, n_steps − 1:

    1. Compute F_eff at (x̂_j, y_j, u, d, p, t_j)
    2. Euler update on x̂:
          x̂_{j+1} = x̂_j + h · f(x̂_j, y_j, u, d, p, t_j)
    3. Newton solve for y at updated x:
          y_{j+1} = Newton( h(x̂_{j+1}, y, u, d, p, t_{j+1}) = 0,  y_init = y_j )
    4. Riccati update:
          G_j    = sigma(x̂_j, y_j, u, d, p, t_j)
          P_{j+1} = P_j + h · (F_eff P_j + P_j F_effᵀ + G_j G_jᵀ)
```

**Filtering**: identical to the CD-EKF, using the extended measurement Jacobian
`Hm = ∂hm/∂x + (∂hm/∂y)(∂y/∂x)` if `hm` depends on `y`.

**Parameters** (extends `ContinuousDiscreteEKF`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteDAEModel` | — | SDAE model |
| `x0` | `(nx,) ndarray` | — | Initial differential state estimate |
| `y0` | `(ny,) ndarray` | — | Initial algebraic state (must satisfy `h = 0`) |
| `P0` | `(nx,nx) ndarray` | — | Initial covariance (differential states only) |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Sub-steps per interval |
| `newton_tol` | `float` | `1e-10` | Newton convergence tolerance |
| `newton_max_iter` | `int` | `50` | Max Newton iterations per sub-step |

**Properties**: `x_hat`, `y_hat`, `P`.

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteDAEEKF

ekf = ContinuousDiscreteDAEEKF(model, x0, y0, P0, dt=1.0)
x_hat, y_hat, P = ekf.step(ym, u, d, p, t, mask=None)
x_hat, y_hat, P = ekf.predict(u, d, p, t)
x_hat, y_hat, P = ekf.update(ym, u, d, p, t, mask=None)
```

---

### 2.4 Optimal Control Problems

#### `CDOptimalControlProblem` — `mbc.control`

Receding-horizon QP for a `LinearContinuousDiscreteModel`.  A typed thin wrapper
around `OptimalControlProblem` that accepts a continuous-discrete model.
Internally wraps the model in a `_CDModelAdapter` that computes ZOH-discretised
matrices `(Ad, Bd, Ed)` from the continuous-time model matrices `(A, B, E)` at
construction time and exposes them as numpy arrays for the QP solver.

The cost function, constraints, batch-form prediction matrices, and QP solver are
identical to `OptimalControlProblem` (see §1.3).  The only difference is the
model type, which provides continuous-time matrices `(A, B, E)` via the
`LinearContinuousDiscreteModel` interface rather than inheriting from
`LinearDiscreteModel`.

**Parameters**: identical to `OptimalControlProblem` with `model` of type
`LinearContinuousDiscreteModel`.

**Usage**:

```python
from mbc.control import CDOptimalControlProblem

ocp = CDOptimalControlProblem(model, N=20, Q=Q_y, R=R_u, P=P_terminal)
U_seq, X_seq = ocp.solve(x0=x_hat, D=D_forecast, x_ref=model.x_ref, u_prev=u_prev)
```

---

#### `CDTrackingOptimalControlProblem` — `mbc.control`

Nonlinear tracking OCP for any `ContinuousDiscreteModel` (NLP formulation,
Ph.D. Ch. 9 — tracking variant).  Unlike `CDOptimalControlProblem` (which is
restricted to linear models solved as a QP), this class handles arbitrary
nonlinear dynamics by solving the open-loop NLP at each sampling time using
`scipy.optimize.minimize` (SLSQP by default).

**Cost function** over prediction horizon N:

```
J = Σ_{k=0}^{N-1} [
        ‖z[k+1] − z_ref‖²_Q
      + ‖u[k]‖²_R
      + ‖Δu[k]‖²_S
      + c_uᵀ u[k]
      + ρ_x (‖max(0, x[k+1] − x_max)‖² + ‖max(0, x_min − x[k+1])‖²)
      + ρ_z (‖max(0, z[k+1] − z_max)‖² + ‖max(0, z_min − z[k+1])‖²)
    ]
  + ‖z[N] − z_ref‖²_P
```

where:
- `z[k] = g(x[k], u[k], d[k], p, t_k)` is the controlled output
- `Δu[k] = u[k] − u[k−1]` is the input rate of movement (ROM)
- `c_u` is a linear input penalty vector
- `ρ_x`, `ρ_z` are soft constraint penalty weights
- `P` defaults to `Q` if not supplied

**Constraints**:

```
x[k+1] = f̄(x[k], u[k], d[k])   (mean dynamics, n_steps Euler steps)
u_min  ≤ u[k] ≤ u_max            (hard input box)
Δu_min ≤ u[k] − u[k−1] ≤ Δu_max (hard input ROM box)
```

Soft state (`x_min`/`x_max`) and output (`z_min`/`z_max`) constraints are encoded
as quadratic penalties in the objective rather than as hard constraints, ensuring
the NLP is always feasible.

**Prediction model**: explicit Euler integration of the mean drift `f` over each
sampling interval:

```
x̂_{k+1} ≈ x̂_k + Σ_{j=0}^{n_steps-1} h · f(x̂_j, u_k, d_k, p, t_k + j·h)
         where  h = dt / n_steps
```

**NLP decision variable**: the flattened input sequence
`u_flat ∈ ℝ^{N·nu}`, reshaped as `u_flat.reshape(N, nu)` inside the objective.

**Warm-starting**: the previous optimal sequence `u_prev` (shape `(N, nu)`) is
shifted by one step (rows 1..N-1, then last row repeated) to produce the initial
guess.  Pass `u_prev=None` on the first call (zero initialisation).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model (with `g` for output) |
| `N` | `int` | — | Prediction horizon |
| `Q` | `(nz, nz) ndarray` | — | Stage output tracking cost |
| `R` | `(nu, nu) ndarray` | — | Stage input cost |
| `P` | `(nz, nz) ndarray` or `None` | `Q` | Terminal output tracking cost |
| `S` | `(nu, nu) ndarray` or `None` | `None` | Input ROM quadratic cost; `None` disables |
| `c_u` | `(nu,) ndarray` or `None` | `None` | Linear input penalty; `None` disables |
| `z_ref` | `(nz,) ndarray` or `None` | zeros | Output reference / setpoint |
| `u_min` | `(nu,) ndarray` or `None` | `None` | Hard lower input bound |
| `u_max` | `(nu,) ndarray` or `None` | `None` | Hard upper input bound |
| `du_min` | `(nu,) ndarray` or `None` | `None` | Hard lower ROM bound |
| `du_max` | `(nu,) ndarray` or `None` | `None` | Hard upper ROM bound |
| `x_min` | `(nx,) ndarray` or `None` | `None` | Soft lower state bound |
| `x_max` | `(nx,) ndarray` or `None` | `None` | Soft upper state bound |
| `rho_x` | `float` | `1e4` | Soft state penalty weight |
| `z_min` | `(nz,) ndarray` or `None` | `None` | Soft lower output bound |
| `z_max` | `(nz,) ndarray` or `None` | `None` | Soft upper output bound |
| `rho_z` | `float` | `1e4` | Soft output penalty weight |
| `n_steps` | `int` | `10` | Euler sub-steps per interval |
| `solver` | `str` | `"SLSQP"` | NLP solver for `scipy.optimize.minimize` |
| `solver_options` | `dict` or `None` | `None` | Options forwarded to the solver |
| `dt` | `float` or `None` | `model.dt` or `1.0` | Sampling interval |

**Public properties**:

| Property | Type | Description |
|----------|------|-------------|
| `N` | `int` | Prediction horizon |
| `nu` | `int` | Input dimension |

**Methods**:

```python
from mbc.control import CDTrackingOptimalControlProblem

ocp = CDTrackingOptimalControlProblem(
    model, N=20, Q=Q_z, R=R_u,
    S=S_rom,                              # ROM penalty
    c_u=np.zeros(model.nu),               # linear input penalty
    z_ref=z_setpoint,                     # output reference
    u_min=u_lo, u_max=u_hi,              # hard input box
    du_min=du_lo, du_max=du_hi,          # hard ROM bounds
    x_min=x_lo, x_max=x_hi, rho_x=1e4,  # soft state constraint
    z_min=z_lo, z_max=z_hi, rho_z=1e4,  # soft output constraint
    dt=1.0, n_steps=10,
)

# Full sequence solve
u_opt, cost = ocp.solve(x0=x_hat, d_trajectory=D, u_prev=u_seq_prev, p=p, t0=t)
# u_opt : (N, nu) ndarray
# cost  : float

# First action only (receding horizon)
u0 = ocp.step(x_hat, D, u_seq_prev, p=p, t0=t)   # returns (nu,) ndarray
```

**Compatibility**: the `solve` / `step` interface is identical to
`EconomicOptimalControlProblem`, so both can be used as the `ocp` argument of
`CDNMPCController` interchangeably.

---

#### `EconomicOptimalControlProblem` — `mbc.control` *(Ph.D. Ch. 9)*

Economic Optimal Control Problem (OCP) for a `ContinuousDiscreteModel`.  Unlike
the tracking OCP, this OCP minimises an arbitrary economic objective comprising
a Lagrange (stage) term `l_e(x, u, d)` and a Mayer (terminal) term `V_f(x_N)`.

This class is an **OCP solver only** — it takes a state estimate `x̂` as input and
returns an optimal input sequence.  To form a complete closed-loop controller,
embed it in a `CDNMPCController` (§2.5) together with a continuous-discrete state
estimator.

**Problem formulation**:

```
min_{u_0, …, u_{N-1}}  J = Σ_{k=0}^{N-1} l_e(x_k, u_k, d_k) + V_f(x_N)
                           + Σ_{k=1}^{N} [ρ_x‖max(0, x_k − x_max)‖²
                                         + ρ_x‖max(0, x_min − x_k)‖²
                                         + ρ_z‖max(0, z_k − z_max)‖²
                                         + ρ_z‖max(0, z_min − z_k)‖²]

subject to:
    x_{k+1} = f̄(x_k, u_k, d_k)       (mean dynamics, Euler integration)
    u_min  ≤ u[k] ≤ u_max              (hard input box)
    Δu_min ≤ u[k] − u[k−1] ≤ Δu_max   (hard input ROM box)
```

Optional ROM and linear input penalties (`S`, `c_u`) can also be included; for
more involved economic problems, embed these directly in the `lagrange` function.

**NLP decision variable layout**: the optimisation variable is the flattened
input sequence `u_flat ∈ ℝ^{N · nu}`, stored in row-major order:

```
u_flat = [u_0[0], …, u_0[nu-1], u_1[0], …, u_1[nu-1], …, u_{N-1}[nu-1]]
```

Reshaped as `u_flat.reshape(N, nu)` inside the objective and constraint functions.

The NLP is solved with `scipy.optimize.minimize` using SLSQP by default.
**Warm-starting**: the previous optimal sequence `u_prev` (shape `(N, nu)`) is
shifted by one step and flattened to produce the initial guess.  If no `u_prev`
is provided (first call), the initial guess is zeros.

**Note on `dt`**: `ContinuousDiscreteModel` does not define `dt` as part of its
abstract interface (unlike `LinearContinuousDiscreteModel`).
`EconomicOptimalControlProblem` therefore accepts `dt` as an explicit constructor
parameter; if omitted it falls back to `model.dt` (if available) or `1.0`.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `N` | `int` | — | Prediction horizon |
| `lagrange` | `(x, u, d) → float` or `None` | `None` | Economic stage cost `l_e` (Lagrange term) |
| `mayer` | `(x) → float` or `None` | `None` | Terminal cost `V_f` (Mayer term) |
| `u_min` | `(nu,) ndarray` or `None` | `None` | Hard lower input bound |
| `u_max` | `(nu,) ndarray` or `None` | `None` | Hard upper input bound |
| `du_min` | `(nu,) ndarray` or `None` | `None` | Hard lower ROM bound |
| `du_max` | `(nu,) ndarray` or `None` | `None` | Hard upper ROM bound |
| `S` | `(nu, nu) ndarray` or `None` | `None` | Input ROM quadratic cost |
| `c_u` | `(nu,) ndarray` or `None` | `None` | Linear input penalty |
| `x_min` | `(nx,) ndarray` or `None` | `None` | Soft lower state bound |
| `x_max` | `(nx,) ndarray` or `None` | `None` | Soft upper state bound |
| `rho_x` | `float` | `1e4` | Soft state penalty weight |
| `z_min` | `(nz,) ndarray` or `None` | `None` | Soft lower output bound |
| `z_max` | `(nz,) ndarray` or `None` | `None` | Soft upper output bound |
| `rho_z` | `float` | `1e4` | Soft output penalty weight |
| `constraints` | `list[dict]` or `None` | `None` | Additional scipy-format constraints |
| `n_steps` | `int` | `10` | Euler sub-steps per interval for prediction |
| `solver` | `str` | `"SLSQP"` | NLP solver name for `scipy.optimize.minimize` |
| `solver_options` | `dict` or `None` | `None` | Options forwarded to the solver |
| `dt` | `float` or `None` | `model.dt` or `1.0` | Sampling interval |

**Public properties**:

| Property | Type | Description |
|----------|------|-------------|
| `N` | `int` | Prediction horizon |
| `nu` | `int` | Input dimension |

**Methods**:

```python
from mbc.control import EconomicOptimalControlProblem

def energy_cost(x, u, d):
    return float(u @ u)   # quadratic energy proxy

def terminal(x):
    return float(x @ x) * 0.1   # terminal state penalty

# Construct the OCP with Lagrange + Mayer formulation
ocp = EconomicOptimalControlProblem(
    model, N=20, dt=1.0,
    lagrange=energy_cost,          # Lagrange (stage cost)
    mayer=terminal,                # Mayer (terminal cost)
    u_min=u_lo, u_max=u_hi,       # hard input box
    du_min=du_lo, du_max=du_hi,   # hard ROM bounds
    x_min=x_lo, x_max=x_hi,       # soft state constraints
    z_min=z_lo, z_max=z_hi,       # soft output constraints
    n_steps=10,
)

# Full sequence solve
u_opt, cost = ocp.solve(x0=x_hat, d_trajectory=d_fcast, u_prev=u_seq_prev, p=p, t0=t)
# u_opt : (N, nu) ndarray — full optimal input sequence
# cost  : float — optimal economic objective value

# First action only (receding horizon)
u0 = ocp.step(x_hat, d_fcast, u_seq_prev, p=p, t0=t)

# To close the loop, pair with an estimator via CDNMPCController (§2.5)
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

Combines a `CDKalmanFilter` (estimator) and a `CDOptimalControlProblem` (OCP) into
a closed-loop receding-horizon controller for a linear continuous-discrete system.

**Receding-horizon policy** — at each measurement time k:

1. **Estimate**: `x̂[k] ← estimator.update(ym[k], d[k])`
2. **Optimise**: `(U*, X*) ← ocp.solve(x̂[k], D, model.x_ref, u_prev)`
3. **Apply**: `u[k] = U*[0:nu]`
4. **Record**: `estimator.record_action(u[k])`

**Usage**:

```python
from mbc.estimation import CDKalmanFilter
from mbc.control import CDOptimalControlProblem, CDMPCController

kf   = CDKalmanFilter(model, n_steps=10)
ocp  = CDOptimalControlProblem(model, N=20, Q=Q_y, R=R_u)
ctrl = CDMPCController(model, estimator=kf, ocp=ocp)

u, U_seq, X_seq = ctrl.step(y, D)   # D = (N*p, 1) stacked disturbance forecast
```

**Closed-loop structure**:

```
           ┌────────────────────────────────────────────────────┐
           │              CDMPCController                        │
  y[k] ───┤──► CDKalmanFilter ──x̂[k]──► CDOptOCP ──U*──►u[k]──┼──► Plant
           │    (ODE integration)         (ZOH + QP)            │
           │         ▲                                          │
           │    record_action(u[k])                             │
           └────────────────────────────────────────────────────┘
```

Note the split: the *estimator* uses the continuous-time matrices `A`, `B`,
`E` directly via ODE integration; the *OCP* obtains ZOH-discretised matrices
`(Ad, Bd, Ed)` via the internal `_CDModelAdapter` (computed from `A`, `B`, `E`,
`dt` at construction time).  Both operate on the same `model` object.

#### `CDNMPCController` — `mbc.control` *(Ph.D. Ch. 9)*

Generic closed-loop CD-NMPC controller.  Composes **any** continuous-discrete
state estimator with **any** OCP that exposes
`solve(x0, d_trajectory, u_prev, p, t0)` into a receding-horizon feedback
controller.

The controller is fully agnostic to the estimator and OCP types.  Any combination
of:

- **Estimators**: `ContinuousDiscreteEKF`, `ContinuousDiscreteUKF`,
  `ContinuousDiscreteEnKF`, `ContinuousDiscreteParticleFilter`,
  `ContinuousDiscreteDAEEKF`, `DelayedObservationFilter` (wrapping any of the above)
- **OCPs**: `CDTrackingOptimalControlProblem`, `EconomicOptimalControlProblem`

can be composed without any changes to the controller code.

**Closed-loop structure**:

```
           ┌──────────────────────────────────────────────────────────┐
           │                    CDNMPCController                       │
  y[k] ───┤──► CD Estimator ──x̂[k]──► Nonlinear OCP ──u_opt──►u[k]──┼──► Plant
           │  (EKF/UKF/EnKF/    │       (CDTrackingOCP                │
           │   PF/DAE-EKF)      │        or EconomicOCP)              │
           │         ▲          │         SLSQP NLP solver            │
           │    warm-start ←────┘                                     │
           └──────────────────────────────────────────────────────────┘
```

**Receding-horizon policy** — at each measurement time k:

1. **Estimate**: `x̂[k] ← estimator.step(ym[k], u[k−1], d[k], p, t_k)` → `(x̂, P)`
2. **Optimise**: `u_opt ← ocp.step(x̂[k], d_trajectory, u_seq_prev, p, t_k)` → `(nu,)`
3. **Apply**: `u[k] = u_opt` (first element of the optimal sequence)
4. **Store**: warm-start buffer shifted for next call

**Required interfaces**:

| Component | Required methods / properties |
|-----------|------------------------------|
| `estimator` | `step(ym, u, d, p, t) → (x_hat, P)` |
| `ocp` | `solve(x0, d_trajectory, u_prev, p, t0) → (u_opt, cost)`, `N`, `nu` |

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `estimator` | CD estimator | Any CD estimator with `step(ym, u, d, p, t)` |
| `ocp` | OCP | Any OCP with `solve`, `N`, `nu` |

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteEKF
from mbc.control import CDTrackingOptimalControlProblem, CDNMPCController

ekf = ContinuousDiscreteEKF(model, x0, P0, dt=1.0)

# Tracking NMPC
ocp = CDTrackingOptimalControlProblem(
    model, N=20, Q=Q_z, R=R_u,
    z_ref=z_setpoint,
    u_min=u_lo, u_max=u_hi,
    dt=1.0,
)
ctrl = CDNMPCController(estimator=ekf, ocp=ocp)
u = ctrl.step(ym, d_trajectory, p=None, t=t_k)

# Economic NMPC (same controller, different OCP)
from mbc.control import EconomicOptimalControlProblem

eocp = EconomicOptimalControlProblem(model, N=20, lagrange=profit_fn, dt=1.0)
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
    history,         # list of {"ym": ndarray, "u": ndarray, "d": ndarray}
    Q,               # (nx,nx) ndarray — process noise covariance
    R,               # (nx,nx) ndarray — measurement noise covariance
)
```

**History format**: each entry `{"ym": (nx,) ndarray, "u": (nu,) ndarray, "d": (nd,) ndarray}`
records one time step.

### `ped_neg_log_likelihood_gradient` — `mbc.identification.likelihood`

Forward finite-difference gradient `∂(−log L)/∂θ` of the PED log-likelihood.
Step size `h = 1e-5` by default.

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
    def g(self, x, u, d, p, t): return np.array([x[1]])
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

**Core dependencies**: `numpy`, `cvxopt`.

**Optional dependencies**: `scipy` (required for `EconomicOptimalControlProblem`, `ParameterEstimator`
with `use_gradient=True`, and future NMPC stubs).

## Running Tests

```bash
pytest tests/
```
