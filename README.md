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

Matrix types: `cvxopt.matrix` for all interfaces to the QP solver; `numpy.ndarray`
for continuous-time computations and nonlinear functions. The boundary between the
two is clearly indicated in each interface.

---

## Contents

- [Part I — Discrete-Discrete Systems](#part-i--discrete-discrete-systems)
  - [1.1 Models](#11-models)
  - [1.2 State Estimators](#12-state-estimators)
  - [1.3 Optimal Control Problems](#13-optimal-control-problems)
  - [1.4 MPC Controllers](#14-mpc-controllers)
- [Part II — Continuous-Discrete Systems](#part-ii--continuous-discrete-systems)
  - [2.1 Models](#21-models)
  - [2.2 Simulators](#22-simulators)
  - [2.3 State Estimators](#23-state-estimators)
  - [2.4 Optimal Control Problems](#24-optimal-control-problems)
  - [2.5 MPC Controllers](#25-mpc-controllers)
- [Part III — System Identification](#part-iii--system-identification)
- [Part IV — Realization](#part-iv--realization)
- [Part V — Monte Carlo Simulation](#part-v--monte-carlo-simulation)
- [Installation](#installation)

---

## Part I — Discrete-Discrete Systems

### 1.1 Models

#### `LinearDiscreteModel` — `mbc.models`

Abstract base class for a linear discrete-time stochastic state-space model:

```
x[k+1] = A(d[k]) x[k] + B(d[k]) u[k] + E(d[k]) d[k] + offset(d[k])
y[k]   = C x[k]
```

where `x ∈ ℝⁿ` is the state, `u ∈ ℝᵐ` is the control input, `d ∈ ℝᵖ` is a
measured disturbance, and `y ∈ ℝˡ` is the observed output.  The output matrix
`C` is time-invariant.

The system matrices `A`, `B`, `E` may depend on `d` to model
**Linear Parameter-Varying (LPV)** systems.  LTI implementations simply ignore
the `d` argument and return constant matrices.  LPV implementations use `d` as a
scheduling variable — for example, when a heat-pump COP or actuator gain varies
with an exogenous signal such as outdoor temperature.

**Abstract interface** — subclasses must implement:

| Member | Type | Description |
|--------|------|-------------|
| `n_x` | `int` | State dimension n |
| `n_u` | `int` | Input dimension m |
| `n_d` | `int` | Disturbance dimension p |
| `C` | `(l, n) matrix` | Output matrix (cvxopt, time-invariant) |
| `x` | `list[float]` | Current state (read/write) |
| `x_ref` | `(n, 1) matrix` | State setpoint / reference |
| `u_bounds` | `(matrix, matrix)` | Input box `(u_min, u_max)`, each `(m, 1)` |
| `discretize(d)` | `→ (A, B, E)` | ZOH-discretised matrices at operating point `d` |

**Concrete members** (overridable):

- `predict_offset(d_np) → (n,) ndarray` — additive offset `offset(d)` in the
  prediction model; default returns zeros.  Override to model known constant
  heat gains or biases not captured by `E d`.
- `params → (p,) ndarray` — flat parameter vector for system identification;
  default returns empty array.
- `with_params(theta) → LinearDiscreteModel` — construct a new model instance
  from parameter vector `θ`; default raises `NotImplementedError`.
- `discretize_jacobian(d, h=1e-5) → (dA, dB, dE)` — forward finite-difference
  Jacobians `∂A_d/∂θ_i`, `∂B_d/∂θ_i`, `∂E_d/∂θ_i` via `with_params`.

**LTI example**:

```python
from cvxopt import matrix
from mbc.models import LinearDiscreteModel

class ThermalRoom(LinearDiscreteModel):
    @property
    def n_x(self): return 2
    @property
    def n_u(self): return 1
    @property
    def n_d(self): return 1
    @property
    def C(self): return matrix([[1.0, 0.0]])
    @property
    def x(self): return [20.0, 15.0]
    @x.setter
    def x(self, val): ...
    @property
    def x_ref(self): return matrix([21.0, 0.0])
    @property
    def u_bounds(self):
        return matrix([0.0]), matrix([1.0])
    def discretize(self, d):
        A = matrix([[0.95, 0.0], [0.0, 1.0]])
        B = matrix([[0.03], [0.0]])
        E = matrix([[0.02], [0.0]])
        return A, B, E
```

**LPV example** — the same interface, but `discretize(d)` returns
matrices that depend on `d[0]`:

```python
def discretize(self, d):
    cop = cop_curve(float(d[0]))      # heat-pump COP varies with outdoor temp
    B = matrix([[0.03 * cop], [0.0]])
    return self._A, B, self._E
```

---

### 1.2 State Estimators

#### `KalmanFilter` — `mbc.estimation`

Standard discrete-time Kalman filter with Joseph-stabilised covariance update,
noise-input-matrix support, and missing-observation handling (M.Sc. thesis, Ch. 5).

**Model**: `LinearDiscreteModel` (LTI or LPV).

**Algorithm** — at each measurement time k, given `y[k]` and `d[k]`:

*Prediction* (time update):

```
A_d, B_d, E_d = model.discretize(d[k-1])

x̂⁻[k]  = A_d x̂[k-1] + B_d u[k-1] + E_d d[k-1] + offset(d[k-1])

P⁻[k]  = A_d P[k-1] A_dᵀ + Q             (standard form, G = I)
       or A_d P[k-1] A_dᵀ + G Q Gᵀ        (noise-separated form, G provided)
```

The model is re-discretised at the previous disturbance `d[k-1]` — for LTI
models this is a no-op; for LPV models this applies the correct scheduling.

*Filtering* (measurement update, Joseph stabilised form):

```
ν[k]  = y[k] − C x̂⁻[k]                   innovation
S[k]  = C P⁻[k] Cᵀ + R                    innovation covariance
K[k]  = P⁻[k] Cᵀ S[k]⁻¹                  Kalman gain
x̂[k]  = x̂⁻[k] + K[k] ν[k]               corrected state

IKC   = I − K[k] C
P[k]  = IKC P⁻[k] IKCᵀ + K[k] R K[k]ᵀ   Joseph form (PSD-preserving)
```

The gain `K` is computed by solving the linear system `S[k] Kᵀ = (P⁻ Cᵀ)ᵀ`
via Cholesky factorisation (`cvxopt.lapack.posv`), which exploits the
positive-definiteness of `S` and avoids forming `S⁻¹` explicitly.

The **Joseph form** `P = IKC P⁻ IKCᵀ + K R Kᵀ` guarantees that the
posterior covariance remains symmetric positive semi-definite in finite-precision
arithmetic, unlike the conventional form `P = (I−KC) P⁻` which can accumulate
numerical skew.

**Bootstrap**: on the first call to `update`, before any prediction has been
run, the state is initialised from the measurement via the Moore–Penrose
pseudoinverse.  When `C` has full column rank (the standard case where `l ≤ n`)
this gives the least-squares solution `x̂ = (CᵀC)⁻¹ Cᵀ y`, obtained by solving
`CᵀC α = Cᵀy` and setting `x̂ = α`.  For `C = I` (full state observation) this
reduces to `x̂ = y`.

**Missing observations** (M.Sc. Ch. 5.5): the optional `mask` argument of
`update(y, d, mask)` controls which output channels are used in the measurement
update.  When `mask[i] = False`, output `i` is excluded.  If all entries are
`False` the measurement update is skipped entirely (prediction-only step), which
is the correct treatment for time steps where no sensor reading is available.
The filter continues to propagate the covariance forward using only the
prediction step.

When only a subset of outputs are available, the filter constructs sub-matrices
`C_sub`, `R_sub`, and `y_sub` restricted to the active rows before calling
`filter(y_sub, x_pred, P_pred, C_sub)`.  The reduced `R_sub` is the
sub-block of `R` corresponding to the active output pairs.

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
| `Q` | `(n,n) matrix` | `0.01·I` | Process noise covariance |
| `R` | `(l,l) matrix` | `0.1·I` | Measurement noise covariance |
| `P0` | `(n,n) matrix` | `I` | Initial state error covariance |
| `noise_matrix` | `(n,g) matrix` or `None` | `None` | Noise input matrix G; `None` uses G=I |

**Usage**:

```python
from mbc.estimation import KalmanFilter

kf = KalmanFilter(model, Q=Q, R=R, P0=P0, noise_matrix=G)

# At each time step:
x_hat = kf.update(y, d, mask=None)   # (n,1) state estimate
kf.record_action(u)                   # store u[k] for next prediction
```

**Public properties**:

| Property | Type | Description |
|----------|------|-------------|
| `x_hat` | `(n,1) matrix` | Current state estimate x̂[k] (copy) |
| `P` | `(n,n) matrix` | Current covariance P[k] (copy) |
| `last_innovation` | `list[float]` or `None` | Most recent innovation ν = y − Cx̂⁻ |

---

### 1.3 Optimal Control Problems

#### `OptimalControlProblem` — `mbc.control`

Finite-horizon quadratic OCP with hard input and soft output constraints,
solved by a condensed (batch/lifted) QP at each step (M.Sc. thesis, Ch. 6).

**Cost function** over prediction horizon N:

```
J(U) = Σ_{k=0}^{N-1} [ ‖y[k+1] − r‖²_Q + ‖u[k]‖²_R + ‖Δu[k]‖²_S ]
     + ‖y[N] − r‖²_P
     + ρ Σ_{k=0}^{N-1} ‖ε[k+1]‖²
```

where:
- `r = C x_ref` is the output setpoint derived from `model.x_ref`
- `Δu[k] = u[k] − u[k-1]` is the input rate of movement (requires `u_prev`)
- `ε[k] ≥ 0` are slack variables for soft output constraint violations
- `ρ` is the violation penalty weight

**Constraints**:

```
u_min ≤ u[k] ≤ u_max                                (hard input box)
y[k+1] ≥ C x_ref − δ − ε[k+1]                       (soft lower output bound)
y[k+1] ≤ C x_ref + δ + ε[k+1]                       (soft upper output bound)
ε[k+1] ≥ 0                                           (slack non-negativity)
```

The output bounds are centred at the reference `C x_ref` with half-width
`δ = y_offset`.  Violations are penalised quadratically via `ρ ‖ε‖²` rather
than infeasible hard constraints, which guarantees the QP is always feasible.

**Batch (lifted) prediction matrices**:

The state trajectory over the horizon is expressed as an affine function of the
input sequence `U = [u[0]; u[1]; …; u[N-1]]` and the disturbance forecast
`D = [d[0]; d[1]; …; d[N-1]]`:

```
X = Ψ x₀ + Γ U + Λ D

where:
  Ψ ∈ ℝᴺⁿˣⁿ    with Ψ_{k} = A^{k+1}
  Γ ∈ ℝᴺⁿˣᴺᵐ   with Γ_{k,j} = A^{k-j} B   (lower-triangular block structure)
  Λ ∈ ℝᴺⁿˣᴺᵖ   with Λ_{k,j} = A^{k-j} E
```

The output predictions are `Y = C̄ X` where `C̄ = blkdiag(C, …, C)`.  The cost
and constraints are expressed entirely in terms of `U` and the slack `ε`,
giving the QP decision variable `z = [U; ε]`:

```
min_z  ½ zᵀ H z + fᵀ z
s.t.   G z ≤ h
```

Solved with `cvxopt.solvers.qp`.  The Hessian `H` is block-diagonal:
`H = blkdiag(CΓᵀ Q̄ CΓ + R̄ + D_diff^T S̄ D_diff, ρ I_{Nl})`.

The model is re-discretised at `d[0]` (first element of the disturbance
forecast) at each `solve` call.  LPV models thus use the current operating-point
matrices; LTI models return the same matrices regardless of `d`.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `LinearDiscreteModel` | — | Plant model |
| `N` | `int` | — | Prediction horizon |
| `Q` | `(l,l) matrix` | — | Stage output tracking cost |
| `R` | `(m,m) matrix` | — | Stage input cost |
| `P` | `(l,l) matrix` | `Q` | Terminal output tracking cost |
| `S` | `(m,m) matrix` or `None` | `None` | Rate-of-movement cost; `None` disables |
| `rho` | `float` | `1e4` | Soft constraint violation penalty |
| `y_offset` | `float` | `2.0` | Half-width δ of soft output band |

**Usage**:

```python
from mbc.control import OptimalControlProblem

ocp = OptimalControlProblem(model, N=20, Q=Q_y, R=R_u, P=P_terminal, S=S_rate)

# D is the stacked disturbance forecast [d[0]; d[1]; ...; d[N-1]], shape (N*p, 1)
U_seq, X_seq = ocp.solve(x0=x_hat, D=D_forecast, x_ref=model.x_ref, u_prev=u_prev)
u_current = U_seq[:model.n_u]   # first element of the optimal sequence
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

1. **Estimate**: `x̂[k] ← estimator.update(y[k], d[k])`
2. **Optimise**: `(U*, X*) ← ocp.solve(x̂[k], D, model.x_ref, u_prev)`
3. **Apply**: `u[k] = U*[0:m]` (first element of the optimal sequence)
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

#### `LinearContinuousDiscreteModel` — `mbc.models`

Abstract base class for a linear continuous-discrete stochastic system.  The
state evolves continuously according to the Itô SDE

```
dx = (A_c x[t] + B_c u[t] + E_c d[t]) dt + G dw[t],   w[t] ~ N(0, Q_c)
```

with observations collected at discrete measurement times t_k:

```
y[k] = C x[k] + v[k],   v[k] ~ N(0, R)
```

Inputs `u` and disturbances `d` are held constant (zero-order hold) over each
sampling interval `[t_k, t_{k+1}]`.

**Notation** (M.Sc. thesis, Ch. 5; Ph.D. thesis, Ch. 5):

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| `n` | — | State dimension |
| `m` | — | Input dimension |
| `p` | — | Disturbance dimension |
| `l` | — | Output dimension |
| `q` | — | Process-noise dimension |
| `A_c` | `(n,n)` | Continuous state matrix |
| `B_c` | `(n,m)` | Continuous input matrix |
| `E_c` | `(n,p)` | Continuous disturbance matrix |
| `G` | `(n,q)` | Noise input matrix |
| `Q_c` | `(q,q)` | Continuous process-noise covariance |
| `C` | `(l,n)` | Output matrix (time-invariant) |
| `R` | `(l,l)` | Measurement noise covariance |
| `dt` | `float` | Sampling interval |

**Abstract interface** — subclasses must implement:

| Member | Type | Description |
|--------|------|-------------|
| `n_x`, `n_u`, `n_d` | `int` | Dimensions |
| `A_c`, `B_c`, `E_c`, `G`, `Q_c` | `(·,·) ndarray` | Continuous-time matrices |
| `C`, `R` | `(·,·) matrix` | Output/noise matrices (cvxopt) |
| `dt` | `float` | Sampling interval |
| `x` | `list[float]` | Current state (read/write) |
| `x_ref` | `(n,1) matrix` | State reference |
| `u_bounds` | `(matrix, matrix)` | Input box `(u_min, u_max)` |

**Concrete utility methods** (provided by the ABC):

*ZOH discretisation* — `discretize(d) → (A_d, B_d, E_d)`:

Uses the augmented-matrix method (no matrix inverse required):

```
expm([[A_c, B_c, E_c],     =  [[A_d, B_d, E_d],
      [ 0,   0,   0 ],          [ 0,  I_m,  0 ],
      [ 0,   0,   0 ]] * dt)    [ 0,   0,  I_p]]
```

where `expm` is the matrix exponential computed via eigendecomposition.

*Discrete noise covariance* — `discretize_noise() → Q_d`:

The exact discrete process-noise covariance via the Van Loan (1978) method:

```
Q_d = ∫₀^{dt} expm(A_c τ) G Q_c Gᵀ expm(A_c τ)ᵀ dτ
```

Computed using the `2n × 2n` augmented matrix:

```
M = [[-A_c,   G Q_c Gᵀ],   * dt
     [  0,    A_cᵀ     ]]

expm(M) = [[expm(-A_c dt),  expm(-A_c dt) Q_d],
           [      0,        expm( A_c dt)     ]]

⟹  Q_d = A_d · expm(M)[:n, n:]
```

These utility methods are used for analysis and for initialising discrete filters
from continuous model parameters.  They are **not** called internally by
`CDKalmanFilter`, which integrates the ODEs directly.  They **are** called
internally by `CDOptimalControlProblem`, which uses ZOH-discretised matrices
for the QP.

**Example**:

```python
import numpy as np
from cvxopt import matrix
from mbc.models import LinearContinuousDiscreteModel

class CSTRLinear(LinearContinuousDiscreteModel):
    @property
    def n_x(self): return 2
    @property
    def n_u(self): return 1
    @property
    def n_d(self): return 1
    @property
    def A_c(self): return np.array([[-1.0, 0.5], [0.0, -2.0]])
    @property
    def B_c(self): return np.array([[1.0], [0.0]])
    @property
    def E_c(self): return np.array([[0.0], [1.0]])
    @property
    def G(self):   return np.eye(2)
    @property
    def Q_c(self): return np.diag([1e-4, 1e-4])
    @property
    def C(self):   return matrix([[1.0, 0.0]])
    @property
    def R(self):   return matrix([[0.05]])
    @property
    def dt(self):  return 60.0   # 1-minute sampling
    @property
    def x(self):   return [0.0, 0.0]
    @x.setter
    def x(self, v): ...
    @property
    def x_ref(self): return matrix([1.0, 0.0])
    @property
    def u_bounds(self): return matrix([0.0]), matrix([2.0])
```

#### `ContinuousDiscreteModel` — `mbc.models`

Abstract base class for a nonlinear continuous-discrete stochastic SDE system
(Ph.D. thesis, Ch. 5):

```
dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c)
y[k] = h(x[k], d[k]) + v[k],                 v[k] ~ N(0, R)
```

**Abstract interface** — subclasses must implement:

| Member | Signature | Description |
|--------|-----------|-------------|
| `f` | `(x, u, d, t) → (nx,)` | Drift function |
| `g` | `(x, u, d, t) → (nx, nw)` | Diffusion matrix |
| `h` | `(x, d) → (ny,)` | Observation function |
| `Q_c` | `(nw, nw) ndarray` | Continuous process-noise covariance |
| `R` | `(ny, ny) ndarray` | Measurement noise covariance |
| `nx`, `nu`, `nd`, `ny` | `int` | Dimensions |

All arrays use `numpy.ndarray`.  This ABC is accepted by all nonlinear estimators
(`ContinuousDiscreteEKF`, `ContinuousDiscreteUKF`, `ContinuousDiscreteEnKF`,
`ContinuousDiscreteParticleFilter`) and by `SDESimulator` and `EconomicNMPC`.

#### `ContinuousDiscreteDAEModel` — `mbc.models`

Extends `ContinuousDiscreteModel` with an algebraic constraint (Ph.D. thesis, Ch. 6):

```
dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
 0 = l(x, z, u, d, t)
y[k] = h(x[k], z[k], d[k]) + v[k]
```

where `z ∈ ℝⁿᶻ` is the algebraic state vector, kept consistent with the
differential state `x` at all times by enforcing `l = 0`.

**Additional abstract members**:

| Member | Signature | Description |
|--------|-----------|-------------|
| `l` | `(x, z, u, d, t) → (nz,)` | Constraint residual; zero when satisfied |
| `nz` | `int` | Algebraic state dimension |

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
x_{j+1} = x_j + h · f(x_j, u, d, t_j) + √h · g(x_j, u, d, t_j) · w_j

where  w_j ~ N(0, Q_c)  and  t_j = t + j·h
```

This is the standard Euler-Maruyama discretisation.  It is first-order strong
and half-order weak convergent.  Appropriate for non-stiff systems.

**Implicit-Explicit (IE) scheme**:

The drift is evaluated implicitly at `t_{j+1}` while the diffusion remains explicit:

```
x_{j+1} = x_j + h · f(x_{j+1}, u, d, t_{j+1}) + √h · g(x_j, u, d, t_j) · w_j
```

The implicit drift term is resolved by **fixed-point iteration**:

```
x^(0) = x_j
x^(ℓ+1) = x_j + h · f(x^(ℓ), u, d, t_{j+1}) + noise_term   (until convergence)
```

The noise term `√h · g(x_j, u, d, t_j) · w_j` is fixed during the inner iteration
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
Extends `SDESimulator` to maintain the algebraic constraint `l(x, z, u, d, t) = 0`
at every sub-step by interleaving Newton iteration for `z` with the Euler step
for `x`.  Finite-difference Jacobians `∂l/∂z` are used by default; analytic
Jacobians can be provided by overriding the relevant method.

**Explicit-Explicit (EE) scheme** (default):

At each sub-step `j`:

```
1. Euler drift update on x (explicit):
      x_trial = x_j + h · f(x_j, z_j, u, d, t_j)

2. Add diffusion noise (explicit):
      x_{j+1} = x_trial + √h · g(x_j, z_j, u, d, t_j) · w_j

3. Newton solve for z at the new x (project onto constraint manifold):
      z_{j+1} = Newton( l(x_{j+1}, z, u, d, t_{j+1}) = 0,  z_init = z_j )
```

Drift and diffusion are evaluated at `(x_j, z_j)` (explicitly), then `z` is
projected back onto `l = 0` at the updated `x_{j+1}`.

**Implicit-Explicit (IE) scheme**:

The drift is solved implicitly in `x`, with `z` updated at each inner iteration
to maintain consistency:

```
1. Inner Newton loop on (x, z):
      Outer iterate: x^(ℓ) with z^(ℓ) = Newton(l(x^(ℓ), z, ...) = 0)
      x^(ℓ+1) = x_j + h · f(x^(ℓ), z^(ℓ), u, d, t_{j+1})
               + √h · g(x_j, z_j, u, d, t_j) · w_j   (noise fixed)
      Repeat until ‖x^(ℓ+1) − x^(ℓ)‖ < tol.

2. Final algebraic solve:
      z_{j+1} = Newton( l(x_{j+1}, z, u, d, t_{j+1}) = 0,  z_init = z^(last) )
```

At each inner iteration, `z` is updated by Newton so that `l(x^(ℓ), z, ...) = 0`
holds, coupling `x` and `z` implicitly.  This is necessary for index-1 DAEs with
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

x_next, z_next = sim.step(x, z, u, d, t)        # one measurement interval
X, Z = sim.simulate(x0, z0, U, D, t0=0.0)        # (T+1,nx) and (T+1,nz)
```

---

### 2.3 State Estimators

All continuous-discrete estimators share the interface:

- `predict(u, d, t) → (x_pred, P_pred)` — propagate from `t` to `t + dt`
- `update(y, d, mask=None) → (x_hat, P)` — measurement update at time `t`
- `step(y, u, d, t, mask=None) → (x_hat, P)` — combined predict + update

---

#### `CDKalmanFilter` — `mbc.estimation`

Kalman filter for a **linear** continuous-discrete stochastic system, implemented
by directly integrating the continuous-time state ODE and matrix Riccati ODE
(Ph.D. thesis, §7.3, specialised to the linear case).  The system matrices
`A_c`, `B_c`, `E_c` are used directly — no ZOH or Van Loan pre-discretisation
is applied inside the filter.

**Model**: `LinearContinuousDiscreteModel`.

**Prediction** — forward-Euler integration of the state and Riccati ODEs over
`[t_{k-1}, t_k]` using `n_steps` sub-steps of size `h = dt / n_steps`:

```
For j = 0, 1, …, n_steps − 1:

    ẋ̂ = A_c x̂ + B_c u + E_c d                    (state ODE, §7.3a)
    Ṗ = A_c P + P A_cᵀ + G Q_c Gᵀ               (Riccati ODE, §7.3b)

    x̂ ← x̂ + h · ẋ̂
    P  ← P  + h · Ṗ

P ← ½(P + Pᵀ)                                    (symmetrise after integration)
```

Inputs `u` and disturbances `d` are the values applied over the previous
sampling interval (zero-order hold).  `G Q_c Gᵀ` is pre-computed and cached.

**Filtering** — Joseph-stabilised measurement update (§7.8–7.11):

```
e[k]  = y[k] − C x̂[k|k-1]                       innovation
R_e   = C P[k|k-1] Cᵀ + R                        innovation covariance
K     = P[k|k-1] Cᵀ R_e⁻¹                        Kalman gain
x̂[k]  = x̂[k|k-1] + K e[k]                       corrected state

IKC   = I − K C
P[k]  = IKC P[k|k-1] IKCᵀ + K R Kᵀ              Joseph form (PSD-preserving)
```

The gain is computed by solving `R_e Kᵀ = C P⁻ᵀ` via `cvxopt.lapack.posv`
(Cholesky on the symmetric positive-definite `R_e`).

**Bootstrap**: on the first call to `update`, the state is initialised from the
measurement via the minimum-norm pseudoinverse.  The system `C C^T α = y` is
solved for `α`, and `x̂ = C^T α` gives the minimum-norm solution
`x̂ = C^T (C C^T)⁻¹ y`.  This reduces to `x̂ = y` when `C = I`.  This form is
correct for full-row-rank `C` (the usual case where `l ≤ n`) and avoids the
rank deficiency that arises in the normal-equation form when `C` has more columns
than rows.

**Missing observations** (M.Sc. Ch. 5.5): see `KalmanFilter` for the identical
masking logic — active-output sub-matrices `C_sub`, `R_sub`, `y_sub` are formed
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
x_hat = kf.update(y, d, mask=None)   # (n,1) cvxopt column
kf.record_action(u)                   # store u[k] for next prediction

# Inspect filter state:
kf.x_hat             # (n,1) state estimate
kf.P                 # (n,n) covariance
kf.last_innovation   # list[float] or None
```

**Internal representation**: state `x̂` and covariance `P` are maintained as
`numpy.ndarray` internally for efficient ODE integration, and converted to
`cvxopt.matrix` on output.

---

#### `ContinuousDiscreteEKF` — `mbc.estimation` *(stub — Ph.D. Ch. 7.1)*

Extended Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Extends the
linear CDKalmanFilter by replacing `A_c` with the Jacobian
`F(t) = ∂f/∂x|_{x̂(t)}` evaluated along the estimated trajectory.

**Prediction** — forward-Euler integration of the **nonlinear** state ODE and
linearised Riccati ODE:

```
For j = 0, 1, …, n_steps − 1:

    F_j = ∂f/∂x evaluated at (x̂_j, u, d, t_j)    (Jacobian, nx × nx)
    G_j = g(x̂_j, u, d, t_j)                        (diffusion matrix, nx × nw)

    x̂_{j+1} = x̂_j + h · f(x̂_j, u, d, t_j)        (nonlinear drift)

    Ṗ_j = F_j P_j + P_j F_jᵀ + G_j Q_c G_jᵀ       (linearised Riccati ODE)
    P_{j+1} = P_j + h · Ṗ_j
```

The Jacobian `F = ∂f/∂x` can be provided analytically by the model or computed
by forward finite differences.

**Filtering** — standard EKF linearised measurement update:

```
H   = ∂h/∂x evaluated at (x̂⁻, d)               (observation Jacobian, ny × nx)
ŷ⁻  = h(x̂⁻, d)                                  (predicted observation)
e   = y − ŷ⁻                                     (innovation)
R_e = H P⁻ Hᵀ + R                               (innovation covariance)
K   = P⁻ Hᵀ R_e⁻¹                               (Kalman gain)
x̂   = x̂⁻ + K e                                  (corrected state)
IKH = I − K H
P   = IKH P⁻ IKHᵀ + K R Kᵀ                      (Joseph form)
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

#### `ContinuousDiscreteUKF` — `mbc.estimation` *(stub — Ph.D. Ch. 7.2)*

Unscented Kalman Filter for a nonlinear `ContinuousDiscreteModel`.  Replaces the
Jacobian linearisation of the EKF with a deterministic sigma-point approximation
(unscented transform), which captures the mean and covariance of a nonlinear
transformation to third order for Gaussian distributions.

**Sigma points** — Van der Merwe scaled sigma-point scheme (`2 nx + 1` points):

```
λ = α² (nx + κ) − nx

χ_0    = x̂                           (mean sigma point)
χ_i    = x̂ + (√((nx+λ) P))_i        for i = 1, …, nx
χ_{nx+i} = x̂ − (√((nx+λ) P))_i     for i = 1, …, nx

where (√M)_i denotes the i-th column of the Cholesky factor of M.
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
    Integrate:  χ̃_i = χ_i + ∫₀^{dt} f(χ_i(τ), u, d, t+τ) dτ   (Euler, n_steps sub-steps)

Predicted mean:       x̂⁻ = Σ_i W_m^i χ̃_i
Predicted covariance: P⁻  = Σ_i W_c^i (χ̃_i − x̂⁻)(χ̃_i − x̂⁻)ᵀ
                            + G Q_c Gᵀ · dt     (diffusion, accumulated per sub-step)
```

**Filtering** — unscented measurement update:

```
Propagate predicted sigma points through the observation function:
    γ_i = h(χ̃_i, d)   for i = 0, …, 2nx

Predicted observation:    ŷ⁻ = Σ_i W_m^i γ_i
Innovation covariance:    S_yy = Σ_i W_c^i (γ_i − ŷ⁻)(γ_i − ŷ⁻)ᵀ + R
Cross-covariance:         S_xy = Σ_i W_c^i (χ̃_i − x̂⁻)(γ_i − ŷ⁻)ᵀ

Kalman gain:  K = S_xy S_yy⁻¹
Corrected state:     x̂ = x̂⁻ + K (y − ŷ⁻)
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

#### `ContinuousDiscreteEnKF` — `mbc.estimation` *(stub — Ph.D. Ch. 7.3)*

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

    X_{j+1}[:,i] = X_j[:,i] + h · f(X_j[:,i], u, d, t_j)
                 + √h · g(X_j[:,i], u, d, t_j) · w_{j,i}

where  w_{j,i} ~ N(0, Q_c)  independently per particle and sub-step.
```

After propagation:

```
x̂⁻ = (1/N) Σ_i X[:,i]                           (ensemble mean)
A  = X − x̂⁻ 1ᵀ                                   (anomaly matrix, nx × N)
P⁻ = (1/(N−1)) A Aᵀ                               (ensemble sample covariance)
```

**Filtering** — perturbed-observations ensemble update:

```
Perturb observations:  y_i = y + v_i,   v_i ~ N(0, R),   i = 1, …, N

Predicted obs for each particle:
    Ŷ[:,i] = h(X⁻[:,i], d)

Innovation matrix:  E = Y − Ŷ   where Y[:,i] = y_i

Cross-covariance:   P_xy = (1/(N−1)) A_x Ŷ_anom^T
Innovation cov:     P_yy = (1/(N−1)) Ŷ_anom Ŷ_anom^T + R

where A_x = X⁻ − x̂⁻ 1ᵀ  and  Ŷ_anom = Ŷ − (mean of Ŷ) 1ᵀ

Kalman gain:  K = P_xy P_yy⁻¹

Update each particle:  X[:,i] ← X⁻[:,i] + K (y_i − Ŷ[:,i])

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

#### `ContinuousDiscreteParticleFilter` — `mbc.estimation` *(stub — Ph.D. Ch. 7.4)*

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

**Filtering** — importance weight update:

```
For each particle i = 1, …, N:

    ŷ_i = h(X⁻[:,i], d)                           (predicted observation)
    ℓ_i = exp(−½ (y − ŷ_i)ᵀ R⁻¹ (y − ŷ_i))      (Gaussian likelihood)
    w_i ← w_i · ℓ_i                                (weight update)

Normalise:  w_i ← w_i / Σ_j w_j

Effective sample size:  N_eff = 1 / Σ_i w_i²
```

When `mask` is provided, only active output channels contribute to the
likelihood computation.

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
handle systems where the state is partially constrained by `l(x, z, u, d, t) = 0`.
The covariance propagation uses an **effective Jacobian** that accounts for the
implicit dependence of `z` on `x` via the constraint.

**Effective Jacobian** (implicit function theorem):

If `l(x, z, ...) = 0` is satisfied, differentiating with respect to `x` gives:

```
∂l/∂x + (∂l/∂z)(∂z/∂x) = 0   ⟹   ∂z/∂x = −(∂l/∂z)⁻¹ (∂l/∂x)
```

The effective drift Jacobian for the Riccati ODE is:

```
F_eff = ∂f/∂x + (∂f/∂z)(∂z/∂x)
      = ∂f/∂x − (∂f/∂z)(∂l/∂z)⁻¹ (∂l/∂x)
```

All Jacobians can be supplied analytically by the model or computed by finite
differences.

**Prediction** — interleaved Euler/Newton:

```
For j = 0, 1, …, n_steps − 1:

    1. Compute F_eff at (x̂_j, z_j, u, d, t_j)
    2. Euler update on x̂:
          x̂_{j+1} = x̂_j + h · f(x̂_j, z_j, u, d, t_j)
    3. Newton solve for z at updated x:
          z_{j+1} = Newton( l(x̂_{j+1}, z, u, d, t_{j+1}) = 0,  z_init = z_j )
    4. Riccati update:
          G_j    = g(x̂_j, z_j, u, d, t_j)
          P_{j+1} = P_j + h · (F_eff P_j + P_j F_effᵀ + G_j Q_c G_jᵀ)
```

**Filtering**: identical to the CD-EKF, using the extended observation function
`H = ∂h/∂x + (∂h/∂z)(∂z/∂x)` if `h` depends on `z`.

**Parameters** (extends `ContinuousDiscreteEKF`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteDAEModel` | — | SDAE model |
| `x0` | `(nx,) ndarray` | — | Initial differential state estimate |
| `z0` | `(nz,) ndarray` | — | Initial algebraic state (must satisfy `l = 0`) |
| `P0` | `(nx,nx) ndarray` | — | Initial covariance (differential states only) |
| `dt` | `float` | — | Sampling interval |
| `n_steps` | `int` | `10` | Sub-steps per interval |
| `newton_tol` | `float` | `1e-10` | Newton convergence tolerance |
| `newton_max_iter` | `int` | `50` | Max Newton iterations per sub-step |

**Properties**: `x_hat`, `z_hat`, `P`.

**Usage**:

```python
from mbc.estimation import ContinuousDiscreteDAEEKF

ekf = ContinuousDiscreteDAEEKF(model, x0, z0, P0, dt=1.0)
x_hat, z_hat, P = ekf.step(y, u, d, t, mask=None)
x_hat, z_hat, P = ekf.predict(u, d, t)
x_hat, z_hat, P = ekf.update(y, d, mask=None)
```

---

### 2.4 Optimal Control Problems

#### `CDOptimalControlProblem` — `mbc.control`

Receding-horizon QP for a `LinearContinuousDiscreteModel`.  A typed thin wrapper
around `OptimalControlProblem` that accepts a continuous-discrete model.
Internally calls `model.discretize(d)` to obtain ZOH-discretised matrices
`(A_d, B_d, E_d)` and delegates to the parent class QP solver.

The cost function, constraints, batch-form prediction matrices, and QP solver are
identical to `OptimalControlProblem` (see §1.3).  The only difference is the
model type, which exposes `discretize(d)` via the `LinearContinuousDiscreteModel`
interface rather than inheriting from `LinearDiscreteModel`.

**Parameters**: identical to `OptimalControlProblem` with `model` of type
`LinearContinuousDiscreteModel`.

**Usage**:

```python
from mbc.control import CDOptimalControlProblem

ocp = CDOptimalControlProblem(model, N=20, Q=Q_y, R=R_u, P=P_terminal)
U_seq, X_seq = ocp.solve(x0=x_hat, D=D_forecast, x_ref=model.x_ref, u_prev=u_prev)
```

#### `EconomicNMPC` — `mbc.control` *(stub — Ph.D. Ch. 9)*

Economic Nonlinear MPC for a `ContinuousDiscreteModel`.  Unlike tracking MPC,
which minimises a quadratic distance to a setpoint, Economic NMPC minimises an
arbitrary economic stage cost `l_e(x, u, d)` that directly represents an
operational criterion (e.g. energy consumption, product yield, operating profit).

**Problem formulation**:

```
min_{u_0, …, u_{N-1}}  J = Σ_{k=0}^{N-1} l_e(x_k, u_k, d_k) + V_f(x_N)

subject to:
    x_{k+1} = f̄(x_k, u_k, d_k)   (mean dynamics, no noise)
    u_k ∈ U                        (input constraints)
    x_k ∈ X                        (optional state constraints)
```

The prediction model `f̄` is obtained by integrating the drift `f` with
`n_steps` forward-Euler sub-steps per sampling interval (noise term omitted for
deterministic prediction).  Constraints are passed as a list of
`scipy.optimize.minimize`-compatible dictionaries
`[{"type": "ineq"/"eq", "fun": ...}]`.

The NLP is solved with `scipy.optimize.minimize` using the SLSQP method by
default.  **Warm-starting**: the previous optimal sequence `u_prev` is shifted
by one step (last element repeated) and used as the initial guess for the NLP,
which significantly reduces solve time in closed-loop operation.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Nonlinear SDE model |
| `N` | `int` | — | Prediction horizon |
| `stage_cost` | `(x, u, d) → float` | — | Economic stage cost `l_e` |
| `terminal_cost` | `(x) → float` or `None` | `None` | Terminal cost `V_f` |
| `constraints` | `list[dict]` or `None` | `None` | scipy-format constraints |
| `n_steps` | `int` | `10` | Euler sub-steps per interval for prediction |
| `solver` | `str` | `"SLSQP"` | NLP solver name for `scipy.optimize.minimize` |
| `solver_options` | `dict` or `None` | `None` | Options forwarded to the solver |

**Methods**:

```python
from mbc.control import EconomicNMPC

def energy_cost(x, u, d):
    return float(u @ u)   # quadratic energy proxy

ocp = EconomicNMPC(model, N=20, stage_cost=energy_cost, n_steps=10)

u_opt, cost = ocp.solve(x0, d_trajectory, u_prev=None)
# u_opt : (N, nu) ndarray — full optimal sequence
# cost  : float — optimal economic objective value

u0 = ocp.step(x0, d_trajectory, u_prev)  # returns (nu,) first action only
```

---

### 2.5 MPC Controllers

#### `CDMPCController` — `mbc.control`

Combines a `CDKalmanFilter` and a `CDOptimalControlProblem` into a receding-horizon
controller for a linear continuous-discrete system.

**Receding-horizon policy** — at each measurement time k:

1. **Estimate**: `x̂[k] ← estimator.update(y[k], d[k])`
2. **Optimise**: `(U*, X*) ← ocp.solve(x̂[k], D, model.x_ref, u_prev)`
3. **Apply**: `u[k] = U*[0:m]`
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

Note the split: the *estimator* uses the continuous-time matrices `A_c`, `B_c`,
`E_c` directly via ODE integration; the *OCP* internally calls `model.discretize(d)`
to obtain ZOH matrices for the QP.  Both operate on the same `model` object.

#### `NMPCController` *(stub)*

Pairs any continuous-discrete state estimator (EKF, UKF, EnKF, PF, or DAE-EKF)
with any nonlinear OCP (`EconomicNMPC`, or a future `TrackingNMPC`) to form a
general nonlinear receding-horizon controller.

**Closed-loop structure**:

```
           ┌────────────────────────────────────────────────────┐
           │              NMPCController                         │
  y[k] ───┤──► CD Estimator ──x̂[k]──► Nonlinear OCP ──►u[k]───┼──► Plant
           │   (EKF/UKF/EnKF/PF)       (SLSQP NLP)             │
           │         ▲                                          │
           │    record_action(u[k])                             │
           └────────────────────────────────────────────────────┘
```

At each step: (1) estimator `step(y, u_prev, d, t)` → `x̂[k]`; (2) OCP
`solve(x̂[k], d_trajectory, u_prev)` → `u_opt`; (3) apply `u[k] = u_opt[0]`.

---

## Part III — System Identification

### `ped_neg_log_likelihood` — `mbc.identification.likelihood`

Evaluates the **prediction-error decomposition (PED)** Kalman-filter negative
log-likelihood for a linear discrete-time model parameterised by `θ`.

For a linear state-space model with `C = I` (full state observation), the
Kalman filter innovations sequence `{ν_k}` is white and Gaussian under the
true model.  The negative log-likelihood is:

```
−log L(θ) = ½ Σ_{k=2}^{T} [ log|S_k| + ν_kᵀ S_k⁻¹ ν_k ]

where:
    x̂_k⁻  = A_d x̂_{k-1} + B_d u_{k-1} + E_d d_{k-1} + offset(d_{k-1})
    P_k⁻   = A_d P_{k-1} A_dᵀ + Q                       (standard G=I form)
    ν_k    = y_k − x̂_k⁻                                 (innovation, C=I)
    S_k    = P_k⁻ + R                                    (innovation covariance, C=I)
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
    Q,               # (n,n) ndarray — process noise covariance
    R,               # (n,n) ndarray — measurement noise covariance
)
```

**History format**: each entry `{"y": (n,) ndarray, "u": (m,) ndarray, "d": (p,) ndarray}`
records one time step.  The model is re-discretised at each step's disturbance.

### `ped_neg_log_likelihood_gradient` — `mbc.identification.likelihood`

Forward finite-difference gradient `∂(−log L)/∂θ` of the PED log-likelihood.
Step size `h = 1e-5` by default.  Can be replaced by analytic propagation through
the Kalman recursion using `model.discretize_jacobian()` for better accuracy and
speed.

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
objective(θ) = −log L(θ|Q, R, history) + regularization_fn(θ)
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

**From impulse response** — `from_impulse_response(h, dt, n)`:

Constructs a minimal nth-order model whose impulse response best fits the sampled
sequence `h[0], h[1], …, h[T-1]` (sampled at interval `dt`) using the Ho-Kalman
algorithm restricted to the SISO case (see §4.2).

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
| `H` | `list[(ny,nu) ndarray]` | Markov parameters `H[0], H[1], …`; length ≥ `2n+1` |
| `n` | `int` | Desired model order |

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

### `MonteCarloSimulation` — `mbc.monte_carlo` *(stub — Ph.D. Ch. 12)*

Closed-loop Monte Carlo framework for assessing controller and estimator
performance under stochastic initial conditions and process noise.

**Trial structure** — each of `N_mc` independent trials proceeds as:

1. Draw initial state: `x₀ⁱ ~ N(x0_mean, x0_cov)`
2. Initialise estimator with `x₀ⁱ` (or skip if `estimator=None`)
3. For each of `T` measurement intervals `k = 0, 1, …, T-1`:
   - **Simulate**: `x_{k+1}^i = simulator.step(x_k^i, u_k^i, D[k], t_k)` (with noise)
   - **Observe**: `y_k^i = model.h(x_{k+1}^i, D[k]) + v_k^i`, `v_k^i ~ N(0, R)`
   - **Estimate**: `x̂_{k+1}^i = estimator.step(y_k^i, u_k^i, D[k], t_{k+1})` (if provided)
   - **Control**: `u_{k+1}^i = controller.step(x̂_{k+1}^i, D[k+1:k+1+N], …)`
4. Record `(X^i, Y^i, U^i)` and cumulative cost `Σ_k l(x_k^i, u_k^i, D[k])`

When `estimator=None`, the true state `x_k^i` is fed directly to the controller
(perfect state-information baseline).

**Reproducibility**: trial `i` uses random seed `seed + i`, ensuring independent
but deterministic realisations.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `ContinuousDiscreteModel` | — | Plant model (for observation function) |
| `simulator` | `SDESimulator` or `SDAESimulator` | — | Plant dynamics integrator |
| `controller` | `object` with `.step()` | — | Feedback controller |
| `estimator` | `object` with `.step()` or `None` | `None` | State estimator; `None` = perfect state info |
| `N_mc` | `int` | `100` | Number of Monte Carlo trials |
| `seed` | `int` or `None` | `None` | Base random seed (trial i uses seed+i) |

**Methods**:

```python
from mbc.monte_carlo import MonteCarloSimulation

mc = MonteCarloSimulation(
    model=plant, simulator=sim,
    controller=ctrl, estimator=ekf,
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

**Optional dependencies**: `scipy` (required for `EconomicNMPC`, `ParameterEstimator`
with `use_gradient=True`, and future NMPC stubs).

## Running Tests

```bash
pytest tests/
```
