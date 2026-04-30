# mbc — Model-Based Control Toolbox

A Python toolbox for linear and nonlinear model-based control, state estimation,
system identification, and realization. Implements algorithms from discrete-discrete
and continuous-discrete stochastic systems, following the notation of the author's
M.Sc. and Ph.D. theses.

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

---

## Part I — Discrete-Discrete Systems

### 1.1 Models

#### `LinearDiscreteModel` (abstract base)

A linear time-invariant (LTI) state-space model in discrete time:

```
x[k+1] = A x[k] + B u[k] + E d[k]
y[k]   = C x[k]           + v[k],    v[k] ~ N(0, R)
```

where `x ∈ ℝⁿ`, `u ∈ ℝᵐ`, `d ∈ ℝᵖ`, `y ∈ ℝˡ`.

**Subclass interface** — implement these abstract properties:

| Property | Type            | Description                    |
|----------|-----------------|--------------------------------|
| `A`      | `(n,n) matrix`  | State transition matrix        |
| `B`      | `(n,m) matrix`  | Input matrix                   |
| `E`      | `(n,p) matrix`  | Disturbance input matrix       |
| `C`      | `(l,n) matrix`  | Output matrix                  |
| `Q`      | `(n,n) matrix`  | Process noise covariance       |
| `R`      | `(l,l) matrix`  | Measurement noise covariance   |
| `x`      | `(n,1) matrix`  | Initial state                  |
| `n_x`   | `int`           | State dimension                |
| `n_u`   | `int`           | Input dimension                |
| `n_d`   | `int`           | Disturbance dimension          |

**Example**:

```python
from cvxopt import matrix
from mbc.models import LinearDiscreteModel

class ThermalModel(LinearDiscreteModel):
    @property
    def A(self): return matrix([[0.95]])
    @property
    def B(self): return matrix([[0.03]])
    @property
    def E(self): return matrix([[0.02]])
    @property
    def C(self): return matrix([[1.0]])
    @property
    def Q(self): return matrix([[1e-4]])
    @property
    def R(self): return matrix([[0.5]])
    @property
    def x(self): return matrix([20.0])
    @property
    def n_x(self): return 1
    @property
    def n_u(self): return 1
    @property
    def n_d(self): return 1
```

#### `LinearDiscreteLPVModel` (LPV extension)

A Linear Parameter-Varying (LPV) extension of `LinearDiscreteModel` where the
system matrices depend on the current disturbance `d[k]`:

```
x[k+1] = A(d[k]) x[k] + B(d[k]) u[k] + E(d[k]) d[k]
y[k]   = C(d[k]) x[k]                            + v[k]
```

Subclasses override `discretize(d)` to return scheduling-dependent matrices
`(A, B, E, C)`. The static `A`, `B`, `E`, `C` properties provide a nominal
linearisation (e.g. at `d = 0`) used for initialisation.

```python
from mbc.models import LinearDiscreteLPVModel

class LPVThermal(LinearDiscreteLPVModel):
    def discretize(self, d):
        # return (A(d), B(d), E(d), C(d)) as cvxopt matrices
        ...
```

---

### 1.2 State Estimators

#### `KalmanFilter`

Standard discrete-time Kalman filter (M.Sc. thesis, Ch. 5).

**Model**: `LinearDiscreteModel` (or `LinearDiscreteLPVModel`).

**Algorithm** — at each time step `k`:

*Prediction*:
```
x̂[k|k-1] = A x̂[k-1] + B u[k-1] + E d[k-1]
P[k|k-1]  = A P[k-1] Aᵀ + G Q Gᵀ
```

*Measurement update* (Joseph stabilised form, §7.11c):
```
e[k]  = y[k] − C x̂[k|k-1]
R_e   = C P[k|k-1] Cᵀ + R
K     = P[k|k-1] Cᵀ R_e⁻¹
x̂[k]  = x̂[k|k-1] + K e[k]
IKC   = I − K C
P[k]  = IKC P[k|k-1] IKCᵀ + K R Kᵀ
```

The Joseph form `P = IKC P⁻ IKCᵀ + K R Kᵀ` guarantees that `P` remains
symmetric positive semi-definite in finite-precision arithmetic.

**Missing observations** (M.Sc. Ch. 5.5): pass `mask` to `update()` to exclude
individual outputs. When `mask[i] = False`, output `i` is dropped from the
measurement update; the filter runs in prediction-only mode when all outputs
are masked.

**Noise separation** (M.Sc. Ch. 5.4): pass `noise_matrix=G` to `__init__()` to
use `G Q Gᵀ` as the process noise contribution. Defaults to `G = I` (i.e. `Q`
is the full covariance).

**Usage**:

```python
from mbc.estimation import KalmanFilter

kf = KalmanFilter(model, P0=None, noise_matrix=None)

x_hat = kf.update(y, d, mask=None)  # returns (n,1) state estimate
kf.record_action(u)                  # store u for next prediction
```

---

### 1.3 Optimal Control Problems

#### `OptimalControlProblem`

Finite-horizon linear-quadratic OCP over a prediction horizon `N` (M.Sc. Ch. 6).

**Cost**:
```
J = Σ_{k=0}^{N-1} [ ½ (y[k]−r)ᵀ Q_y (y[k]−r) + ½ Δu[k]ᵀ R_u Δu[k] ]
  + ½ (y[N]−r)ᵀ Q_yN (y[N]−r)
```

where `Δu[k] = u[k] − u[k-1]` is the incremental input.

**Constraints**:
```
u_min ≤ u[k] ≤ u_max    ∀k
Δu_min ≤ Δu[k] ≤ Δu_max  ∀k
y_min ≤ y[k] ≤ y_max    ∀k
```

The OCP is transcribed into a condensed QP and solved with `cvxopt.solvers.qp`.

**Usage**:

```python
from mbc.control import OptimalControlProblem

ocp = OptimalControlProblem(model, N=20, Q_y=..., R_u=..., Q_yN=...)
u_opt = ocp.solve(x0, r, d, u_prev)  # returns first optimal input
```

---

### 1.4 MPC Controllers

#### `MPCController`

Combines a `KalmanFilter` and an `OptimalControlProblem` into a receding-horizon
feedback controller.

**Closed-loop structure**:

```
         ┌─────────────────────────────────────┐
    r ──►│                                     │
         │         MPCController               │
  y[k] ─┤─► KalmanFilter ──► OCP ─► u[k] ────┼──► Plant ──► y[k+1]
         │      x̂[k]            │              │
         │                      └── record ────┤
         └─────────────────────────────────────┘
```

At each step: (1) `KalmanFilter.update(y, d)` → `x̂[k]`; (2)
`OCP.solve(x̂[k], r, d, u_prev)` → `u[k]`; (3) `KalmanFilter.record_action(u)`.

**Usage**:

```python
from mbc.control import MPCController

ctrl = MPCController(model, N=20, Q_y=..., R_u=..., Q_yN=...)
u = ctrl.step(y, r, d)  # full observe-optimise-record cycle
```

---

## Part II — Continuous-Discrete Systems

### 2.1 Models

#### `LinearContinuousDiscreteModel` (abstract base)

A linear continuous-discrete stochastic state-space model (Ph.D. Ch. 5):

```
dx = (A_c x + B_c u + E_c d) dt + G dw,   w ~ N(0, Q_c dt)
y[k] = C x[k] + v[k],                      v[k] ~ N(0, R)
```

where `x ∈ ℝⁿ`, `u ∈ ℝᵐ`, `d ∈ ℝᵖ`, `y ∈ ℝˡ`, `G ∈ ℝⁿˣᵍ`.

**Subclass interface** — implement these abstract properties:

| Property | Type            | Description                          |
|----------|-----------------|--------------------------------------|
| `A_c`    | `(n,n) ndarray` | Continuous state matrix              |
| `B_c`    | `(n,m) ndarray` | Continuous input matrix              |
| `E_c`    | `(n,p) ndarray` | Continuous disturbance matrix        |
| `G`      | `(n,q) ndarray` | Noise input matrix                   |
| `Q_c`    | `(q,q) ndarray` | Continuous process noise covariance  |
| `C`      | `(l,n) matrix`  | Output matrix (cvxopt)               |
| `R`      | `(l,l) matrix`  | Measurement noise covariance (cvxopt)|
| `dt`     | `float`         | Sampling interval                    |
| `x`      | `(n,) ndarray`  | Initial state                        |
| `n_x`   | `int`           | State dimension                      |
| `n_u`   | `int`           | Input dimension                      |
| `n_d`   | `int`           | Disturbance dimension                |

**Utility methods** (ZOH discretisation — for use outside the filter/OCP):

```python
A_d, B_d, E_d = model.discretize()       # Van Loan ZOH
Q_d           = model.discretize_noise() # exact discrete Q via Van Loan (1978)
```

These methods are provided for convenience (e.g. for analysis or initialising
a discrete filter from continuous matrices) and are **not** called internally by
`CDKalmanFilter` or `CDOptimalControlProblem`, which operate directly on the
continuous matrices `A_c`, `B_c`, `E_c`.

**Example**:

```python
import numpy as np
from cvxopt import matrix
from mbc.models import LinearContinuousDiscreteModel

class CSTRModel(LinearContinuousDiscreteModel):
    @property
    def A_c(self): return np.array([[-0.5, 0.0], [1.0, -1.0]])
    @property
    def B_c(self): return np.array([[1.0], [0.0]])
    @property
    def E_c(self): return np.zeros((2, 1))
    @property
    def G(self):   return np.eye(2)
    @property
    def Q_c(self): return np.diag([1e-3, 1e-3])
    @property
    def C(self):   return matrix([[1.0, 0.0]])
    @property
    def R(self):   return matrix([[0.1]])
    @property
    def dt(self):  return 1.0
    @property
    def x(self):   return np.zeros(2)
    @property
    def n_x(self): return 2
    @property
    def n_u(self): return 1
    @property
    def n_d(self): return 1
```

#### `ContinuousDiscreteModel` (nonlinear abstract base)

Nonlinear continuous-discrete SDE model (Ph.D. Ch. 5):

```
dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c dt)
y[k] = h(x[k], d[k]) + v[k],                  v[k] ~ N(0, R)
```

**Subclass interface**: implement `f`, `g`, `h`, `Q_c`, `R`, `nx`, `nu`, `ny`.

#### `ContinuousDiscreteDAEModel` (nonlinear DAE abstract base)

Nonlinear continuous-discrete stochastic DAE model (Ph.D. Ch. 6):

```
dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
 0 = l(x, z, u, d, t)
y[k] = h(x[k], z[k], d[k]) + v[k]
```

Extends `ContinuousDiscreteModel`; additionally implement `l` and `nz`.

---

### 2.2 Simulators

#### `SDESimulator`

Euler-Maruyama simulator for `ContinuousDiscreteModel` (Ph.D. Ch. 5).

Between measurement times `t_k` and `t_{k+1} = t_k + dt`, the SDE is integrated
with `n_steps` sub-steps of size `h = dt / n_steps`.

**Explicit-Explicit (EE) scheme** — drift and diffusion evaluated at the start
of each sub-step:

```
x_{j+1} = x_j + h · f(x_j, u, d, t_j) + √h · g(x_j, u, d, t_j) · w_j
```

where `w_j ~ N(0, Q_c)`. This is the standard Euler-Maruyama scheme; it is
first-order strong and half-order weak.

**Implicit-Explicit (IE) scheme** — drift evaluated implicitly at `t_{j+1}`,
diffusion evaluated explicitly:

```
x_{j+1} = x_j + h · f(x_{j+1}, u, d, t_{j+1}) + √h · g(x_j, u, d, t_j) · w_j
```

The implicit drift term is resolved by fixed-point iteration
`x^(ℓ+1) = x_j + h · f(x^(ℓ), ...) + noise` until convergence. The IE scheme
has better stability properties for stiff systems and damps spurious oscillations
that EE can introduce.

**Usage**:

```python
from mbc.simulation import SDESimulator

sim = SDESimulator(model, dt=1.0, n_steps=10, scheme="EE", seed=42)

x_next = sim.step(x, u, d, t)           # one interval
X      = sim.simulate(x0, U, D, t0=0.)  # full horizon, returns (T+1, nx)
```

#### `SDAESimulator`

Euler-Maruyama simulator for `ContinuousDiscreteDAEModel` (Ph.D. Ch. 6).

At each sub-step the algebraic constraint `l(x, z, u, d, t) = 0` is enforced
so that `z` remains consistent with `x`.

**Explicit-Explicit (EE) scheme**:

```
1. Euler step on x:  x_{j+1} = x_j + h · f(x_j, z_j, u, d, t_j)
                               + √h · g(x_j, z_j, u, d, t_j) · w_j
2. Newton solve for z_{j+1}: l(x_{j+1}, z_{j+1}, u, d, t_{j+1}) = 0
```

Drift and diffusion are evaluated at `(x_j, z_j)` (explicit), then `z` is
projected back onto the constraint manifold at the new `x`.

**Implicit-Explicit (IE) scheme**:

```
1. Inner Newton loop: solve for x_{j+1} implicitly
   x^(ℓ+1) = x_j + h · f(x^(ℓ), z^(ℓ), u, d, t_{j+1})
             + √h · g(x_j, z_j, u, d, t_j) · w_j
   where z^(ℓ) is obtained by Newton-solving l(x^(ℓ), z, ...) = 0
   at each inner iteration, coupling x and z implicitly.
2. Final Newton solve: l(x_{j+1}, z_{j+1}, u, d, t_{j+1}) = 0
```

The IE scheme is necessary for index-1 DAEs with stiff algebraic coupling.

**Usage**:

```python
from mbc.simulation import SDAESimulator

sim = SDAESimulator(model, dt=1.0, n_steps=10, scheme="EE")

x_next, z_next = sim.step(x, z, u, d, t)
X, Z            = sim.simulate(x0, z0, U, D, t0=0.)
```

---

### 2.3 State Estimators

#### `CDKalmanFilter`

Kalman filter for a linear continuous-discrete stochastic system (Ph.D. Ch. 7.3
specialised to linear `f`).

**Prediction** — continuous ODE integration over `[t_{k-1}, t_k]` using
`n_steps` forward-Euler sub-steps of size `h = dt / n_steps`:

```
dx̂/dt = A_c x̂ + B_c u + E_c d                     (state ODE, §7.3a)
dP/dt  = A_c P + P A_cᵀ + G Q_c Gᵀ                 (Riccati ODE, §7.3b)
```

Inputs `u` and disturbances `d` are held constant (zero-order hold) over each
interval. The system matrices `A_c`, `B_c`, `E_c` are used directly — no ZOH
or Van Loan discretisation is applied inside the filter.

**Filtering** — Joseph stabilised measurement update:

```
e[k]  = y[k] − C x̂[k|k-1]
R_e   = C P[k|k-1] Cᵀ + R
K     = P[k|k-1] Cᵀ R_e⁻¹
x̂[k]  = x̂[k|k-1] + K e[k]
IKC   = I − K C
P[k]  = IKC P[k|k-1] IKCᵀ + K R Kᵀ
```

**Bootstrap**: on the first call, the state estimate is initialised via the
minimum-norm pseudoinverse `x̂ = Cᵀ (C Cᵀ)⁻¹ y`, which reduces to `x̂ = y`
when `C = I`.

**Missing observations** (M.Sc. Ch. 5.5): `update(y, d, mask)` excludes
individual outputs. Prediction-only mode when all outputs are masked.

**Usage**:

```python
from mbc.estimation import CDKalmanFilter

kf = CDKalmanFilter(model, P0=None, n_steps=10)

x_hat = kf.update(y, d, mask=None)   # observe
kf.record_action(u)                   # store u for next prediction step
```

#### `ContinuousDiscreteEKF` *(stub — Ph.D. Ch. 7.1)*

Extended Kalman Filter for nonlinear `ContinuousDiscreteModel`.

State and covariance are propagated via continuous-time ODEs:

```
dx̂/dt = f(x̂, u, d, t)
dP/dt  = F(t) P + P F(t)ᵀ + G Q_c Gᵀ,   F = ∂f/∂x|_{x̂}
```

integrated with forward Euler. Measurement update uses the linearised
observation `H = ∂h/∂x|_{x̂}` in the standard EKF correction.

#### `ContinuousDiscreteUKF` *(stub — Ph.D. Ch. 7.2)*

Unscented Kalman Filter for nonlinear `ContinuousDiscreteModel`.

Uses Van der Merwe scaled sigma points `{χᵢ, wᵢ}` (`2nx + 1` points).
Each sigma point is propagated through the drift ODE via forward Euler.
The prior mean and covariance are reconstructed from propagated points,
with `G Q_c Gᵀ dt` added per sub-step. Measurement update uses the
unscented transform cross-covariance `S_xy` and innovation covariance
`S_yy`.

#### `ContinuousDiscreteEnKF` *(stub — Ph.D. Ch. 7.3)*

Ensemble Kalman Filter for nonlinear `ContinuousDiscreteModel`.

An ensemble of `N` particles is propagated via Euler-Maruyama:

```
xᵢ_{j+1} = xᵢ_j + h · f(xᵢ_j, u, d, t_j) + √h · G · wᵢ_j,   wᵢ_j ~ N(0, Q_c)
```

The ensemble-estimated covariance replaces `P` in the measurement update.
Observations are perturbed: `yᵢ = y + vᵢ`, `vᵢ ~ N(0, R)`.

#### `ContinuousDiscreteParticleFilter` *(stub — Ph.D. Ch. 7.4)*

Particle Filter for nonlinear `ContinuousDiscreteModel`.

`N` particles are propagated through the SDE via Euler-Maruyama. Weights
are updated by the Gaussian likelihood `p(y | xᵢ)`. Systematic resampling
is triggered when the effective sample size `N_eff = 1/Σ(wᵢ)²` falls below
`N/2`.

#### `ContinuousDiscreteDAEEKF` *(stub — Ph.D. Ch. 8)*

EKF for `ContinuousDiscreteDAEModel`. The effective Jacobian for the
prediction ODE accounts for the implicit constraint via the implicit
function theorem:

```
F_eff = ∂f/∂x − (∂f/∂z)(∂l/∂z)⁻¹(∂l/∂x)
```

At each integration sub-step, `z` is updated by Newton iteration on
`l(x, z, u, d, t) = 0`, interleaved with the Euler step on `x`.

---

### 2.4 Optimal Control Problems

#### `CDOptimalControlProblem`

Finite-horizon OCP for `LinearContinuousDiscreteModel` (Ph.D. Ch. 9 linear
specialisation).

**Continuous-time cost**:

```
J = ∫_0^T [ (y(t)−r)ᵀ Q_c (y(t)−r) + u(t)ᵀ R_c u(t) ] dt
  + (y(T)−r)ᵀ Q_T (y(T)−r)
```

Over the prediction horizon `[0, N·dt]`, the continuous cost is approximated
by integrating the ODE dynamics with `n_steps` forward-Euler sub-steps and
summing the quadrature contributions. Working directly in continuous time
(no ZOH pre-computation), the effective discrete weights per interval are:

```
Q̃_xx ≈ Q_c · dt,   Q̃_xu ≈ 0,   Q̃_uu ≈ R_c · dt
```

The transcribed QP is solved with `cvxopt.solvers.qp`. Constraints on `u`,
`Δu`, and `y` are supported.

**Usage**:

```python
from mbc.control import CDOptimalControlProblem

ocp = CDOptimalControlProblem(model, N=20, Q_c=..., R_c=..., Q_T=...)
u_opt = ocp.solve(x0, r, d, u_prev)
```

#### `TrackingNMPCProblem` *(stub)*

Finite-horizon tracking OCP for nonlinear `ContinuousDiscreteModel`.

**Cost**:

```
J = Σ_{k=0}^{N-1} [ (h(x[k], d[k])−r)ᵀ Q_y (h(x[k], d[k])−r) + u[k]ᵀ R_u u[k] ]
  + (h(x[N], d[N])−r)ᵀ Q_yN (h(x[N], d[N])−r)
```

Prediction uses forward Euler integration of `f(x, u, d, t)` with
`n_steps` sub-steps. The NLP is solved with `scipy.optimize.minimize`
(SLSQP). State and input constraints are supported.

**Usage**:

```python
from mbc.control import TrackingNMPCProblem

ocp = TrackingNMPCProblem(model, N=20, Q_y=..., R_u=..., Q_yN=..., n_steps=10)
u_opt = ocp.solve(x0, r, d, u_prev)  # first optimal input
```

#### `EconomicNMPCProblem` *(stub — Ph.D. Ch. 9)*

Finite-horizon economic OCP for nonlinear `ContinuousDiscreteModel`.

**Cost**:

```
J = Σ_{k=0}^{N-1} l_e(x[k], u[k], d[k]) + V_f(x[N])
```

where `l_e` is an arbitrary economic stage cost (e.g. energy, yield, profit)
and `V_f` is an optional terminal cost. Unlike tracking MPC, `l_e` need not
be quadratic. The NLP is solved with SLSQP.

**Usage**:

```python
from mbc.control import EconomicNMPCProblem

def stage_cost(x, u, d): return energy(x, u)

ocp = EconomicNMPCProblem(model, N=20, stage_cost=stage_cost, n_steps=10)
u_opt, cost = ocp.solve(x0, d_trajectory, u_prev)
```

---

### 2.5 MPC Controllers

#### `CDMPCController`

Combines a `CDKalmanFilter` and a `CDOptimalControlProblem` into a linear
continuous-discrete receding-horizon controller.

**Usage**:

```python
from mbc.control import CDMPCController

ctrl = CDMPCController(model, N=20, Q_c=..., R_c=..., Q_T=...)
u = ctrl.step(y, r, d)
```

#### `NMPCController` *(stub)*

Combines any continuous-discrete state estimator (EKF, UKF, EnKF, PF, or
DAE-EKF) with any nonlinear OCP (tracking or economic) to form a nonlinear
receding-horizon controller.

**Closed-loop structure**:

```
          ┌──────────────────────────────────────────────────┐
     r ──►│                                                  │
          │              NMPCController                      │
   y[k] ─┤─► CD Estimator ──► Nonlinear OCP ─► u[k] ───────┼──► Plant ──► y[k+1]
          │      x̂[k]              │                        │
          │                        └──── record_action ──────┤
          └──────────────────────────────────────────────────┘
```

At each step: (1) estimator `update(y, d)` → `x̂[k]`; (2) OCP `solve(x̂[k], ...)`
→ `u[k]`; (3) estimator `record_action(u)`.

**Usage**:

```python
from mbc.control import NMPCController

ctrl = NMPCController(estimator=ekf, ocp=tracking_ocp)
u = ctrl.step(y, r, d)   # for tracking OCP

ctrl = NMPCController(estimator=ukf, ocp=economic_ocp)
u = ctrl.step(y, d)      # for economic OCP (no reference)
```

---

## Part III — System Identification

### `MaximumLikelihoodEstimator`

Identifies model parameters by maximising the log-likelihood of the
innovations sequence from a Kalman filter (M.Sc. Ch. 6, Ph.D. Ch. 10).

For a Gaussian state-space model the log-likelihood is:

```
log p(y_{1:T} | θ) = −½ Σ_k [ log det R_e[k] + e[k]ᵀ R_e[k]⁻¹ e[k] ]
```

where `e[k] = y[k] − C x̂[k|k-1]` are the Kalman innovations and
`R_e[k] = C P[k|k-1] Cᵀ + R` is the innovation covariance. Optimised
over parameter vector `θ` using `scipy.optimize.minimize`.

**Usage**:

```python
from mbc.identification import MaximumLikelihoodEstimator

mle = MaximumLikelihoodEstimator(model_class, param_bounds)
theta_opt = mle.fit(Y, U, D)
```

---

## Part IV — Realization

Algorithms for constructing state-space models from transfer functions or
input-output data (M.Sc. Ch. 2–4).

### 4.1 SISO Realization

#### `SISORealization` *(stub — M.Sc. Ch. 2–3)*

Constructs a discrete-time SISO state-space model from a rational transfer
function or sampled impulse/step response data.

**Deterministic transfer function** `H(z) = B(z)/A(z)`:

```
A(z) y[k] = B(z) u[k]
```

where `A(z) = z^n + a_1 z^{n-1} + ... + a_n` and
`B(z) = b_0 z^n + b_1 z^{n-1} + ... + b_n`.

Supported canonical forms: **observable** (default) and **controllable**.

**Stochastic transfer function (ARMAX)**:

For a process driven by both a deterministic input and additive coloured noise,
the transfer function takes the ARMAX form:

```
A(z) y[k] = B(z) u[k] + C(z) e[k],   e[k] ~ N(0, σ²)
```

where `C(z) = z^n + c_1 z^{n-1} + ... + c_n` is the MA (moving-average)
polynomial that colours the white noise `e[k]`.

The ARMAX model is realised by augmenting the state-space with a noise channel:

```
x[k+1] = A x[k] + B u[k] + G e[k]
y[k]   = C x[k] + D u[k] + e[k]
```

The matrix `G` is realised from the `C`-numerator coefficients in **exactly
the same way** as `B` is realised from the `B`-numerator coefficients — both
share the same denominator `A(z)`, so the canonical state-space construction
applies identically to both channels. In the observable canonical form:

```
G = [c_1 − a_1, c_2 − a_2, ..., c_n − a_n]ᵀ
```

which are the residuals between the C-polynomial and the A-polynomial
coefficients.

**Usage**:

```python
from mbc.realization import SISORealization

# Deterministic
sys = SISORealization.from_transfer_function(num, den, form="observable")

# Stochastic (ARMAX)
sys = SISORealization.from_transfer_function(num, den, noise_num=c_coeffs)

# From impulse response
sys = SISORealization.from_impulse_response(h, dt=1.0, n=4)

A, B, C, D, G = sys.A, sys.B, sys.C, sys.D, sys.G
```

### 4.2 MIMO Realization

#### `MIMORealization` *(stub — M.Sc. Ch. 4)*

Constructs a MIMO state-space model from a sequence of Markov parameters
(pulse response matrices) using the Ho-Kalman algorithm.

**Markov parameters**: `H[k] = C Aᵏ⁻¹ B` for `k = 1, 2, ..., 2n`.

The Ho-Kalman algorithm forms the block Hankel matrix:

```
H = [[H[1],  H[2],  ... H[n]  ],
     [H[2],  H[3],  ... H[n+1]],
     [  ⋮      ⋮          ⋮   ],
     [H[n], H[n+1], ... H[2n] ]]
```

and extracts `(A, B, C)` from the singular-value decomposition
`H = U Σ Vᵀ` truncated to rank `n`.

**Usage**:

```python
from mbc.realization import MIMORealization

sys = MIMORealization.from_markov_parameters(H_list, n=4)
A, B, C, D = sys.A, sys.B, sys.C, sys.D
```

---

## Part V — Monte Carlo Simulation

### `MonteCarloSimulation` *(stub — Ph.D. Ch. 12)*

Closed-loop Monte Carlo framework for assessing controller and estimator
performance under uncertainty.

Each of `N_mc` trials:
1. Draws initial state `x₀ⁱ ~ N(x0_mean, x0_cov)`.
2. Simulates the SDE/SDAE plant forward over `T` steps.
3. Calls `estimator.update(y, d)` → `x̂[k]`.
4. Calls `controller.step(x̂[k], ...)` → `u[k]`.
5. Records state, output, and input trajectories.

Results are stored in a `MonteCarloResult` dataclass:

```python
@dataclass
class MonteCarloResult:
    X     : np.ndarray  # (N_mc, T+1, nx) — state trajectories
    Y     : np.ndarray  # (N_mc, T,   ny) — output trajectories
    U     : np.ndarray  # (N_mc, T,   nu) — input trajectories
    costs : np.ndarray  # (N_mc,)         — total cost per run
```

**Usage**:

```python
from mbc.monte_carlo import MonteCarloSimulation

mc = MonteCarloSimulation(
    model=model, simulator=sim,
    controller=ctrl, estimator=kf,
    N_mc=200, seed=0,
)
result = mc.run(x0_mean, x0_cov, D=D, T=100)

# Analyse
import numpy as np
print("Mean cost:", result.costs.mean())
print("95th pct: ", np.percentile(result.costs, 95))
```

---

## Installation

```bash
pip install -e .
```

**Dependencies**: `numpy`, `cvxopt`, `scipy` (for NMPC/identification stubs).

## Running Tests

```bash
pytest tests/
```
