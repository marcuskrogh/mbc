# mbc — Model-Based Control Toolbox

A Python toolbox for linear and nonlinear model-based control, estimation, and system identification.  The library is organised around the abstractions developed in two theses:

- **M.Sc. thesis** — linear discrete-time and continuous-discrete systems, Kalman filtering, state-space realisation, system identification, and linear MPC.
- **Ph.D. thesis** — nonlinear continuous-discrete SDE/SDAE models, CD-EKF/UKF/EnKF/PF estimators, Economic NMPC, and Monte Carlo closed-loop simulation.

---

## Package structure

```
mbc/
├── models.py              Model ABCs
├── control/
│   ├── ocp.py             OptimalControlProblem  (discrete-time QP)
│   ├── mpc.py             MPCController
│   ├── cd_ocp.py          CDOptimalControlProblem
│   ├── cd_mpc.py          CDMPCController
│   └── enmpc.py           EconomicNMPC            (stub)
├── estimation/
│   ├── kalman.py          KalmanFilter            (discrete-time)
│   ├── cd_kalman.py       CDKalmanFilter          (continuous-discrete)
│   ├── ekf.py             ContinuousDiscreteEKF   (stub)
│   ├── ukf.py             ContinuousDiscreteUKF   (stub)
│   ├── enkf.py            ContinuousDiscreteEnKF  (stub)
│   ├── pf.py              ContinuousDiscreteParticleFilter (stub)
│   └── ekf_dae.py         ContinuousDiscreteDAEEKF (stub)
├── identification/
│   ├── likelihood.py      PED log-likelihood
│   └── estimator.py       ParameterEstimator
├── realization/
│   ├── siso.py            SISORealization         (stub)
│   └── mimo.py            MIMORealization         (stub)
├── simulation/
│   ├── sde.py             SDESimulator            (stub)
│   └── sdae.py            SDAESimulator           (stub)
└── monte_carlo/
    └── simulation.py      MonteCarloSimulation    (stub)
```

---

## Models

### 1. `LinearDiscreteModel` (M.Sc. Ch. 5)

Abstract base for linear time-invariant or LPV discrete-time systems:

```
x[k+1] = A(d) x[k] + B(d) u[k] + E(d) d[k] + offset(d)
y[k]   = C x[k]
```

| Symbol | Dimension | Description                              |
|--------|-----------|------------------------------------------|
| x      | n         | State vector                             |
| u      | m         | Control input (ZOH over dt)              |
| d      | p         | Exogenous disturbance                    |
| y      | l         | Measured output                          |
| A, B, E| n×n, n×m, n×p | Discrete state-space matrices       |
| C      | l×n       | Output matrix (time-invariant)           |
| offset | n         | Additive constant (e.g. known bias)      |

Subclasses implement `discretize(d)` returning the cvxopt matrices `(A_d, B_d, E_d)`.  The `d` argument lets LPV subclasses schedule matrices on the current operating point.

For system identification, subclasses additionally implement `params` (parameter vector θ), `with_params(θ)` (model factory), and optionally `predict_offset(d_np)`.

---

### 2. `LinearContinuousDiscreteModel` (M.Sc. Ch. 5, Ph.D. Ch. 7.3)

Abstract base for linear continuous-time systems observed at discrete times:

```
dx = (A_c x + B_c u + E_c d) dt + G dw,   w ~ N(0, Q_c)
y[k] = C x[k] + v[k],                     v[k] ~ N(0, R)
```

| Symbol | Dimension | Description                                  |
|--------|-----------|----------------------------------------------|
| A_c    | n×n       | Continuous state matrix                      |
| B_c    | n×m       | Continuous input matrix                      |
| E_c    | n×p       | Continuous disturbance matrix                |
| G      | n×q       | Noise input matrix (maps dw to state space)  |
| Q_c    | q×q       | Continuous process-noise covariance          |
| C      | l×n       | Output matrix                                |
| R      | l×l       | Measurement noise covariance                 |
| dt     | —         | Sampling interval                            |

Subclasses provide `A_c`, `B_c`, `E_c`, `G`, `Q_c`, `C`, `R`, `dt` as abstract properties.

**Concrete methods provided by the base class:**

**ZOH discretisation** (`discretize(d)`) — augmented matrix method (no matrix inverse required):

```
[A_d | B_d | E_d] = expm([[A_c, B_c, E_c],
                           [ 0,   0,   0 ],
                           [ 0,   0,   0 ]] · dt)[:n, :]
```

**Discrete process-noise covariance** (`discretize_noise()`) — Van Loan (1978) method:

```
Q_d = ∫₀^dt expm(A_c τ) G Q_c Gᵀ expm(A_cᵀ τ) dτ
```

Computed via the 2n×2n augmented matrix:

```
M = [[-A_c,  G Q_c Gᵀ],
     [  0,    A_cᵀ  ]] · dt

E = expm(M)
Q_d = E[n:, n:]ᵀ · E[:n, n:]          (symmetrised)
```

---

### 3. `ContinuousDiscreteModel` (Ph.D. Ch. 5)

Abstract base for nonlinear continuous-discrete stochastic systems:

```
dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c)
y_k = h(x_k, d_k) + v_k,                    v_k ~ N(0, R)
```

Subclasses implement `f`, `g`, `h`, `Q_c`, `R`, `nx`, `nu`, `nd`, `ny`.

---

### 4. `ContinuousDiscreteDAEModel` (Ph.D. Ch. 6)

Extends `ContinuousDiscreteModel` with differential-algebraic structure:

```
dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
0  = l(x, z, u, d, t)
y_k = h(x_k, z_k, d_k) + v_k
```

| Symbol | Dimension | Description              |
|--------|-----------|--------------------------|
| z      | nz        | Algebraic state vector   |
| l      | nz        | Algebraic constraint     |

Subclasses additionally implement `l` and `nz`.

---

## Estimation

### 5. `KalmanFilter` (M.Sc. Ch. 5)

Discrete-time Kalman filter with Joseph-stabilised covariance update.

**Prediction step:**

```
x̂⁻[k]  = A x̂[k−1] + B u[k−1] + E d[k−1]
P⁻[k]   = A P[k−1] Aᵀ + Q                    (standard form, G = I)
P⁻[k]   = A P[k−1] Aᵀ + G Q Gᵀ              (noise-separated form, M.Sc. Ch. 5.4)
```

**Measurement update (Joseph stabilised form):**

```
S[k]   = C P⁻[k] Cᵀ + R
K[k]   = P⁻[k] Cᵀ S[k]⁻¹
x̂[k]   = x̂⁻[k] + K[k] (y[k] − C x̂⁻[k])
P[k]   = (I − K[k] C) P⁻[k] (I − K[k] C)ᵀ + K[k] R K[k]ᵀ
```

The Joseph form guarantees P remains symmetric positive semi-definite regardless of finite-precision errors (in contrast to the simpler `P = (I − KC) P⁻` form).

**Missing observations (M.Sc. Ch. 5.5):**  The `mask` parameter of `update(y, d, mask)` accepts a boolean list of length l.  When `mask[i] = False`, output i is excluded from the measurement update.  When all outputs are masked the filter performs a prediction-only step.

**Noise-separated form (M.Sc. Ch. 5.4):**  Pass `noise_matrix=G` (cvxopt, n×q) to `__init__`.  The prediction covariance then uses `G Q Gᵀ` instead of `Q`, where Q is interpreted as the noise in the process-noise subspace.

---

### 6. `CDKalmanFilter` (Ph.D. Ch. 7.3, linear special case)

Kalman filter for `LinearContinuousDiscreteModel`.  Follows the CD-EKF formulation of Ph.D. Ch. 7.3, specialised to linear dynamics where no Jacobian is required.

**Prediction — continuous ODE integration (§7.3a–b):**

The state estimate and error covariance are propagated by integrating ODEs over the sampling interval [t_{k-1}, t_k]:

```
dx̂/dt = A_c x̂ + B_c u + E_c d             (state ODE, §7.3a)
dP/dt = A_c P + P A_cᵀ + G Q_c Gᵀ         (matrix Riccati ODE, §7.3b)
```

Both ODEs are integrated with `n_steps` forward-Euler sub-steps of size `h = dt / n_steps`.  Inputs u and disturbances d are held constant over the interval (zero-order hold), but the system matrices are **never discretised**.

**Measurement update (§7.8–7.11, Joseph stabilised):**

```
e_k    = y_k − C x̂[k|k−1]                 (innovation, §7.8)
R_e    = C P[k|k−1] Cᵀ + R                (innovation covariance, §7.9)
K      = P[k|k−1] Cᵀ R_e⁻¹               (Kalman gain, §7.10)
x̂[k]   = x̂[k|k−1] + K e_k               (state update, §7.11a)
P[k]   = (I − K C) P[k|k−1] (I−KC)ᵀ + K R Kᵀ   (Joseph form, §7.11c)
```

**Key difference from `KalmanFilter`:**  The prediction never computes ZOH-discretised matrices or the Van Loan Q_d.  The continuous-time dynamics are integrated directly, which is more accurate when dt is large relative to the system time constants.

---

### 7. `ContinuousDiscreteEKF` (Ph.D. Ch. 7.1) — *stub*

Extended Kalman filter for nonlinear `ContinuousDiscreteModel`.

**Prediction — continuous ODE integration (§7.3a–b):**

```
dx̂/dt = f(x̂, u, d, t)                      (state ODE, §7.3a)
dP/dt = F(t) P + P F(t)ᵀ + σ(t) σ(t)ᵀ     (Riccati ODE, §7.3b)
```

where `F(t) = ∂f/∂x|_{x=x̂}` (Jacobian of drift) and `σ(t) = g(x̂, u, d, t)`.

**Measurement update:**

```
H = ∂h/∂x|_{x=x̂}
e = y − h(x̂, d)
R_e = H P Hᵀ + R
K = P Hᵀ R_e⁻¹
x̂ ← x̂ + K e
P ← (I − KH) P (I−KH)ᵀ + K R Kᵀ
```

---

### 8. `ContinuousDiscreteUKF` (Ph.D. Ch. 7.2) — *stub*

Unscented Kalman filter for nonlinear `ContinuousDiscreteModel`.

Sigma points `{χ_i, W_i}` (2n+1 points, van der Merwe weights) are propagated through the nonlinear state ODE via numerical integration.  Covariance is reconstructed from the propagated sigma points.  The measurement update uses the unscented transform.

---

### 9. `ContinuousDiscreteEnKF` (Ph.D. Ch. 7.3) — *stub*

Ensemble Kalman filter for nonlinear `ContinuousDiscreteModel`.

An ensemble of N particles `{x^(i)}_{i=1}^N` is propagated through the nonlinear SDE via Euler-Maruyama integration.  Covariance is estimated from the ensemble.  The measurement update uses the perturbed-observations form.

---

### 10. `ContinuousDiscreteParticleFilter` (Ph.D. Ch. 7.4) — *stub*

Particle filter for nonlinear `ContinuousDiscreteModel`.

N particles propagated through the nonlinear SDE via Euler-Maruyama.  Particle weights updated by the likelihood `p(y_k | x_k^(i))`.  Systematic resampling when effective sample size `N_eff = 1 / Σ_i (w_i)²` falls below N/2.

---

### 11. `ContinuousDiscreteDAEEKF` (Ph.D. Ch. 8) — *stub*

CD-EKF for `ContinuousDiscreteDAEModel`.  Algebraic variables z are solved at each integration step by Newton iteration on `l(x, z, u, d, t) = 0`, interleaved with the continuous ODE integration for x and P.

---

## Control

### 12. `OptimalControlProblem` (M.Sc. Ch. 5)

Finite-horizon QP for MPC, formulated in the lifted (batch) form.

**Objective:**

```
min_U  J = Σ_{k=0}^{N-1} [ ‖y[k+1] − r‖²_Q  +  ‖u[k]‖²_R  +  ‖Δu[k]‖²_S ]
           + ‖y[N] − r‖²_P
           + ρ Σ_{k=0}^{N-1} ‖ε[k+1]‖²
```

**Subject to:**

```
x[k+1] = A x[k] + B u[k] + E d[k]               (dynamics)
u_min ≤ u[k] ≤ u_max                             (hard input bounds)
y_min − ε[k+1] ≤ y[k+1] ≤ y_max + ε[k+1]        (soft output bounds)
ε[k+1] ≥ 0                                       (slack non-negativity)
```

where `Δu[k] = u[k] − u[k−1]` is the input rate of movement and ε are slack variables penalised by ρ.

**Batch prediction:**

```
X = Ψ x₀ + Γ U + Λ D
Y = C̄ X
```

where Ψ ∈ ℝ^{Nn×n}, Γ ∈ ℝ^{Nn×Nm}, Λ ∈ ℝ^{Nn×Np} are the standard prediction matrices.  The QP decision variable is z = [U; ε] and solved via `cvxopt.solvers.qp`.

---

### 13. `MPCController` (M.Sc. Ch. 5)

Receding-horizon MPC composed of a `KalmanFilter` and `OptimalControlProblem`.

**Control loop at each measurement time t_k:**

```
1. Estimate:   x̂[k] ← KalmanFilter.update(y[k], d[k])
2. Optimise:   U*   ← OCP.solve(x̂[k], D, x_ref)
3. Apply:      u[k]  = U*[0:m]     (first element of optimal sequence)
4. Record:     KalmanFilter.record_action(u[k])
```

---

### 14. `CDOptimalControlProblem` (M.Sc. Ch. 5 / Ph.D. Ch. 9)

Thin subclass of `OptimalControlProblem` typed for `LinearContinuousDiscreteModel`.  Uses ZOH discretisation (`model.discretize(d)`) at each solve step.  The OCP is inherently a finite-dimensional QP so ZOH is the correct interface; continuous-time integration is not needed here.

---

### 15. `CDMPCController` (M.Sc. Ch. 5)

Receding-horizon MPC composed of `CDKalmanFilter` and `CDOptimalControlProblem`.

**Control loop at each measurement time t_k:**

```
1. Estimate:   x̂[k] ← CDKalmanFilter.update(y[k], d[k])
2. Optimise:   U*   ← CDOptimalControlProblem.solve(x̂[k], D, x_ref)
3. Apply:      u[k]  = U*[0:m]
4. Record:     CDKalmanFilter.record_action(u[k])
```

The estimator integrates continuous ODEs; the OCP uses ZOH matrices.  This is the correct separation: the estimator should not discretise the system (accuracy), but the QP must be finite-dimensional (tractability).

---

### 16. `EconomicNMPC` (Ph.D. Ch. 9) — *stub*

Economic NMPC for `ContinuousDiscreteModel`.  Minimises an economic stage cost `l_e(x, u, d)` (energy, yield, profit) rather than a quadratic distance to a setpoint.

```
min_{U}   Σ_{k=0}^{N-1} l_e(x[k], u[k], d[k]) + V_f(x[N])
s.t.      x[k+1] = f_d(x[k], u[k], d[k])      (discretised dynamics)
          g(x[k], u[k]) ≤ 0                    (path constraints)
          g_T(x[N]) ≤ 0                         (terminal constraint)
```

Solved by an NLP solver (default: `scipy.optimize.minimize` with SLSQP).

---

## System Identification

### 17. PED log-likelihood (M.Sc. Ch. 6)

Prediction-error decomposition negative log-likelihood for a `LinearDiscreteModel`:

```
-log L(θ) = ½ Σ_k [ log|S_k| + ν_k^T S_k⁻¹ ν_k ]
```

where the one-step-ahead innovation and innovation covariance are obtained from a Kalman filter run through the data:

```
ν_k  = y_k − x̂_k⁻        (innovation, assumes C = I)
S_k  = P_k⁻ + R            (innovation covariance, C = I)
```

The `model_factory(θ)` callable maps the parameter vector to a model instance.  `ped_neg_log_likelihood(model_factory, theta, history, Q, R)` evaluates the negative log-likelihood given a recorded history of `{y, u, d}` tuples.

---

### 18. `ParameterEstimator` (M.Sc. Ch. 6)

Minimises the PED negative log-likelihood using Nelder-Mead (derivative-free):

```
θ* = argmin_{θ} [ -log L(θ) ]
```

Wraps `ped_neg_log_likelihood` and `_nelder_mead` into a single `fit(history)` method.

---

## Realization (M.Sc. Ch. 2–4)

### 19. `SISORealization` — *stub* (M.Sc. Ch. 2–3)

State-space realization of a SISO system.

**From transfer function** — observable and controllable canonical forms.

Given `H(z) = (b_0 z^r + … + b_r) / (a_0 z^n + … + a_n)`:

*Observable canonical form:*

```
A = [[ 0,  0, …,  0,  -a_n/a_0   ],
     [ 1,  0, …,  0,  -a_{n-1}/a_0],
     [ 0,  1, …,  0,      ⋮       ],
     [ ⋮,     ⋱,  ⋮,      ⋮       ],
     [ 0,  0, …,  1,  -a_1/a_0   ]]

B = [b̃_r, b̃_{r-1}, …, b̃_0]ᵀ    (Markov-corrected coefficients)

C = [0, 0, …, 0, 1]

D = b_0/a_0  (when deg(num) = deg(den))
```

**From impulse response** — Hankel matrix realization.

Given samples `h[0], h[1], …, h[T-1]`, form the Hankel matrix and apply SVD truncation to order n to recover (A, B, C, D).

---

### 20. `MIMORealization` — *stub* (M.Sc. Ch. 4)

Ho–Kalman realization from Markov parameters.

Given the Markov parameters `{H_0, H_1, H_2, …}` where `H_k = C A^{k-1} B`:

1. Form the block Hankel matrix from `H_1, H_2, …`
2. Compute rank-n truncated SVD: `Hankel = U_n Σ_n V_nᵀ`
3. Recover: `C = U_n Σ_n^{1/2}`, `B = Σ_n^{1/2} V_nᵀ`
4. Shift Hankel by one block row; recover `A = Σ_n^{-1/2} U_nᵀ H_shifted V_n Σ_n^{-1/2}`
5. Set `D = H_0`

---

## Simulation (Ph.D. Ch. 5–6)

### 21. `SDESimulator` — *stub* (Ph.D. Ch. 5)

Euler-Maruyama integration of `ContinuousDiscreteModel`.

**Explicit-Explicit (EE) scheme:**

```
x_{j+1} = x_j + f(x_j, u, d, t_j) h + g(x_j, u, d, t_j) √h w_j,   w_j ~ N(0, Q_c)
```

**Implicit-Explicit (IE) scheme:**

```
x_{j+1} = x_j + f(x_{j+1}, u, d, t_{j+1}) h + g(x_j, u, d, t_j) √h w_j
```

where `h = dt / n_steps` and the implicit drift `f(x_{j+1}, …)` is solved by fixed-point or Newton iteration.

---

### 22. `SDAESimulator` — *stub* (Ph.D. Ch. 6)

Euler-Maruyama integration of `ContinuousDiscreteDAEModel`.

At each sub-step:

1. Apply Euler drift update to x
2. Solve `l(x, z, u, d, t) = 0` for z via Newton iteration (initialised from previous z)
3. Add diffusion noise term

---

## Monte Carlo Simulation (Ph.D. Ch. 12)

### 23. `MonteCarloSimulation` — *stub*

Closed-loop Monte Carlo framework.

Runs N_mc independent trials; each trial:

1. Draw `x₀ ~ N(x₀_mean, x₀_cov)`
2. Propagate through `SDESimulator` (or `SDAESimulator`)
3. Apply controller via `controller.step(y, D)`
4. Optionally apply estimator via `estimator.step(y, u, d, t)`
5. Accumulate trajectories X, Y, U and total cost

Returns a `MonteCarloResult` dataclass with arrays of shape `(N_mc, T+1, nx)`, `(N_mc, T, ny)`, `(N_mc, T, nu)`, and `(N_mc,)` costs.

---

## Dependencies

| Package  | Purpose                                          |
|----------|--------------------------------------------------|
| numpy    | Numerical arrays; ODE integration internally    |
| cvxopt   | QP solver; Kalman filter linear algebra          |

No scipy dependency.  Matrix exponentials are computed via eigendecomposition (`_expm` in `_utils.py`).

---

## Notation summary

| Symbol | Meaning                                         |
|--------|-------------------------------------------------|
| n      | State dimension                                 |
| m      | Input dimension                                 |
| p      | Disturbance dimension                           |
| l      | Output dimension                                |
| q      | Process noise dimension                         |
| N      | MPC prediction horizon                          |
| T      | Simulation horizon                              |
| dt     | Sampling interval                               |
| A_c, B_c, E_c | Continuous-time state-space matrices   |
| A_d, B_d, E_d | ZOH-discretised state-space matrices   |
| G      | Noise input matrix (maps process noise to state)|
| Q_c    | Continuous process-noise covariance             |
| Q_d    | Discrete process-noise covariance (Van Loan)    |
| R      | Measurement noise covariance                    |
| P      | State error covariance                          |
| K      | Kalman gain                                     |
| ν, e   | Kalman innovation                               |
| θ      | Parameter vector (system identification)        |
