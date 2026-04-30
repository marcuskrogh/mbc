# mbc — Model-Based Control Toolbox

A Python toolbox for linear and nonlinear model-based control, estimation, and system identification, built around abstractions from two theses covering linear continuous-discrete systems (M.Sc.) and nonlinear SDE/SDAE systems (Ph.D.).

---

## Package structure

```
mbc/
├── models.py              Model ABCs
├── control/
│   ├── ocp.py             OptimalControlProblem      (discrete-time QP)
│   ├── mpc.py             MPCController              (discrete-time)
│   ├── cd_ocp.py          CDOptimalControlProblem    (continuous-discrete QP)
│   ├── cd_mpc.py          CDMPCController
│   └── enmpc.py           EconomicNMPC               (stub)
├── estimation/
│   ├── kalman.py          KalmanFilter               (discrete-time)
│   ├── cd_kalman.py       CDKalmanFilter             (continuous-discrete linear)
│   ├── ekf.py             ContinuousDiscreteEKF      (stub)
│   ├── ukf.py             ContinuousDiscreteUKF      (stub)
│   ├── enkf.py            ContinuousDiscreteEnKF     (stub)
│   ├── pf.py              ContinuousDiscreteParticleFilter (stub)
│   └── ekf_dae.py         ContinuousDiscreteDAEEKF   (stub)
├── identification/
│   ├── likelihood.py      PED log-likelihood
│   └── estimator.py       ParameterEstimator
├── realization/
│   ├── siso.py            SISORealization            (stub)
│   └── mimo.py            MIMORealization            (stub)
├── simulation/
│   ├── sde.py             SDESimulator               (stub)
│   └── sdae.py            SDAESimulator              (stub)
└── monte_carlo/
    └── simulation.py      MonteCarloSimulation       (stub)
```

---

## 1. Models

### 1.1 `LinearDiscreteModel`

Abstract base for a linear time-invariant or LPV discrete-time system:

```
x[k+1] = A(d) x[k] + B(d) u[k] + E(d) d[k] + offset(d)
y[k]   = C x[k]
```

| Symbol   | Dimension     | Description                          |
|----------|---------------|--------------------------------------|
| x        | n             | State vector                         |
| u        | m             | Control input (ZOH over dt)          |
| d        | p             | Exogenous disturbance                |
| y        | l             | Measured output                      |
| A, B, E  | n×n, n×m, n×p | Discrete state-space matrices        |
| C        | l×n           | Output matrix (time-invariant)       |
| offset   | n             | Additive constant (optional bias)    |

**Interface contract:**

- `discretize(d)` → `(A_d, B_d, E_d)` as cvxopt matrices.  LTI implementations ignore `d`; LPV subclasses may schedule matrices on the current operating point.
- `predict_offset(d_np)` → `(n,)` ndarray additive offset.  Default: zero vector.
- `params` → flat parameter vector θ for system identification.
- `with_params(θ)` → new model instance from θ (used for finite-difference Jacobians).
- `discretize_jacobian(d, h)` → `(dA, dB, dE)` lists of Jacobians ∂A_d/∂θ_i, ∂B_d/∂θ_i, ∂E_d/∂θ_i via forward finite differences.

---

### 1.2 `LinearContinuousDiscreteModel`

Abstract base for a linear continuous-time system observed at discrete measurement times:

```
dx = (A_c x + B_c u + E_c d) dt + G dw,   w ~ N(0, Q_c)
y[k] = C x[k] + v[k],                     v[k] ~ N(0, R)
```

| Symbol | Dimension | Description                                  |
|--------|-----------|----------------------------------------------|
| A_c    | n×n       | Continuous state matrix                      |
| B_c    | n×m       | Continuous input matrix                      |
| E_c    | n×p       | Continuous disturbance matrix                |
| G      | n×q       | Noise input matrix                           |
| Q_c    | q×q       | Continuous process-noise covariance (q×q)    |
| C      | l×n       | Output matrix (time-invariant, cvxopt)       |
| R      | l×l       | Measurement noise covariance (cvxopt)        |
| dt     | —         | Sampling interval                            |

Subclasses provide `A_c`, `B_c`, `E_c`, `G`, `Q_c` as numpy arrays and `C`, `R` as cvxopt matrices.

**Concrete methods provided by the base class:**

**ZOH discretisation** `discretize(d)` — augmented matrix method (no matrix inverse required):

```
[A_d | B_d | E_d] = expm([[A_c, B_c, E_c],
                           [ 0,   0,   0 ],
                           [ 0,   0,   0 ]] · dt)[:n, :]
```

Returns `(A_d, B_d, E_d)` as cvxopt matrices.  Used by `CDOptimalControlProblem`; the `CDKalmanFilter` does **not** call this.

**Exact discrete process-noise covariance** `discretize_noise()` — Van Loan (1978) method:

```
Q_d = ∫₀^dt expm(A_c τ) G Q_c Gᵀ expm(A_cᵀ τ) dτ
```

Computed via the 2n×2n augmented matrix:

```
E = expm([[-A_c,  G Q_c Gᵀ],
          [  0,    A_cᵀ   ]] · dt)

Q_d = E[n:2n, n:2n]ᵀ · E[0:n, n:2n]    (symmetrised)
```

Returns `Q_d` as a cvxopt matrix.  Available for diagnostics; not used in `CDKalmanFilter`.

---

### 1.3 `ContinuousDiscreteModel`

Abstract base for a nonlinear continuous-discrete stochastic system:

```
dx = f(x, u, d, t) dt + g(x, u, d, t) dw,   w ~ N(0, Q_c)
y_k = h(x_k, d_k) + v_k,                    v_k ~ N(0, R)
```

| Symbol | Dimension | Description                              |
|--------|-----------|------------------------------------------|
| f      | ℝⁿˣ       | Drift function (state dynamics)          |
| g      | ℝⁿˣˣⁿʷ   | Diffusion function (noise coupling)      |
| h      | ℝⁿʸ       | Observation function                     |
| Q_c    | nw×nw     | Continuous process-noise covariance      |
| R      | ny×ny     | Measurement noise covariance             |

Subclasses implement `f`, `g`, `h`, `Q_c`, `R`, `nx`, `nu`, `nd`, `ny`.

---

### 1.4 `ContinuousDiscreteDAEModel`

Extends `ContinuousDiscreteModel` with differential-algebraic structure:

```
dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
0  = l(x, z, u, d, t)
y_k = h(x_k, z_k, d_k) + v_k
```

| Symbol | Dimension | Description                           |
|--------|-----------|---------------------------------------|
| z      | nz        | Algebraic state (solved at each step) |
| l      | nz        | Algebraic constraint residual         |

Subclasses additionally implement `l` and `nz`.  Subclasses should override `h` to accept `(x, z, d)` if outputs depend on z.

---

## 2. Estimation

All estimators share a common interface: `predict(u, d, t)`, `update(y, d, mask)`, `step(y, u, d, t, mask)`.  The `mask` parameter is a boolean array of length ny; outputs with `mask[i] = False` are excluded from the measurement update.  `step` is a convenience wrapper calling `predict` then `update`.

---

### 2.1 `KalmanFilter`

Discrete-time Kalman filter.  State and covariance are propagated using the ZOH-discretised model matrices.

**Prediction step:**

```
x̂⁻[k]  = A x̂[k−1] + B u[k−1] + E d[k−1]
P⁻[k]   = A P[k−1] Aᵀ + Q                   (standard form, G = I)
P⁻[k]   = A P[k−1] Aᵀ + G Q Gᵀ             (noise-separated form)
```

The noise-separated form is activated by passing `noise_matrix=G` (cvxopt, n×q) to `__init__`.  Q is then interpreted as a q×q noise covariance and G maps noise into the state space.

**Measurement update — Joseph stabilised form:**

```
S[k]   = C P⁻[k] Cᵀ + R
K[k]   = P⁻[k] Cᵀ S[k]⁻¹
x̂[k]   = x̂⁻[k] + K[k] (y[k] − C x̂⁻[k])
P[k]   = (I − K[k] C) P⁻[k] (I − K[k] C)ᵀ + K[k] R K[k]ᵀ
```

The Joseph form guarantees P remains symmetric positive semi-definite regardless of floating-point arithmetic.  The simpler form `P = (I − KC) P⁻` is numerically equivalent but loses positive semi-definiteness in practice.

**Missing observations:**

When `mask[i] = False`, output i is excluded from the update: C and R are subsetted to active rows/columns only, and the reduced observation vector is used.  When all outputs are masked the filter performs a prediction-only step.

---

### 2.2 `CDKalmanFilter`

Kalman filter for a `LinearContinuousDiscreteModel`.  Follows the CD-EKF framework (Section 2.3 below), specialised to the linear case where no Jacobian linearisation is needed.

**Prediction — forward-Euler integration of state and Riccati ODEs:**

Over each sampling interval [t_{k-1}, t_k] with `n_steps` sub-steps of size h = dt / n_steps:

```
dx̂/dt = A_c x̂ + B_c u + E_c d             (state ODE)
dP/dt  = A_c P + P A_cᵀ + G Q_c Gᵀ         (matrix Riccati ODE)
```

Both ODEs are integrated by forward Euler:

```
x̂(t + h) ← x̂(t) + h [A_c x̂(t) + B_c u + E_c d]
P(t + h)  ← P(t)  + h [A_c P(t) + P(t) A_cᵀ + G Q_c Gᵀ]
```

u and d are held constant over the interval (zero-order hold).  The system is **never** discretised; no ZOH or Van Loan computation occurs in the filter.

**Measurement update — Joseph stabilised form:**

```
e[k]   = y[k] − C x̂[k|k−1]
R_e    = C P[k|k−1] Cᵀ + R
K      = P[k|k−1] Cᵀ R_e⁻¹
x̂[k]   = x̂[k|k−1] + K e[k]
P[k]   = (I − KC) P[k|k−1] (I − KC)ᵀ + K R Kᵀ
```

Solved via Cholesky factorisation of R_e (Lapack `posv`).

**Initialisation:**  On the first call to `update(y, d)`, x̂ is bootstrapped from the measurement via the minimum-norm pseudoinverse: x̂ = Cᵀ (C Cᵀ)⁻¹ y.  This avoids requiring a prior state estimate and reduces to x̂ = y when C = I.

**Key difference from `KalmanFilter`:**  The prediction integrates the continuous-time Riccati equation directly.  ZOH / Van Loan discretisation is more accurate when dt is small, but the continuous integration is more accurate when dt is large relative to the dominant time constants of A_c and therefore should always be preferred for the linear continuous-discrete case.

---

### 2.3 `ContinuousDiscreteEKF` *(stub)*

Extended Kalman filter for a nonlinear `ContinuousDiscreteModel`.  Extends `CDKalmanFilter` to nonlinear dynamics by linearising (computing Jacobians) at each integration sub-step.

**Prediction — forward Euler on state and Riccati ODEs:**

```
dx̂/dt = f(x̂, u, d, t)                            (state ODE)
dP/dt  = F(t) P + P F(t)ᵀ + σ(t) σ(t)ᵀ           (matrix Riccati ODE)
```

where at each sub-step:

```
F(t) = ∂f/∂x |_{x = x̂(t), u, d, t}     (Jacobian of drift, nx × nx)
σ(t) = g(x̂(t), u, d, t)                 (diffusion matrix, nx × nw)
```

F is computed by finite-difference perturbation of f by default; subclasses may override with an analytic Jacobian.  Forward Euler steps:

```
x̂(t + h) ← x̂(t) + h f(x̂(t), u, d, t)
P(t + h)  ← P(t)  + h [F P + P Fᵀ + σ σᵀ]
```

**Measurement update — linearised Joseph form:**

```
H      = ∂h/∂x |_{x = x̂[k|k−1], d}    (observation Jacobian, ny × nx)
e      = y − h(x̂[k|k−1], d)
R_e    = H P[k|k−1] Hᵀ + R
K      = P[k|k−1] Hᵀ R_e⁻¹
x̂[k]   = x̂[k|k−1] + K e
P[k]   = (I − KH) P[k|k−1] (I − KH)ᵀ + K R Kᵀ
```

H is also computed by finite-difference perturbation of h by default.

---

### 2.4 `ContinuousDiscreteUKF` *(stub)*

Unscented Kalman filter for a nonlinear `ContinuousDiscreteModel`.  Replaces the Jacobian linearisation of the CD-EKF with a deterministic sigma-point approximation.  Better captures nonlinearities up to third order in the Gaussian assumption.

**Sigma-point set — van der Merwe parameterisation:**

Given tuning parameters α (spread, typically 1×10⁻³), β = 2 (Gaussian optimum), κ = 0:

```
λ  = α² (nx + κ) − nx

W_m^0  = λ / (nx + λ)
W_c^0  = λ / (nx + λ)  +  (1 − α² + β)
W_m^i  = W_c^i = 1 / (2 (nx + λ))    for i = 1, …, 2 nx
```

Sigma points formed from current estimate (x̂, P):

```
χ_0      = x̂
χ_i      = x̂ + [√((nx + λ) P)]_i     for i = 1, …, nx
χ_{nx+i} = x̂ − [√((nx + λ) P)]_i     for i = 1, …, nx
```

where [√M]_i is the i-th column of the matrix square root of M (via Cholesky).

**Prediction — propagate sigma points through drift ODE:**

Each sigma point is integrated independently over [t_{k-1}, t_k]:

```
dχ_i/dt = f(χ_i, u, d, t)     for i = 0, …, 2 nx
```

using n_steps forward-Euler sub-steps.

Predicted mean and covariance:

```
x̂[k|k−1] = Σ_i W_m^i χ_i(t_k)
P[k|k−1]  = Σ_i W_c^i (χ_i − x̂)(χ_i − x̂)ᵀ  +  G Q_c Gᵀ · dt
```

The term G Q_c Gᵀ · dt adds the diffusion contribution (first-order approximation; exact when G is constant and Q_c is time-invariant).

**Measurement update — unscented transform:**

New sigma points are generated from (x̂[k|k−1], P[k|k−1]) and mapped through h:

```
γ_i = h(χ_i, d)    for i = 0, …, 2 nx
```

Predicted observation mean and innovation covariances:

```
ŷ     = Σ_i W_m^i γ_i
S_yy  = Σ_i W_c^i (γ_i − ŷ)(γ_i − ŷ)ᵀ  +  R
S_xy  = Σ_i W_c^i (χ_i − x̂)(γ_i − ŷ)ᵀ
```

Update:

```
K      = S_xy S_yy⁻¹
x̂[k]   = x̂[k|k−1] + K (y − ŷ)
P[k]   = P[k|k−1] − K S_yy Kᵀ
```

---

### 2.5 `ContinuousDiscreteEnKF` *(stub)*

Ensemble Kalman filter for a nonlinear `ContinuousDiscreteModel`.  Maintains N particles; replaces the analytical Gaussian assumption with a sample-based approximation.  No Jacobians required.

**Initialisation:**  Draw initial ensemble from N(x₀, P₀):

```
X^(i) ~ N(x₀, P₀)    for i = 1, …, N
```

**Prediction — Euler-Maruyama propagation of each particle:**

For each particle i, integrate the SDE over [t_{k-1}, t_k] with n_steps sub-steps of size h:

```
x^(i)(t+h) = x^(i)(t) + h f(x^(i)(t), u, d, t)
             + √h g(x^(i)(t), u, d, t) w_h^(i),   w_h^(i) ~ N(0, Q_c)
```

Predicted ensemble mean and sample covariance:

```
x̄     = (1/N) Σ_i x^(i)(t_k)
P_pred = (1/(N−1)) Σ_i (x^(i) − x̄)(x^(i) − x̄)ᵀ
```

**Measurement update — perturbed observations:**

Generate observation perturbations: ỹ^(i) = y + v^(i), v^(i) ~ N(0, R)

Compute ensemble-based innovation covariance and cross-covariance:

```
y^(i)_pred = h(x^(i), d)
ȳ          = (1/N) Σ_i y^(i)_pred

S_xx = (1/(N−1)) Σ_i (x^(i) − x̄)(x^(i) − x̄)ᵀ
S_xy = (1/(N−1)) Σ_i (x^(i) − x̄)(y^(i)_pred − ȳ)ᵀ
S_yy = (1/(N−1)) Σ_i (y^(i)_pred − ȳ)(y^(i)_pred − ȳ)ᵀ  +  R
```

Kalman gain and particle update:

```
K      = S_xy S_yy⁻¹
x^(i)  ← x^(i)  +  K (ỹ^(i) − h(x^(i), d))    for i = 1, …, N
```

Posterior mean and covariance computed from the updated ensemble.

---

### 2.6 `ContinuousDiscreteParticleFilter` *(stub)*

Sequential Monte Carlo estimator for a nonlinear `ContinuousDiscreteModel`.  Approximates the full non-Gaussian posterior distribution over the state.  Makes no Gaussian assumption; accurate for strongly nonlinear, multimodal distributions.

**Initialisation:**  Draw N particles from N(x₀, P₀) with uniform weights:

```
x^(i) ~ N(x₀, P₀),    w^(i) = 1/N    for i = 1, …, N
```

**Prediction — Euler-Maruyama propagation (identical to EnKF):**

```
x^(i)(t+h) = x^(i)(t) + h f(x^(i), u, d, t)
             + √h g(x^(i), u, d, t) w_h^(i)
```

Weights do not change during prediction.

**Measurement update — sequential importance reweighting:**

Update weights by the observation likelihood under a Gaussian noise model:

```
w̃^(i) = w^(i) · p(y | x^(i))

p(y | x^(i)) = |2π R|^{−½} exp(−½ (y − h(x^(i), d))ᵀ R⁻¹ (y − h(x^(i), d)))
```

Normalise:

```
w^(i) ← w̃^(i) / Σ_j w̃^(j)
```

**Resampling — systematic resampling:**

Compute effective sample size:

```
N_eff = 1 / Σ_i (w^(i))²
```

When N_eff < resample_threshold · N (default threshold = 0.5), draw N new particles from the discrete distribution defined by {x^(i), w^(i)} using systematic resampling and reset all weights to 1/N.

Systematic resampling draws positions U^(j) = (U + j − 1) / N, j = 1, …, N where U ~ Uniform(0, 1/N), then selects particles by the inverse CDF of the weight distribution.

**State estimate:**

```
x̂ = Σ_i w^(i) x^(i)
P  = Σ_i w^(i) (x^(i) − x̂)(x^(i) − x̂)ᵀ
```

---

### 2.7 `ContinuousDiscreteDAEEKF` *(stub)*

CD-EKF for a `ContinuousDiscreteDAEModel`.  Extends the CD-EKF to handle systems where the differential state x is coupled to an algebraic state z via the constraint l(x, z, u, d, t) = 0.

**Prediction — interleaved Euler/Newton integration:**

At each sub-step of size h:

1. **Euler step for x:**

   ```
   x̂_new = x̂ + h f(x̂, ẑ, u, d, t)
   ```

2. **Newton solve for z** (enforcing the algebraic constraint at t + h):

   ```
   Initialise:  ẑ_new = ẑ
   Iterate:     ẑ_new ← ẑ_new − (∂l/∂z)⁻¹ l(x̂_new, ẑ_new, u, d, t+h)
   ```

   until |l| < newton_tol (default 10⁻¹⁰) or newton_max_iter iterations.

3. **Riccati step** using the effective state Jacobian:

   The implicit function theorem on the constraint l = 0 gives:

   ```
   ∂z/∂x = −(∂l/∂z)⁻¹ (∂l/∂x)
   ```

   Effective state Jacobian of the differential dynamics:

   ```
   F_eff = ∂f/∂x  +  (∂f/∂z)(∂z/∂x)
         = ∂f/∂x  −  (∂f/∂z)(∂l/∂z)⁻¹(∂l/∂x)
   ```

   Riccati update:

   ```
   P_new = P + h [F_eff P + P F_eff ᵀ + G Q_c Gᵀ]
   ```

   All partial derivatives are computed by finite differences by default.

**Measurement update — EKF Joseph form:**

If h depends on both x and z:

```
H = ∂h/∂x  +  (∂h/∂z)(∂z/∂x)    (effective observation Jacobian)
```

Update equations identical to CD-EKF (Section 2.3).

---

## 3. Control

### 3.1 `OptimalControlProblem`

Finite-horizon quadratic programme for discrete-time MPC.  State and input costs are evaluated at each discrete time step.

**System:**

```
x[k+1] = A x[k] + B u[k] + E d[k],    y[k] = C x[k]
```

**Objective — tracking a per-output reference r over horizon N:**

```
J(U) = Σ_{k=0}^{N−1} [ ‖y[k+1] − r‖²_Q  +  ‖u[k]‖²_R  +  ‖Δu[k]‖²_S ]
       + ‖y[N] − r‖²_P
       + ρ Σ_{k=0}^{N−1} ‖ε[k+1]‖²
```

where:
- Q — output tracking weight (l×l)
- R — input magnitude weight (m×m)
- S — input rate-of-movement weight (m×m); Δu[k] = u[k] − u[k−1]
- P — terminal output weight (l×l); default = Q
- ρ — soft output constraint penalty scalar

**Constraints:**

```
u_min ≤ u[k] ≤ u_max                               (hard input box)
y_min − ε[k+1] ≤ y[k+1] ≤ y_max + ε[k+1]           (soft output box)
ε[k+1] ≥ 0
```

The output bounds are `[r − δ, r + δ]` where δ is the `y_offset` parameter.

**Batch (lifted) prediction:**

Stacking x[1], …, x[N] gives:

```
X = Ψ x₀ + Γ U + Λ D
Y = C̄ X
```

where Ψ ∈ ℝ^{Nn×n}, Γ ∈ ℝ^{Nn×Nm}, Λ ∈ ℝ^{Nn×Np} are the prediction matrices, and C̄ = blkdiag(C, …, C) ∈ ℝ^{Nl×Nn}.

The QP decision variable is z = [U; ε] ∈ ℝ^{N(m+l)}:

```
min_z   ½ zᵀ H z + fᵀ z
s.t.    G_qp z ≤ h_qp
```

Solved via `cvxopt.solvers.qp`.

---

### 3.2 `MPCController`

Discrete-time receding-horizon MPC composed of a `KalmanFilter` and `OptimalControlProblem`.

**Control loop at each measurement time t_k:**

```
1. Estimate:  x̂[k] ← KalmanFilter.update(y[k], d[k])
2. Optimise:  U*, X* ← OCP.solve(x̂[k], D, x_ref)
3. Apply:     u[k] = U*[0:m]           (receding horizon — first action only)
4. Record:    KalmanFilter.record_action(u[k])
```

---

### 3.3 `CDOptimalControlProblem`

Optimal control problem for a `LinearContinuousDiscreteModel`.  The key difference from Section 3.1 is that the stage cost is defined in **continuous time**: weights Q_c and R_c have units of (1/time), meaning they accumulate per unit time rather than per step.

**Continuous-time stage cost formulation:**

Over the interval [t_k, t_{k+1}], the stage cost integral with ZOH input u[k] is:

```
l_k = ∫_{t_k}^{t_{k+1}} [ ‖y(t) − r‖²_{Q_c}  +  ‖u[k] − u_r‖²_{R_c}
                           + ‖Δu[k]‖²_{S_c}    ] dt
```

For the state-dependent term, the exact result using the continuous state trajectory x(τ) = A_d(τ) x[k] + B_d(τ) u[k] + E_d(τ) d[k] (where A_d(τ) = expm(A_c τ)) is:

```
Q̃_xx = ∫₀^{dt} A_d(τ)ᵀ Cᵀ Q_c C A_d(τ) dτ       (exact state weight)
Q̃_xu = ∫₀^{dt} A_d(τ)ᵀ Cᵀ Q_c C B_d(τ) dτ       (exact cross weight)
Q̃_uu = ∫₀^{dt} B_d(τ)ᵀ Cᵀ Q_c C B_d(τ) dτ  +  R_c · dt   (exact input weight)
```

The input and rate terms integrate trivially because u[k] is piecewise constant:

```
R̃ = R_c · dt,    S̃ = S_c · dt
```

**Total objective over horizon N:**

```
J = Σ_{k=0}^{N−1} [ x[k]ᵀ Q̃_xx x[k]  +  (u[k]−u_r)ᵀ Q̃_uu (u[k]−u_r)
                    + Δu[k]ᵀ S̃ Δu[k] ]
    + ‖y[N] − r‖²_{P_f}
```

The terminal weight P_f is **not** scaled by dt — it penalises the terminal state at a single point in time rather than an integral.

**Practical approximation for constant Q_c:**

When dt is small relative to the dominant time constants of A_c:

```
Q̃_xx ≈ Cᵀ Q_c C · dt,    Q̃_xu ≈ 0
```

The current implementation uses this approximation, converting Q_c → Q_c · dt, R_c → R_c · dt, S_c → S_c · dt before passing to the underlying QP.  The exact integrals can replace this when higher accuracy is needed.

**Tuning implications:**

Using the same numerical value for Q in the discrete OCP (Section 3.1) and Q_c in the CD-OCP produces very different closed-loop behaviour.  For sampling-rate independent tuning, set Q_c = Q_d / dt and R_c = R_d / dt.  This ensures that the total cost over N steps is the same regardless of the chosen dt.

**Constraints and batch form:**  Identical to Section 3.1 (input/output box constraints, slack variables, QP via `cvxopt.solvers.qp`).

---

### 3.4 `CDMPCController`

Receding-horizon MPC composed of `CDKalmanFilter` and `CDOptimalControlProblem`.

**Control loop at each measurement time t_k:**

```
1. Estimate:  x̂[k] ← CDKalmanFilter.update(y[k], d[k])
2. Optimise:  U*, X* ← CDOptimalControlProblem.solve(x̂[k], D, x_ref)
3. Apply:     u[k] = U*[0:m]
4. Record:    CDKalmanFilter.record_action(u[k])
```

The estimator integrates continuous-time ODEs (no discretisation); the OCP uses ZOH matrices from `model.discretize(d)`.  This separation is correct: the filter should track the continuous-time system accurately, while the QP optimiser requires a finite-dimensional problem.

---

### 3.5 `EconomicNMPC` *(stub)*

Economic NMPC for a `ContinuousDiscreteModel`.  Minimises an economic stage cost l_e(x, u, d) rather than a quadratic tracking error.

**Continuous-time economic objective:**

```
min_{u(·)}  ∫_{t_0}^{t_N} l_e(x(t), u(t), d(t)) dt  +  V_f(x(t_N))
```

**Discretised via multiple shooting over N intervals:**

At each shooting node k, the state x[k] is a free decision variable.  For each interval [t_k, t_{k+1}] with ZOH input u[k]:

- **Stage cost** (Euler quadrature): `l_e(x[k], u[k], d[k]) · dt`
- **Continuity constraint**: `x[k+1] = f̄(x[k], u[k], d[k])` where f̄ integrates the mean dynamics (drift only, no noise) over dt via n_steps Euler sub-steps.

**Full NLP:**

```
min_{x[0],…,x[N], u[0],…,u[N−1]}
    Σ_{k=0}^{N−1} l_e(x[k], u[k], d[k]) · dt  +  V_f(x[N])

s.t.  x[k+1] = f̄(x[k], u[k], d[k])            k = 0, …, N−1
      x[0] = x_init                             (initial condition)
      u_min ≤ u[k] ≤ u_max                      (input box)
      g(x[k], u[k]) ≤ 0                         (path constraints, optional)
      g_T(x[N]) ≤ 0                              (terminal constraint, optional)
```

Solved via `scipy.optimize.minimize` with SLSQP (default).  Warm-starting: the previous optimal sequence shifted by one step is used as the initial guess.

**Closed-loop step:** `step(x0, d_trajectory, u_prev)` solves the NLP and returns only u[0] (receding-horizon principle).

---

## 4. System Identification

### 4.1 PED log-likelihood

Prediction-error decomposition (PED) negative log-likelihood for a `LinearDiscreteModel`.  Measures how well the model explains a recorded dataset via the one-step-ahead Kalman filter innovations.

Assumes C = I (state observed directly).  For each time step the Kalman filter produces:

```
ν_k  = y_k − x̂_k⁻                (one-step-ahead innovation)
S_k  = P_k⁻ + R                   (innovation covariance)
```

The PED negative log-likelihood is:

```
-log L(θ) = ½ Σ_k [ log|S_k| + ν_kᵀ S_k⁻¹ ν_k ]  +  const
```

This is the exact Gaussian likelihood under the model.  Maximising L(θ) over θ gives the maximum-likelihood estimate of the model parameters.

**Function signature:**

```python
ped_neg_log_likelihood(model_factory, theta, history, Q, R)
```

- `model_factory(θ)` → model instance (`LinearDiscreteModel`)
- `theta` — current parameter vector
- `history` — list of `{"y": (n,), "u": (m,), "d": (p,)}` dicts
- `Q`, `R` — process and measurement noise covariances (numpy)

Returns a scalar; `_INVALID_LIKELIHOOD = 1×10¹⁰` is returned on numerical failure.

---

### 4.2 `ParameterEstimator`

Wraps the PED likelihood in a Nelder-Mead optimiser to find the maximum-likelihood parameter vector:

```
θ* = argmin_θ [ -log L(θ) ]
```

**Method `fit(history)`:**  Runs Nelder-Mead starting from `model.params`.  Returns `θ*` and the achieved likelihood.  Uses the `_nelder_mead` implementation in `_utils.py` (no scipy dependency).

---

## 5. Realization

State-space realization algorithms construct a minimal `(A, B, C, D)` system from input-output data or frequency-domain representations.

---

### 5.1 `SISORealization` *(stub)*

Constructs a SISO state-space model from either a transfer function or sampled impulse/step response data.

**Realised system:**

```
x[k+1] = A x[k] + B u[k]
y[k]   = C x[k] + D u[k]
```

#### 5.1.1 From transfer function — `from_transfer_function(num, den, form)`

Given:

```
H(z) = (b_0 z^r + b_1 z^{r−1} + … + b_r)
       ─────────────────────────────────────
       (a_0 z^n + a_1 z^{n−1} + … + a_n)
```

Normalise a_0 = 1.  Zero-pad num to length n+1 if r < n.  Compute direct-term:

```
d_val = b_0 / a_0    (= 0 when r < n)
```

Corrected numerator coefficients (removing the contribution of d_val):

```
b̃_i = b_i − d_val · a_i    for i = 0, …, n
```

**Observable canonical form** (default):

```
        [ 0   0   …   0  −a_n  ]
        [ 1   0   …   0  −a_{n−1}]
A_obs = [ 0   1   …   0  −a_{n−2}]
        [ ⋮       ⋱   ⋮     ⋮    ]
        [ 0   0   …   1  −a_1  ]

B_obs = [b̃_n,  b̃_{n−1},  …,  b̃_1]ᵀ

C_obs = [0, 0, …, 0, 1]

D_obs = d_val
```

**Controllable canonical form:**

```
        [ 0    1    0   …   0  ]
        [ 0    0    1   …   0  ]
A_ctl = [ ⋮              ⋱   ⋮  ]
        [ 0    0    0   …   1  ]
        [−a_n −a_{n−1} … −a_1]

B_ctl = [0, 0, …, 0, 1]ᵀ

C_ctl = [b̃_n,  b̃_{n−1},  …,  b̃_1]

D_ctl = d_val
```

The two forms are related by a similarity transformation.

#### 5.1.2 From impulse response — `from_impulse_response(h, dt, n)`

Given sampled impulse response `h[0], h[1], …, h[T−1]`:

Form the block Hankel matrix (p rows, q cols, p·q ≥ n):

```
      [h[1]    h[2]    …  h[q]  ]
H   = [h[2]    h[3]    …  h[q+1]]
      [ ⋮               ⋱    ⋮   ]
      [h[p]  h[p+1]   …  h[p+q−1]]
```

Shifted Hankel (used to recover A):

```
      [h[2]    h[3]    …  h[q+1]]
H'  = [h[3]    h[4]    …  h[q+2]]
      [ ⋮               ⋱    ⋮   ]
      [h[p+1] h[p+2]  …  h[p+q] ]
```

Truncated SVD: H = U Σ Vᵀ; keep rank-n approximation:

```
O_n = U_n √Σ_n       (observability factor, p × n)
C_n = √Σ_n V_nᵀ      (controllability factor, n × q)
```

Recover system matrices:

```
C = O_n[0:1, :]             (first row of O_n)
B = C_n[:, 0:1]             (first column of C_n)
A = pinv(O_n) @ H' @ pinv(C_n)
D = h[0]
```

---

### 5.2 `MIMORealization` *(stub)*

Ho–Kalman realization from a sequence of Markov parameters (impulse-response matrices).

**Markov parameters:**

```
H[0] = D,    H[k] = C A^{k−1} B    for k ≥ 1
```

Shape of each H[k]: ny × nu.

**Algorithm — `from_markov_parameters(H, n)`:**

1. Form block Hankel matrices (p block rows, q block cols, p·q·min(ny,nu) ≥ n):

   ```
         [H[1]   H[2]   …   H[q]  ]
   Hankel = [H[2]   H[3]   …   H[q+1]]
         [ ⋮              ⋱     ⋮    ]
         [H[p]  H[p+1]  …  H[p+q−1]]
   ```

   Shifted Hankel H_shift: replace H[k] → H[k+1] throughout.

2. Truncated SVD: Hankel = U Σ Vᵀ; keep rank-n:

   ```
   O_n = U_n √Σ_n           (block observability factor, p·ny × n)
   C_n = √Σ_n V_nᵀ           (block controllability factor, n × q·nu)
   ```

3. Recover system matrices:

   ```
   C = O_n[0:ny, :]          (first ny rows)
   B = C_n[:, 0:nu]          (first nu columns)
   A = pinv(O_n) @ H_shift @ pinv(C_n)
   D = H[0]
   ```

---

## 6. Simulation

### 6.1 `SDESimulator` *(stub)*

Euler-Maruyama numerical integrator for a `ContinuousDiscreteModel`.  Advances the state stochastically from one measurement time to the next.

**Itô SDE being integrated:**

```
dx = f(x, u, d, t) dt + g(x, u, d, t) dw,    w ~ N(0, Q_c)
```

Over one measurement interval [t_k, t_{k+1}] with n_steps sub-steps of size h = dt / n_steps.

#### Explicit-Explicit (EE) scheme (default):

Drift and diffusion both evaluated at the beginning of the sub-step:

```
x_{j+1} = x_j + h f(x_j, u, d, t_j) + √h g(x_j, u, d, t_j) w_j

w_j ~ N(0, Q_c)
```

This is the standard Euler-Maruyama method.  First-order strong convergence (√h) and first-order weak convergence (h).

#### Implicit-Explicit (IE) scheme:

Drift evaluated implicitly at t_{j+1}; diffusion evaluated explicitly at t_j:

```
x_{j+1} = x_j + h f(x_{j+1}, u, d, t_{j+1}) + √h g(x_j, u, d, t_j) w_j
```

The implicit equation for x_{j+1} is solved by fixed-point or Newton iteration:

```
x^{(0)}_{j+1} = x_j    (initialise at current state)
x^{(s+1)}_{j+1} = x_j + h f(x^{(s)}_{j+1}, u, d, t_{j+1}) + √h g(x_j, u, d, t_j) w_j
```

Iterate until convergence.  Better stability for stiff systems.

**`step(x, u, d, t)`** → x at t + dt (one realisation).

**`simulate(x0, U, D, t0)`** → X of shape (T+1, nx); applies `step` T times.

---

### 6.2 `SDAESimulator` *(stub)*

Euler-Maruyama integrator for a `ContinuousDiscreteDAEModel`.  At each sub-step the algebraic constraint is enforced by Newton iteration.

**System being integrated:**

```
dx = f(x, z, u, d, t) dt + g(x, z, u, d, t) dw
0  = l(x, z, u, d, t)
```

**Sub-step procedure (EE scheme):**

1. Euler drift update for x:

   ```
   x_new = x + h f(x, z, u, d, t)
   ```

2. Newton solve for z consistent with x_new:

   ```
   z^{(0)} = z    (initialise at previous z)
   z^{(s+1)} = z^{(s)} − (∂l/∂z)⁻¹ l(x_new, z^{(s)}, u, d, t+h)
   ```

   until |l| < newton_tol or newton_max_iter iterations.

3. Add diffusion noise:

   ```
   x_new += √h g(x, z, u, d, t) w,    w ~ N(0, Q_c)
   ```

**`step(x, z, u, d, t)`** → (x_next, z_next) at t + dt.

**`simulate(x0, z0, U, D, t0)`** → (X, Z) with shapes (T+1, nx) and (T+1, nz).

---

## 7. Monte Carlo Simulation

### 7.1 `MonteCarloSimulation` *(stub)*

Closed-loop Monte Carlo framework for stochastic performance assessment under uncertainty.

**Purpose:**  Run N_mc independent closed-loop trials to estimate the distribution of state trajectories, outputs, inputs, and cost under different initial conditions and process noise realisations.

**Each trial i = 1, …, N_mc:**

```
1. Draw initial state:  x₀^(i) ~ N(x₀_mean, x₀_cov)

2. For k = 0, 1, …, T−1:
   a. Observe:   y_k^(i) = h(x_k^(i), d_k) + v_k,    v_k ~ N(0, R)
   b. Estimate:  x̂_k^(i) = estimator.step(y_k, u_{k−1}, d_k, t_k)
                 (or x̂ = x_true when estimator is None)
   c. Control:   u_k^(i) = controller.step(x̂_k, d_k, ...)
   d. Propagate: x_{k+1}^(i) = simulator.step(x_k^(i), u_k^(i), d_k, t_k)

3. Total cost:  J^(i) = Σ_k l(x_k^(i), u_k^(i), d_k) · dt
```

**Result:**  `MonteCarloResult` dataclass:

```
X      : (N_mc, T+1, nx)  — state trajectories
Y      : (N_mc, T, ny)    — noisy output trajectories
U      : (N_mc, T, nu)    — applied input trajectories
costs  : (N_mc,)          — total cost per trial
```

**Reproducibility:**  Trial i uses random seed `seed + i`, allowing independent trials with deterministic replay.

---

## 8. Dependencies

| Package | Purpose                                              |
|---------|------------------------------------------------------|
| numpy   | Arrays; internal ODE/SDE integration                |
| cvxopt  | QP solver (`solvers.qp`); Kalman filter Cholesky     |

No scipy dependency.  Matrix exponentials use `_expm` (eigendecomposition) in `_utils.py`.

---

## 9. Notation summary

| Symbol      | Meaning                                              |
|-------------|------------------------------------------------------|
| n, nx       | State dimension                                      |
| m, nu       | Input dimension                                      |
| p, nd       | Disturbance dimension                                |
| l, ny       | Output dimension                                     |
| q, nw       | Process noise dimension                              |
| nz          | Algebraic state dimension (SDAE only)                |
| N           | MPC / NMPC prediction horizon (steps)                |
| T           | Simulation horizon (steps)                           |
| N_mc        | Number of Monte Carlo trials                         |
| dt          | Sampling interval (s)                                |
| h           | ODE sub-step size = dt / n_steps                     |
| n_steps     | Number of sub-steps per sampling interval            |
| A_c, B_c, E_c | Continuous-time state-space matrices               |
| A_d, B_d, E_d | ZOH-discretised matrices                           |
| G           | Noise input matrix (maps process noise to state)     |
| Q_c         | Continuous process-noise covariance (q×q, per time)  |
| Q_d         | Discrete process-noise covariance (Van Loan)         |
| Q           | Discrete tracking weight or discrete Q_d             |
| R           | Measurement noise covariance                         |
| P           | State error covariance                               |
| K           | Kalman gain                                          |
| ν, e        | Kalman innovation (y − Cx̂)                          |
| F           | Jacobian ∂f/∂x (EKF linearisation)                  |
| H           | Jacobian ∂h/∂x (EKF measurement linearisation)      |
| χ_i, W_i    | UKF sigma points and weights                         |
| w^(i)       | Particle filter weights                              |
| N_eff       | Effective sample size = 1/Σ(w^(i))²                 |
| θ           | Parameter vector (system identification)             |
| l_e         | Economic stage cost                                  |
| V_f         | Terminal cost                                        |
