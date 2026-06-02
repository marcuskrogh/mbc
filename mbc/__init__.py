"""
mbc – model-based control library.

Provides generic, reusable algorithms for linear and nonlinear
model-based control, estimation, identification, and simulation.

  mbc.models
      Abstract model interfaces:

      * ``DiscreteLinearSDE``              – linear discrete-time system (ZOH).
      * ``DiscreteLinearisedSDE``          – linearised discrete-time system
                                             with deviation-variable formulation.
      * ``ContinuousDiscreteSDE``          – nonlinear SDE (Ph.D. Ch. 5).
      * ``ContinuousDiscreteLinearSDE``    – linear continuous-discrete SDE
                                             (M.Sc. Ch. 5).
      * ``ContinuousDiscreteLinearisedSDE`` – linearised continuous-discrete SDE
                                              with deviation-variable formulation.
      * ``ContinuousDiscreteSDAE``         – nonlinear SDAE (Ph.D. Ch. 6).

  mbc.estimation
      State-estimation algorithms:

      * ``KalmanFilter``                    – discrete-time KF (Joseph form).
        Supports noise-separated covariance (M.Sc. Ch. 5.4) and missing
        observations (M.Sc. Ch. 5.5).
      * ``ContinuousDiscreteKalmanFilter``                  – KF for linear continuous-discrete
                                              systems; ZOH + Van Loan Q_d (M.Sc. Ch. 5).
      * ``ContinuousDiscreteEKF``           – CD-EKF (Ph.D. Ch. 7.1).
      * ``ContinuousDiscreteUKF``           – CD-UKF (Ph.D. Ch. 7.2).
      * ``ContinuousDiscreteEnKF``          – CD-EnKF (Ph.D. Ch. 7.3).
      * ``ContinuousDiscreteParticleFilter``– CD-PF  (Ph.D. Ch. 7.4).
      * ``ContinuousDiscreteDAEEKF``        – CD-EKF for SDAE (Ph.D. Ch. 8).

  mbc.control
      Optimal control algorithms:

      * ``OptimalControlProblem``         – receding-horizon QP (tracking MPC, discrete).
      * ``MPCController``                 – KalmanFilter + OptimalControlProblem.
      * ``CDOptimalControlProblem``       – receding-horizon QP for linear CD systems.
      * ``CDMPCController``               – ContinuousDiscreteKalmanFilter + CDOptimalControlProblem.
      * ``CDTrackingOptimalControlProblem`` – nonlinear tracking OCP for CD systems
                                             (NLP; input/state/output constraints,
                                             ROM penalty + constraints, linear input
                                             penalty).
      * ``EconomicOptimalControlProblem`` – economic nonlinear OCP (Ph.D. Ch. 9).
                                           Accepts Mayer + Lagrange functions and
                                           the same constraint set as the tracking OCP.
      * ``CDNMPCController``              – generic estimator + OCP controller
                                           (works with any CD estimator and any OCP).

  mbc.identification
      System-identification / parameter-estimation utilities:

      * ``ped_neg_log_likelihood``          – PED Kalman log-likelihood (linear discrete).
      * ``ped_neg_log_likelihood_gradient`` – finite-difference gradient.
      * ``ParameterEstimator``              – multi-start optimiser (linear discrete).
      * ``cd_ped_neg_log_likelihood``       – CD-EKF PED log-likelihood (nonlinear CD).
      * ``cd_ped_neg_log_likelihood_gradient`` – finite-difference gradient (nonlinear CD).
      * ``CDParameterEstimator``            – multi-start optimiser (nonlinear CD).
      * ``EstimationResult``                – result dataclass.

  mbc.realization
      State-space realization from I/O data (M.Sc. Ch. 2–4):

      * ``SISORealization`` – from transfer function or impulse response.
      * ``MIMORealization`` – Ho–Kalman from Markov parameters.

  mbc.simulation
      Numerical integration of SDE/SDAE models (Ph.D. Ch. 5–6):

      * ``ContinuousDiscreteSDESimulator``  – Euler-Maruyama for SDE systems.
      * ``ContinuousDiscreteSDAESimulator`` – Euler-Maruyama for SDAE systems.

  mbc.monte_carlo
      Closed-loop Monte Carlo simulation framework (Ph.D. Ch. 12):

      * ``MonteCarloSimulation`` – N_mc independent closed-loop trials.
      * ``MonteCarloResult``     – results container.
"""

from .models import (
    DiscreteLinearSDE,
    DiscreteLinearisedSDE,
    ContinuousDiscreteSDE,
    ContinuousDiscreteLinearSDE,
    ContinuousDiscreteLinearisedSDE,
    ContinuousDiscreteSDAE,
)
from .estimation import (
    IntegrationScheme,
    EstimatorParams,
    DiscreteEstimator,
    ContinuousDiscreteEstimator,
    ContinuousDiscreteDAEEstimator,
    DiscreteLinearKFParams,
    ContinuousDiscreteLinearKFParams,
    ContinuousDiscreteEKFParams,
    ContinuousDiscreteUKFParams,
    ContinuousDiscreteEnKFParams,
    ContinuousDiscretePFParams,
    ContinuousDiscreteDAEEKFParams,
    DiscreteLinearKF,
    ContinuousDiscreteLinearKF,
    ContinuousDiscreteEKF,
    ContinuousDiscreteUKF,
    ContinuousDiscreteEnKF,
    ContinuousDiscretePF,
    ContinuousDiscreteDAEEKF,
    DelayedObservationFilter,
)
from .control import (
    OptimalControlProblem,
    MPCController,
    CDOptimalControlProblem,
    CDTrackingOptimalControlProblem,
    CDMPCController,
    EconomicOptimalControlProblem,
    CDNMPCController,
)
from .identification.estimator import ParameterEstimator, CDParameterEstimator, EstimationResult
from .identification.likelihood import (
    ped_neg_log_likelihood,
    ped_neg_log_likelihood_gradient,
    cd_ped_neg_log_likelihood,
    cd_ped_neg_log_likelihood_gradient,
)
from .realization import SISORealization, MIMORealization
from .simulation import (
    SimulatorParams,
    ContinuousDiscreteSimulator,
    ContinuousDiscreteDAESimulator,
    ContinuousDiscreteSDESimulatorParams,
    ContinuousDiscreteSDAESimulatorParams,
    ContinuousDiscreteSDESimulator,
    ContinuousDiscreteSDAESimulator,
    SDESimulator,
    SDAESimulator,
)
from .monte_carlo import MonteCarloSimulation, MonteCarloResult

__all__ = [
    # Models
    "DiscreteLinearSDE",
    "DiscreteLinearisedSDE",
    "ContinuousDiscreteSDE",
    "ContinuousDiscreteLinearSDE",
    "ContinuousDiscreteLinearisedSDE",
    "ContinuousDiscreteSDAE",
    # Estimation — integration scheme
    "IntegrationScheme",
    # Estimation — abstract bases
    "EstimatorParams",
    "DiscreteEstimator",
    "ContinuousDiscreteEstimator",
    "ContinuousDiscreteDAEEstimator",
    # Estimation — parameter structures
    "DiscreteLinearKFParams",
    "ContinuousDiscreteLinearKFParams",
    "ContinuousDiscreteEKFParams",
    "ContinuousDiscreteUKFParams",
    "ContinuousDiscreteEnKFParams",
    "ContinuousDiscretePFParams",
    "ContinuousDiscreteDAEEKFParams",
    # Estimation — estimators
    "DiscreteLinearKF",
    "ContinuousDiscreteLinearKF",
    "ContinuousDiscreteEKF",
    "ContinuousDiscreteUKF",
    "ContinuousDiscreteEnKF",
    "ContinuousDiscretePF",
    "ContinuousDiscreteDAEEKF",
    "DelayedObservationFilter",
    # Control
    "OptimalControlProblem",
    "MPCController",
    "CDOptimalControlProblem",
    "CDTrackingOptimalControlProblem",
    "CDMPCController",
    "EconomicOptimalControlProblem",
    "CDNMPCController",
    # Identification
    "ParameterEstimator",
    "CDParameterEstimator",
    "EstimationResult",
    "ped_neg_log_likelihood",
    "ped_neg_log_likelihood_gradient",
    "cd_ped_neg_log_likelihood",
    "cd_ped_neg_log_likelihood_gradient",
    # Realization
    "SISORealization",
    "MIMORealization",
    # Simulation — abstract bases
    "SimulatorParams",
    "ContinuousDiscreteSimulator",
    "ContinuousDiscreteDAESimulator",
    # Simulation — parameter structures
    "ContinuousDiscreteSDESimulatorParams",
    "ContinuousDiscreteSDAESimulatorParams",
    # Simulation — simulators
    "ContinuousDiscreteSDESimulator",
    "ContinuousDiscreteSDAESimulator",
    # Simulation — backward-compatible aliases
    "SDESimulator",
    "SDAESimulator",
    # Monte Carlo
    "MonteCarloSimulation",
    "MonteCarloResult",
]
