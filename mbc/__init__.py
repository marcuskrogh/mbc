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

      * ``DiscreteLinearOCP``             – receding-horizon QP for discrete-time linear systems.
      * ``DiscreteLinearisedOCP``         – QP for linearised discrete-time systems with SS coordinate shifting.
      * ``ContinuousLinearOCP``           – receding-horizon QP for linear continuous-discrete systems.
      * ``ContinuousOCP``                 – economic/tracking NLP for nonlinear CD systems.
      * ``ContinuousLinearisedOCP``       – QP for linearised CD systems with SS coordinate shifting.
      * ``MPCController``    – DiscreteLinearKF + DiscreteLinearOCP.
      * ``CDMPCController``  – ContinuousDiscreteLinearKF + ContinuousLinearOCP.
      * ``CDNMPCController`` – generic estimator + OCP controller.

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

      * ``SDESimulator``  – Euler-Maruyama for SDE systems.
      * ``SDAESimulator`` – Euler-Maruyama for SDAE systems.

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
    OCP,
    DiscreteLinearOCP,
    DiscreteLinearisedOCP,
    ContinuousLinearOCP,
    ContinuousOCP,
    ContinuousLinearisedOCP,
    MPCController,
    CDMPCController,
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
from .simulation import SDESimulator, SDAESimulator
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
    # Control — abstract base
    "OCP",
    # Control — OCP canonical names
    "DiscreteLinearOCP",
    "DiscreteLinearisedOCP",
    "ContinuousLinearOCP",
    "ContinuousOCP",
    "ContinuousLinearisedOCP",
    # Control — MPC controllers
    "MPCController",
    "CDMPCController",
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
    # Simulation
    "SDESimulator",
    "SDAESimulator",
    # Monte Carlo
    "MonteCarloSimulation",
    "MonteCarloResult",
]
