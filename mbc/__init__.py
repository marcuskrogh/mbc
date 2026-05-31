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
      * ``ContinuousDiscreteKalmanFilter``  – KF for linear continuous-discrete systems.
      * ``ContinuousDiscreteEKF``           – CD-EKF (Ph.D. Ch. 7.1).
      * ``ContinuousDiscreteUKF``           – CD-UKF (Ph.D. Ch. 7.2).
      * ``ContinuousDiscreteEnKF``          – CD-EnKF (Ph.D. Ch. 7.3).
      * ``ContinuousDiscreteParticleFilter``– CD-PF  (Ph.D. Ch. 7.4).
      * ``ContinuousDiscreteDAEEKF``        – CD-EKF for SDAE (Ph.D. Ch. 8).

  mbc.ocp
      Optimal Control Problems:

      * ``DiscreteLinearOCP``             – receding-horizon QP (tracking MPC, discrete).
      * ``ContinuousLinearOCP``           – receding-horizon QP for linear CD systems.
      * ``ContinuousNonlinearOCP``        – economic nonlinear OCP (Ph.D. Ch. 9).

  mbc.control
      Model Predictive Control:

      * ``MPCController``                 – KalmanFilter + DiscreteLinearOCP.
      * ``CDMPCController``               – ContinuousDiscreteKalmanFilter + ContinuousLinearOCP.
      * ``CDLinearizedMPCController``     – successive-linearisation MPC.
      * ``CDNMPCController``              – generic estimator + OCP controller.
      * ``EconomicOptimalControlProblem`` – alias for ContinuousNonlinearOCP.

  mbc.identification
      System-identification / parameter-estimation utilities.

  mbc.realization
      State-space realization from I/O data (M.Sc. Ch. 2–4).

  mbc.simulation
      Numerical integration of SDE/SDAE models (Ph.D. Ch. 5–6).

  mbc.monte_carlo
      Closed-loop Monte Carlo simulation framework (Ph.D. Ch. 12).
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
    KalmanFilter,
    ContinuousDiscreteKalmanFilter,
    ContinuousDiscreteEKF,
    ContinuousDiscreteUKF,
    ContinuousDiscreteEnKF,
    ContinuousDiscreteParticleFilter,
    ContinuousDiscreteDAEEKF,
)
from .ocp import (
    OCP,
    DiscreteLinearOCPBase,
    DiscreteLinearisedOCPBase,
    ContinuousLinearOCPBase,
    ContinuousLinearisedOCPBase,
    ContinuousNonlinearOCPBase,
    DiscreteLinearOCP,
    DiscreteLinearisedOCP,
    ContinuousLinearOCP,
    ContinuousLinearisedOCP,
    ContinuousNonlinearOCP,
    NLPConstraint,
    NLPProblem,
    NLPScalingPolicy,
    NLPSolverBackend,
    ScipyNLPBackend,
    IpoptNLPBackend,
    QPProblem,
    QPResult,
    QPSolverBackend,
    HighsQPBackend,
    OSQPBackend,
    make_qp_backend,
)
from .control import (
    MPCController,
    CDMPCController,
    CDLinearizedMPCController,
    linearize_cd_model,
    discretize_cd_linearization,
    EconomicOptimalControlProblem,
    CDNMPCController,
    # Legacy control aliases
    OptimalControlProblem,
    CDOptimalControlProblem,
    CDTrackingOptimalControlProblem,
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
    # Estimation
    "KalmanFilter",
    "ContinuousDiscreteKalmanFilter",
    "ContinuousDiscreteEKF",
    "ContinuousDiscreteUKF",
    "ContinuousDiscreteEnKF",
    "ContinuousDiscreteParticleFilter",
    "ContinuousDiscreteDAEEKF",
    # OCP abstract bases
    "OCP",
    "DiscreteLinearOCPBase",
    "DiscreteLinearisedOCPBase",
    "ContinuousLinearOCPBase",
    "ContinuousLinearisedOCPBase",
    "ContinuousNonlinearOCPBase",
    # OCP concrete classes
    "DiscreteLinearOCP",
    "DiscreteLinearisedOCP",
    "ContinuousLinearOCP",
    "ContinuousLinearisedOCP",
    "ContinuousNonlinearOCP",
    # NLP solver
    "NLPConstraint",
    "NLPProblem",
    "NLPScalingPolicy",
    "NLPSolverBackend",
    "ScipyNLPBackend",
    "IpoptNLPBackend",
    # QP solver
    "QPProblem",
    "QPResult",
    "QPSolverBackend",
    "HighsQPBackend",
    "OSQPBackend",
    "make_qp_backend",
    # MPC controllers
    "MPCController",
    "CDMPCController",
    "CDLinearizedMPCController",
    "linearize_cd_model",
    "discretize_cd_linearization",
    "EconomicOptimalControlProblem",
    "CDNMPCController",
    # Legacy control aliases
    "OptimalControlProblem",
    "CDOptimalControlProblem",
    "CDTrackingOptimalControlProblem",
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
