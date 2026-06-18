"""
Abstract base classes for optimal control problems.

OCPs are named by optimisation *time domain* only:

* **Discrete-time** — finite-horizon QP over a discrete prediction model.
* **Continuous-time** — NLP with continuous-time dynamics discretised inside
  the solver (direct simultaneous transcription).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OptimalControlProblem(ABC):
    """
    Common abstract base for all optimal control problems.

    Every concrete OCP must expose the prediction horizon ``N`` and the input
    dimension ``nu`` so that closed-loop controller wrappers can query them
    without knowing the concrete OCP type.
    """

    @property
    @abstractmethod
    def N(self) -> int:
        """Prediction horizon (number of control intervals)."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Input dimension nᵘ."""


class DiscreteOptimalControlProblem(OptimalControlProblem):
    """
    Abstract base for discrete-time optimal control problems.

    Concrete subclasses solve a finite-horizon QP (or equivalent convex program)
    over a discrete prediction model.
    """


class ContinuousOptimalControlProblem(OptimalControlProblem):
    """
    Abstract base for continuous-time optimal control problems.

    Concrete subclasses formulate a finite-horizon NLP whose dynamics are
    discretised internally (e.g. direct simultaneous transcription).
    """
