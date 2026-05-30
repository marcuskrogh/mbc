"""
Abstract base class for all optimal control problems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OCP(ABC):
    """
    Abstract base class for all optimal control problems.

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
