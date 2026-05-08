"""ABCs for multivariate (n, D) pipeline components."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MVGenerator(ABC):
    """Base class for multivariate data-generating processes."""

    @abstractmethod
    def __call__(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        """Return an (n, d) array."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> dict: ...

    @property
    def A_matrices(self) -> dict[str, np.ndarray]:
        """Return the A matrix/matrices used in the last call."""
        raise NotImplementedError


class MVEstimator(ABC):
    """Maps an (n, d) series to an estimate."""

    @abstractmethod
    def __call__(self, y: np.ndarray) -> np.ndarray | float | dict: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def params(self) -> dict:
        return {}

    @property
    def is_pairwise(self) -> bool:
        """True if the estimate is a (d, d) matrix (source to target)."""
        return False