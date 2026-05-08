"""Shared abstractions used by both univariate and multivariate pipelines."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Sensitivity(ABC):
    """A post-generation transformation applied to a series."""

    @abstractmethod
    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> dict: ...

    @property
    @abstractmethod
    def identity_params(self) -> dict:
        """Parameter values that represent 'no transformation applied'."""
        ...
