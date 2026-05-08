"""ABCs for univariate (1,D) pipeline components."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Generator(ABC):
    """Base class for univariate data-generating processes."""

    @abstractmethod
    def __call__(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def params(self) -> dict: ...


class Estimator(ABC):
    """Maps a 1-D series to a scalar estimate."""

    @abstractmethod
    def __call__(self, y: np.ndarray) -> float: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def params(self) -> dict:
        """Override in subclasses with tuneable parameters."""
        return {}