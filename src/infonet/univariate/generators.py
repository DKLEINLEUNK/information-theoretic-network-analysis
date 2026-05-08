"""Univariate data-generating processes."""
from __future__ import annotations

import numpy as np

from infonet.univariate.base import Generator


class AR1(Generator):
    def __init__(self, phi: float = 0.5, sigma: float = 1.0):
        self.phi = phi
        self.sigma = sigma

    @property
    def name(self) -> str:
        return "AR1"

    @property
    def params(self) -> dict:
        return {"phi": self.phi, "sigma": self.sigma}

    def __call__(self, n: int, rng: np.random.Generator) -> np.ndarray:
        burnin = 10_000
        y = np.empty(n + burnin)
        y[0] = rng.normal(0, self.sigma)
        for t in range(1, n + burnin):
            y[t] = self.phi * y[t - 1] + rng.normal(0, self.sigma)
        return y[burnin:]


class NAR(Generator):
    """Nonlinear AR(1) with logistic map and Irwin-Hall noise."""

    def __init__(
        self,
        r: float = 3.8,
        sigma: float = 0.02,
        K: int = 40,
        half_w: float = 1 / 40,
        divergence_threshold: float = 100.0,
    ):
        if not 0.0 <= r <= 4.0:
            raise ValueError("r must be in [0, 4] for the logistic map")
        self.r = r
        self.sigma = sigma
        self.K = K
        self.half_w = half_w
        self.divergence_threshold = divergence_threshold

    @property
    def name(self) -> str:
        return "NAR"

    @property
    def params(self) -> dict:
        return {"r": self.r, "sigma": self.sigma, "K": self.K, "half_w": self.half_w}

    def __call__(self, n: int, rng: np.random.Generator) -> np.ndarray:
        burnin = 10_000
        y = np.empty(n + burnin)
        y[0] = rng.uniform(0.01, 0.99)
        for t in range(1, n + burnin):
            eps = np.sum(rng.uniform(-self.half_w, self.half_w, size=self.K))
            det = self.r * y[t - 1] * (1.0 - y[t - 1])
            y[t] = det + self.sigma * eps
            if abs(y[t]) > self.divergence_threshold:
                y[t] = np.nan
        return y[burnin:]

    @staticmethod
    def irwin_hall_sample(rng: np.random.Generator, K: int, half_w: float) -> float:
        """Draw one sample from the Irwin-Hall noise distribution."""
        return float(np.sum(rng.uniform(-half_w, half_w, size=K)))