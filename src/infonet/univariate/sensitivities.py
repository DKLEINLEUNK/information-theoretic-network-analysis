"""Univariate-only stationarity sensitivities."""
from __future__ import annotations

import numpy as np

from infonet.core.base import Sensitivity
from infonet.univariate.generators import NAR


class Missingness(Sensitivity):
    """Replace a fraction of observations with NaN (MCAR)."""

    def __init__(self, frac: float = 0.1):
        if not 0.0 < frac < 1.0:
            raise ValueError("frac must be in (0, 1)")
        self.frac = frac

    @property
    def name(self) -> str:
        return "Missingness"

    @property
    def params(self) -> dict:
        return {"frac": self.frac}

    @property
    def identity_params(self) -> dict:
        return {"frac": 0.0}

    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = y.copy()
        mask = rng.random(len(out)) < self.frac
        out[mask] = np.nan
        return out


class MeanShift(Sensitivity):
    """Inject a step change in the mean at the midpoint.

        y_t -> y_t + delta  for t > T/2

    """

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    @property
    def name(self) -> str:
        return "MeanShift"

    @property
    def params(self) -> dict:
        return {"delta": self.delta}

    @property
    def identity_params(self) -> dict:
        return {"delta": 0.0}

    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = y.copy()
        mid = len(out) // 2
        out[mid:] += self.delta
        return out


class VarianceChange(Sensitivity):
    """Resimulate the second half with a different noise std dev.

    The first half is kept intact. The second half is regenerated from
    y[mid-1] using the *same* dependence parameter but a different sigma,
    so only the innovation variance changes.
    """

    def __init__(
        self,
        process: str = "ar1",
        factor: float = 2.0,
        sigma_original: float = 1.0,
        phi: float = 0.5,
        r: float = 3.8,
        K: int = 40,
        half_w: float = 1 / 40,
    ):
        if process not in ("ar1", "nar"):
            raise ValueError("process must be 'ar1' or 'nar'")
        if factor <= 0:
            raise ValueError("factor must be positive")
        self.process = process
        self.factor = factor
        self.sigma_original = sigma_original
        self.phi = phi
        self.r = r
        self.K = K
        self.half_w = half_w

    @property
    def name(self) -> str:
        return "VarianceChange"

    @property
    def params(self) -> dict:
        return {"factor": self.factor}

    @property
    def identity_params(self) -> dict:
        return {"factor": 1.0}

    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = y.copy()
        mid = len(out) // 2
        sigma_new = self.sigma_original * self.factor

        if self.process == "ar1":
            for t in range(mid, len(out)):
                out[t] = self.phi * out[t - 1] + rng.normal(0, sigma_new)
        elif self.process == "nar":
            for t in range(mid, len(out)):
                eps = NAR.irwin_hall_sample(rng, self.K, self.half_w)
                det = self.r * out[t - 1] * (1.0 - out[t - 1])
                out[t] = det + sigma_new * eps

        return out


class DependenceChange(Sensitivity):
    """Shift dependence structure at the midpoint by resimulating with
    an altered parameter (phi for AR1, r for NAR).
    """

    def __init__(
        self,
        process: str = "ar1",
        shift: float = 0.2,
        sigma: float = 1.0,
        phi_original: float = 0.5,
        r_original: float = 3.8,
        K: int = 40,
        half_w: float = 1 / 40,
    ):
        if process not in ("ar1", "nar"):
            raise ValueError("process must be 'ar1' or 'nar'")
        self.process = process
        self.shift = shift
        self.sigma = sigma
        self.phi_original = phi_original
        self.r_original = r_original
        self.K = K
        self.half_w = half_w

    @property
    def name(self) -> str:
        return "DependenceChange"

    @property
    def params(self) -> dict:
        return {"shift": self.shift}

    @property
    def identity_params(self) -> dict:
        return {"shift": 0.0}

    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = y.copy()
        mid = len(out) // 2

        if self.process == "ar1":
            phi_new = np.clip(self.phi_original + self.shift, -0.99, 0.99)
            for t in range(mid, len(out)):
                out[t] = phi_new * out[t - 1] + rng.normal(0, self.sigma)
        elif self.process == "nar":
            r_new = np.clip(self.r_original + self.shift, 0.0, 4.0)
            for t in range(mid, len(out)):
                eps = NAR.irwin_hall_sample(rng, self.K, self.half_w)
                det = r_new * out[t - 1] * (1.0 - out[t - 1])
                out[t] = det + self.sigma * eps

        return out