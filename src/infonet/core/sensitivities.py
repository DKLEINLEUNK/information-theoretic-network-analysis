"""Sensitivities shared between univariate and multivariate pipelines."""
from __future__ import annotations

import numpy as np

from infonet.core.base import Sensitivity


class Ordinal(Sensitivity):
    """Discretise into ordinal categories via equal-width bins.

    Operates on a 1-D series. The multivariate pipeline applies this
    column-wise, so each variable gets its own [min, max] partitioning.
    Constant series (lo == hi) collapse to a single bin (all zeros).
    """

    def __init__(self, n_bins: int = 5):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "Ordinal"

    @property
    def params(self) -> dict:
        return {"n_bins": self.n_bins}

    @property
    def identity_params(self) -> dict:
        # Continuous data = infinitely many bins
        return {"n_bins": np.inf}

    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        lo, hi = np.nanmin(y), np.nanmax(y)
        if lo == hi:
            return np.zeros_like(y, dtype=float)
        edges = np.linspace(lo, hi, self.n_bins + 1)
        return np.clip(np.digitize(y, edges[1:-1]), 0, self.n_bins - 1).astype(float)