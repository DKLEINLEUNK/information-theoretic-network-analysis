"""Multivariate data-generating processes and the stable A-matrix sampler."""
from __future__ import annotations

import numpy as np

from infonet.multivariate.base import MVGenerator


def var1_matrix_spectrum(
    n: int,
    max_radius: float = 0.95,
    upper_scale: float = 0.5,
    get_info: bool = False,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
):
    """Generate a stable real-valued VAR(1) coefficient matrix A with purely real eigenvalues."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0 < max_radius < 1):
        raise ValueError("max_radius must satisfy 0 < max_radius < 1")

    if rng is None:
        rng = np.random.default_rng(seed)

    eigenvalues = rng.uniform(-max_radius, max_radius, size=n)

    S = np.zeros((n, n), dtype=float)
    np.fill_diagonal(S, eigenvalues)

    if n >= 2:
        iu = np.triu_indices(n, k=1)
        S[iu] = rng.normal(loc=0.0, scale=upper_scale, size=iu[0].size)

    perm = rng.permutation(n)
    A = S[np.ix_(perm, perm)]

    eigvals_A = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigvals_A)))
    is_stable = spectral_radius < 1.0 - 1e-12

    if get_info:
        info = {
            "S": S,
            "perm": perm,
            "eigenvalues": np.array(eigenvalues, dtype=float),
            "spectral_radius": spectral_radius,
            "is_stable": bool(is_stable),
        }
        return A, info
    return A


class VAR1(MVGenerator):
    """Vector AR(1):  y_t = A @ y_{t-1} + eps_t,  eps_t ~ N(theta, sigma^2 I)."""

    def __init__(
        self,
        A: np.ndarray | None = None,
        sigma: float = 1.0,
        max_radius: float = 0.95,
        A_upper_scale: float = 0.5,
    ):
        self._A_fixed = A
        self.sigma = sigma
        self.max_radius = max_radius
        self.A_upper_scale = A_upper_scale
        self._A_used: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "VAR1"

    @property
    def params(self) -> dict:
        return {
            "sigma": self.sigma,
            "max_radius": self.max_radius,
            "A_upper_scale": self.A_upper_scale,
        }

    @property
    def A_matrices(self) -> dict[str, np.ndarray]:
        return {"A": self._A_used}

    def __call__(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        if self._A_fixed is not None:
            A = self._A_fixed
            if A.shape != (d, d):
                raise ValueError(
                    f"Fixed A has shape {A.shape} but d={d} was requested"
                )
        else:
            A = var1_matrix_spectrum(
                n=d,
                max_radius=self.max_radius,
                upper_scale=self.A_upper_scale,
                get_info=False,
                rng=rng,
            )
        self._A_used = A

        burn_in = d * 1000
        total = burn_in + n

        y = np.empty((total, d))
        y[0] = rng.normal(0, self.sigma, size=d)
        for t in range(1, total):
            y[t] = A @ y[t - 1] + rng.normal(0, self.sigma, size=d)

        return y[burn_in:]


class NVAR(MVGenerator):
    """Nonlinear VAR with logistic-map dynamics."""

    def __init__(
        self,
        r: float = 4.0,
        sigma: float = 0.1,
        boundary: str = "mod",
        max_radius: float = 0.95,
        A_upper_scale: float = 0.5,
    ):
        if not 0.0 <= r <= 4.0:
            raise ValueError("r must be in [0, 4]")
        if boundary not in ("mod", "reflect"):
            raise ValueError("boundary must be 'mod' or 'reflect'")
        self.r = r
        self.sigma = sigma
        self.boundary = boundary
        self.max_radius = max_radius
        self.A_upper_scale = A_upper_scale
        self._A_used: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"NVAR_r{self.r}"

    @property
    def params(self) -> dict:
        return {
            "r": self.r,
            "sigma": self.sigma,
            "boundary": self.boundary,
            "max_radius": self.max_radius,
            "A_upper_scale": self.A_upper_scale,
        }

    @property
    def A_matrices(self) -> dict[str, np.ndarray]:
        return {"A": self._A_used}

    def __call__(self, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
        A = var1_matrix_spectrum(
            n=d,
            max_radius=self.max_radius,
            upper_scale=self.A_upper_scale,
            get_info=False,
            rng=rng,
        )
        self._A_used = A

        constrain = self._mod if self.boundary == "mod" else self._reflect

        burn_in = d * 1000
        total = burn_in + n

        y = np.empty((total, d))
        y[0] = rng.uniform(0.0, 1.0, size=d)

        for t in range(1, total):
            u = A @ y[t - 1]
            v = self.r * u * (1.0 - u)
            eps = rng.normal(0, self.sigma, size=d)
            y[t] = constrain(v + eps)

        return y[burn_in:]

    @staticmethod
    def _mod(x):
        return x % 1.0

    @staticmethod
    def _reflect(x):
        x = np.abs(x)
        n_floor = np.floor(x).astype(int)
        frac = x - n_floor
        return np.where(n_floor % 2 == 1, 1.0 - frac, frac)
