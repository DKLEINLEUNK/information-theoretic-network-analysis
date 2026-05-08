"""Analytical AIS ground truth for univariate generators.

- AR(1):  AIS = -0.5 * ln(1 - phi^2)   (closed form)
- NAR:    AIS = h[X_t] - h[noise]      (numerical decomposition)
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def ais_true_ar1(phi: float) -> float:
    """Analytical AIS for a stationary Gaussian AR(1) process."""
    return -0.5 * np.log(1.0 - phi ** 2)


def _noise_density(sigma, K=40, half_w=1 / 40, n_grid=8192):
    """Density of the Irwin-Hall noise: sigma * sum_k U(-half_w, half_w)."""
    from scipy.signal import fftconvolve

    support = sigma * K * half_w
    pad = 0.05 * max(support, 0.001)
    x = np.linspace(-support - pad, support + pad, n_grid)
    dx = x[1] - x[0]
    hw = sigma * half_w
    if hw < 1e-15:
        f = np.zeros(n_grid)
        f[n_grid // 2] = 1.0 / dx
        return x, f
    f = np.where(np.abs(x) <= hw + dx / 2, 1.0 / (2 * hw), 0.0)
    f /= np.sum(f) * dx
    density = f.copy()
    for _ in range(K - 1):
        density = fftconvolve(density, f, mode="same") * dx
    density = np.maximum(density, 0)
    density /= np.sum(density) * dx
    return x, density


@lru_cache(maxsize=32)
def _noise_density_cached(sigma: float, K: int, half_w: float, n_grid: int):
    """Cached noise density. Returns tuples (numpy arrays are wrapped)."""
    x, f = _noise_density(sigma, K, half_w, n_grid)
    return x, f


def _diff_entropy(x, f):
    """Differential entropy of a density f on grid x."""
    dx = x[1] - x[0]
    mask = f > 1e-30
    return -np.sum(f[mask] * np.log(f[mask])) * dx


def _stationary_density(r, noise_x, noise_f, n_grid=2000, n_iter=350):
    """Stationary density of the stochastic logistic map via Chapman-Kolmogorov."""
    from scipy.interpolate import interp1d

    noise_lo, noise_hi = noise_x[0], noise_x[-1]
    x_lo = min(0.0, noise_lo) - 0.15
    x_hi = max(r / 4.0 + noise_hi, 1.0) + 0.15
    x = np.linspace(x_lo, x_hi, n_grid)
    dx = x[1] - x[0]
    noise_interp = interp1d(
        noise_x, noise_f, kind="linear", bounds_error=False, fill_value=0.0
    )
    g = r * x * (1.0 - x)
    shifts = x[:, None] - g[None, :]
    P = noise_interp(shifts)
    P = np.maximum(P, 0.0)
    col_sums = np.sum(P, axis=0) * dx
    col_sums[col_sums < 1e-30] = 1.0
    P /= col_sums[None, :]
    rho = np.where((x >= 0) & (x <= 1), 1.0, 0.0)
    rho /= max(np.sum(rho) * dx, 1e-30)
    for _ in range(n_iter):
        rho_new = P @ (rho * dx)
        total = np.sum(rho_new) * dx
        if total > 1e-15:
            rho_new /= total
        rho = rho_new
    return x, rho


def ais_true_nar(r, sigma=0.02, K=40, half_w=1 / 40):
    """Analytical AIS for the stochastic logistic map via numerical decomposition."""
    x_noise, f_noise = _noise_density_cached(sigma, K, half_w, 16384) # = 2^14
    h_noise = _diff_entropy(x_noise, f_noise)
    x_s, rho_s = _stationary_density(r, x_noise, f_noise)
    h_X = _diff_entropy(x_s, rho_s)
    return h_X - h_noise