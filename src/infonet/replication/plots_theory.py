"""Theoretical CV-of-theta admissibility-bounds figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from infonet.replication.data import PHI_VALUES


ALPHA_DEFAULT = 0.05
DELTA_DEFAULT = 0.50
N_DEFAULT = 100

PANEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

H_BATRA = 672


def _cv_from_u(u: float, N: int) -> float:
    """CV(thetâ) as a function of u = thetah and sample size N."""
    if u < 1e-12:
        return float(np.sqrt(2 / N))
    return float(np.sqrt(np.expm1(2 * u) / (N * u ** 2)))


def _lhs(u: float, N: int) -> float:
    """LHS of the admissibility equation: N · u² / (e^{2u} − 1)."""
    if u < 1e-12:
        return N / 2
    return N * u ** 2 / np.expm1(2 * u)


def _solve_u_bounds(N: int, rhs: float) -> tuple[float, float, float]:
    """Solve `_lhs(u, N) == rhs` and return (u_lo, u_opt, u_hi)."""
    u_grid = np.linspace(0.001, 15, 100_000)
    lhs_grid = np.array([_lhs(u, N) for u in u_grid])
    u_opt = float(u_grid[np.argmax(lhs_grid)])

    u_lo = brentq(lambda u: _lhs(u, N) - rhs, 1e-6, u_opt)
    u_hi = brentq(lambda u: _lhs(u, N) - rhs, u_opt, 15)
    return float(u_lo), u_opt, float(u_hi)


def _plot_one_panel(
    ax,
    phi: float,
    color: str,
    *,
    u_lo: float,
    u_opt: float,
    u_hi: float,
    cv_threshold: float,
    N: int,
) -> None:
    theta = -np.log(phi) / 24
    h_lo = u_lo / theta
    h_hi = u_hi / theta
    h_opt = u_opt / theta

    h_min = h_lo * 0.15
    h_max = h_hi * 2.5
    h_grid = np.linspace(h_min, h_max, 2000)
    cv_grid = np.array([_cv_from_u(theta * h, N) for h in h_grid])

    ax.plot(h_grid, cv_grid, color=color, linewidth=2)

    ax.axhline(y=cv_threshold, color="gray", linestyle="--", linewidth=0.8)

    ax.axvspan(h_lo, h_hi, alpha=0.12, color=color)
    ax.axvline(x=h_lo, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(x=h_hi, color=color, linestyle=":", linewidth=0.8, alpha=0.5)

    cv_opt = _cv_from_u(theta * h_opt, N)
    ax.plot(
        h_opt, cv_opt,
        marker="*", color=color, markersize=14,
        markeredgecolor="black", markeredgewidth=0.5, zorder=5,
    )

    ax.text(h_lo, cv_threshold * 5.2, rf"$h_{{lo}}={h_lo:.1f}$",
            fontsize=8, ha="center", color=color)
    ax.text(h_hi, cv_threshold * 5.2, rf"$h_{{hi}}={h_hi:.1f}$",
            fontsize=8, ha="center", color=color)
    ax.text(h_opt, cv_opt - cv_threshold * 0.6, rf"$h^*={h_opt:.1f}$",
            fontsize=8, ha="center", color=color)

    ax.annotate(
        rf"$h={H_BATRA}$",
        xy=(h_max * 0.92, cv_threshold * 4.5),
        fontsize=9, color="red", ha="right",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        xytext=(h_max * 0.75, cv_threshold * 3.5),
    )

    ax.set_title(rf"$\phi = {phi}$", fontsize=13)
    ax.set_xlabel(r"$h$", fontsize=12)
    ax.set_ylim(0, cv_threshold * 6)
    ax.set_xlim(h_min, h_max)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_sampling_bounds(
    *,
    alpha: float = ALPHA_DEFAULT,
    delta: float = DELTA_DEFAULT,
    N: int = N_DEFAULT,
    phis: list[float] = PHI_VALUES,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Four-panel figure: CV(thetâ) vs h, with admissible region shaded."""
    z_alpha = norm.ppf(1 - alpha / 2)
    rhs = (z_alpha / delta) ** 2
    cv_threshold = delta / z_alpha

    u_lo, u_opt, u_hi = _solve_u_bounds(N, rhs)

    fig, axes = plt.subplots(1, len(phis),
                             figsize=(4 * len(phis), 4.5),
                             sharey=True)
    if len(phis) == 1:
        axes = [axes]

    for ax, phi, color in zip(axes, phis, PANEL_COLORS):
        _plot_one_panel(
            ax, phi, color,
            u_lo=u_lo, u_opt=u_opt, u_hi=u_hi,
            cv_threshold=cv_threshold, N=N,
        )

    axes[0].set_ylabel(
        r"$\mathrm{cv}(\hat{\theta})$",
        fontsize=13, rotation=0, labelpad=30,
    )

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)