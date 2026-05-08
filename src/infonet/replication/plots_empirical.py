"""Empirical replication plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from infonet.replication.data import PHI_VALUES


JITTER_WIDTH = 0.05

FS = {
    "axis":   20,
    "tick":   18,
    "legend": 14,
    "ylabel_bias": 26,
    "ylabel_phi":  20,
}


def _median_relative_bias_per_phi(
    df: pd.DataFrame,
    phi_col: str = "AR_Parameter",
    bias_col: str = "bias",
    phis: list[float] = PHI_VALUES,
) -> list[float]:
    """Median of |bias| / phi for each phi. Used by both median-bias plots."""
    return [
        float(np.median(df[df[phi_col] == p][bias_col].abs() / p))
        for p in phis
    ]


def _set_phi_xaxis(ax, phis: list[float] = PHI_VALUES) -> None:
    """The phi x-axis is shared across all four empirical plots."""
    ax.set_xlim(-0.05, 0.95)
    ax.set_xticks(phis)
    ax.set_xlabel(r"$\phi$", fontsize=FS["axis"])
    ax.tick_params(axis="both", labelsize=FS["tick"])


def _draw_median_curve(ax, phis: list[float], medians: list[float]) -> None:
    """The red-X-with-line motif used by figures 3 and 4."""
    ax.scatter(
        phis, medians,
        s=150, marker="X", color="red",
        edgecolor="black", zorder=10,
    )
    ax.plot(
        phis, medians,
        color="red", linewidth=4, zorder=7,
    )
    ax.axhline(
        y=0, color="gray", linestyle="--", linewidth=2,
        zorder=5, alpha=0.6,
    )


def plot_original_estimates(
    df: pd.DataFrame,
    *,
    phi_col: str = "AR_Parameter",
    estimate_col: str = "Estimate_rescaled",
    seed: int = 42,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Scatter rescaled estimates per phi, with horizontal jitter."""
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for phi in PHI_VALUES:
        sub = df[df[phi_col] == phi]
        x = phi + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(sub))
        ax.scatter(
            x, sub[estimate_col],
            s=50, alpha=0.20, color="C0",
            edgecolors="none", zorder=1,
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    _set_phi_xaxis(ax)
    ax.set_ylabel(
        r"$\widehat{\phi}$",
        fontsize=FS["ylabel_phi"], rotation=0, labelpad=15,
    )

    sns.despine(ax=ax)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_corrected_estimates(
    df: pd.DataFrame,
    *,
    phi_col: str = "dtar_true",
    estimate_col: str = "dtar_true_hat",
    converged_col: str = "converged",
    seed: int = 42,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Re-ran scatter, with non-converged fits flagged as red triangles."""
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for phi in PHI_VALUES:
        sub = df[df[phi_col] == phi]
        sub_conv = sub[sub[converged_col] == 1]
        sub_nconv = sub[sub[converged_col] == 0]

        if len(sub_conv) > 0:
            x = phi + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH,
                                  size=len(sub_conv))
            ax.scatter(
                x, sub_conv[estimate_col],
                s=50, alpha=0.20, color="C0",
                edgecolors="none", zorder=1,
            )

        if len(sub_nconv) > 0:
            x = phi + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH,
                                  size=len(sub_nconv))
            ax.scatter(
                x, sub_nconv[estimate_col],
                s=100, alpha=0.35, color="red",
                edgecolors="none", marker="v", zorder=2,
            )

    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    _set_phi_xaxis(ax)
    ax.set_ylabel(
        r"$\widehat{\phi}$",
        fontsize=FS["ylabel_phi"], rotation=0, labelpad=15,
    )
    sns.despine(ax=ax)

    legend_handle = Line2D(
        [], [],
        marker="v", color="red", linestyle="None",
        markersize=10, alpha=0.6, label="No convergence",
    )
    ax.legend(handles=[legend_handle], loc="upper right",
              fontsize=FS["legend"], frameon=True)

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_median_relative_bias(
    df: pd.DataFrame,
    *,
    phi_col: str = "AR_Parameter",
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Standalone red-X-with-line median curve."""
    medians = _median_relative_bias_per_phi(df, phi_col=phi_col)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.scatterplot(
        x=PHI_VALUES, y=medians,
        s=300, marker="X", color="red", edgecolor="black",
        ax=ax, zorder=15,
    )
    sns.lineplot(
        x=PHI_VALUES, y=medians,
        color="red", ax=ax, zorder=10, linewidth=4,
    )
    ax.axhline(y=0, color="gray", zorder=1,
               linestyle="--", linewidth=4)

    ax.set_ylim(-2, 18)
    ax.set_yticks([0, 5, 10, 15])
    _set_phi_xaxis(ax)
    ax.set_ylabel(
        r"$\frac{\left|\widehat{\phi} - \phi \right|}{\phi}$",
        fontsize=28, rotation=0, labelpad=40,
    )

    sns.despine(ax=ax)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_relative_bias_with_scatter(
    df: pd.DataFrame,
    *,
    phi_col: str = "AR_Parameter",
    relative_bias_col: str = "bias_relative",
    seed: int = 42,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Combined: per-rep scatter of relative bias plus the median curve."""
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    for phi in PHI_VALUES:
        sub = df[df[phi_col] == phi]
        x = phi + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(sub))
        ax.scatter(
            x, sub[relative_bias_col],
            s=50, alpha=0.15, color="C0",
            edgecolors="none", zorder=1,
        )

    medians = _median_relative_bias_per_phi(df, phi_col=phi_col)
    _draw_median_curve(ax, PHI_VALUES, medians)

    ax.set_yticks([0, 5, 10, 15])
    _set_phi_xaxis(ax)
    ax.set_ylabel(
        r"$\frac{\left|\widehat{\phi} - \phi \right|}{\phi}$",
        fontsize=FS["ylabel_bias"], rotation=0, labelpad=35,
    )

    legend_handle = Line2D(
        [], [],
        marker="X", color="red", markeredgecolor="black",
        linestyle="None", markersize=10, label="Median",
    )
    ax.legend(handles=[legend_handle], loc="upper right",
              fontsize=FS["legend"], frameon=True)

    sns.despine(ax=ax)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)