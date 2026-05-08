"""Joint heatmaps over (N, n_bins): coefficient of variation and convergence rate."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from infonet.plotting.labels import (
    annotate_heatmap_cells,
    apply_col_header,
)
from infonet.plotting.style import (
    AIS_LEGEND_LABELS,
    FONT,
)
from infonet.plotting.univariate._common import standard_column_order



def _build_pivot(
    sub: pd.DataFrame,
    *,
    n_grid: list[int],
    n_bins_grid: list[int],
    statistic: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (n_bins, n) value pivot and matching valid-count pivot."""
    if statistic == "cv":
        pivot = (
            sub.groupby(["n_bins", "n"])
            .apply(lambda g: g["estimate"].std() / g["true_ais"].iloc[0])
            .unstack("n")
            .reindex(index=n_bins_grid, columns=n_grid)
        )
    elif statistic == "rel_bias":
        pivot = (
            sub.groupby(["n_bins", "n"])
            .apply(lambda g: (g["estimate"] - g["true_ais"].iloc[0]).median()
                             / g["true_ais"].iloc[0])
            .unstack("n")
            .reindex(index=n_bins_grid, columns=n_grid)
        )
    elif statistic == "converge_rate":
        pivot = (
            sub.groupby(["n_bins", "n"])["estimate"]
            .apply(lambda x: x.notna().mean())
            .unstack("n")
            .reindex(index=n_bins_grid, columns=n_grid)
            .fillna(0)
        )
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    count_pivot = (
        sub.groupby(["n_bins", "n"])["estimate"]
        .apply(lambda x: x.notna().sum())
        .unstack("n")
        .reindex(index=n_bins_grid, columns=n_grid)
        .fillna(0)
    )
    return pivot, count_pivot


def _draw_panel(
    ax,
    values: np.ndarray,
    n_grid: list[int],
    n_bins_grid: list[int],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    annotate: bool,
    annot_kwargs: dict,
):
    im = ax.imshow(
        values,
        aspect="equal", origin="lower",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )

    if annotate:
        annotate_heatmap_cells(
            ax, values,
            vmin=vmin, vmax=vmax, cmap=cmap,
            **annot_kwargs,
        )

    bad_rows, bad_cols = np.where(np.isnan(values))
    ax.scatter(bad_cols, bad_rows,
               marker="x", color="red", s=40, linewidths=1.5)

    ax.set_xticks(np.arange(len(n_grid)))
    ax.set_xticklabels(n_grid, rotation=45, ha="right",
                       fontsize=FONT["annot_n"])
    ax.set_yticks(np.arange(len(n_bins_grid)))
    ax.set_yticklabels(n_bins_grid, fontsize=FONT["annot_n"])
    return im


def _plot_grid(
    df: pd.DataFrame,
    *,
    statistic: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    cbar_pct: bool,
    estimators: list[str] | None,
    annotate: bool,
    annot_kwargs: dict,
    min_valid_frac: float,
    savepath: str | Path | None,
    show: bool,
) -> None:
    if estimators is None:
        estimators = sorted(df["estimator"].unique())
    col_order = standard_column_order(df)
    n_grid = sorted(df["n"].unique())
    n_bins_grid = sorted(df["n_bins"].unique())

    total_reps = df["rep"].nunique()
    min_valid = max(1, int(total_reps * min_valid_frac))

    fig, axes = plt.subplots(
        len(estimators), len(col_order),
        figsize=(4 * len(col_order), 4 * len(estimators)),
        squeeze=False,
    )

    im = None
    for row, est_name in enumerate(estimators):
        for col, gen_name in enumerate(col_order):
            ax = axes[row, col]
            sub = df[(df["generator"] == gen_name)
                     & (df["estimator"] == est_name)]

            pivot, count_pivot = _build_pivot(
                sub, n_grid=n_grid, n_bins_grid=n_bins_grid,
                statistic=statistic,
            )
            insufficient = count_pivot.values < min_valid
            pivot_masked = pivot.mask(
                pd.DataFrame(insufficient,
                             index=pivot.index, columns=pivot.columns)
            )

            im = _draw_panel(
                ax, pivot_masked.values,
                n_grid, n_bins_grid,
                cmap=cmap, vmin=vmin, vmax=vmax,
                annotate=annotate, annot_kwargs=annot_kwargs,
            )

            if row == len(estimators) - 1:
                ax.set_xlabel(r"$T$", fontsize=FONT["tick"])
            if col == 0:
                ax.set_ylabel(r"$B$", fontsize=FONT["tick"])
                ax.text(
                    -0.35, 0.5, AIS_LEGEND_LABELS.get(est_name, est_name),
                    transform=ax.transAxes,
                    fontsize=FONT["subtitle"], fontweight="bold",
                    rotation=90, ha="center", va="center",
                )
            if row == 0:
                apply_col_header(ax, gen_name)

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.ax.tick_params(labelsize=FONT["annot_n"])
    cbar.set_label(cbar_label, fontsize=FONT["tick"])
    if cbar_pct:
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_rel_bias_grid(
    df: pd.DataFrame,
    *,
    estimators: list[str] | None = None,
    vlim: float = 0.5,
    annotate: bool = True,
    min_valid_frac: float = 0.75,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap of median relative bias ((est - true) / true) over (N by n_bins)."""
    _plot_grid(
        df,
        statistic="rel_bias",
        cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        cbar_label="Median relative bias",
        cbar_pct=True,
        estimators=estimators,
        annotate=annotate,
        annot_kwargs=dict(
            fmt="{:+.0f}", fontsize=6, scale=100.0, suffix="%",
            luminance_threshold=0.35,
        ),
        min_valid_frac=min_valid_frac,
        savepath=savepath, show=show,
    )

def plot_cv_grid(
    df: pd.DataFrame,
    *,
    estimators: list[str] | None = None,
    vmax: float = 0.5,
    annotate: bool = True,
    min_valid_frac: float = 0.75,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap of coefficient of variation (SD / true_ais) over (N, n_bins)."""
    _plot_grid(
        df,
        statistic="cv",
        cmap="viridis", vmin=0.0, vmax=vmax,
        cbar_label="Coefficient of Variation",
        cbar_pct=True,
        estimators=estimators,
        annotate=annotate,
        annot_kwargs=dict(
            fmt="{:.0f}", fontsize=6, scale=100.0,
            luminance_threshold=0.25,
        ),
        min_valid_frac=min_valid_frac,
        savepath=savepath, show=show,
    )


def plot_convergence_grid(
    df: pd.DataFrame,
    *,
    estimators: list[str] | None = None,
    annotate: bool = True,
    min_valid_frac: float = 0.0,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Heatmap of convergence rate (fraction of non-NaN replicates)."""
    has_failures = (
        df.groupby("estimator")["estimate"]
        .apply(lambda x: x.isna().any())
    )
    candidate = sorted(df["estimator"].unique()) if estimators is None else estimators
    failing = [e for e in candidate if has_failures.get(e, False)]
    if not failing:
        print("All estimators converged in all cells — skipping convergence plot.")
        return

    _plot_grid(
        df,
        statistic="converge_rate",
        cmap="RdYlGn", vmin=0.0, vmax=1.0,
        cbar_label="Convergence rate",
        cbar_pct=True,
        estimators=failing,
        annotate=annotate,
        annot_kwargs=dict(
            fmt="{:.0f}", fontsize=6, scale=100.0,
            luminance_threshold=1.0, yellow_aware=True,
        ),
        min_valid_frac=min_valid_frac,
        savepath=savepath, show=show,
    )
