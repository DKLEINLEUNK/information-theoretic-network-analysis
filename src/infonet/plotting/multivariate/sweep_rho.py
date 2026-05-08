"""Spearman rho (edge calibration) sweeps over n and n_bins (rank-recovery view)"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from infonet.plotting.labels import (
    bins_label,
    bins_sort_key,
    col_title,
    generator_sort_key,
    ordered_bins,
    push_legend_outside,
    row_label,
)
from infonet.plotting.metrics import compute_rho_per_rep
from infonet.plotting.style import (
    ESTIMATOR_LEGEND_LABELS,
    ESTIMATOR_PALETTE,
    FONT,
)


def plot_n_only(
    df: pd.DataFrame,
    n_bins: int | float = np.inf,
    savepath: str | None = None,
) -> None:
    """Spearman ρ as a function of n at a fixed ordinality level."""
    rho_df = compute_rho_per_rep(df)
    rho_df = rho_df[rho_df["sens.n_bins"] == n_bins]
    if rho_df.empty:
        avail = sorted(
            compute_rho_per_rep(df)["sens.n_bins"].unique(),
            key=bins_sort_key,
        )
        raise ValueError(f"No rows found for n_bins={n_bins}. Available: {avail}")

    generators = sorted(rho_df["generator"].unique(), key=generator_sort_key)
    n_gen = len(generators)
    has_multi_est = rho_df["estimator"].nunique() > 1
    estimators = (list(ESTIMATOR_PALETTE.keys()) if has_multi_est
                  else [rho_df["estimator"].iloc[0]])

    fig, axes = plt.subplots(
        1, n_gen, figsize=(5.0 * n_gen, 4.5),
        sharey=True, squeeze=False,
    )

    for col, gen in enumerate(generators):
        ax = axes[0, col]
        sub = rho_df[rho_df["generator"] == gen].copy()
        sub["n_str"] = sub["n"].astype(str)

        sns.boxplot(
            data=sub, x="n_str", y="rho",
            hue="estimator" if has_multi_est else None,
            hue_order=estimators if has_multi_est else None,
            palette=ESTIMATOR_PALETTE if has_multi_est else None,
            order=[str(n) for n in sorted(sub["n"].unique())],
            ax=ax,
            linewidth=0.8,
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
            legend=False,
        )

        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)
        header, params = col_title(gen)
        ax.set_title(f"{header} {params}",
                     fontsize=FONT["subtitle"], fontweight="bold")
        ax.set_xlabel("T", fontsize=FONT["axis"])
        ax.set_ylabel("Spearman ρ" if col == 0 else "", fontsize=FONT["axis"])
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    if has_multi_est:
        push_legend_outside(axes[0, -1], ESTIMATOR_LEGEND_LABELS)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ord_only(
    df: pd.DataFrame,
    n: int | None = None,
    savepath: str | None = None,
) -> None:
    """Grouped boxplot of Spearman ρ by n_bins at a fixed n."""
    rho_df = compute_rho_per_rep(df)

    if n is None:
        n = int(np.max(rho_df["n"]))
    rho_df = rho_df[rho_df["n"] == n].copy()

    if rho_df.empty:
        avail = sorted(compute_rho_per_rep(df)["n"].unique())
        raise ValueError(f"No rows found for n={n}. Available: {avail}")

    _, bins_labels = ordered_bins(rho_df)
    rho_df["bins_label"] = rho_df["sens.n_bins"].map(bins_label)
    rho_df["bins_label"] = pd.Categorical(
        rho_df["bins_label"], categories=bins_labels, ordered=True,
    )

    generators = sorted(rho_df["generator"].unique(), key=generator_sort_key)
    n_gen = len(generators)
    has_multi_est = rho_df["estimator"].nunique() > 1

    fig, axes = plt.subplots(
        1, n_gen, figsize=(5.0 * n_gen, 4.5),
        sharey=True, squeeze=False,
    )

    for col, gen in enumerate(generators):
        ax = axes[0, col]
        sub = rho_df[rho_df["generator"] == gen]

        sns.boxplot(
            data=sub, x="bins_label", y="rho",
            hue="estimator" if has_multi_est else None,
            hue_order=list(ESTIMATOR_PALETTE.keys()) if has_multi_est else None,
            palette=ESTIMATOR_PALETTE if has_multi_est else None,
            ax=ax,
            legend=(col == n_gen - 1) and has_multi_est,
            width=0.6 if not has_multi_est else 0.8,
            whis=(5, 95),
            capprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.2},
            boxprops={"linewidth": 1.0},
            medianprops={"linewidth": 1.5, "color": "black"},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )

        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="k", lw=0.8, alpha=0.5)

        header, params = col_title(gen)
        ax.set_title(f"{header} {params}",
                     fontsize=FONT["subtitle"], fontweight="bold")
        ax.set_xlabel("B", fontsize=FONT["axis"])
        ax.set_ylabel("Spearman ρ" if col == 0 else "", fontsize=FONT["axis"])
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    if has_multi_est:
        push_legend_outside(axes[0, -1], ESTIMATOR_LEGEND_LABELS)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_joint(
    df: pd.DataFrame,
    savepath: str | None = None,
    annotate: bool = False,
    min_valid_frac: float = 0.75,
) -> None:
    """Heatmap of median Spearman ρ across n × n_bins, one panel per
    (estimator, generator) cell."""
    rho_df = compute_rho_per_rep(df)
    if rho_df.empty:
        raise ValueError("No rows found.")

    _, bins_labels = ordered_bins(rho_df)
    rho_df["bins_label"] = rho_df["sens.n_bins"].map(bins_label)

    n_levels = sorted(rho_df["n"].unique())
    estimators = rho_df["estimator"].unique().tolist()
    generators = sorted(rho_df["generator"].unique(), key=generator_sort_key)
    n_est = len(estimators)
    n_gen = len(generators)

    total_reps = rho_df["rep"].nunique()
    min_valid = max(1, int(total_reps * min_valid_frac))

    fig, axes = plt.subplots(
        n_est, n_gen,
        figsize=(4 * n_gen, 4 * n_est),
        squeeze=False,
    )

    im = None
    for row, est_name in enumerate(estimators):
        for col, gen_name in enumerate(generators):
            ax = axes[row, col]
            cell = rho_df[
                (rho_df["estimator"] == est_name) &
                (rho_df["generator"] == gen_name)
            ]

            pivot = (
                cell.groupby(["bins_label", "n"])["rho"]
                .median().unstack("n")
                .reindex(index=bins_labels, columns=n_levels)
            )
            count_pivot = (
                cell.groupby(["bins_label", "n"])["rho"]
                .apply(lambda x: x.notna().sum()).unstack("n")
                .reindex(index=bins_labels, columns=n_levels).fillna(0)
            )

            insufficient = count_pivot.values < min_valid
            pivot_masked = pivot.mask(
                pd.DataFrame(insufficient, index=pivot.index,
                             columns=pivot.columns)
            )

            im = ax.imshow(
                pivot_masked.values,
                aspect="equal", origin="lower",
                cmap="RdYlGn", vmin=0.0, vmax=1.0,
            )

            if annotate:
                for i in range(pivot_masked.shape[0]):
                    for j in range(pivot_masked.shape[1]):
                        val = pivot_masked.values[i, j]
                        if np.isnan(val):
                            continue
                        dist_from_center = abs(val - 0.5)
                        tc = "white" if dist_from_center > 0.30 else "black"
                        ax.text(j, i, f"{val:.2f}",
                                ha="center", va="center",
                                color=tc, fontsize=8)

            bad_rows, bad_cols = np.where(np.isnan(pivot_masked.values))
            ax.scatter(bad_cols, bad_rows,
                       marker="x", color="red", s=40, linewidths=1.5)

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right",
                               fontsize=10)
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=10)

            if row == n_est - 1:
                ax.set_xlabel("T", fontsize=11)
            if col == 0:
                ax.set_ylabel("Number of bins", fontsize=11)
                ax.text(
                    -0.35, 0.5, row_label(est_name),
                    transform=ax.transAxes,
                    fontsize=14, fontweight="bold",
                    rotation=90, ha="center", va="center",
                )
            if row == 0:
                header, params = col_title(gen_name)
                ax.set_title(params, fontsize=12)
                ax.annotate(
                    header,
                    xy=(0.5, 1.125), xycoords="axes fraction",
                    fontsize=14, fontweight="bold",
                    ha="center", va="bottom",
                )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Median Spearman rho", fontsize=14)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)