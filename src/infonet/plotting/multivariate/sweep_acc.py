"""Edge detection accuracy sweeps over n and n_bins (TPR/FPR)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

from infonet.plotting.labels import (
    apply_col_header,
    apply_row_label,
    bins_label,
    bins_sort_key,
    generator_sort_key,
    ordered_bins,
    push_legend_outside,
)
from infonet.plotting.metrics import compute_metrics_per_rep
from infonet.plotting.style import (
    ESTIMATOR_LEGEND_LABELS,
    ESTIMATOR_PALETTE,
    FONT,
    text_color,
)


_METRICS = [
    ("sensitivity", "Sensitivity\n(TPR)"),
    ("fpr",         "False Positive\nRate (FPR)"),
]


def plot_n_only_metrics(
    df: pd.DataFrame,
    n_bins: int | float = np.inf,
    savepath: str | None = None,
) -> None:
    """Sensitivity and FPR as functions of n at a fixed ordinality level."""
    met_df = compute_metrics_per_rep(df)
    met_df = met_df[met_df["sens.n_bins"] == n_bins]

    if met_df.empty:
        avail = sorted(
            compute_metrics_per_rep(df)["sens.n_bins"].unique(),
            key=bins_sort_key,
        )
        raise ValueError(f"No rows found for n_bins={n_bins}. Available: {avail}")

    generators = sorted(met_df["generator"].unique(), key=generator_sort_key)
    n_gen = len(generators)
    has_multi_est = met_df["estimator"].nunique() > 1
    estimators = (list(ESTIMATOR_PALETTE.keys()) if has_multi_est
                  else [met_df["estimator"].iloc[0]])

    n_rows = len(_METRICS)
    fig, axes = plt.subplots(
        n_rows, n_gen,
        figsize=(5.0 * n_gen, 4.0 * n_rows),
        sharey="row", sharex=False, squeeze=False,
    )

    for row, (metric, ylabel) in enumerate(_METRICS):
        for col, gen in enumerate(generators):
            ax = axes[row, col]
            sub = met_df[met_df["generator"] == gen].copy()
            sub["n_str"] = sub["n"].astype(str)
            n_order = [str(n) for n in sorted(sub["n"].unique())]

            sns.boxplot(
                data=sub, x="n_str", y=metric,
                hue="estimator" if has_multi_est else None,
                hue_order=estimators if has_multi_est else None,
                palette=ESTIMATOR_PALETTE if has_multi_est else None,
                order=n_order, ax=ax,
                linewidth=0.8,
                flierprops=dict(marker="o", markersize=2, alpha=0.4),
                legend=False,
            )

            if metric == "sensitivity":
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(-0.01, 1.05)
                ax.axhline(0.10, color="red", ls="--", lw=0.8, alpha=0.5)

            if row == 0:
                apply_col_header(ax, gen)

            if row == n_rows - 1:
                ax.set_xlabel("n (samples)", fontsize=FONT["axis"])
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            ax.set_ylabel(ylabel if col == 0 else "", fontsize=FONT["axis"])
            ax.tick_params(labelsize=FONT["annot_n"])
            ax.grid(True, alpha=0.3, axis="y")

    if has_multi_est:
        push_legend_outside(axes[0, -1], ESTIMATOR_LEGEND_LABELS)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ord_only_metrics(
    df: pd.DataFrame,
    n: int | None = None,
    savepath: str | None = None,
) -> None:
    """Grouped boxplots of sensitivity and FPR by n_bins at a fixed n."""
    met_df = compute_metrics_per_rep(df)

    if n is None:
        n = int(np.max(met_df["n"]))
    met_df = met_df[met_df["n"] == n].copy()

    if met_df.empty:
        avail = sorted(compute_metrics_per_rep(df)["n"].unique())
        raise ValueError(f"No rows found for n={n}. Available: {avail}")

    _, bins_labels = ordered_bins(met_df)
    met_df["bins_label"] = met_df["sens.n_bins"].map(bins_label)
    met_df["bins_label"] = pd.Categorical(
        met_df["bins_label"], categories=bins_labels, ordered=True,
    )

    generators = sorted(met_df["generator"].unique(), key=generator_sort_key)
    n_gen = len(generators)
    has_multi_est = met_df["estimator"].nunique() > 1

    n_rows = len(_METRICS)
    fig, axes = plt.subplots(
        n_rows, n_gen,
        figsize=(5.0 * n_gen, 4.0 * n_rows),
        sharey="row", sharex=True, squeeze=False,
    )

    for row, (metric, ylabel) in enumerate(_METRICS):
        for col, gen in enumerate(generators):
            ax = axes[row, col]
            sub = met_df[met_df["generator"] == gen]

            sns.boxplot(
                data=sub, x="bins_label", y=metric,
                hue="estimator" if has_multi_est else None,
                hue_order=list(ESTIMATOR_PALETTE.keys()) if has_multi_est else None,
                palette=ESTIMATOR_PALETTE if has_multi_est else None,
                ax=ax,
                legend=(col == n_gen - 1) and (row == 0) and has_multi_est,
                width=0.6 if not has_multi_est else 0.8,
                whis=(5, 95),
                capprops={"linewidth": 1.5},
                whiskerprops={"linewidth": 1.2},
                boxprops={"linewidth": 1.0},
                medianprops={"linewidth": 1.5, "color": "black"},
                flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            )

            if metric == "sensitivity":
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.set_ylim(-0.01, 1.05)
                ax.axhline(0.10, color="red", ls="--", lw=0.8, alpha=0.5)

            if row == 0:
                apply_col_header(ax, gen)

            if row == n_rows - 1:
                ax.set_xlabel("Discretization", fontsize=FONT["axis"])
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            else:
                ax.set_xlabel("")

            ax.set_ylabel(ylabel if col == 0 else "", fontsize=FONT["axis"])
            ax.tick_params(labelsize=FONT["annot_n"])
            ax.grid(True, alpha=0.3, axis="y")

    if has_multi_est:
        push_legend_outside(axes[0, -1], ESTIMATOR_LEGEND_LABELS)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_joint_metrics(
    df: pd.DataFrame,
    metric: str = "sensitivity",
    savepath: str | None = None,
    annotate: bool = False,
    min_valid_frac: float = 0.75,
    fpr_vmax: float = 0.15,
) -> None:
    """Heatmap of median sensitivity or FPR across n × n_bins."""
    if metric not in ("sensitivity", "fpr"):
        raise ValueError("metric must be 'sensitivity' or 'fpr'")

    met_df = compute_metrics_per_rep(df)
    if met_df.empty:
        raise ValueError("No rows found.")

    _, bins_labels = ordered_bins(met_df)
    met_df["bins_label"] = met_df["sens.n_bins"].map(bins_label)

    n_levels = sorted(met_df["n"].unique())
    estimators = met_df["estimator"].unique().tolist()
    generators = sorted(met_df["generator"].unique(), key=generator_sort_key)
    n_est = len(estimators)
    n_gen = len(generators)

    total_reps = met_df["rep"].nunique()
    min_valid = max(1, int(total_reps * min_valid_frac))

    if metric == "sensitivity":
        cmap_choice = "RdYlGn"
        vmin_choice, vmax_choice = 0.0, 1.0
    else:
        cmap_choice = "Reds"
        vmin_choice, vmax_choice = 0.0, fpr_vmax

    fig, axes = plt.subplots(
        n_est, n_gen,
        figsize=(4 * n_gen, 4 * n_est),
        squeeze=False,
    )

    im = None
    cmap_obj = plt.get_cmap(cmap_choice)
    norm = Normalize(vmin=vmin_choice, vmax=vmax_choice)

    for row, est_name in enumerate(estimators):
        for col, gen_name in enumerate(generators):
            ax = axes[row, col]
            cell = met_df[
                (met_df["estimator"] == est_name) &
                (met_df["generator"] == gen_name)
            ]

            pivot = (
                cell.groupby(["bins_label", "n"])[metric]
                .median().unstack("n")
                .reindex(index=bins_labels, columns=n_levels)
            )
            count_pivot = (
                cell.groupby(["bins_label", "n"])[metric]
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
                cmap=cmap_choice,
                vmin=vmin_choice, vmax=vmax_choice,
            )

            if annotate:
                fmt = ".3f" if vmax_choice < 0.5 else ".2f"
                for i in range(pivot_masked.shape[0]):
                    for j in range(pivot_masked.shape[1]):
                        val = pivot_masked.values[i, j]
                        if np.isnan(val):
                            continue
                        clipped = np.clip(val, vmin_choice, vmax_choice)
                        rgba = cmap_obj(norm(clipped))
                        ax.text(j, i, f"{val:{fmt}}",
                                ha="center", va="center",
                                color=text_color(rgba), fontsize=8)

            bad_rows, bad_cols = np.where(np.isnan(pivot_masked.values))
            ax.scatter(bad_cols, bad_rows,
                       marker="x", color="red", s=40, linewidths=1.5)

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right",
                               fontsize=FONT["annot_n"])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=FONT["annot_n"])

            if row == n_est - 1:
                ax.set_xlabel("N", fontsize=FONT["tick"])
            if col == 0:
                ax.set_ylabel("Number of bins", fontsize=FONT["tick"])
                apply_row_label(ax, est_name)
            if row == 0:
                apply_col_header(ax, gen_name)

    label = "Median Sensitivity" if metric == "sensitivity" else "Median FPR"
    cbar = fig.colorbar(im, ax=axes, shrink=0.8)
    cbar.ax.tick_params(labelsize=FONT["annot_n"])
    cbar.set_label(label, fontsize=FONT["tick"])

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)