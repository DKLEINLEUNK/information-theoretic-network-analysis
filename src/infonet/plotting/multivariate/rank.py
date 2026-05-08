"""Rank-recovery (edge calibration) grid for the multivariate sweep."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

from infonet.plotting.labels import (
    apply_col_header,
    apply_row_label,
    generator_sort_key,
)
from infonet.plotting.style import ESTIMATOR_PALETTE, FONT


def _compute_rank_heatmap(sub: pd.DataFrame, reps: list) -> tuple:
    """Return (proportions DataFrame, rho_values, n_ranks) or (None, [], 0)
    if no rep had at least 2 true-positive edges to rank."""
    rho_values = []
    all_pairs = []
    tp_counts = []

    for rep in reps:
        rep_sub = sub[sub["rep"] == rep]
        tp = rep_sub.loc[rep_sub["true_pos"]].copy()
        if len(tp) < 2:
            continue

        tp["rank_abs_A"]   = tp["abs_true_A"].rank(method="first").astype(int)
        tp["rank_abs_est"] = tp["abs_estimate"].rank(method="first").astype(int)

        rho, _ = spearmanr(tp["abs_true_A"], tp["abs_estimate"])
        rho_values.append(rho)
        all_pairs.append(tp[["rank_abs_A", "rank_abs_est"]])
        tp_counts.append(len(tp))

    if not all_pairs:
        return None, [], 0

    pairs_df = pd.concat(all_pairs, ignore_index=True)
    max_observed = pairs_df[["rank_abs_A", "rank_abs_est"]].max().max()
    max_expected = max(tp_counts)
    if max_observed > max_expected:
        raise ValueError(
            f"Rank exceeded per-rep TP count "
            f"({max_observed} > {max_expected}). Filter upstream."
        )

    n_ranks = max_expected
    rank_axis = np.arange(1, n_ranks + 1)

    counts = (
        pairs_df.groupby(["rank_abs_est", "rank_abs_A"])
        .size().unstack(fill_value=0)
        .reindex(index=rank_axis, columns=rank_axis, fill_value=0)
    )
    col_sums = counts.sum(axis=0)
    proportions = counts.divide(col_sums.where(col_sums > 0, 1), axis=1)
    return proportions, rho_values, n_ranks


def _draw_rank_heatmap(ax, proportions, n_ranks: int) -> None:
    sns.heatmap(
        proportions, cmap="YlOrRd", vmin=0, vmax=1,
        ax=ax, cbar=False, square=True,
        linewidths=0.5, linecolor="white",
    )
    ax.invert_yaxis()
    ax.plot([0.5, n_ranks - 0.5], [0.5, n_ranks - 0.5],
            "k--", lw=1.0, alpha=0.5)


def _draw_rho_hist(ax, rho_values: list, color) -> None:
    if not rho_values:
        return
    median_rho = np.median(rho_values)
    ax.hist(rho_values, bins=np.linspace(-1, 1, 21),
            color=color, alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(median_rho, color="k", ls="--", lw=2, alpha=0.7)
    ax.text(0.02, 0.95, f"median ρ = {median_rho:.2f}",
            transform=ax.transAxes, fontsize=FONT["annot_n"],
            ha="left", va="top")


def plot_rank_recovery_grid(
    df: pd.DataFrame,
    estimator_names: list[str] | None = None,
    generators: list[str] | None = None,
    n: int | None = None,
    n_bins: int | float | None = None,
    title: str | None = None,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """Rank-recovery grid: rows = estimators (heatmap + rho histogram),
    columns = generators."""
    if n is not None:
        df = df[df["n"] == n]
    if n_bins is not None:
        df = df[df["sens.n_bins"] == n_bins]
    if df.empty:
        raise ValueError(f"No rows match n={n}, n_bins={n_bins}.")

    if estimator_names is None:
        estimator_names = sorted(df["estimator"].unique())
    if generators is None:
        generators = sorted(df["generator"].unique(), key=generator_sort_key)

    reps = sorted(df["rep"].unique())
    n_est = len(estimator_names)
    n_gen = len(generators)

    colors = [
        ESTIMATOR_PALETTE.get(name, plt.cm.tab10(i / max(n_est, 10)))
        for i, name in enumerate(estimator_names)
    ]

    fig = plt.figure(figsize=(4.5 * n_gen + 0.8, 6.5 * n_est))
    gs = GridSpec(
        2 * n_est, n_gen + 1, figure=fig,
        height_ratios=[3, 1] * n_est,
        width_ratios=[1] * n_gen + [0.05],
        hspace=0.45, wspace=0.25,
    )

    for est_idx, (est_name, color) in enumerate(zip(estimator_names, colors)):
        row_hm   = 2 * est_idx
        row_hist = 2 * est_idx + 1

        for gen_idx, gen_name in enumerate(generators):
            sub = df[
                (df["estimator"] == est_name) &
                (df["generator"] == gen_name) &
                (df["source"] != df["target"])
            ].copy()
            sub["abs_true_A"]   = sub["true.A"].abs()
            sub["abs_estimate"] = sub["estimate"].abs()
            sub["true_pos"]     = sub["abs_true_A"] > 0

            proportions, rho_values, n_ranks = _compute_rank_heatmap(sub, reps)

            ax_hm = fig.add_subplot(gs[row_hm, gen_idx])
            if proportions is not None:
                _draw_rank_heatmap(ax_hm, proportions, n_ranks)
            else:
                ax_hm.text(0.5, 0.5, "no data",
                           ha="center", va="center",
                           transform=ax_hm.transAxes,
                           fontsize=FONT["annot_n"])

            ax_hm.set_xlabel("Rank of |True A|", fontsize=FONT["tick"])
            ax_hm.set_ylabel("Rank of |Estimate|" if gen_idx == 0 else "",
                             fontsize=FONT["tick"])
            ax_hm.tick_params(labelsize=FONT["tick"])

            if gen_idx == 0:
                apply_row_label(ax_hm, est_name)
            if est_idx == 0:
                apply_col_header(ax_hm, gen_name)

            ax_hist = fig.add_subplot(gs[row_hist, gen_idx])
            _draw_rho_hist(ax_hist, rho_values, color)
            ax_hist.set_xlabel("Spearman ρ", fontsize=FONT["tick"])
            ax_hist.set_ylabel("Count" if gen_idx == 0 else "",
                               fontsize=FONT["tick"])
            ax_hist.tick_params(labelsize=FONT["annot_n"])
            ax_hist.set_xlim(-1, 1)

    heatmap_rows = list(range(0, 2 * n_est, 2))
    cbar_ax = fig.add_subplot(gs[heatmap_rows[0]:heatmap_rows[-1] + 1, -1])
    sm = ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Proportion per True Rank", fontsize=FONT["tick"])
    cbar.ax.tick_params(labelsize=FONT["annot_n"])

    if title:
        fig.suptitle(title, fontsize=FONT["suptitle"],
                     fontweight="bold", y=1.01)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)