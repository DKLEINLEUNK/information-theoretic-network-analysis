"""Confusion-matrix grid for the multivariate sweep"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from infonet.plotting.labels import (
    apply_col_header,
    apply_row_label,
    generator_sort_key,
)
from infonet.plotting.style import (
    CONFUSION_CMAP,
    CONFUSION_NORM,
    FONT,
    text_color,
)


def confusion_counts(
    df: pd.DataFrame,
    estimator_name: str,
    exclude_diagonal: bool = True,
    verbose: bool = True,
) -> dict:
    """Compute TN/FP/FN/TP counts and derived rates for one estimator.

    Detection rule: the `reject` column. Ground truth: ``|true.A| > 0``.
    """
    sub = df.loc[df["estimator"] == estimator_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for estimator '{estimator_name}'")
    if exclude_diagonal:
        sub = sub[sub["source"] != sub["target"]]

    if "reject" not in sub.columns:
        raise ValueError(
            f"Estimator '{estimator_name}' has no 'reject' column."
        )

    nan_mask = sub["reject"].isna()
    if verbose and nan_mask.any():
        n_nan = int(nan_mask.sum())
        print(
            f"[confusion_counts] {estimator_name}: "
            f"{n_nan}/{len(sub)} rows have NaN in 'reject' "
            "(treated as not detected)."
        )

    true_pos = sub["true.A"].abs() > 0
    est_pos = sub["reject"].fillna(False).astype(bool)

    tp = int((true_pos & est_pos).sum())
    fp = int((~true_pos & est_pos).sum())
    fn = int((true_pos & ~est_pos).sum())
    tn = int((~true_pos & ~est_pos).sum())

    counts = np.array([[tn, fp], [fn, tp]])
    row_sums = counts.sum(axis=1, keepdims=True)
    rates = np.where(row_sums > 0, 100.0 * counts / row_sums, 0.0)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    tnr = 1.0 - fpr if not np.isnan(fpr) else float("nan")
    fnr = 1.0 - tpr if not np.isnan(tpr) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (2 * precision * tpr / (precision + tpr)
          if (precision + tpr) > 0 else float("nan"))

    return {
        "counts": counts, "rates": rates, "n_cells": int(counts.sum()),
        "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr,
        "precision": precision, "recall": tpr, "f1": f1,
    }


def _draw_confusion_cell(ax, stats: dict) -> None:
    rates = stats["rates"]
    counts = stats["counts"]

    sns.heatmap(
        rates, annot=False, fmt="",
        cmap=CONFUSION_CMAP, norm=CONFUSION_NORM,
        xticklabels=["Est = 0", "Est ≠ 0"],
        yticklabels=["True = 0", "True ≠ 0"],
        ax=ax, cbar=False,
    )
    ax.tick_params(labelsize=FONT["tick"])

    for i in range(2):
        for j in range(2):
            cx, cy = j + 0.5, i + 0.5
            tc = text_color(CONFUSION_CMAP(CONFUSION_NORM(rates[i, j])))
            ax.text(cx, cy - 0.10, f"{rates[i, j]:.0f}%",
                    ha="center", va="center", color=tc,
                    fontsize=FONT["annot_pct"], fontweight="bold",
                    transform=ax.transData)
            ax.text(cx, cy + 0.18, f"(n={counts[i, j]})",
                    ha="center", va="center", color=tc,
                    fontsize=FONT["annot_n"], transform=ax.transData)
    ax.set_aspect("equal")


def plot_confusion_grid(
    df: pd.DataFrame,
    estimator_names: list[str] | None = None,
    generators: list[str] | None = None,
    n: int | None = None,
    n_bins: int | float | None = None,
    title: str | None = None,
    savepath: str | Path | None = None,
    show: bool = True,
) -> dict:
    """Confusion-matrix grid: rows = estimators, columns = generators.

    CLEANUP: fixed the `show` argument — original called plt.close() in
    BOTH branches of the if/else, so `show=True` never actually showed.
    Now `show=True` calls plt.show(); `show=False` closes silently.
    """
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

    n_rows = len(estimator_names)
    n_cols = len(generators)
    cell_size = 4.2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size * n_cols, cell_size * n_rows),
        squeeze=False,
    )

    all_stats: dict = {}
    for row, name in enumerate(estimator_names):
        for col, gen in enumerate(generators):
            ax = axes[row, col]
            df_cell = df[(df["estimator"] == name) & (df["generator"] == gen)]
            if df_cell.empty:
                ax.set_visible(False)
                continue

            stats = confusion_counts(df_cell, name)
            all_stats[(name, gen)] = stats
            _draw_confusion_cell(ax, stats)

            if col == 0:
                apply_row_label(ax, name)
            if row == 0:
                apply_col_header(ax, gen)
            ax.set_xlabel("")
            ax.set_ylabel("")

    if title:
        fig.suptitle(title, fontsize=FONT["suptitle"],
                     fontweight="bold", y=1.01)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return all_stats