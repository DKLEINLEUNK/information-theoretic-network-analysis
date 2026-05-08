"""Bias vs ordinal bin count B at fixed (phi, r, N)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infonet.plotting.labels import ais_column_metadata, style_axis
from infonet.plotting.style import (
    AIS_LEGEND_LABELS,
    AIS_MARKERS,
    AIS_PALETTE,
    FONT,
    GRAY,
)
from infonet.plotting.univariate._common import (
    add_figure_legend,
    panel_header,
    standard_column_order,
)


# Per-panel x-axis upper limits (lower fixed at 2)
DEFAULT_XLIM_BY_GENERATOR: dict[str, tuple[int, int]] = {
    "AR1":      (2, 15),
    "NAR_r2.8": (2, 15),
    "NAR_r3.2": (2, 25),
    "NAR_r3.6": (2, 60),
}


def _nice_round_ticks(x_max: float) -> tuple[np.ndarray, int]:
    """Even integer ticks starting at 0, ending at a round upper bound."""
    candidate_steps = [1, 2, 5, 10, 15, 20, 25, 50, 100]
    for step in candidate_steps:
        n_ticks = int(np.ceil(x_max / step)) + 1
        if 4 <= n_ticks <= 6:
            upper = step * (n_ticks - 1)
            return np.arange(0, upper + 1, step), upper
    step = candidate_steps[-1]
    upper = step * int(np.ceil(x_max / step))
    return np.arange(0, upper + 1, step), upper


def _plot_panel(ax, sub: pd.DataFrame, gen_name: str,
                header: str, params_str: str,
                xlim: tuple[int, int],
                show_ylabel: bool = False) -> None:
    for est, grp in sub.groupby("estimator"):
        s = grp.groupby("n_bins")["bias"].agg(
            mean=lambda x: np.nanmean(x),
            lo=lambda x: np.nanquantile(x, 0.025),
            hi=lambda x: np.nanquantile(x, 0.975),
        ).reset_index()

        ax.plot(
            s["n_bins"], s["mean"],
            label=AIS_LEGEND_LABELS[est],
            color=AIS_PALETTE[est],
            lw=1.5,
            marker=AIS_MARKERS[est],
            markersize=5 if est == "AIS_Kraskov" else 4,
            alpha=0.8,
        )
        ax.fill_between(s["n_bins"], s["lo"], s["hi"],
                        alpha=0.15, color=AIS_PALETTE[est])

    ax.axhline(0, color=GRAY, ls="--", lw=1, alpha=0.6, zorder=1)

    ticks, upper = _nice_round_ticks(xlim[1])
    ax.set_xlim(0, upper)
    ax.set_xticks(ticks)

    panel_header(ax, header, params_str)
    ax.set_xlabel(r"$B$", fontsize=FONT["axis"])
    if show_ylabel:
        ax.set_ylabel("Bias (nats)", fontsize=FONT["axis"])


def plot_bias_bin_count(
    df: pd.DataFrame,
    *,
    phi: float = 0.8,
    xlim_by_generator: dict[str, tuple[int, int]] | None = None,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """1 by N figure: one panel per generator, bias vs n_bins with 95% PI bands."""
    if xlim_by_generator is None:
        xlim_by_generator = DEFAULT_XLIM_BY_GENERATOR

    col_order = standard_column_order(df)
    n_cols = len(col_order)

    fig, axes = plt.subplots(
        1, n_cols, figsize=(3.4 * n_cols, 4), sharey=True,
    )

    for col, gen_name in enumerate(col_order):
        sub = df[df["generator"] == gen_name]
        true_ais = float(sub["true_ais"].iloc[0])
        header, params = ais_column_metadata(
            gen_name, phi=phi, true_ais=true_ais,
        )
        xlim = xlim_by_generator.get(gen_name, (2, int(sub["n_bins"].max())))
        _plot_panel(
            axes[col], sub, gen_name, header, params, xlim,
            show_ylabel=(col == 0),
        )
        style_axis(axes[col])

    add_figure_legend(fig, axes[0])
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)