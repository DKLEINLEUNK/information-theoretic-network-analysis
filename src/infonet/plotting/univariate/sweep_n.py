"""Bias vs sample size N at fixed (phi, r)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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


def _plot_panel(ax, sub: pd.DataFrame, header: str, params_str: str,
                show_ylabel: bool = False) -> None:
    for est, grp in sub.groupby("estimator"):
        s = grp.groupby("n")["bias"].agg(
            mean="mean",
            lo=lambda x: x.quantile(0.025),
            hi=lambda x: x.quantile(0.975),
        ).reset_index()

        ax.semilogx(
            s["n"], s["mean"],
            label=AIS_LEGEND_LABELS[est],
            color=AIS_PALETTE[est],
            lw=1.5,
            marker=AIS_MARKERS[est],
            markersize=5 if est == "AIS_Kraskov" else 4,
            alpha=0.8,
        )
        ax.fill_between(s["n"], s["lo"], s["hi"],
                        alpha=0.15, color=AIS_PALETTE[est])

    ax.axhline(0, color=GRAY, ls="--", lw=1, alpha=0.6, zorder=1)
    panel_header(ax, header, params_str)
    ax.set_xlim(7, 1e4)
    ax.set_xlabel(r"$T$", fontsize=FONT["axis"])
    if show_ylabel:
        ax.set_ylabel("Bias (nats)", fontsize=FONT["axis"])


def plot_bias_sample_size(
    df: pd.DataFrame,
    *,
    phi: float = 0.8,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """1 by N figure: one panel per generator, bias vs n with 95% PI bands."""
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
        _plot_panel(axes[col], sub, header, params, show_ylabel=(col == 0))
        style_axis(axes[col])

    add_figure_legend(fig, axes[0])
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)