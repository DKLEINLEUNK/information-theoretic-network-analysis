"""AIS estimators vs generator parameter (phi for AR, r for NAR)."""
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
)


# Vertical reference lines at the simulation parameter values
AR_REFERENCE = [0.8]
NAR_REFERENCES = [2.8, 3.2, 3.6]


def _plot_panel(ax, sub: pd.DataFrame, header: str, params_str: str,
                x_label: str) -> None:
    truth = sub[sub["estimator"] == "true"].sort_values("param")
    ax.plot(truth["param"], truth["estimate"],
            color="black", lw=1.0, label="Exact", zorder=5)

    for est in ("AIS_Gaussian", "AIS_Kraskov"):
        s = sub[sub["estimator"] == est].sort_values("param")
        ax.plot(
            s["param"], s["estimate"],
            color=AIS_PALETTE[est],
            lw=1.5,
            marker=AIS_MARKERS[est],
            markersize=5 if est == "AIS_Kraskov" else 4,
            label=AIS_LEGEND_LABELS[est],
            alpha=0.8,
            zorder=4 if est == "AIS_Gaussian" else 3,
        )

    panel_header(ax, header, params_str)
    ax.set_xlabel(x_label, fontsize=FONT["axis"])


def plot_ais_over_parameter(
    df: pd.DataFrame,
    *,
    sigma_ar: float = 1.0,
    sigma_nar: float = 0.15,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """1×2 figure: AR (phi) on the left, NAR (r) on the right."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ar_sub = df[df["kind"] == "ar"]
    nar_sub = df[df["kind"] == "nar"]

    header_ar, params_ar = ais_column_metadata("AR1", sigma_ar=sigma_ar)
    _plot_panel(axes[0], ar_sub, header_ar, params_ar, x_label=r"$\phi$")
    axes[0].set_ylabel(r"$\mathcal{A}$ (nats)", fontsize=FONT["axis"])

    header_nar, params_nar = ais_column_metadata("NAR_r0", sigma_nar=sigma_nar)
    # ais_column_metadata("NAR_r0") returns "(r = 0.0)" — overwrite for the
    # parameter sweep where we want just sigma in the subtitle.
    params_nar = rf"($\sigma = {sigma_nar}$)"
    _plot_panel(axes[1], nar_sub, "NAR", params_nar, x_label=r"$r$")

    for ax in axes:
        style_axis(ax)

    for phi in AR_REFERENCE:
        axes[0].axvline(phi, color=GRAY, ls="--", lw=1, alpha=0.6, zorder=1)
    for r in NAR_REFERENCES:
        axes[1].axvline(r, color=GRAY, ls="--", lw=1, alpha=0.6, zorder=1)

    add_figure_legend(fig, axes[0])
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)