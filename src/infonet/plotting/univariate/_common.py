"""Helpers shared across the univariate plotters.

All four plotters use the same column-of-panels layout (AR | NAR_r2.8 |
NAR_r3.2 | NAR_r3.6) with a bold header above each panel and a single
figure-level legend on the right.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from infonet.plotting.style import FONT


def standard_column_order(df: pd.DataFrame) -> list[str]:
    """Return generators in the canonical AR-first, NAR-by-r order."""
    gens = df["generator"].unique().tolist()
    order = []
    if "AR1" in gens:
        order.append("AR1")
    nar = sorted(
        (g for g in gens if g.startswith("NAR_r")),
        key=lambda g: float(g.split("_r")[1]),
    )
    return order + nar


def add_figure_legend(fig, ref_ax) -> None:
    """Single right-side legend pulled from a reference axis."""
    handles, labels = ref_ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=FONT["legend"],
        frameon=False,
    )


def panel_header(ax, header: str, params_str: str) -> None:
    """The bold-header / parametric-subtitle pattern shared by every panel."""
    ax.set_title(params_str, fontsize=FONT["title"])
    ax.annotate(
        header, xy=(0.5, 1.1), xycoords="axes fraction",
        fontsize=FONT["subtitle"], fontweight="bold",
        ha="center", va="bottom",
    )