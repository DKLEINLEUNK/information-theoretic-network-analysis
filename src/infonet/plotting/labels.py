"""Label/axis helpers used by all plotters."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infonet.plotting.style import (
    ESTIMATOR_LABELS,
    FONT,
)


def bins_label(x) -> str:
    """Pretty label for the n_bins factor. None / NaN / inf to inf."""
    if x is None:
        return "∞"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "∞"
    return f"{int(x)}"


def bins_sort_key(x):
    """Sort key for n_bins: continuous goes last."""
    if x is None:
        return float("inf")
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return float("inf")
    return float(x)


def ordered_bins(rho_df: pd.DataFrame) -> tuple[list, list[str]]:
    """Return (bins_levels, deduplicated bins_labels) in plot order."""
    levels = sorted(rho_df["sens.n_bins"].unique(), key=bins_sort_key)
    labels = [bins_label(b) for b in levels]
    seen = set()
    labels = [b for b in labels if not (b in seen or seen.add(b))]
    return levels, labels


# ── Generator / estimator labelling ─────────────────────────────────
def generator_sort_key(gen: str) -> tuple:
    """Sort key so VAR1 comes first, then NVAR_r* in ascending r order."""
    if gen.startswith("VAR"):
        return (0, 0.0)
    m = re.match(r"NVAR_r(.+)", gen)
    if m:
        try:
            return (1, float(m.group(1)))
        except ValueError:
            return (1, float("inf"))
    return (2, 0.0)


def col_title(gen_name: str) -> tuple[str, str]:
    """Split a generator name into (bold header, parameter line)."""
    if gen_name == "VAR1":
        return "VAR", "(σ = 1.0)"
    if gen_name.startswith("NVAR_r"):
        r = float(gen_name.split("_r")[1])
        return "NVAR", rf"(r = {r})"
    return gen_name, ""


def row_label(est_name: str) -> str:
    """Display label for the estimator row (compact form)."""
    labels = {
        "FullCondTE_KSG":           "KSG",
        "FullCondTE_KSG_Perm":      "KSG",
        "FullCondTE_Gaussian":      "Gaussian",
        "FullCondTE_Gaussian_Sig":  "Gaussian",
        "TE_Kraskov":               "KSG",
        "TE_Gaussian":              "Gaussian",
        "IDTxl_MultivTE_Gaussian":  "Gaussian",
        "IDTxl_MultivTE_Kraskov":   "KSG",
    }
    return labels.get(est_name, est_name)


def split_estimator_label(name: str) -> tuple[str, str]:
    """'FullCondTE_KSG_Perm' -> ('TE KSG', '(α=0.10)')"""
    label = ESTIMATOR_LABELS.get(name, name)
    for kernel in ("Gaussian", "KSG"):
        if kernel in label:
            return f"TE {kernel}", "(α=0.10)"
    return label, ""


# ── Axis-side decorations ───────────────────────────────────────────
def apply_row_label(ax, est_name: str, x_offset: float = -0.30) -> None:
    """Bold estimator name + params to the left of the leftmost column."""
    display, params = split_estimator_label(est_name)
    ax.text(x_offset - 0.15, 0.5, display,
            transform=ax.transAxes,
            fontsize=FONT["subtitle"], fontweight="bold",
            rotation=90, ha="center", va="center")
    ax.text(x_offset - 0.05, 0.5, params,
            transform=ax.transAxes,
            fontsize=FONT["annot_n"],
            rotation=90, ha="center", va="center")


def apply_col_header(ax, gen_name: str) -> None:
    """Bold generator name + params above the topmost row."""
    display, params = col_title(gen_name)
    ax.set_title(params, fontsize=FONT["subtitle"])
    ax.annotate(display,
                xy=(0.5, 1.1), xycoords="axes fraction",
                fontsize=FONT["subtitle"], fontweight="bold",
                ha="center", va="bottom")


# ── Legend handling ─────────────────────────────────────────────────
def relabel_legend(ax, label_map: dict[str, str]) -> None:
    """Replace auto-generated legend labels with mapped display names."""
    leg = ax.get_legend()
    if leg is None:
        return
    for text in leg.get_texts():
        raw = text.get_text()
        text.set_text(label_map.get(raw, raw))


def push_legend_outside(ax, label_map: dict[str, str]) -> None:
    """Relabel the legend and push it outside the rightmost axis."""
    leg = ax.get_legend()
    if leg is None:
        return
    relabel_legend(ax, label_map)
    ax.legend(
        leg.legend_handles,
        [t.get_text() for t in ax.get_legend().get_texts()],
        title="Estimator", loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=14,
    )


def style_axis(ax, labelsize: int | None = None, n_ticks: int | None = None):
    """Apply the gray-spine, gray-tick aesthetic shared by all plotters.

    Optional `n_ticks` limits the number of ticks per axis (used by the
    stationarity density plots).
    """
    from infonet.plotting.style import FONT, GRAY
    from matplotlib.ticker import MaxNLocator

    ax.tick_params(
        labelsize=labelsize if labelsize is not None else FONT["tick"],
        colors=GRAY,
    )
    for spine in ax.spines.values():
        spine.set_color(GRAY)

    if n_ticks is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, prune=None))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks, prune=None))


def ais_column_metadata(
    gen_name: str,
    *,
    phi: float | None = None,
    sigma_ar: float | None = None,
    sigma_nar: float | None = None,
    true_ais: float | None = None,
) -> tuple[str, str]:
    """Return (header, params_str) for an AR1 / NAR_r* univariate column."""
    if gen_name == "AR1":
        header = "AR"
        if true_ais is not None and phi is not None:
            params = rf"($\phi = {phi}$, $\mathcal{{A}} = {true_ais:.2f}$)"
        elif sigma_ar is not None:
            params = rf"($\sigma = {sigma_ar}$)"
        else:
            params = ""
        return header, params

    if gen_name.startswith("NVAR_r") or gen_name.startswith("NAR_r"):
        # NAR_r3.6 or NVAR_r3.6 → r=3.6
        r = float(gen_name.split("_r")[1])
        header = "NAR" if gen_name.startswith("NAR") else "NVAR"
        if true_ais is not None:
            params = rf"($r = {r}$, $\mathcal{{A}} = {true_ais:.2f}$)"
        elif sigma_nar is not None:
            params = rf"($\sigma = {sigma_nar}$)"
        else:
            params = rf"($r = {r}$)"
        return header, params

    return gen_name, ""


def annotate_heatmap_cells(
    ax,
    values,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    fmt: str = "{:.0f}",
    fontsize: int = 6,
    scale: float = 1.0,
    suffix: str = "",
    luminance_threshold: float = 0.35,
    yellow_aware: bool = False,
):
    """Annotate each cell of a heatmap `imshow` with its value."""
    import numpy as np

    cmap_obj = plt.get_cmap(cmap)
    span = max(vmax - vmin, 1e-12)

    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if not np.isfinite(v):
                continue

            normed = float(np.clip((v - vmin) / span, 0.0, 1.0))
            rgba = cmap_obj(normed)

            if yellow_aware and 0.4 <= normed <= 0.6:
                color = "black"
            else:
                color = text_color_for(rgba, luminance_threshold)

            ax.text(
                j, i, fmt.format(v * scale) + suffix,
                ha="center", va="center",
                color=color, fontsize=fontsize,
            )


def text_color_for(rgba, threshold: float = 0.35) -> str:
    """Variant of `style.text_color` that takes a custom luminance threshold."""
    r, g, b, _ = rgba

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "white" if L < threshold else "black"