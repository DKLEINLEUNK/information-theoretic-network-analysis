"""Shared style constants and color helpers."""
from __future__ import annotations

import matplotlib.pyplot as plt


# Estimator: color
ESTIMATOR_PALETTE = {
    "FullCondTE_Gaussian_Sig": "#D85A30",
    "FullCondTE_KSG_Perm":     "#534AB7",
}

AIS_PALETTE = {
    "AIS_Gaussian": "#D85A30",
    "AIS_Kraskov":  "#534AB7",
}

# Estimator: legend label
ESTIMATOR_LEGEND_LABELS = {
    "FullCondTE_Gaussian_Sig": "Gaussian",
    "FullCondTE_KSG_Perm":     "KSG",
}

AIS_LEGEND_LABELS = {
    "AIS_Gaussian": "Gaussian",
    "AIS_Kraskov":  "KSG",
}

# Estimator: display label (i.e. row headers)
ESTIMATOR_LABELS = {
    "FullCondTE_Gaussian_Sig": "TE Gaussian",
    "FullCondTE_KSG_Perm":     "TE KSG",
}

AIS_MARKERS = {
    "AIS_Gaussian": "o",
    "AIS_Kraskov":  "^",
}

# Centralised font sizes
FONT = {
    "suptitle":   18,
    "subtitle":   18,
    "title":      16,
    "axis":       16,
    "tick":       14,
    "legend":     14,
    "annot_n":    14,
    "annot_pct":  14,
    "cbar_label": 11,
}

# Misc
GRAY = "#666666"

CONFUSION_CMAP = plt.get_cmap("Blues")
CONFUSION_NORM = plt.Normalize(vmin=0, vmax=100)

def text_color(bg_rgba) -> str:
    """Pick black or white text for readable contrast on a colored cell.

    Uses relative luminance per the WCAG formula. Threshold tuned by eye.
    """
    r, g, b, _ = bg_rgba

    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    L = 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)
    return "white" if L < 0.35 else "black"