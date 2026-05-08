"""Bias-only stationarity plots for the main text."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infonet.plotting.labels import style_axis
from infonet.plotting.style import (
    AIS_LEGEND_LABELS,
    AIS_MARKERS,
    AIS_PALETTE,
    FONT,
)
from infonet.plotting.univariate._common import (
    add_figure_legend,
    panel_header,
)


X_LABELS = {
    "mean_shift":        "Mean Shift",
    "variance_change":   "Variance Shift",
    "dependence_change": "Dependence Shift",
}


DEFAULT_GENERATOR_PARAMS = {
    "AR1": {"phi": 0.8, "sigma": 1.0},
    "NAR": {"r": 3.2, "sigma": 0.15},
}


def _params_str(gen_name: str, gen_params: dict) -> tuple[str, str]:
    p = gen_params[gen_name]
    if gen_name == "AR1":
        header = "AR"
        params = rf"($\phi = {p['phi']}$, $\sigma = {p['sigma']}$)"
    else:
        header = "NAR"
        params = rf"($r = {p['r']}$, $\sigma = {p['sigma']}$)"
    return header, params


def _agg_with_pi(grouped, col: str) -> pd.DataFrame:
    """Mean and 95% percentile interval for `col` across each group."""
    return (
        grouped[col]
        .agg(
            mean=lambda x: np.nanmean(x),
            lo=lambda x: np.nanquantile(x, 0.025),
            hi=lambda x: np.nanquantile(x, 0.975),
        )
        .reset_index()
    )


def _plot_ais_panel(ax, sub: pd.DataFrame, gen_name: str,
                    gen_params: dict) -> None:
    """One column of the figure: estimates + PI bands + analytical truth."""
    for est, grp in sub.groupby("estimator"):
        s = _agg_with_pi(grp.groupby("severity"), "estimate")
        ax.plot(
            s["severity"], s["mean"],
            label=AIS_LEGEND_LABELS[est],
            color=AIS_PALETTE[est],
            alpha=0.8, lw=1.8,
            marker=AIS_MARKERS[est], markersize=6,
            markeredgecolor="white", markeredgewidth=0.8,
        )
        ax.fill_between(s["severity"], s["lo"], s["hi"],
                        alpha=0.15, color=AIS_PALETTE[est])

    truth = (
        sub.groupby("severity")["true_ais_perturbed"]
        .first().reset_index().sort_values("severity")
    )
    ax.plot(
        truth["severity"], truth["true_ais_perturbed"],
        color="black", ls="--", lw=1.2, label="Average", zorder=5,
    )

    header, params = _params_str(gen_name, gen_params)
    panel_header(ax, header, params)


def plot_bias_under_perturbation(
    df: pd.DataFrame,
    perturbation: str,
    *,
    gen_params: dict | None = None,
    min_y_span: float = 0.5,
    savepath: str | Path | None = None,
    show: bool = True,
) -> None:
    """1 by 2 figure: AIS estimates under one perturbation, AR (left) and NAR (right)."""
    if perturbation not in X_LABELS:
        raise ValueError(
            f"Unknown perturbation: {perturbation}. "
            f"Expected one of {list(X_LABELS)}."
        )

    if gen_params is None:
        gen_params = DEFAULT_GENERATOR_PARAMS

    sub_all = df[df["perturbation"] == perturbation]
    if sub_all.empty:
        raise ValueError(f"No rows for perturbation={perturbation!r}.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for col, gen in enumerate(("AR1", "NAR")):
        sub = sub_all[sub_all["generator"] == gen]
        _plot_ais_panel(axes[col], sub, gen, gen_params)

        if col == 0:
            axes[col].set_ylabel("AIS (nats)", fontsize=FONT["axis"])
        axes[col].set_xlabel(X_LABELS[perturbation], fontsize=FONT["axis"])

        # Enforce minimum y-span so weak perturbations don't get visually
        # over-magnified by autoscaling.
        ylim = axes[col].get_ylim()
        span = ylim[1] - ylim[0]
        if span < min_y_span:
            centre = (ylim[0] + ylim[1]) / 2
            axes[col].set_ylim(centre - min_y_span / 2,
                               centre + min_y_span / 2)

        style_axis(axes[col])

    add_figure_legend(fig, axes[0])
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_perturbations(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    gen_params: dict | None = None,
    show: bool = False,
) -> None:
    """Convenience wrapper: render the bias plot for all three perturbations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pert in ("mean_shift", "variance_change", "dependence_change"):
        plot_bias_under_perturbation(
            df, pert,
            gen_params=gen_params,
            savepath=output_dir / f"bias_{pert}.png",
            show=show,
        )