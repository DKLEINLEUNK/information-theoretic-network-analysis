"""Generate all univariate plots from the combined parquets."""
from pathlib import Path

import numpy as np
import pandas as pd

from infonet.plotting.univariate import (
    plot_ais_over_parameter,
    plot_bias_sample_size,
    plot_bias_bin_count,
    plot_rel_bias_grid,
    plot_cv_grid,
    plot_convergence_grid,
    plot_all_perturbations
)


PLOTS_DIR = Path("./plots/univariate")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load(p: Path) -> pd.DataFrame:
    if not p.exists():
        print(f"  skipping (missing): {p}")
        return None
    return pd.read_parquet(p)


if __name__ == "__main__":
    df_param = _load(Path("./results/univariate/parameter_sweep/combined.parquet"))
    if df_param is not None:
        plot_ais_over_parameter(
            df_param,
            savepath=PLOTS_DIR / "ais_over_parameter.png",
            show=False,
        )

    df_n = _load(Path("./results/univariate/samples_sweep/combined.parquet"))
    if df_n is not None:
        plot_bias_sample_size(
            df_n,
            savepath=PLOTS_DIR / "bias_sample_size.png",
            show=False,
        )

    df_bins = _load(Path("./results/univariate/bins_sweep/combined.parquet"))
    if df_bins is not None:
        plot_bias_bin_count(
            df_bins,
            savepath=PLOTS_DIR / "bias_bin_count.png",
            show=False,
        )

    df_joint = _load(Path("./results/univariate/joint_sweep/combined.parquet"))
    if df_joint is not None:
        df_joint["estimate"] = df_joint["estimate"].replace([np.inf, -np.inf], np.nan)  # Sanity check needed for convergence plot
        plot_rel_bias_grid(
            df_joint,
            savepath=PLOTS_DIR / "joint_rel_bias.png", 
            show=False
        )
        plot_cv_grid(
            df_joint,
            savepath=PLOTS_DIR / "joint_cv.png",
            show=False,
        )
        plot_convergence_grid(
            df_joint,
            savepath=PLOTS_DIR / "joint_convergence.png",
            show=False,
        )

    df_stat = _load(Path("./results/univariate/stationarity/combined.parquet"))
    if df_stat is not None:
        plot_all_perturbations(
            df_stat, 
            output_dir=PLOTS_DIR
        )
