"""Generate detection-accuracy (TPR/FPR) plots from the combined sweep results."""
from pathlib import Path

import pandas as pd
import numpy as np

from infonet.plotting.multivariate import (
    plot_n_only_metrics,
    plot_ord_only_metrics,
    plot_joint_metrics,
)


COMBINED_PATH = Path("./results/multivariate/combined.parquet")
PLOTS_DIR = Path("./plots/multivariate")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    df = pd.read_parquet(COMBINED_PATH)

    plot_n_only_metrics(
        df, n_bins=np.inf,
        savepath=PLOTS_DIR / "metrics_sample_size.png",
    )
    plot_ord_only_metrics(
        df,
        savepath=PLOTS_DIR / "metrics_bin_size.png",
    )
    plot_joint_metrics(
        df, metric="sensitivity",
        savepath=PLOTS_DIR / "metrics_joint_sensitivity.png",
    )
    plot_joint_metrics(
        df, metric="fpr",
        savepath=PLOTS_DIR / "metrics_joint_fpr.png",
    )
    plot_joint_metrics(
        df, metric="sensitivity", annotate=True,
        savepath=PLOTS_DIR / "metrics_joint_sensitivity_annotated.png",
    )
    plot_joint_metrics(
        df, metric="fpr", annotate=True,
        savepath=PLOTS_DIR / "metrics_joint_fpr_annotated.png",
    )