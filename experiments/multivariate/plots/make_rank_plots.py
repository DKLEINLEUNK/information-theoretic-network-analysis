"""Generate rank-recovery plots from the combined sweep results."""
from pathlib import Path

import pandas as pd
import numpy as np

from infonet.plotting.multivariate import (
    plot_confusion_grid,
    plot_rank_recovery_grid,
    plot_n_only,
    plot_ord_only,
    plot_joint,
)


COMBINED_PATH = Path("./results/multivariate/combined.parquet")
PLOTS_DIR = Path("./plots/multivariate")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    df = pd.read_parquet(COMBINED_PATH)

    plot_confusion_grid(
        df, 
        n=1000, 
        n_bins=np.inf,
        savepath=PLOTS_DIR / "confusion_baseline.png",
        show=False,
    )

    plot_rank_recovery_grid(
        df, 
        n=1000, 
        n_bins=np.inf,
        savepath=PLOTS_DIR / "rank_recovery_baseline.png",
        show=False,
    )

    plot_n_only(
        df, 
        n_bins=np.inf,
        savepath=PLOTS_DIR / "rank_true_sample_size.png"
    )

    plot_ord_only(
        df,
        savepath=PLOTS_DIR / "rank_true_bin_size.png"
    )

    # plot_joint(
    #     df,
    #     savepath=PLOTS_DIR / "rank_true_joint.png"
    # )

    plot_joint(
        df, 
        annotate=True,
        savepath=PLOTS_DIR / "rank_true_joint_annotated.png"
    )
