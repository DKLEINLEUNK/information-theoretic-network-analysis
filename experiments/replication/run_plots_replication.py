"""Generate the four empirical replication figures from CSV inputs."""
from pathlib import Path

from infonet.replication import (
    load_batra_data,
    load_replication_data,
    plot_original_estimates,
    plot_corrected_estimates,
    plot_median_relative_bias,
    plot_relative_bias_with_scatter,
)


# ── Paths ───────────────────────────────────────────────────────────

DATA_DIR  = Path("./experiments/replication/data/Results_OpenMx")
REPL_CSV  = Path("./experiments/replication/data/batra_results_replication.csv")
PLOTS_DIR = Path("./plots/replication")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    df_batra = load_batra_data(DATA_DIR)
    df_repl = load_replication_data(REPL_CSV)

    plot_original_estimates(
        df_batra,
        savepath=PLOTS_DIR / "batra_2_original.png",
        show=False,
    )
    plot_corrected_estimates(
        df_repl,
        savepath=PLOTS_DIR / "batra_2_corrected.png",
        show=False,
    )
    plot_median_relative_bias(
        df_batra,
        savepath=PLOTS_DIR / "batra_replication.png",
        show=False,
    )
    plot_relative_bias_with_scatter(
        df_batra,
        savepath=PLOTS_DIR / "batra_3_relative_bias.png",
        show=False,
    )