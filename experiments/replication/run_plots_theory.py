"""Generate the theoretical CV-bounds figure (no data inputs)."""
from pathlib import Path

from infonet.replication import plot_sampling_bounds


PLOTS_DIR = Path("./plots/replication")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    plot_sampling_bounds(
        savepath=PLOTS_DIR / "sampling_bounds.png",
        show=False,
    )