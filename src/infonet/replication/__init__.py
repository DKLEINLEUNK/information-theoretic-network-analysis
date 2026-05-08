"""Replication study of Batra et al. empirical replication plots plus the theoretical CV-bounds analysis."""
from infonet.replication.data import (
    PHI_VALUES,
    load_batra_data,
    load_replication_data,
)
from infonet.replication.plots_empirical import (
    plot_original_estimates,
    plot_corrected_estimates,
    plot_median_relative_bias,
    plot_relative_bias_with_scatter,
)
from infonet.replication.plots_theory import plot_sampling_bounds

__all__ = [
    "PHI_VALUES",
    "load_batra_data",
    "load_replication_data",
    "plot_original_estimates",
    "plot_corrected_estimates",
    "plot_median_relative_bias",
    "plot_relative_bias_with_scatter",
    "plot_sampling_bounds",
]