"""Data loading for the Batra replication."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PHI_VALUES: list[float] = [0.05, 0.20, 0.50, 0.80]


def _load_one_condition(path: Path) -> pd.DataFrame:
    """Load one Batra condition file and apply the standard filters/rescaling."""
    df = pd.read_csv(path)
    df = df[df["Sample_Freq"] == "Monthly"]    # only Δ = 672
    df = df[df["parameter"] == "beta_ct"]
    # 24/672 rescaling: from monthly drift to hourly AR(1) coefficient
    df["Estimate_rescaled"] = np.exp(df["Estimate"] * 24 / 672)
    return df


def load_batra_data(
    data_dir: Path | str,
    *,
    pattern: str = "data_cond{}.csv",
    conditions: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    """Load and concatenate the four Batra condition files."""
    data_dir = Path(data_dir)
    parts = [
        _load_one_condition(data_dir / pattern.format(c))
        for c in conditions
    ]
    df = pd.concat(parts, ignore_index=True)

    df["bias"] = df["Estimate_rescaled"] - df["AR_Parameter"]
    df["bias_relative"] = df["bias"].abs() / df["AR_Parameter"]
    return df


def load_replication_data(path: Path | str) -> pd.DataFrame:
    """Load the corrected-replication CSV (`batra_results_replication.csv`)."""
    return pd.read_csv(path)