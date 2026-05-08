"""Combine sweep checkpoints into a single parquet file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CHECKPOINT_DIR = Path("./results/multivariate/checkpoints")
DEFAULT_COMBINED_PATH = Path("./results/multivariate/combined.parquet")


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file, then rename — never leaves a corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def combine(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    output_path: Path | None = DEFAULT_COMBINED_PATH,
    save: bool = True,
) -> pd.DataFrame:
    """Concatenate every checkpoint into one DataFrame.

    Continuous-baseline rows arrive with NaN in `sens.n_bins` (no Ordinal
    sensitivity was applied); we normalise those to inf so downstream
    label/sort helpers see a single canonical value.
    """
    files = sorted(Path(checkpoint_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    if "sens.n_bins" in df.columns:
        df["sens.n_bins"] = df["sens.n_bins"].fillna(np.inf)

    if save and output_path is not None:
        _atomic_write_parquet(df, Path(output_path))
    return df


def verify_counts(
    df: pd.DataFrame,
    d: int = 5,
    expected_reps: int | None = None,
    verbose: bool = True,
) -> dict:
    """Verify per-cell counts match expectations."""
    n_pairs = df.groupby(["estimator", "generator"]).ngroups

    if expected_reps is None:
        expected_reps = int(
            df.groupby(["n", "sens.n_bins"])["rep"].nunique().max()
        )

    expected_per_cell = n_pairs * expected_reps * (d * d)

    counts = df.groupby(["n", "sens.n_bins"]).size()
    mismatches = counts[counts != expected_per_cell]

    if verbose:
        print(f"Total rows: {len(df)}")
        print(
            f"Expected per (n, sens.n_bins): "
            f"{n_pairs} (estimator, generator) pairs × "
            f"{expected_reps} reps * {d*d} edges = {expected_per_cell}"
        )
        if len(mismatches):
            print(f"\n⚠ {len(mismatches)} cells do not match:")
            print(mismatches)
        else:
            print(
                f"\n✓ All {len(counts)} cells match expected count "
                f"of {expected_per_cell}"
            )

    return {
        "n_rows": len(df),
        "expected_per_cell": expected_per_cell,
        "n_reps_inferred": expected_reps,
        "counts": counts,
        "mismatches": mismatches,
    }
