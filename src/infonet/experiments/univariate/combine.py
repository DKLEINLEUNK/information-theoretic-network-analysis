"""Combine sweep checkpoints into a single parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file, then rename — never leaves a corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def combine(
    checkpoint_dir: Path,
    output_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Concatenate every checkpoint into one DataFrame."""
    checkpoint_dir = Path(checkpoint_dir)
    files = sorted(checkpoint_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Univariate convention: continuous baseline rows have NaN n_bins.
    # Normalise to inf so downstream sort/label helpers see one value.
    if "n_bins" in df.columns:
        df["n_bins"] = df["n_bins"].fillna(np.inf)

    if save and output_path is not None:
        _atomic_write_parquet(df, Path(output_path))
    return df


def verify_counts(
    df: pd.DataFrame,
    group_cols: list[str],
    expected_reps: int | None = None,
    verbose: bool = True,
) -> dict:
    """Verify per-cell counts match expectations for a univariate sweep."""
    n_pairs = df.groupby(["estimator", "generator"]).ngroups

    if expected_reps is None:
        expected_reps = int(
            df.groupby(group_cols)["rep"].nunique().max()
        )

    expected_per_cell = n_pairs * expected_reps
    counts = df.groupby(group_cols).size()
    mismatches = counts[counts != expected_per_cell]

    if verbose:
        print(f"Total rows: {len(df)}")
        print(
            f"Expected per {tuple(group_cols)} cell: "
            f"{n_pairs} (estimator, generator) pairs * "
            f"{expected_reps} reps = {expected_per_cell}"
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
