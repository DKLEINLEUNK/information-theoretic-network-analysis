"""Combine checkpoints into `combined.parquet` without re-running sweep."""
from infonet.experiments.multivariate import combine, verify_counts


if __name__ == "__main__":
    df = combine()
    verify_counts(df)
