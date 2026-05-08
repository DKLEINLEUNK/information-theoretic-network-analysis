"""Run the joint (N, n_bins) sweep."""
from infonet.experiments.univariate.sweeps.joint import run

if __name__ == "__main__":
    df = run()
    print(f"Done. {len(df)} rows.")