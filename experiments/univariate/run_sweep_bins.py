"""Run the bias-vs-n_bins sweep."""
from infonet.experiments.univariate.sweeps.bin_size import run

if __name__ == "__main__":
    df = run()
    print(f"Done. {len(df)} rows.")