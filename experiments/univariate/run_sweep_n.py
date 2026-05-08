"""Run the bias-vs-N sweep."""
from infonet.experiments.univariate.sweeps.sample_size import run

if __name__ == "__main__":
    df = run()
    print(f"Done. {len(df)} rows.")