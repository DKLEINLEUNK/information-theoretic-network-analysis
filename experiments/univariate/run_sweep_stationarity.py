"""Run the stationarity-perturbation sweep."""
from infonet.experiments.univariate.sweeps.stationarity import run

if __name__ == "__main__":
    df = run()
    print(f"Done. {len(df)} rows.")