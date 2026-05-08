"""Run the AIS-vs-parameter sweep (phi for AR, r for NAR)."""
from infonet.experiments.univariate.sweeps.parameter import run

if __name__ == "__main__":
    df = run()
    print(f"Done. {len(df)} rows.")