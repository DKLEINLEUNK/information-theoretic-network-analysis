"""CLI wrapper for multivariate sweep."""
from infonet.experiments.multivariate.sweep import run_all, combine
from tqdm.auto import tqdm


if __name__ == "__main__":
    print("Starting multivariate sweep: it is safe to interupt the sweep at any point.")
    print()
    run_all(estimator_group="gaussian", n_workers=8)
    run_all(estimator_group="ksg", n_workers=3)
    df = combine()
    tqdm.write(f"Done. {len(df)} rows.")