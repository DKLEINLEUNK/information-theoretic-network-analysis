"""Joint sweep: AIS bias / variability over the (N, n_bins) grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from infonet.core import Ordinal
from infonet.univariate import (
    AR1, NAR, Pipeline,
    AISGaussian, AISKraskov,
    ais_true_ar1, ais_true_nar,
)
from infonet.experiments.univariate.combine import _atomic_write_parquet


PHI = 0.8
SIGMA_AR = 1.0
R_NAR_VALUES = (2.8, 3.2, 3.6)
SIGMA_NAR = 0.15
K_NAR = 40
HALF_W_NAR = 1 / 40

N_GRID_DEFAULT = np.logspace(1, 4, 7, dtype=int)
N_BINS_DEFAULT = np.array([2, 3, 5, 10, 20, 50, 100])
REPS_DEFAULT = 50
SEED_DEFAULT = 42

DEFAULT_OUTPUT_DIR = Path("./results/univariate/joint_sweep")


def _build_generators() -> dict[str, object]:
    gens = {"AR1": AR1(phi=PHI, sigma=SIGMA_AR)}
    for r in R_NAR_VALUES:
        gens[f"NAR_r{r}"] = NAR(r=r, sigma=SIGMA_NAR,
                                K=K_NAR, half_w=HALF_W_NAR)
    return gens


def _build_true_ais() -> dict[str, float]:
    out = {"AR1": ais_true_ar1(PHI)}
    for r in R_NAR_VALUES:
        out[f"NAR_r{r}"] = ais_true_nar(
            r, sigma=SIGMA_NAR, K=K_NAR, half_w=HALF_W_NAR
        )
    return out


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_grid=N_GRID_DEFAULT,
    n_bins_grid=N_BINS_DEFAULT,
    reps: int = REPS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    generators = _build_generators()
    true_ais = _build_true_ais()

    rows = []
    plan = [
        (gen_name, gen, int(n), int(b))
        for gen_name, gen in generators.items()
        for n in n_grid
        for b in n_bins_grid
    ]

    for gen_name, gen, n, n_bins in tqdm(plan, desc="cells"):
        results = (
            Pipeline()
            .generators(gen)
            .sensitivities(Ordinal(n_bins=n_bins))
            .estimators(AISGaussian(), AISKraskov())
            .run(n=n, seed=seed, reps=reps)
        )
        for r in results:
            rows.append({
                "generator": gen_name,
                "n": n,
                "n_bins": n_bins,
                "rep": r.params["rep"],
                "estimator": r.estimator_name,
                "estimate": r.estimate,
                "true_ais": true_ais[gen_name],
            })

    df = pd.DataFrame(rows)
    df["estimate"] = df["estimate"].replace([np.inf, -np.inf], np.nan)
    df["bias"] = df["estimate"] - df["true_ais"]

    _atomic_write_parquet(df, output_dir / "combined.parquet")
    return df