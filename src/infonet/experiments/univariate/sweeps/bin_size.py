"""Bin size sweep: AIS bias vs ordinal bin count B at fixed (phi, r, N).

Output schema (long format):
    generator   : "AR1" | f"NAR_r{r}"
    n_bins      : int
    rep         : int
    estimator   : "AIS_Gaussian" | "AIS_Kraskov"
    estimate    : float
    true_ais    : float
"""
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

N_FIXED = 10_000
N_BINS_DEFAULT = np.array([
    2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 14, 16, 18, 20,
    25, 30, 40, 50, 60, 70, 80, 90, 100,
])
REPS_DEFAULT = 10
SEED_DEFAULT = 42

DEFAULT_OUTPUT_DIR = Path("./results/univariate/bins_sweep")


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
    n_bins_grid=N_BINS_DEFAULT,
    n_fixed: int = N_FIXED,
    reps: int = REPS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    generators = _build_generators()
    true_ais = _build_true_ais()

    rows = []
    for gen_name, gen in tqdm(generators.items(), desc="generators"):
        for n_bins in n_bins_grid:
            results = (
                Pipeline()
                .generators(gen)
                .sensitivities(Ordinal(n_bins=int(n_bins)))
                .estimators(AISGaussian(), AISKraskov())
                .run(n=n_fixed, seed=seed, reps=reps)
            )
            for r in results:
                rows.append({
                    "generator": gen_name,
                    "n_bins": int(n_bins),
                    "rep": r.params["rep"],
                    "estimator": r.estimator_name,
                    "estimate": r.estimate,
                    "true_ais": true_ais[gen_name],
                })

    df = pd.DataFrame(rows)
    df["bias"] = df["estimate"] - df["true_ais"]
    df["bias"] = df["bias"].replace([np.inf, -np.inf], np.nan)

    _atomic_write_parquet(df, output_dir / "combined.parquet")
    return df