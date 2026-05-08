"""Parameter sweep: φ for AR(1), r for NAR.

Output schema: one row per (kind, generator, param, estimator).
    kind        : "ar" | "nar"
    generator   : "AR1" | f"NAR_r{r}"
    param       : φ (for AR) or r (for NAR)
    estimator   : "AIS_Gaussian" | "AIS_Kraskov" | "true"
    estimate    : float
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from infonet.univariate import (
    AR1, NAR,
    AISGaussian, AISKraskov,
    ais_true_ar1, ais_true_nar,
)
from infonet.experiments.univariate.combine import _atomic_write_parquet


# ─── Defaults (override via run() kwargs) ───────────────────────────

PHI_GRID = np.linspace(0.0, 0.95, 41)
SIGMA_AR = 1.0

R_GRID = np.linspace(2.8, 3.8, 41)
SIGMA_NAR = 0.15
K_NAR = 40
HALF_W_NAR = 1 / 40

N_DEFAULT = 10**5
SEED_DEFAULT = 42

DEFAULT_OUTPUT_DIR = Path("./results/univariate/parameter_sweep")


def _sweep_one(
    kind: str,
    generator_factory,
    param_grid,
    true_ais_fn,
    n: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    gauss = AISGaussian()
    ksg = AISKraskov()

    for param in tqdm(param_grid, desc=f"Sweeping {kind}"):
        gen = generator_factory(param)
        rng = np.random.default_rng(seed)
        y = gen(n, rng)

        gen_name = "AR1" if kind == "ar" else f"NAR_r{param}"
        common = {"kind": kind, "generator": gen_name, "param": float(param)}

        rows.append({**common, "estimator": "AIS_Gaussian",
                     "estimate": float(gauss(y))})
        rows.append({**common, "estimator": "AIS_Kraskov",
                     "estimate": float(ksg(y))})
        rows.append({**common, "estimator": "true",
                     "estimate": float(true_ais_fn(param))})

    return pd.DataFrame(rows)


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    phi_grid=PHI_GRID,
    r_grid=R_GRID,
    n: int = N_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> pd.DataFrame:
    """Run the φ and r sweeps and write a single combined.parquet."""
    output_dir = Path(output_dir)

    df_ar = _sweep_one(
        kind="ar",
        generator_factory=lambda phi: AR1(phi=phi, sigma=SIGMA_AR),
        param_grid=phi_grid,
        true_ais_fn=ais_true_ar1,
        n=n, seed=seed,
    )
    df_nar = _sweep_one(
        kind="nar",
        generator_factory=lambda r: NAR(r=r, sigma=SIGMA_NAR,
                                        K=K_NAR, half_w=HALF_W_NAR),
        param_grid=r_grid,
        true_ais_fn=lambda r: ais_true_nar(
            r, sigma=SIGMA_NAR, K=K_NAR, half_w=HALF_W_NAR
        ),
        n=n, seed=seed,
    )

    df = pd.concat([df_ar, df_nar], ignore_index=True)
    df.attrs["sigma_ar"] = SIGMA_AR
    df.attrs["sigma_nar"] = SIGMA_NAR

    _atomic_write_parquet(df, output_dir / "combined.parquet")
    return df