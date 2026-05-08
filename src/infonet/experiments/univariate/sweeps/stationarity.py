"""Stationarity-perturbation sweep: AIS bias under MeanShift, VarianceChange, and DependenceChange across a range of severities.

Output schema:
    generator           : "AR1" | "NAR"
    perturbation        : "mean_shift" | "variance_change" | "dependence_change"
    severity            : float
    rep                 : int
    estimator           : "AIS_Gaussian" | "AIS_Kraskov"
    estimate            : float
    true_ais_baseline   : float — AIS of the unperturbed generator
    true_ais_perturbed  : float — AIS of the perturbed (mixed) series
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from infonet.univariate import (
    AR1, NAR, Pipeline,
    AISGaussian, AISKraskov,
    MeanShift, VarianceChange, DependenceChange,
    ais_true_ar1, ais_true_nar,
)
from infonet.experiments.univariate.combine import _atomic_write_parquet


PHI = 0.8
SIGMA_AR = 1.0

R_NAR = 3.2
SIGMA_NAR = 0.15
K_NAR = 40
HALF_W_NAR = 1 / 40


SWEEP_SEVERITIES: dict[str, list[float]] = {
    "mean_shift":        np.linspace(-3.0, 3.0, 13).tolist(),
    "variance_change":   np.linspace(0.1, 1.9, 13).tolist(),
    "dependence_change": np.linspace(-0.4, 0.4, 13).tolist(),
}


N_SWEEP_DEFAULT = 10_000
REPS_DEFAULT = 10
SEED_DEFAULT = 42

DEFAULT_OUTPUT_DIR = Path("./results/univariate/stationarity")


def _calibrate_marginal_scales(seed: int) -> dict[str, float]:
    """Marginal SD of each generator. Used to scale the mean-shift grid
    so severity is in units of σ_X (not σ_innovation)."""
    sigma_ar = SIGMA_AR / np.sqrt(1.0 - PHI ** 2)
    rng = np.random.default_rng(seed)
    nar_gen = NAR(r=R_NAR, sigma=SIGMA_NAR, K=K_NAR, half_w=HALF_W_NAR)
    y = nar_gen(100_000, rng)
    sigma_nar = float(np.std(y[~np.isnan(y)]))
    return {"AR1": sigma_ar, "NAR": sigma_nar}


def _precompute_true_ais() -> dict:
    """Look-up table for analytical AIS at baseline and at perturbed parameter values."""
    return {
        "baseline": {
            "AR1": ais_true_ar1(PHI),
            "NAR": ais_true_nar(R_NAR, sigma=SIGMA_NAR,
                                K=K_NAR, half_w=HALF_W_NAR),
        },
        "variance": {
            f: ais_true_nar(R_NAR, sigma=SIGMA_NAR * f,
                            K=K_NAR, half_w=HALF_W_NAR)
            for f in SWEEP_SEVERITIES["variance_change"]
            if f > 0
        },
        "dependence_ar": {
            s: ais_true_ar1(np.clip(PHI + s, -0.99, 0.99))
            for s in SWEEP_SEVERITIES["dependence_change"]
        },
        "dependence_nar": {
            s: ais_true_nar(np.clip(R_NAR + s, 0.01, 3.99),
                            sigma=SIGMA_NAR, K=K_NAR, half_w=HALF_W_NAR)
            for s in SWEEP_SEVERITIES["dependence_change"]
        },
    }


def _expected_true_ais(true_ais: dict, gen_name: str,
                       perturbation: str, severity: float) -> float:
    """Mixture-AIS for the perturbed series."""
    baseline = true_ais["baseline"][gen_name]

    if perturbation == "mean_shift":
        return baseline

    if perturbation == "variance_change":
        if gen_name == "AR1":
            return baseline
        if severity not in true_ais["variance"]:
            return baseline
        perturbed = true_ais["variance"][severity]
        return 0.5 * baseline + 0.5 * perturbed

    if perturbation == "dependence_change":
        key = "dependence_ar" if gen_name == "AR1" else "dependence_nar"
        perturbed = true_ais[key][severity]
        return 0.5 * baseline + 0.5 * perturbed

    raise ValueError(f"Unknown perturbation: {perturbation}")


def _make_generator(gen_name: str):
    if gen_name == "AR1":
        return AR1(phi=PHI, sigma=SIGMA_AR)
    if gen_name == "NAR":
        return NAR(r=R_NAR, sigma=SIGMA_NAR, K=K_NAR, half_w=HALF_W_NAR)
    raise ValueError(f"Unknown generator: {gen_name}")


def _make_sensitivity(perturbation: str, gen_name: str, severity: float,
                      mean_scales: dict[str, float]):
    if perturbation == "mean_shift":
        return MeanShift(delta=severity * mean_scales[gen_name])

    if perturbation == "variance_change":
        if gen_name == "AR1":
            return VarianceChange(process="ar1", factor=severity,
                                  sigma_original=SIGMA_AR, phi=PHI)
        return VarianceChange(process="nar", factor=severity,
                              sigma_original=SIGMA_NAR, r=R_NAR,
                              K=K_NAR, half_w=HALF_W_NAR)

    if perturbation == "dependence_change":
        if gen_name == "AR1":
            return DependenceChange(process="ar1", shift=severity,
                                    sigma=SIGMA_AR, phi_original=PHI)
        return DependenceChange(process="nar", shift=severity,
                                sigma=SIGMA_NAR, r_original=R_NAR,
                                K=K_NAR, half_w=HALF_W_NAR)

    raise ValueError(f"Unknown perturbation: {perturbation}")


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n: int = N_SWEEP_DEFAULT,
    reps: int = REPS_DEFAULT,
    seed: int = SEED_DEFAULT,
    severities: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Run the stationarity-perturbation sweep and write combined.parquet."""
    output_dir = Path(output_dir)
    severities = severities or SWEEP_SEVERITIES

    mean_scales = _calibrate_marginal_scales(seed=seed)
    true_ais = _precompute_true_ais()

    plan = [
        (perturbation, gen_name, sev)
        for gen_name in ("AR1", "NAR")
        for perturbation, sev_list in severities.items()
        for sev in sev_list
    ]

    rows = []
    for perturbation, gen_name, severity in tqdm(plan, desc="Sweeping"):
        gen = _make_generator(gen_name)
        sens = _make_sensitivity(perturbation, gen_name, severity, mean_scales)

        try:
            results = (
                Pipeline()
                .generators(gen)
                .sensitivities(sens)
                .estimators(AISGaussian(), AISKraskov())
                .run(n=n, seed=seed, reps=reps)
            )
        except Exception as e:
            print(f"\nFAILED: {perturbation}, {gen_name}, sev={severity}: {e}")
            continue

        true_ais_here = _expected_true_ais(
            true_ais, gen_name, perturbation, severity
        )
        baseline = true_ais["baseline"][gen_name]

        for r in results:
            if not r.sensitivities:
                continue  # baseline rows — sensitivity must have been applied
            rows.append({
                "generator": gen_name,
                "perturbation": perturbation,
                "severity": severity,
                "rep": r.params["rep"],
                "estimator": r.estimator_name,
                "estimate": r.estimate,
                "true_ais_baseline": baseline,
                "true_ais_perturbed": true_ais_here,
            })

    df = pd.DataFrame(rows)
    df["estimate"] = df["estimate"].replace([np.inf, -np.inf], np.nan)

    _atomic_write_parquet(df, output_dir / "combined.parquet")
    return df
