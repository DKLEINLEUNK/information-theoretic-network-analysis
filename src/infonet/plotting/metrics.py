"""Per-rep aggregations from the long-format estimate DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_rho_per_rep(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho per (estimator, generator, n, sens.n_bins, rep) group."""
    sub = df[df["source"] != df["target"]].copy()
    sub["abs_true_A"]   = sub["true.A"].abs()
    sub["abs_estimate"] = sub["estimate"].abs()
    sub = sub[sub["abs_true_A"] > 0]

    group_cols = ["estimator", "generator", "n", "sens.n_bins", "rep"]
    records = []
    for keys, grp in sub.groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue
        rho, _ = spearmanr(grp["abs_true_A"], grp["abs_estimate"])
        records.append(dict(zip(group_cols, keys), rho=rho))
    return pd.DataFrame(records)


def compute_metrics_per_rep(df: pd.DataFrame) -> pd.DataFrame:
    """TPR (sensitivity) and FPR per group, derived from `reject`."""
    sub = df[df["source"] != df["target"]].copy()
    sub["true_pos"] = sub["true.A"].abs() > 0
    sub["est_pos"]  = sub["reject"].fillna(False).astype(bool)

    group_cols = ["estimator", "generator", "n", "sens.n_bins", "rep"]
    records = []
    for keys, grp in sub.groupby(group_cols, dropna=False):
        tp = int((grp["true_pos"] & grp["est_pos"]).sum())
        fn = int((grp["true_pos"] & ~grp["est_pos"]).sum())
        fp = int((~grp["true_pos"] & grp["est_pos"]).sum())
        tn = int((~grp["true_pos"] & ~grp["est_pos"]).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr  = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        records.append(dict(
            zip(group_cols, keys),
            sensitivity=sens,
            fpr=fpr,
        ))
    return pd.DataFrame(records)