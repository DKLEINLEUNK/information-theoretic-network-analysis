"""Multivariate pipeline results to long-format DataFrame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from infonet.multivariate.pipeline import MVPipelineResult


def mv_results_to_df(results: list[MVPipelineResult]) -> pd.DataFrame:
    """Convert multivariate pipeline results to long-format DataFrame."""
    rows = []
    for r in results:
        base = {
            "generator": r.generator_name,
            **{f"gen.{k}": v for k, v in r.generator_params.items()},
            "sensitivities": " → ".join(r.sensitivities) if r.sensitivities else "(none)",
            **r.sensitivity_params,
            "estimator": r.estimator_name,
            **{f"est.{k}": v for k, v in r.estimator_params.items()},
            **r.params,
        }

        est_val = r.estimate

        if isinstance(est_val, dict):
            if "estimate" not in est_val:
                raise ValueError("Dict estimator output must contain an 'estimate' key.")
            primary = est_val["estimate"]
            if not (isinstance(primary, np.ndarray) and primary.ndim == 2):
                raise ValueError(
                    "Dict 'estimate' must be a 2-D (d, d) ndarray; "
                    f"got {type(primary).__name__} shape {getattr(primary, 'shape', None)}"
                )
            n_tgt, n_src = primary.shape
            if n_tgt != n_src:
                raise ValueError(f"Pairwise estimate must be square; got {primary.shape}")

            extras = {k: v for k, v in est_val.items() if k != "estimate"}
            for k, v in extras.items():
                if not (isinstance(v, np.ndarray) and v.shape == primary.shape):
                    raise ValueError(
                        f"Dict entry '{k}' must be a (d, d) ndarray matching 'estimate'; "
                        f"got shape {getattr(v, 'shape', None)}"
                    )

            for target in range(n_tgt):
                for source in range(n_src):
                    row = {
                        **base,
                        "source": source, "target": target, "variable": np.nan,
                        "estimate": primary[target, source],
                    }
                    for k, v in extras.items():
                        row[k] = v[target, source]
                    for mat_name, mat in r.A_true.items():
                        row[f"true.{mat_name}"] = mat[target, source]
                    rows.append(row)

        elif isinstance(est_val, np.ndarray) and est_val.ndim == 2:
            n_tgt, n_src = est_val.shape
            if n_tgt != n_src:
                raise ValueError(f"Pairwise estimate must be square; got {est_val.shape}")
            for target in range(n_tgt):
                for source in range(n_src):
                    row = {
                        **base,
                        "source": source, "target": target, "variable": np.nan,
                        "estimate": est_val[target, source],
                    }
                    for mat_name, mat in r.A_true.items():
                        row[f"true.{mat_name}"] = mat[target, source]
                    rows.append(row)

        elif isinstance(est_val, np.ndarray) and est_val.ndim == 1:
            for i, val in enumerate(est_val):
                rows.append({
                    **base,
                    "source": np.nan, "target": np.nan, "variable": i,
                    "estimate": val,
                })

        else:
            rows.append({
                **base,
                "source": np.nan, "target": np.nan, "variable": np.nan,
                "estimate": float(est_val),
            })

    df = pd.DataFrame(rows)

    for col in ("source", "target", "variable"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    return df