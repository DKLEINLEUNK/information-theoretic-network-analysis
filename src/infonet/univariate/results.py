"""Univariate pipeline results to DataFrame."""
from __future__ import annotations

import pandas as pd

from infonet.univariate.pipeline import PipelineResult


def results_to_df(results: list[PipelineResult]) -> pd.DataFrame:
    """Convert pipeline results to a long-format pandas DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "generator": r.generator_name,
            **{f"gen.{k}": v for k, v in r.generator_params.items()},
            "sensitivities": " → ".join(r.sensitivities) if r.sensitivities else "(none)",
            **r.sensitivity_params,
            "estimator": r.estimator_name,
            **r.params,
            "estimate": r.estimate,
        })
    return pd.DataFrame(rows)