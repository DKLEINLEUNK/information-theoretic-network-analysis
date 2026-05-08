"""Univariate pipeline result container and builder."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from infonet.core.base import Sensitivity
from infonet.core.pipeline import (
    flatten,
    sensitivity_combos,
    collect_identity_defaults,
)
from infonet.univariate.base import Generator, Estimator


@dataclass
class PipelineResult:
    """Immutable record of a single univariate pipeline run."""
    generator_name: str
    generator_params: dict
    sensitivities: list[str]
    sensitivity_params: dict
    estimator_name: str
    params: dict               # n, seed, rep
    data_raw: np.ndarray
    data_transformed: np.ndarray
    estimate: float


@dataclass
class Pipeline:
    """Declarative pipeline builder for univariate experiments."""

    _generators: list[Generator] = field(default_factory=list)
    _sensitivities: list[Sensitivity] = field(default_factory=list)
    _estimators: list[Estimator] = field(default_factory=list)

    def generators(self, *gens) -> "Pipeline":
        self._generators = flatten(gens)
        return self

    def sensitivities(self, *sens) -> "Pipeline":
        self._sensitivities = flatten(sens)
        return self

    def estimators(self, *ests) -> "Pipeline":
        self._estimators = flatten(ests)
        return self

    def run(
        self,
        n: int = 500,
        seed: int = 42,
        reps: int = 1,
    ) -> list[PipelineResult]:
        rng = np.random.default_rng(seed)
        combos = sensitivity_combos(self._sensitivities)
        identity_defaults = collect_identity_defaults(self._sensitivities)
        results: list[PipelineResult] = []

        for gen in self._generators:
            for rep in range(reps):
                y_raw = gen(n, rng)

                for combo in combos:
                    y = y_raw.copy()
                    applied_names: list[str] = []
                    sens_params: dict = dict(identity_defaults)
                    for s in combo:
                        y = s(y, rng)
                        applied_names.append(s.name)
                        for k, v in s.params.items():
                            sens_params[f"sens.{k}"] = v

                    for est in self._estimators:
                        results.append(PipelineResult(
                            generator_name=gen.name,
                            generator_params=gen.params,
                            sensitivities=applied_names,
                            sensitivity_params=sens_params,
                            estimator_name=est.name,
                            params={"n": n, "seed": seed, "rep": rep},
                            data_raw=y_raw,
                            data_transformed=y,
                            estimate=est(y),
                        ))

        return results
