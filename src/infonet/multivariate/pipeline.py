"""Multivariate pipeline result container and builder."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from infonet.core.base import Sensitivity
from infonet.core.pipeline import (
    flatten,
    sensitivity_combos,
    collect_identity_defaults,
)
from infonet.multivariate.base import MVGenerator, MVEstimator


@dataclass(frozen=True)
class MVPipelineResult:
    """Immutable record of one multivariate pipeline run."""

    generator_name: str
    generator_params: dict
    sensitivities: list[str]
    sensitivity_params: dict
    estimator_name: str
    estimator_params: dict
    params: dict
    data_raw: np.ndarray
    data_transformed: np.ndarray
    estimate: np.ndarray | float | dict
    A_true: dict[str, np.ndarray]


@dataclass
class MultivariatePipeline:
    """Declarative pipeline builder for multivariate experiments."""

    _generators: list[MVGenerator] = field(default_factory=list)
    _sensitivities: list[Sensitivity] = field(default_factory=list)
    _estimators: list[MVEstimator] = field(default_factory=list)
    _verbose: bool = True

    def generators(self, *gens) -> "MultivariatePipeline":
        self._generators = flatten(gens)
        return self

    def sensitivities(self, *sens) -> "MultivariatePipeline":
        self._sensitivities = flatten(sens)
        return self

    def estimators(self, *ests) -> "MultivariatePipeline":
        self._estimators = flatten(ests)
        return self

    def verbose(self, flag: bool = True) -> "MultivariatePipeline":
        self._verbose = flag
        return self

    def run(
        self,
        n: int = 500,
        d: int = 3,
        seed: int = 42,
        reps: int = 1,
    ) -> list[MVPipelineResult]:
        rng = np.random.default_rng(seed)
        combos = sensitivity_combos(self._sensitivities)
        identity_defaults = collect_identity_defaults(self._sensitivities)
        results: list[MVPipelineResult] = []

        for gen in self._generators:
            for rep in range(reps):
                t0 = time.perf_counter()
                y_raw = gen(n, d, rng)
                A_true = gen.A_matrices

                for combo in combos:
                    y = y_raw.copy()
                    applied_names: list[str] = []
                    sens_params: dict = dict(identity_defaults)
                    for s in combo:
                        for col in range(d):
                            y[:, col] = s(y[:, col], rng)
                        applied_names.append(s.name)
                        for k, v in s.params.items():
                            sens_params[f"sens.{k}"] = v

                    for est in self._estimators:
                        results.append(MVPipelineResult(
                            generator_name=gen.name,
                            generator_params=gen.params,
                            sensitivities=applied_names,
                            sensitivity_params=sens_params,
                            estimator_name=est.name,
                            estimator_params=est.params,
                            params={"n": n, "d": d, "seed": seed, "rep": rep},
                            data_raw=y_raw,
                            data_transformed=y,
                            estimate=est(y),
                            A_true=A_true,
                        ))

                if self._verbose:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"\r  {gen.name} rep {rep + 1}/{reps}  ({elapsed:.1f}s)",
                        end="", flush=True,
                    )
            if self._verbose:
                print()

        return results
