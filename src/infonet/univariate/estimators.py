"""Univariate AIS estimators."""
from __future__ import annotations

import numpy as np

from infonet.jidt.runtime import ensure_jvm, to_jarray_1d
from infonet.univariate.base import Estimator


def _drop_nan(y: np.ndarray) -> np.ndarray:
    return y[~np.isnan(y)] if np.any(np.isnan(y)) else y


class AISGaussian(Estimator):
    """JIDT Gaussian AIS calculator at history k=1."""

    @property
    def name(self) -> str:
        return "AIS_Gaussian"

    def __call__(self, y: np.ndarray) -> float:
        from jpype import JPackage

        ensure_jvm()
        calc_cls = JPackage(
            "infodynamics.measures.continuous.gaussian"
        ).ActiveInfoStorageCalculatorGaussian
        calc = calc_cls()
        calc.setProperty("k_HISTORY", "1")
        calc.initialise()

        valid = _drop_nan(y)
        calc.setObservations(to_jarray_1d(valid))
        return float(np.asarray(calc.computeAverageLocalOfObservations(), dtype=float))


class AISKraskov(Estimator):
    """JIDT Kraskov (KSG) AIS calculator at history k=1."""

    @property
    def name(self) -> str:
        return "AIS_Kraskov"

    def __call__(self, y: np.ndarray) -> float:
        from jpype import JPackage

        ensure_jvm()
        calc_cls = JPackage(
            "infodynamics.measures.continuous.kraskov"
        ).ActiveInfoStorageCalculatorKraskov
        calc = calc_cls()
        calc.setProperty("k_HISTORY", "1")
        calc.initialise()

        valid = _drop_nan(y)
        calc.setObservations(to_jarray_1d(valid))
        return float(np.asarray(calc.computeAverageLocalOfObservations(), dtype=float))


class AISDiscrete(Estimator):
    """JIDT discrete AIS calculator. Returns AIS in nats (converted from bits)."""

    def __init__(self, base: int = 2, k: int = 1):
        self._base = base
        self._k = k

    @property
    def name(self) -> str:
        return "AIS_Discrete"

    def __call__(self, y: np.ndarray) -> float:
        from jpype import JPackage, JArray, JInt

        ensure_jvm()
        valid = _drop_nan(y)
        ints = np.round(valid).astype(int)
        ints = np.clip(ints, 0, self._base - 1)

        calc_cls = JPackage(
            "infodynamics.measures.discrete"
        ).ActiveInformationCalculatorDiscrete
        calc = calc_cls(self._base, self._k)
        calc.initialise()
        calc.addObservations(JArray(JInt, 1)(ints.tolist()))
        bits = float(np.asarray(calc.computeAverageLocalOfObservations(), dtype=float))
        return bits * np.log(2)


class OLS_AR1(Estimator):
    """OLS AR(1) coefficient, converted to AIS (nats).

    Estimates phi via OLS, then returns -0.5 * ln(1 - phi_hat^2).
    """

    @property
    def name(self) -> str:
        return "OLS_AR1"

    def __call__(self, y: np.ndarray) -> float:
        valid = _drop_nan(y)
        x_lag = valid[:-1]
        x_now = valid[1:]
        phi_hat = float(np.dot(x_lag, x_now) / np.dot(x_lag, x_lag))
        phi_c = np.clip(phi_hat, -0.999, 0.999)
        return -0.5 * np.log(1.0 - phi_c ** 2)


class YuleWalker_AR1(Estimator):
    """Yule-Walker AR(1) coefficient: phi = r(1) / r(0)."""

    @property
    def name(self) -> str:
        return "YuleWalker_AR1"

    def __call__(self, y: np.ndarray) -> float:
        valid = _drop_nan(y)
        y_dm = valid - valid.mean()
        r0 = np.dot(y_dm, y_dm)
        r1 = np.dot(y_dm[:-1], y_dm[1:])
        return float(r1 / r0)