"""Multivariate estimators: VAR coefficients, transfer entropy, IDTxl wrappers."""
from __future__ import annotations

import numpy as np

from infonet.jidt.runtime import ensure_jvm, to_jarray_2d
from infonet.multivariate.base import MVEstimator


#  Linear VAR estimators

class OLS_VAR1(MVEstimator):
    """OLS per equation:  y_t[i] = A[i, :] @ y_{t-1} + e_i."""

    @property
    def name(self) -> str:
        return "OLS_VAR1"

    @property
    def is_pairwise(self) -> bool:
        return True

    def __call__(self, y: np.ndarray) -> np.ndarray:
        mask = ~np.any(np.isnan(y), axis=1)
        yc = y[mask]
        X = yc[:-1]
        Y = yc[1:]
        # lstsq solves Y ≈ X @ B, so y_t ≈ B.T @ y_{t-1}
        # We want A such that y_t = A @ y_{t-1}, hence A = B.T
        B = np.linalg.lstsq(X, Y, rcond=None)[0]
        return B.T


class SparseVAR(MVEstimator):
    """Lasso-regularised VAR(1) per equation."""

    def __init__(
        self,
        alpha: float | str = "cv",
        cv: int = 5,
        max_iter: int = 5000,
    ):
        if isinstance(alpha, str) and alpha != "cv":
            raise ValueError("alpha must be a float or the string 'cv'")
        self.alpha_ = alpha
        self.cv = cv
        self.max_iter = max_iter
        self._alphas_selected: list[float] | None = None

    @property
    def name(self) -> str:
        return "SparseVAR_CV" if self.alpha_ == "cv" else "SparseVAR"

    @property
    def params(self) -> dict:
        p = {"alpha": self.alpha_, "max_iter": self.max_iter}
        if self.alpha_ == "cv":
            p["cv"] = self.cv
        return p

    @property
    def is_pairwise(self) -> bool:
        return True

    def __call__(self, y: np.ndarray) -> np.ndarray:
        from sklearn.linear_model import Lasso, LassoCV

        mask = ~np.any(np.isnan(y), axis=1)
        yc = y[mask]
        X = yc[:-1]
        Y = yc[1:]
        d = y.shape[1]
        A_hat = np.zeros((d, d))
        alphas_selected: list[float] = []

        for i in range(d):
            if self.alpha_ == "cv":
                model = LassoCV(cv=self.cv, fit_intercept=False, max_iter=self.max_iter)
                model.fit(X, Y[:, i])
                alphas_selected.append(float(model.alpha_))
            else:
                model = Lasso(alpha=self.alpha_, fit_intercept=False, max_iter=self.max_iter)
                model.fit(X, Y[:, i])
            A_hat[i, :] = model.coef_

        if self.alpha_ == "cv":
            self._alphas_selected = alphas_selected
        return A_hat


#  Bivariate transfer entropy

class _TransferEntropyBase(MVEstimator):
    """Shared base for JIDT bivariate TE estimators."""

    def __init__(
        self,
        k_history: int = 1,
        k_tau: int = 1,
        normalise: bool = True,
    ):
        self.k_history = k_history
        self.k_tau = k_tau
        self.normalise = normalise

    @property
    def is_pairwise(self) -> bool:
        return True

    @property
    def params(self) -> dict:
        return {
            "k_history": self.k_history,
            "k_tau": self.k_tau,
            "normalise": self.normalise,
        }

    def _calc_cls(self):
        raise NotImplementedError

    def _significance(self, calc):
        raise NotImplementedError

    def __call__(self, y: np.ndarray) -> dict:
        from jpype import JArray, JDouble

        ensure_jvm()
        calc_cls = self._calc_cls()

        d = y.shape[1]
        te_matrix = np.zeros((d, d))
        pvalues = np.full((d, d), np.nan)

        for t in range(d):
            for s in range(d):
                if s == t:
                    continue

                calc = calc_cls()
                calc.setProperty("k_HISTORY", str(self.k_history))
                calc.setProperty("k_TAU", str(self.k_tau))
                calc.setProperty("NORMALISE", "true" if self.normalise else "false")
                calc.initialise()

                s_col = y[:, s]
                t_col = y[:, t]
                valid = ~(np.isnan(s_col) | np.isnan(t_col))
                source = JArray(JDouble, 1)(s_col[valid].tolist())
                target = JArray(JDouble, 1)(t_col[valid].tolist())

                calc.setObservations(source, target)
                te_matrix[t, s] = float(calc.computeAverageLocalOfObservations())

                meas_dist = self._significance(calc)
                pvalues[t, s] = float(meas_dist.pValue)

        return {"estimate": te_matrix, "pvalue": pvalues}


class TransferEntropy(_TransferEntropyBase):
    """Bivariate TE via JIDT Kraskov (KSG) estimator."""

    def __init__(
        self,
        k_history: int = 1,
        k_tau: int = 1,
        n_surrogates: int = 100,
        normalise: bool = True,
    ):
        super().__init__(k_history=k_history, k_tau=k_tau, normalise=normalise)
        self.n_surrogates = n_surrogates

    @property
    def name(self) -> str:
        return "TE_Kraskov"

    @property
    def params(self) -> dict:
        return {**super().params, "n_surrogates": self.n_surrogates}

    def _calc_cls(self):
        from jpype import JPackage
        return JPackage(
            "infodynamics.measures.continuous.kraskov"
        ).TransferEntropyCalculatorKraskov

    def _significance(self, calc):
        return calc.computeSignificance(self.n_surrogates)


class TransferEntropyGaussian(_TransferEntropyBase):
    """Bivariate TE via JIDT Gaussian estimator."""

    @property
    def name(self) -> str:
        return "TE_Gaussian"

    def _calc_cls(self):
        from jpype import JPackage
        return JPackage(
            "infodynamics.measures.continuous.gaussian"
        ).TransferEntropyCalculatorGaussian

    def _significance(self, calc):
        return calc.computeSignificance()


#  Fully conditional TE

def conditional_te_gaussian(
    y: np.ndarray,
    source: int,
    target: int,
    conditioning: list[int],
    normalise: bool = True,
) -> float:
    """Lag-1 conditional TE using JIDT's Gaussian conditional MI estimator."""
    from jpype import JPackage

    ensure_jvm()
    CondMICalc = JPackage(
        "infodynamics.measures.continuous.gaussian"
    ).ConditionalMutualInfoCalculatorMultiVariateGaussian

    mask = ~np.any(np.isnan(y), axis=1)
    yc = y[mask]

    target_future = yc[1:, target:target + 1]
    source_past = yc[:-1, source:source + 1]
    cond_cols = [target] + list(conditioning)
    cond_past = yc[:-1, cond_cols]

    calc = CondMICalc()
    calc.setProperty("NORMALISE", "true" if normalise else "false")
    calc.initialise(1, 1, len(cond_cols))
    calc.setObservations(
        to_jarray_2d(target_future),
        to_jarray_2d(source_past),
        to_jarray_2d(cond_past),
    )
    return float(calc.computeAverageLocalOfObservations())


def conditional_te_gaussian_with_significance(
    y: np.ndarray,
    source: int,
    target: int,
    conditioning: list[int],
    normalise: bool = True,
) -> tuple[float, float]:
    """Same as `conditional_te_gaussian` but returns ``(te_obs, pvalue)``."""
    from jpype import JPackage

    ensure_jvm()
    CondMICalc = JPackage(
        "infodynamics.measures.continuous.gaussian"
    ).ConditionalMutualInfoCalculatorMultiVariateGaussian

    mask = ~np.any(np.isnan(y), axis=1)
    yc = y[mask]

    target_future = yc[1:, target:target + 1]
    source_past = yc[:-1, source:source + 1]
    cond_cols = [target] + list(conditioning)
    cond_past = yc[:-1, cond_cols]

    calc = CondMICalc()
    calc.setProperty("NORMALISE", "true" if normalise else "false")
    calc.initialise(1, 1, len(cond_cols))
    calc.setObservations(
        to_jarray_2d(target_future),
        to_jarray_2d(source_past),
        to_jarray_2d(cond_past),
    )

    te_obs = float(calc.computeAverageLocalOfObservations())
    meas_dist = calc.computeSignificance()
    return te_obs, float(meas_dist.pValue)


def conditional_te_ksg(
    y: np.ndarray,
    source: int,
    target: int,
    conditioning: list[int],
    k: int = 4,
    normalise: bool = True,
) -> float:
    """Lag-1 conditional TE via JIDT's KSG conditional MI estimator."""
    from jpype import JPackage

    ensure_jvm()
    CondMICalc = JPackage(
        "infodynamics.measures.continuous.kraskov"
    ).ConditionalMutualInfoCalculatorMultiVariateKraskov1

    mask = ~np.any(np.isnan(y), axis=1)
    yc = y[mask]

    target_future = yc[1:, target:target + 1]
    source_past = yc[:-1, source:source + 1]
    cond_cols = [target] + list(conditioning)
    cond_past = yc[:-1, cond_cols]

    calc = CondMICalc()
    calc.setProperty("k", str(k))
    calc.setProperty("NORMALISE", "true" if normalise else "false")
    calc.initialise(1, 1, len(cond_cols))
    calc.setObservations(
        to_jarray_2d(target_future),
        to_jarray_2d(source_past),
        to_jarray_2d(cond_past),
    )
    return float(calc.computeAverageLocalOfObservations())


def conditional_te_ksg_with_significance(
    y: np.ndarray,
    source: int,
    target: int,
    conditioning: list[int],
    k: int = 4,
    n_perm: int = 500,
    normalise: bool = True,
) -> tuple[float, float, np.ndarray]:
    """Same as `conditional_te_ksg` but with JIDT surrogate test."""
    from jpype import JPackage

    ensure_jvm()
    CondMICalc = JPackage(
        "infodynamics.measures.continuous.kraskov"
    ).ConditionalMutualInfoCalculatorMultiVariateKraskov1

    mask = ~np.any(np.isnan(y), axis=1)
    yc = y[mask]

    target_future = yc[1:, target:target + 1]
    source_past = yc[:-1, source:source + 1]
    cond_cols = [target] + list(conditioning)
    cond_past = yc[:-1, cond_cols]

    calc = CondMICalc()
    calc.setProperty("k", str(k))
    calc.setProperty("NORMALISE", "true" if normalise else "false")
    calc.initialise(1, 1, len(cond_cols))
    calc.setObservations(
        to_jarray_2d(target_future),
        to_jarray_2d(source_past),
        to_jarray_2d(cond_past),
    )

    te_obs = float(calc.computeAverageLocalOfObservations())
    meas_dist = calc.computeSignificance(n_perm)
    pvalue = float(meas_dist.pValue)
    null = np.array(meas_dist.distribution, dtype=float)
    return te_obs, pvalue, null


# Multiple-comparison correction

def _apply_correction(
    pvalues: np.ndarray, alpha: float, correction: str
) -> np.ndarray:
    """Apply correction to off-diagonal p-values; return rejection mask."""
    d = pvalues.shape[0]
    off_diag = ~np.eye(d, dtype=bool)
    pvec = pvalues[off_diag]
    m = pvec.size

    if correction == "none":
        rej_vec = pvec < alpha
    elif correction == "bonferroni":
        rej_vec = pvec < (alpha / m)
    elif correction == "fdr_bh":
        order = np.argsort(pvec)
        ranked = pvec[order]
        thresholds = alpha * np.arange(1, m + 1) / m
        passed = ranked <= thresholds
        if passed.any():
            k_max = np.max(np.where(passed)[0])
            rej_ordered = np.zeros(m, dtype=bool)
            rej_ordered[: k_max + 1] = True
        else:
            rej_ordered = np.zeros(m, dtype=bool)
        rej_vec = np.zeros(m, dtype=bool)
        rej_vec[order] = rej_ordered
    else:
        raise ValueError(correction)

    reject = np.zeros_like(pvalues, dtype=bool)
    reject[off_diag] = rej_vec
    return reject


# Estimator classes

class FullConditionalTE_Gaussian(MVEstimator):
    """Fully-conditional lag-1 TE via JIDT's Gaussian conditional MI."""

    def __init__(
        self,
        normalise: bool = True,
        true_mask: np.ndarray | None = None,
    ):
        self.normalise = normalise
        self.true_mask = true_mask

    @property
    def name(self) -> str:
        return "FullCondTE_Gaussian"

    @property
    def params(self) -> dict:
        return {"normalise": self.normalise}

    @property
    def is_pairwise(self) -> bool:
        return True

    def __call__(self, y: np.ndarray) -> dict:
        d = y.shape[1]
        te = np.zeros((d, d))
        for t in range(d):
            for s in range(d):
                if s == t:
                    continue
                if self.true_mask is not None and not self.true_mask[t, s]:
                    continue
                conditioning = [k for k in range(d) if k != s and k != t]
                te[t, s] = conditional_te_gaussian(
                    y, source=s, target=t,
                    conditioning=conditioning, normalise=self.normalise,
                )
        return {"estimate": te}


class FullCondTE_Gaussian_SigTest(MVEstimator):
    """FullConditionalTE_Gaussian plus per-edge analytic significance test and multiple-comparison correction."""

    def __init__(
        self,
        normalise: bool = True,
        alpha: float = 0.05,
        correction: str = "fdr_bh",
    ):
        if correction not in ("none", "bonferroni", "fdr_bh"):
            raise ValueError(
                "correction must be 'none', 'bonferroni', or 'fdr_bh'"
            )
        self.normalise = normalise
        self.alpha = alpha
        self.correction = correction

    @property
    def name(self) -> str:
        return "FullCondTE_Gaussian_Sig"

    @property
    def params(self) -> dict:
        return {
            "normalise": self.normalise,
            "alpha": self.alpha,
            "correction": self.correction,
        }

    @property
    def is_pairwise(self) -> bool:
        return True

    def __call__(self, y: np.ndarray) -> dict:
        d = y.shape[1]
        te = np.zeros((d, d))
        pvalues = np.full((d, d), np.nan)

        for t in range(d):
            for s in range(d):
                if s == t:
                    continue
                conditioning = [kk for kk in range(d) if kk != s and kk != t]
                te_obs, pval = conditional_te_gaussian_with_significance(
                    y, source=s, target=t,
                    conditioning=conditioning, normalise=self.normalise,
                )
                te[t, s] = te_obs
                pvalues[t, s] = pval

        reject = _apply_correction(pvalues, self.alpha, self.correction)
        np.fill_diagonal(te, 0.0)
        np.fill_diagonal(reject, False)
        return {"estimate": te, "pvalue": pvalues, "reject": reject}


class FullConditionalTE_KSG(MVEstimator):
    """Fully-conditional lag-1 TE via JIDT KSG conditional MI."""

    def __init__(
        self,
        k: int = 4,
        normalise: bool = True,
        true_mask: np.ndarray | None = None,
    ):
        self.k = k
        self.normalise = normalise
        self.true_mask = true_mask

    @property
    def name(self) -> str:
        return "FullCondTE_KSG"

    @property
    def params(self) -> dict:
        return {"k": self.k, "normalise": self.normalise}

    @property
    def is_pairwise(self) -> bool:
        return True

    def __call__(self, y: np.ndarray) -> dict:
        d = y.shape[1]
        te = np.zeros((d, d))
        for t in range(d):
            for s in range(d):
                if s == t:
                    continue
                if self.true_mask is not None and not self.true_mask[t, s]:
                    continue
                conditioning = [k for k in range(d) if k != s and k != t]
                te[t, s] = conditional_te_ksg(
                    y, source=s, target=t,
                    conditioning=conditioning, k=self.k, normalise=self.normalise,
                )
        return {"estimate": te}


class FullCondTE_KSG_PermTest(MVEstimator):
    """FullConditionalTE_KSG plus per-edge surrogate significance test and multiple-comparison correction."""

    def __init__(
        self,
        k: int = 4,
        normalise: bool = True,
        n_perm: int = 500,
        alpha: float = 0.05,
        correction: str = "fdr_bh",
        seed: int | None = None,
    ):
        if correction not in ("none", "bonferroni", "fdr_bh"):
            raise ValueError(
                "correction must be 'none', 'bonferroni', or 'fdr_bh'"
            )
        self.k = k
        self.normalise = normalise
        self.n_perm = n_perm
        self.alpha = alpha
        self.correction = correction
        self.seed = seed

    @property
    def name(self) -> str:
        return "FullCondTE_KSG_Perm"

    @property
    def params(self) -> dict:
        return {
            "k": self.k,
            "normalise": self.normalise,
            "n_perm": self.n_perm,
            "alpha": self.alpha,
            "correction": self.correction,
        }

    @property
    def is_pairwise(self) -> bool:
        return True

    def __call__(self, y: np.ndarray) -> dict:
        d = y.shape[1]
        te = np.zeros((d, d))
        pvalues = np.full((d, d), np.nan)

        for t in range(d):
            for s in range(d):
                if s == t:
                    continue
                conditioning = [kk for kk in range(d) if kk != s and kk != t]
                te_obs, pval, _ = conditional_te_ksg_with_significance(
                    y, source=s, target=t,
                    conditioning=conditioning,
                    k=self.k, n_perm=self.n_perm, normalise=self.normalise,
                )
                te[t, s] = te_obs
                pvalues[t, s] = pval

        reject = _apply_correction(pvalues, self.alpha, self.correction)
        np.fill_diagonal(te, 0.0)
        np.fill_diagonal(reject, False)
        return {"estimate": te, "pvalue": pvalues, "reject": reject}


#  IDTxl wrappers (not used in the end)

class _IDTxlTEBase(MVEstimator):
    """Shared base for IDTxl network-inference TE estimators."""

    def __init__(
        self,
        cmi_estimator: str = "JidtGaussianCMI",
        min_lag_sources: int = 1,
        max_lag_sources: int = 1,
        max_lag_target: int | None = None,
        alpha_omnibus: float = 0.05,
        alpha_max_stat: float = 0.05,
        alpha_min_stat: float = 0.05,
        n_perm_omnibus: int = 500,
        n_perm_max_stat: int = 200,
        n_perm_min_stat: int = 200,
        n_perm_max_seq: int = 500,
        fdr_correction: bool = True,
        verbose: bool = False,
        extra_settings: dict | None = None,
    ):
        self.cmi_estimator = cmi_estimator
        self.min_lag_sources = min_lag_sources
        self.max_lag_sources = max_lag_sources
        self.max_lag_target = max_lag_target
        self.alpha_omnibus = alpha_omnibus
        self.alpha_max_stat = alpha_max_stat
        self.alpha_min_stat = alpha_min_stat
        self.n_perm_omnibus = n_perm_omnibus
        self.n_perm_max_stat = n_perm_max_stat
        self.n_perm_min_stat = n_perm_min_stat
        self.n_perm_max_seq = n_perm_max_seq
        self.fdr_correction = fdr_correction
        self.verbose = verbose
        self.extra_settings = dict(extra_settings) if extra_settings else {}

    @property
    def is_pairwise(self) -> bool:
        return True

    @property
    def params(self) -> dict:
        return {
            "cmi_estimator": self.cmi_estimator,
            "min_lag_sources": self.min_lag_sources,
            "max_lag_sources": self.max_lag_sources,
            "max_lag_target": self.max_lag_target,
            "alpha_omnibus": self.alpha_omnibus,
            "alpha_max_stat": self.alpha_max_stat,
            "alpha_min_stat": self.alpha_min_stat,
            "n_perm_omnibus": self.n_perm_omnibus,
            "n_perm_max_stat": self.n_perm_max_stat,
            "n_perm_min_stat": self.n_perm_min_stat,
            "n_perm_max_seq": self.n_perm_max_seq,
            "fdr_correction": self.fdr_correction,
            **{f"extra.{k}": v for k, v in self.extra_settings.items()},
        }

    def _analysis_cls(self):
        raise NotImplementedError

    def _build_settings(self) -> dict:
        settings = {
            "cmi_estimator": self.cmi_estimator,
            "min_lag_sources": self.min_lag_sources,
            "max_lag_sources": self.max_lag_sources,
            "alpha_omnibus": self.alpha_omnibus,
            "alpha_max_stat": self.alpha_max_stat,
            "alpha_min_stat": self.alpha_min_stat,
            "n_perm_omnibus": self.n_perm_omnibus,
            "n_perm_max_stat": self.n_perm_max_stat,
            "n_perm_min_stat": self.n_perm_min_stat,
            "n_perm_max_seq": self.n_perm_max_seq,
            "fdr_correction": self.fdr_correction,
            "verbose": self.verbose,
        }
        if self.max_lag_target is not None:
            settings["max_lag_target"] = self.max_lag_target
        settings.update(self.extra_settings)
        return settings

    def __call__(self, y: np.ndarray) -> dict:
        from idtxl.data import Data

        mask = ~np.any(np.isnan(y), axis=1)
        yc = y[mask]
        data = Data(yc, dim_order="sp", normalise=True)

        analyser = self._analysis_cls()()
        results = analyser.analyse_network(
            settings=self._build_settings(), data=data,
        )

        adj_obj = results.get_adjacency_matrix(
            weights="max_te_lag", fdr=self.fdr_correction,
        )
        if hasattr(adj_obj, "edge_matrix") and hasattr(adj_obj, "weight_matrix"):
            edge_src_tgt = np.asarray(adj_obj.edge_matrix, dtype=bool)
            weight_src_tgt = np.asarray(adj_obj.weight_matrix, dtype=float)
        else:
            arr = np.asarray(adj_obj, dtype=float)
            edge_src_tgt = arr > np.iinfo(np.int32).min + 1
            weight_src_tgt = np.where(edge_src_tgt, arr, 0.0)

        reject = edge_src_tgt.T.copy()
        estimate = weight_src_tgt.T.copy()
        np.fill_diagonal(reject, False)
        np.fill_diagonal(estimate, 0.0)

        return {"estimate": estimate, "reject": reject}


class IDTxlBivariateTE(_IDTxlTEBase):
    """IDTxl bivariate TE with greedy embedding + hierarchical stats."""

    @property
    def name(self) -> str:
        suffix = self.cmi_estimator.replace("Jidt", "").replace("CMI", "")
        return f"IDTxl_BivTE_{suffix}"

    def _analysis_cls(self):
        from idtxl.bivariate_te import BivariateTE
        return BivariateTE


class IDTxlMultivariateTE(_IDTxlTEBase):
    """IDTxl multivariate (conditional) TE with greedy embedding +
    hierarchical stats."""

    @property
    def name(self) -> str:
        suffix = self.cmi_estimator.replace("Jidt", "").replace("CMI", "")
        return f"IDTxl_MultivTE_{suffix}"

    def _analysis_cls(self):
        from idtxl.multivariate_te import MultivariateTE
        return MultivariateTE
