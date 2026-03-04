from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
import warnings
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, polygamma
from scipy.stats import chi2, f as f_dist, median_abs_deviation, norm, t as t_dist
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import DomainWarning

_USE_CYTHON_DISP = os.environ.get("PYDESEQ2_DISABLE_CYTHON", "").strip().lower() not in {
    "1",
    "true",
    "yes",
    "on",
}
if _USE_CYTHON_DISP:
    try:
        from . import _core_cy as _core_cy_mod
    except Exception:  # pragma: no cover - optional acceleration module.
        _fit_disp_cy = None
        _fit_disp_grid_cy = None
        _fit_glm_cy = None

        def _disp_openmp_enabled() -> bool:
            return False
    else:
        _fit_disp_cy = getattr(_core_cy_mod, "fit_disp_core", None)
        _fit_disp_grid_cy = getattr(_core_cy_mod, "fit_disp_grid_core", None)
        _fit_glm_cy = getattr(_core_cy_mod, "fit_glm_core", None)
        _openmp_enabled_fn = getattr(_core_cy_mod, "openmp_enabled", None)

        if callable(_openmp_enabled_fn):

            def _disp_openmp_enabled() -> bool:
                return bool(_openmp_enabled_fn())

        else:

            def _disp_openmp_enabled() -> bool:
                return False
else:  # pragma: no cover - environment disables acceleration path.
    _fit_disp_cy = None
    _fit_disp_grid_cy = None
    _fit_glm_cy = None

    def _disp_openmp_enabled() -> bool:
        return False


LOG2E = float(np.log2(np.e))
LN2 = float(np.log(2.0))
_CYTHON_FALLBACK_WARNED: set[str] = set()
_TRUTHY_ENV = {"1", "true", "yes", "on"}


def _is_noisy_matmul_fallback(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("matmul" in msg) and (
        ("input operand" in msg) or ("mismatch in its core dimension" in msg)
    )


def _python_fallback_enabled() -> bool:
    return (
        os.environ.get("PYDESEQ2_ALLOW_PYTHON_FALLBACK", "").strip().lower() in _TRUTHY_ENV
        or os.environ.get("PYDESEQ2_DISABLE_CYTHON", "").strip().lower() in _TRUTHY_ENV
    )


def _warn_cython_fallback_once(backend: str, exc: Optional[Exception] = None) -> None:
    """Warn once per backend when falling back to the slow Python implementation."""
    if (exc is not None) and _is_noisy_matmul_fallback(exc):
        return
    if backend in _CYTHON_FALLBACK_WARNED:
        return
    _CYTHON_FALLBACK_WARNED.add(backend)
    show_details = os.environ.get("PYDESEQ2_SHOW_FALLBACK_EXCEPTIONS", "").strip().lower() in _TRUTHY_ENV
    if show_details and (exc is not None):
        msg = (
            f"Cython {backend} backend failed; falling back to Python implementation "
            f"(much slower): {exc}"
        )
    else:
        msg = (
            f"Cython {backend} backend is unavailable or failed; falling back to Python "
            "implementation (much slower). "
            "Set PYDESEQ2_SHOW_FALLBACK_EXCEPTIONS=1 to show exception details."
        )
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _handle_cython_backend_issue(backend: str, exc: Optional[Exception] = None) -> None:
    """Raise by default; only allow fallback when explicitly requested."""
    if _python_fallback_enabled():
        _warn_cython_fallback_once(backend, exc)
        return

    if exc is None:
        detail = f"Cython {backend} backend is unavailable."
    else:
        detail = f"Cython {backend} backend failed: {exc}"
    raise RuntimeError(
        f"{detail} Python fallback is disabled by default because it is much slower. "
        "Fix the Cython backend, or rerun with PYDESEQ2_ALLOW_PYTHON_FALLBACK=1 "
        "(or PYDESEQ2_DISABLE_CYTHON=1) to explicitly allow slow fallback."
    ) from exc


class DESeq2Error(RuntimeError):
    """Raised for invalid DESeq2 model state or inputs."""


def _as_float_matrix(x: Union[np.ndarray, pd.DataFrame], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise DESeq2Error(f"{name} must be a 2D matrix, got shape {arr.shape}.")
    return arr


def _cython_num_threads() -> int:
    raw = os.environ.get("PYDESEQ2_NUM_THREADS")
    if raw is None:
        return 0
    raw = raw.strip()
    if raw == "":
        return 0
    try:
        val = int(raw)
    except ValueError:
        warnings.warn(
            "Ignoring invalid PYDESEQ2_NUM_THREADS value; expected positive integer.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    return val if val > 0 else 0


def _row_vars(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= 1:
        return np.zeros(x.shape[0], dtype=float)
    return np.var(x, axis=1, ddof=1)


def _trimmed_mean_rows(x: np.ndarray, proportion: float) -> np.ndarray:
    if x.ndim != 2:
        raise DESeq2Error("trimmed mean expects a 2D matrix.")
    n = x.shape[1]
    if n == 0:
        return np.full(x.shape[0], np.nan, dtype=float)
    k = int(math.floor(proportion * n))
    if 2 * k >= n:
        return np.mean(x, axis=1)
    x_sorted = np.sort(x, axis=1)
    return np.mean(x_sorted[:, k : n - k], axis=1)


def _check_full_rank(model_matrix: np.ndarray) -> None:
    rank = np.linalg.matrix_rank(model_matrix)
    if rank < model_matrix.shape[1]:
        raise DESeq2Error("Model matrix is not full rank.")


def _nbinom_logpmf(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    size = 1.0 / alpha
    if y.ndim == 2 and size.ndim == 1:
        size = size[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (
            gammaln(y + size)
            - gammaln(size)
            - gammaln(y + 1.0)
            + size * (np.log(size) - np.log(size + mu))
            + y * (np.log(mu) - np.log(size + mu))
        )
    return out


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    mask = np.isfinite(pvalues)
    m = int(mask.sum())
    if m == 0:
        return out
    pv = pvalues[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.empty_like(ranked)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = prev
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out_nonzero = np.empty_like(adjusted)
    out_nonzero[order] = adjusted
    out[mask] = out_nonzero
    return out


def _adjust_pvalues(pvalues: np.ndarray, method: str = "BH") -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    mask = np.isfinite(pvalues)
    if not np.any(mask):
        return out
    method_key = method.lower()
    method_map = {
        "bh": "fdr_bh",
        "fdr_bh": "fdr_bh",
        "by": "fdr_by",
        "fdr_by": "fdr_by",
        "bonferroni": "bonferroni",
        "holm": "holm",
        "sidak": "sidak",
        "none": None,
    }
    if method_key not in method_map:
        raise DESeq2Error(f"Unsupported p-value adjustment method '{method}'.")
    sm_method = method_map[method_key]
    if sm_method is None:
        out[mask] = pvalues[mask]
        return out
    _, p_adj, _, _ = multipletests(pvalues[mask], method=sm_method)
    out[mask] = p_adj
    return out


def _weighted_quantile(x: np.ndarray, weights: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if np.sum(keep) == 0:
        return np.nan
    x = x[keep]
    w = w[keep]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return np.nan
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x))


def _match_upper_quantile_for_variance(x: np.ndarray, upper_quantile: float = 0.05) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    sd_est = np.quantile(np.abs(x), 1.0 - upper_quantile) / norm.ppf(1.0 - upper_quantile / 2.0)
    return float(sd_est**2)


def _match_weighted_upper_quantile_for_variance(
    x: np.ndarray, weights: np.ndarray, upper_quantile: float = 0.05
) -> float:
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    q = _weighted_quantile(np.abs(x), weights, 1.0 - upper_quantile)
    if not np.isfinite(q):
        return float("nan")
    sd_est = q / norm.ppf(1.0 - upper_quantile / 2.0)
    return float(sd_est**2)


def _make_names(names: Sequence[str]) -> list[str]:
    out = []
    for name in names:
        n = str(name)
        if n == "(Intercept)":
            n = "Intercept"
        out.append(n)
    return out


def _make_valid_name(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z_.]", ".", str(name))
    if re.match(r"^[0-9]", s):
        s = "X" + s
    return s


def _build_design_matrix(
    metadata: pd.DataFrame, design: str
) -> Tuple[np.ndarray, list[str], pd.DataFrame]:
    try:
        design_df = patsy.dmatrix(design, metadata, return_type="dataframe")
    except Exception as exc:  # pragma: no cover - passthrough for user diagnostics.
        raise DESeq2Error(f"Failed to build design matrix from design '{design}': {exc}") from exc
    model_matrix = np.asarray(design_df, dtype=float)
    coef_names = _make_names(list(design_df.columns))
    _check_full_rank(model_matrix)
    if model_matrix.shape[0] == model_matrix.shape[1]:
        raise DESeq2Error(
            "Number of samples equals number of coefficients; there are no replicates to estimate dispersion."
        )
    return model_matrix, coef_names, design_df


def estimate_size_factors_for_matrix(
    counts: Union[np.ndarray, pd.DataFrame],
    locfunc: Callable[[np.ndarray], float] = np.median,
    geo_means: Optional[np.ndarray] = None,
    control_genes: Optional[Union[np.ndarray, Sequence[int], Sequence[bool]]] = None,
    type: str = "ratio",
) -> np.ndarray:
    """Estimate DESeq2 size factors using the median-ratio method."""
    cts = _as_float_matrix(counts, "counts")
    if np.any(cts < 0):
        raise DESeq2Error("counts must be non-negative.")
    if type not in {"ratio", "poscounts"}:
        raise DESeq2Error("type must be either 'ratio' or 'poscounts'.")

    incoming_geo_means = geo_means is not None
    if geo_means is None:
        if type == "ratio":
            with np.errstate(divide="ignore"):
                loggeomeans = np.mean(np.log(cts), axis=1)
        else:
            with np.errstate(divide="ignore"):
                lc = np.log(cts)
            lc[~np.isfinite(lc)] = 0.0
            loggeomeans = np.mean(lc, axis=1)
            all_zero = np.sum(cts, axis=1) == 0
            loggeomeans[all_zero] = -np.inf
    else:
        geo_means = np.asarray(geo_means, dtype=float)
        if geo_means.shape[0] != cts.shape[0]:
            raise DESeq2Error("geo_means must be as long as the number of genes.")
        with np.errstate(divide="ignore"):
            loggeomeans = np.log(geo_means)

    if np.all(np.isinf(loggeomeans)):
        raise DESeq2Error(
            "Every gene contains at least one zero; cannot compute geometric means with type='ratio'."
        )

    if control_genes is None:
        cts_sub = cts
        loggeomeans_sub = loggeomeans
    else:
        idx = np.asarray(control_genes)
        if idx.dtype == bool:
            if idx.shape[0] != cts.shape[0]:
                raise DESeq2Error("Boolean control_genes must be the same length as the number of genes.")
        else:
            idx = idx.astype(int)
        cts_sub = cts[idx, :]
        loggeomeans_sub = loggeomeans[idx]

    sf = np.empty(cts.shape[1], dtype=float)
    for j in range(cts_sub.shape[1]):
        cnts = cts_sub[:, j]
        keep = np.isfinite(loggeomeans_sub) & (cnts > 0)
        if not np.any(keep):
            sf[j] = np.nan
            continue
        vals = np.log(cnts[keep]) - loggeomeans_sub[keep]
        sf[j] = float(np.exp(locfunc(vals)))

    if incoming_geo_means:
        if np.any(sf <= 0) or np.any(~np.isfinite(sf)):
            raise DESeq2Error("Size factors must be finite and positive.")
        sf = sf / np.exp(np.mean(np.log(sf)))
    return sf


def _model_matrix_groups(model_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rounded = np.round(model_matrix, decimals=12)
    _, inverse, counts = np.unique(
        rounded, axis=0, return_inverse=True, return_counts=True
    )
    return inverse, counts[inverse]


def _n_or_more_in_cell(model_matrix: np.ndarray, n: int) -> np.ndarray:
    _, counts = _model_matrix_groups(model_matrix)
    return counts >= n


def _trim_bin_idx(n: int) -> int:
    if n <= 3:
        return 0
    if n <= 23:
        return 1
    return 2


def _trimmed_cell_variance(cnts: np.ndarray, cells: np.ndarray) -> np.ndarray:
    trimratio = [1.0 / 3.0, 1.0 / 4.0, 1.0 / 8.0]
    scale_c = [2.04, 1.86, 1.51]
    levels = np.unique(cells)
    cell_means = np.empty((cnts.shape[0], levels.shape[0]), dtype=float)
    for i, lvl in enumerate(levels):
        idx = cells == lvl
        n = int(np.sum(idx))
        tr = trimratio[_trim_bin_idx(n)]
        cell_means[:, i] = _trimmed_mean_rows(cnts[:, idx], tr)

    level_to_col = {lvl: i for i, lvl in enumerate(levels)}
    qmat = np.empty_like(cnts)
    for j, lvl in enumerate(cells):
        qmat[:, j] = cell_means[:, level_to_col[lvl]]
    sqerror = (cnts - qmat) ** 2

    var_est = np.empty((cnts.shape[0], levels.shape[0]), dtype=float)
    for i, lvl in enumerate(levels):
        idx = cells == lvl
        n = int(np.sum(idx))
        bin_idx = _trim_bin_idx(n)
        tr = trimratio[bin_idx]
        var_est[:, i] = scale_c[bin_idx] * _trimmed_mean_rows(sqerror[:, idx], tr)
    return np.max(var_est, axis=1)


def _trimmed_variance(x: np.ndarray) -> np.ndarray:
    rm = _trimmed_mean_rows(x, 1.0 / 8.0)
    sqerror = (x - rm[:, None]) ** 2
    return 1.51 * _trimmed_mean_rows(sqerror, 1.0 / 8.0)


def _robust_method_of_moments_disp(cnts_norm: np.ndarray, model_matrix: np.ndarray) -> np.ndarray:
    three_or_more = _n_or_more_in_cell(model_matrix, n=3)
    if np.any(three_or_more):
        row_groups, _ = _model_matrix_groups(model_matrix)
        keep_levels = np.unique(row_groups[three_or_more])
        keep_samples = np.isin(row_groups, keep_levels)
        cnts_sub = cnts_norm[:, keep_samples]
        cells_sub = row_groups[keep_samples]
        v = _trimmed_cell_variance(cnts_sub, cells_sub)
    else:
        v = _trimmed_variance(cnts_norm)
    m = np.mean(cnts_norm, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (v - m) / (m**2)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    return np.maximum(alpha, 0.04)


def _record_max_cooks(model_matrix: np.ndarray, cooks: np.ndarray, num_rows: int) -> np.ndarray:
    samples_for_cooks = _n_or_more_in_cell(model_matrix, n=3)
    m, p = model_matrix.shape
    if (m > p) and np.any(samples_for_cooks):
        return np.max(cooks[:, samples_for_cooks], axis=1)
    return np.full(num_rows, np.nan, dtype=float)


@dataclass
class _DispFit:
    func: Callable[[np.ndarray], np.ndarray]
    fit_type: str
    var_log_disp_ests: float


class DESeq2:
    """Pythonic DESeq2-like analysis object.

    Parameters
    ----------
    counts:
        Count matrix with genes as rows and samples as columns.
    metadata:
        Sample metadata indexed by sample names.
    design:
        Patsy design formula, e.g. ``"~ condition"``.
    weights:
        Optional observation weights with the same shape as `counts`.
    """

    def __init__(
        self,
        counts: Union[np.ndarray, pd.DataFrame],
        metadata: pd.DataFrame,
        design: str,
        weights: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> None:
        if isinstance(counts, pd.DataFrame):
            counts_df = counts.copy()
            gene_names = counts_df.index.astype(str)
            sample_names = counts_df.columns.astype(str)
        else:
            cts = _as_float_matrix(counts, "counts")
            gene_names = pd.Index([f"gene_{i}" for i in range(cts.shape[0])], dtype=str)
            sample_names = pd.Index([f"sample_{j}" for j in range(cts.shape[1])], dtype=str)
            counts_df = pd.DataFrame(cts, index=gene_names, columns=sample_names)

        if not isinstance(metadata, pd.DataFrame):
            raise DESeq2Error("metadata must be a pandas DataFrame.")
        if metadata.index.dtype != "object":
            metadata = metadata.copy()
            metadata.index = metadata.index.astype(str)

        sample_names = pd.Index(sample_names, dtype=str)
        if set(metadata.index) != set(sample_names):
            raise DESeq2Error(
                "metadata index must match count matrix sample names."
            )
        metadata = metadata.loc[sample_names].copy()

        self.counts_df_ = counts_df
        self.metadata_ = metadata
        self.design_ = design
        self.gene_names_ = pd.Index(gene_names, dtype=str)
        self.sample_names_ = sample_names
        self.counts_ = np.asarray(counts_df.values, dtype=float)
        if np.any(self.counts_ < 0):
            raise DESeq2Error("counts must be non-negative.")

        self.is_integer_counts_ = np.allclose(self.counts_, np.round(self.counts_))
        self.model_matrix_standard_, self.coef_names_standard_, self.design_df_ = _build_design_matrix(
            self.metadata_, self.design_
        )
        self.model_matrix_ = self.model_matrix_standard_.copy()
        self.coef_names_ = list(self.coef_names_standard_)
        self.model_matrix_type_: str = "standard"
        self.disp_model_matrix_ = self.model_matrix_standard_.copy()

        if weights is not None:
            w = _as_float_matrix(weights, "weights")
            if w.shape != self.counts_.shape:
                raise DESeq2Error("weights must have the same shape as counts.")
            self.weights_ = np.maximum(w, 1e-6)
        else:
            self.weights_ = None

        # normalization
        self.size_factors_: Optional[np.ndarray] = None
        self.normalization_factors_: Optional[np.ndarray] = None

        # base stats
        self.base_mean_: Optional[np.ndarray] = None
        self.base_var_: Optional[np.ndarray] = None
        self.all_zero_: Optional[np.ndarray] = None

        # dispersion
        self.mu_: Optional[np.ndarray] = None
        self.disp_gene_est_: Optional[np.ndarray] = None
        self.disp_gene_iter_: Optional[np.ndarray] = None
        self.disp_fit_: Optional[np.ndarray] = None
        self.dispersions_: Optional[np.ndarray] = None
        self.disp_map_: Optional[np.ndarray] = None
        self.disp_iter_: Optional[np.ndarray] = None
        self.disp_outlier_: Optional[np.ndarray] = None
        self.disp_prior_var_: Optional[float] = None
        self.dispersion_fit_: Optional[_DispFit] = None

        # regression/test outputs
        self.beta_: Optional[np.ndarray] = None
        self.beta_se_: Optional[np.ndarray] = None
        self.beta_conv_: Optional[np.ndarray] = None
        self.beta_iter_: Optional[np.ndarray] = None
        self.mle_beta_: Optional[np.ndarray] = None
        self.mle_coef_names_: Optional[list[str]] = None
        self.wald_stat_: Optional[np.ndarray] = None
        self.wald_pvalue_: Optional[np.ndarray] = None
        self.lrt_stat_: Optional[np.ndarray] = None
        self.lrt_pvalue_: Optional[np.ndarray] = None
        self.deviance_: Optional[np.ndarray] = None
        self.hat_diagonals_: Optional[np.ndarray] = None
        self.cooks_: Optional[np.ndarray] = None
        self.max_cooks_: Optional[np.ndarray] = None
        self.beta_prior_: bool = False
        self.beta_prior_var_: Optional[np.ndarray] = None
        self.test_: Optional[str] = None
        self.replace_counts_: Optional[np.ndarray] = None
        self.replace_cooks_: Optional[np.ndarray] = None
        self.original_counts_: Optional[np.ndarray] = None
        self.replace_mask_: Optional[np.ndarray] = None
        self.replaceable_samples_: Optional[np.ndarray] = None
        self.score_null_design_: Optional[str] = None
        self.score_null_model_matrix_: Optional[np.ndarray] = None
        self.score_null_coef_names_: Optional[list[str]] = None
        self.score_null_beta_: Optional[np.ndarray] = None
        self.score_null_mu_: Optional[np.ndarray] = None
        self.score_null_beta_conv_: Optional[np.ndarray] = None
        self.score_null_beta_iter_: Optional[np.ndarray] = None
        self.score_null_minmu_: Optional[float] = None
        self.optimizer_stats_: Optional[dict[str, Union[int, float, str]]] = None

        self.internal_to_result_name_: dict[str, str] = {}
        self.result_name_to_internal_: dict[str, str] = {}
        self._refresh_result_name_map()

    @property
    def n_genes(self) -> int:
        return int(self.counts_.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.counts_.shape[1])

    @property
    def n_coefs(self) -> int:
        return int(self.model_matrix_.shape[1])

    def _size_or_norm_factors(self) -> np.ndarray:
        if self.normalization_factors_ is not None:
            return self.normalization_factors_
        if self.size_factors_ is None:
            raise DESeq2Error("Size factors are not estimated yet.")
        return np.repeat(self.size_factors_[None, :], self.n_genes, axis=0)

    def _normalized_counts(self) -> np.ndarray:
        nf = self._size_or_norm_factors()
        return self.counts_ / nf

    def _ensure_base_stats(self) -> None:
        cts_norm = self._normalized_counts()
        if self.weights_ is not None:
            cts_norm = self.weights_ * cts_norm
        self.base_mean_ = np.mean(cts_norm, axis=1)
        self.base_var_ = _row_vars(cts_norm)
        self.all_zero_ = np.sum(self.counts_, axis=1) == 0

    @staticmethod
    def _linear_model_mu(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        q, r = np.linalg.qr(x, mode="reduced")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            mu = (y @ q) @ q.T
        return np.where(np.isfinite(mu), mu, 0.0)

    def _linear_model_mu_normalized(self, counts: np.ndarray, nf: np.ndarray, x: np.ndarray) -> np.ndarray:
        norm_counts = counts / nf
        muhat = self._linear_model_mu(norm_counts, x)
        return muhat * nf

    @staticmethod
    def _rough_disp_estimate(y_norm: np.ndarray, x: np.ndarray) -> np.ndarray:
        mu = DESeq2._linear_model_mu(y_norm, x)
        mu = np.maximum(mu, 1.0)
        m, p = x.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.sum(((y_norm - mu) ** 2 - mu) / (mu**2), axis=1) / (m - p)
        est = np.where(np.isfinite(est), est, 0.0)
        return np.maximum(est, 0.0)

    def _moments_disp_estimate(self, base_var: np.ndarray, base_mean: np.ndarray) -> np.ndarray:
        if self.normalization_factors_ is not None:
            xim = np.mean(1.0 / np.mean(self.normalization_factors_, axis=0))
        else:
            if self.size_factors_ is None:
                raise DESeq2Error("Size factors are required.")
            xim = np.mean(1.0 / self.size_factors_)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (base_var - xim * base_mean) / (base_mean**2)
        out = np.where(np.isfinite(out), out, 0.0)
        return out

    def estimate_size_factors(
        self,
        type: str = "ratio",
        geo_means: Optional[np.ndarray] = None,
        control_genes: Optional[Union[np.ndarray, Sequence[int], Sequence[bool]]] = None,
    ) -> "DESeq2":
        self.size_factors_ = estimate_size_factors_for_matrix(
            self.counts_,
            geo_means=geo_means,
            control_genes=control_genes,
            type=type,
        )
        return self

    def _design_vars(self) -> list[str]:
        vars_in_design = []
        for col in self.metadata_.columns:
            if re.search(rf"\b{re.escape(str(col))}\b", self.design_) is not None:
                vars_in_design.append(str(col))
        return vars_in_design

    def _design_has_interactions(self) -> bool:
        return (":" in self.design_) or ("*" in self.design_)

    def _refresh_result_name_map(self) -> None:
        internal_to_result: dict[str, str] = {}
        result_to_internal: dict[str, str] = {}
        design_vars = self._design_vars()
        factor_info: dict[str, list[str]] = {}
        for var in design_vars:
            if var in self.metadata_.columns and isinstance(self.metadata_[var].dtype, pd.CategoricalDtype):
                factor_info[var] = [str(x) for x in self.metadata_[var].cat.categories]

        for internal in self.coef_names_:
            alias = internal
            if internal in {"Intercept", "(Intercept)"}:
                alias = "Intercept"
            else:
                matched = False
                for var, levels in factor_info.items():
                    if len(levels) <= 1:
                        continue
                    ref = levels[0]
                    for lvl in levels[1:]:
                        if internal in {
                            f"{var}[T.{lvl}]",
                            f"{var}[{lvl}]",
                            _make_valid_name(f"{var}[T.{lvl}]"),
                            _make_valid_name(f"{var}[{lvl}]"),
                        }:
                            alias = _make_valid_name(f"{var}_{lvl}_vs_{ref}")
                            matched = True
                            break
                    if matched:
                        break
                if not matched and self.model_matrix_type_ == "expanded":
                    for var, levels in factor_info.items():
                        for lvl in levels:
                            if internal == _make_valid_name(f"{var}{lvl}"):
                                alias = _make_valid_name(f"{var}{lvl}")
                                matched = True
                                break
                        if matched:
                            break

            internal_to_result[internal] = alias

        for internal, alias in internal_to_result.items():
            result_to_internal[internal] = internal
            result_to_internal[alias] = internal
        self.internal_to_result_name_ = internal_to_result
        self.result_name_to_internal_ = result_to_internal

    def results_names(self) -> list[str]:
        return [self.internal_to_result_name_.get(n, n) for n in self.coef_names_]

    def _internal_coef_name(self, name: str) -> str:
        if name in self.result_name_to_internal_:
            return self.result_name_to_internal_[name]
        raise DESeq2Error(
            f"Unknown coefficient '{name}'. Available: {self.results_names()}"
        )

    def _make_expanded_model_matrix(self) -> tuple[np.ndarray, list[str]]:
        # Expanded model matrix: intercept + one column per factor level + numeric covariates.
        # This is used for beta-prior Wald fitting (no interaction terms).
        design_vars = self._design_vars()
        n = self.n_samples
        cols: list[np.ndarray] = [np.ones(n, dtype=float)]
        names: list[str] = ["Intercept"]
        for var in design_vars:
            if var not in self.metadata_.columns:
                continue
            series = self.metadata_[var]
            if isinstance(series.dtype, pd.CategoricalDtype):
                levels = [str(x) for x in series.cat.categories]
                values = series.astype(str).to_numpy()
                for lvl in levels:
                    cols.append((values == lvl).astype(float))
                    names.append(_make_valid_name(f"{var}{lvl}"))
            else:
                cols.append(pd.to_numeric(series, errors="raise").to_numpy(dtype=float))
                names.append(_make_valid_name(var))
        x = np.column_stack(cols)
        if np.any(np.sum(np.abs(x), axis=0) == 0):
            raise DESeq2Error("Expanded model matrix has all-zero columns and cannot be fit.")
        return x, names

    def _estimate_beta_prior_var(
        self,
        mle_betas: np.ndarray,
        upper_quantile: float = 0.05,
        beta_prior_method: str = "weighted",
    ) -> np.ndarray:
        if self.base_mean_ is None:
            self._ensure_base_stats()
        if self.disp_fit_ is None:
            if self.dispersions_ is not None:
                disp_fit = np.full(self.n_genes, np.nanmean(self.dispersions_), dtype=float)
            else:
                disp_fit = np.ones(self.n_genes, dtype=float)
        else:
            disp_fit = self.disp_fit_
        with np.errstate(divide="ignore", invalid="ignore"):
            varlogk = 1.0 / self.base_mean_ + disp_fit
            weights = 1.0 / varlogk
        weights = np.where(np.isfinite(weights), weights, 0.0)

        beta_prior_var = np.zeros(self.n_coefs, dtype=float)
        for j in range(self.n_coefs):
            x = mle_betas[:, j]
            use_finite = np.isfinite(x) & (np.abs(x) < 10.0)
            if np.sum(use_finite) == 0:
                beta_prior_var[j] = 1e6
                continue
            if beta_prior_method == "weighted":
                v = _match_weighted_upper_quantile_for_variance(
                    x[use_finite], weights[use_finite], upper_quantile
                )
            elif beta_prior_method == "quantile":
                v = _match_upper_quantile_for_variance(x[use_finite], upper_quantile)
            else:
                raise DESeq2Error("beta_prior_method must be one of {'weighted', 'quantile'}.")
            beta_prior_var[j] = 1e6 if (not np.isfinite(v) or v <= 0) else v

        for j, name in enumerate(self.coef_names_):
            if name in {"Intercept", "(Intercept)"}:
                beta_prior_var[j] = 1e6
        return beta_prior_var

    def _expand_beta_prior_var(
        self,
        standard_prior_var: np.ndarray,
        expanded_coef_names: Sequence[str],
    ) -> np.ndarray:
        standard_prior_var = np.asarray(standard_prior_var, dtype=float)
        if standard_prior_var.shape[0] != len(self.coef_names_standard_):
            raise DESeq2Error("standard_prior_var length does not match standard coefficients.")

        prior_by_internal = {
            internal: float(v) for internal, v in zip(self.coef_names_standard_, standard_prior_var)
        }
        expanded_prior = np.zeros(len(expanded_coef_names), dtype=float)

        design_vars = self._design_vars()
        factor_info: dict[str, list[str]] = {}
        for var in design_vars:
            if var in self.metadata_.columns and isinstance(self.metadata_[var].dtype, pd.CategoricalDtype):
                factor_info[var] = [str(x) for x in self.metadata_[var].cat.categories]

        for j, name in enumerate(expanded_coef_names):
            if name in {"Intercept", "(Intercept)"}:
                expanded_prior[j] = 1e6
                continue
            assigned = False
            for var, levels in factor_info.items():
                if name.startswith(_make_valid_name(var)):
                    priors = []
                    ref = levels[0]
                    for lvl in levels[1:]:
                        for candidate in (
                            f"{var}[T.{lvl}]",
                            f"{var}[{lvl}]",
                            _make_valid_name(f"{var}[T.{lvl}]"),
                            _make_valid_name(f"{var}[{lvl}]"),
                        ):
                            if candidate in prior_by_internal:
                                priors.append(prior_by_internal[candidate])
                                break
                    if len(priors) > 0:
                        expanded_prior[j] = float(np.mean(priors))
                        assigned = True
                    break
            if assigned:
                continue
            if name in prior_by_internal:
                expanded_prior[j] = prior_by_internal[name]
            else:
                expanded_prior[j] = 1e6
        expanded_prior = np.where((expanded_prior > 0) & np.isfinite(expanded_prior), expanded_prior, 1e6)
        return expanded_prior

    def _build_contrast_vector(
        self, contrast: Union[np.ndarray, Sequence[float], dict[str, float]]
    ) -> np.ndarray:
        if isinstance(contrast, dict):
            vec = np.zeros(self.n_coefs, dtype=float)
            for key, value in contrast.items():
                internal = self._internal_coef_name(key)
                vec[self.coef_names_.index(internal)] = float(value)
            return vec
        vec = np.asarray(contrast, dtype=float)
        if vec.ndim != 1 or vec.shape[0] != self.n_coefs:
            raise DESeq2Error("Numeric contrast must have one value per model coefficient.")
        return vec

    def _contrast_all_zero_character(
        self, factor: str, numerator: str, denominator: str
    ) -> np.ndarray:
        if factor not in self.metadata_.columns:
            return np.zeros(self.n_genes, dtype=bool)
        f = self.metadata_[factor].astype(str).to_numpy()
        mask = np.isin(f, [numerator, denominator])
        if not np.any(mask):
            return np.zeros(self.n_genes, dtype=bool)
        cts_sub = self.counts_[:, mask]
        return np.sum(cts_sub == 0, axis=1) == cts_sub.shape[1]

    def _contrast_all_zero_numeric(self, cvec: np.ndarray) -> np.ndarray:
        if np.all(cvec >= 0) or np.all(cvec <= 0):
            return np.zeros(self.n_genes, dtype=bool)
        contrast_binary = np.where(cvec == 0, 0.0, 1.0)
        which_samples = np.where(self.model_matrix_ @ contrast_binary == 0, 0.0, 1.0)
        zero_test = self.counts_ @ which_samples
        return zero_test == 0

    def _character_contrast_to_vector(
        self, contrast_factor: str, num_level: str, denom_level: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if contrast_factor not in self.metadata_.columns:
            raise DESeq2Error(f"{contrast_factor} should be a factor in metadata.")
        var = self.metadata_[contrast_factor]
        if not isinstance(var.dtype, pd.CategoricalDtype):
            raise DESeq2Error(f"{contrast_factor} is not a categorical variable.")
        levels = [str(x) for x in var.cat.categories]
        if num_level not in levels or denom_level not in levels:
            raise DESeq2Error(
                f"Levels '{num_level}' and '{denom_level}' must be in {contrast_factor} levels {levels}."
            )
        if num_level == denom_level:
            raise DESeq2Error("Numerator and denominator levels should be different.")

        cvec = np.zeros(self.n_coefs, dtype=float)
        if self.model_matrix_type_ == "expanded":
            num_name = _make_valid_name(f"{contrast_factor}{num_level}")
            den_name = _make_valid_name(f"{contrast_factor}{denom_level}")
            num_internal = self._internal_coef_name(num_name)
            den_internal = self._internal_coef_name(den_name)
            cvec[self.coef_names_.index(num_internal)] = 1.0
            cvec[self.coef_names_.index(den_internal)] = -1.0
        else:
            base = levels[0]
            if denom_level == base:
                key = _make_valid_name(f"{contrast_factor}_{num_level}_vs_{base}")
                internal = self._internal_coef_name(key)
                cvec[self.coef_names_.index(internal)] = 1.0
            elif num_level == base:
                key = _make_valid_name(f"{contrast_factor}_{denom_level}_vs_{base}")
                internal = self._internal_coef_name(key)
                cvec[self.coef_names_.index(internal)] = -1.0
            else:
                num_key = _make_valid_name(f"{contrast_factor}_{num_level}_vs_{base}")
                den_key = _make_valid_name(f"{contrast_factor}_{denom_level}_vs_{base}")
                num_internal = self._internal_coef_name(num_key)
                den_internal = self._internal_coef_name(den_key)
                cvec[self.coef_names_.index(num_internal)] = 1.0
                cvec[self.coef_names_.index(den_internal)] = -1.0
        all_zero = self._contrast_all_zero_character(contrast_factor, num_level, denom_level)
        return cvec, all_zero

    def _parse_contrast_input(
        self,
        contrast: Optional[
            Union[
                np.ndarray,
                Sequence[float],
                dict[str, float],
                Sequence[str],
                Sequence[Sequence[str]],
            ]
        ],
        list_values: tuple[float, float] = (1.0, -1.0),
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if contrast is None:
            return None, None

        if isinstance(contrast, dict):
            cvec = self._build_contrast_vector(contrast)
            return cvec, self._contrast_all_zero_numeric(cvec)

        if isinstance(contrast, np.ndarray):
            if contrast.dtype.kind in {"U", "S", "O"} and contrast.ndim == 1 and contrast.shape[0] == 3:
                cvec, all_zero = self._character_contrast_to_vector(
                    str(contrast[0]), str(contrast[1]), str(contrast[2])
                )
                return cvec, all_zero
            cvec = self._build_contrast_vector(contrast)
            return cvec, self._contrast_all_zero_numeric(cvec)

        if isinstance(contrast, (list, tuple)):
            if len(contrast) == 3 and all(isinstance(x, str) for x in contrast):
                cvec, all_zero = self._character_contrast_to_vector(
                    str(contrast[0]), str(contrast[1]), str(contrast[2])
                )
                return cvec, all_zero

            if len(contrast) in {1, 2} and all(
                isinstance(x, (list, tuple, np.ndarray)) for x in contrast
            ):
                if len(contrast) == 1:
                    contrast = [contrast[0], []]
                num_names = [str(x) for x in contrast[0]]
                den_names = [str(x) for x in contrast[1]]
                if len(num_names) + len(den_names) == 0:
                    raise DESeq2Error("At least one side of list contrast must be non-empty.")
                overlap = set(num_names).intersection(set(den_names))
                if overlap:
                    raise DESeq2Error(
                        f"Contrast terms must not appear in both numerator and denominator: {sorted(overlap)}"
                    )
                cvec = np.zeros(self.n_coefs, dtype=float)
                for name in num_names:
                    internal = self._internal_coef_name(name)
                    cvec[self.coef_names_.index(internal)] += float(list_values[0])
                for name in den_names:
                    internal = self._internal_coef_name(name)
                    cvec[self.coef_names_.index(internal)] += float(list_values[1])
                return cvec, self._contrast_all_zero_numeric(cvec)

            try:
                cvec = self._build_contrast_vector(np.asarray(contrast, dtype=float))
                return cvec, self._contrast_all_zero_numeric(cvec)
            except Exception as exc:
                raise DESeq2Error(
                    "Unsupported contrast format. Use numeric vector, dict, "
                    "character 3-tuple (factor, numerator, denominator), or list-contrast."
                ) from exc

        raise DESeq2Error("Unsupported contrast type.")

    def _coef_covariance_row(
        self,
        row_idx: int,
        ridge_log2: np.ndarray,
    ) -> np.ndarray:
        if self.beta_ is None or self.dispersions_ is None:
            raise DESeq2Error("Model fit is incomplete; missing beta or dispersions.")
        nf = self._size_or_norm_factors()
        eta = self.model_matrix_ @ (self.beta_[row_idx, :] * LN2)
        eta = np.clip(np.where(np.isfinite(eta), eta, 0.0), -30.0, 30.0)
        mu_row = nf[row_idx, :] * np.exp(eta)
        alpha = self.dispersions_[row_idx]
        if not np.isfinite(alpha):
            return np.full((self.n_coefs, self.n_coefs), np.nan, dtype=float)
        if self.weights_ is not None:
            w = self.weights_[row_idx, :] / (1.0 / np.maximum(mu_row, 1e-12) + alpha)
        else:
            w = 1.0 / (1.0 / np.maximum(mu_row, 1e-12) + alpha)
        xtwx = self.model_matrix_.T @ (self.model_matrix_ * w[:, None])
        ridge_nat = np.diag(ridge_log2 / (LN2**2))
        try:
            xtwxr_inv = np.linalg.inv(xtwx + ridge_nat)
        except np.linalg.LinAlgError:
            xtwxr_inv = np.linalg.pinv(xtwx + ridge_nat)
        sigma_nat = xtwxr_inv @ xtwx @ xtwxr_inv
        sigma_log2 = (LOG2E**2) * sigma_nat
        return sigma_log2

    @staticmethod
    def _filtered_p(
        filter_stat: np.ndarray, pvalues: np.ndarray, theta: np.ndarray, method: str
    ) -> np.ndarray:
        filt = np.asarray(filter_stat, dtype=float)
        pv = np.asarray(pvalues, dtype=float)
        finite_filt = filt[np.isfinite(filt)]
        if finite_filt.size == 0:
            return np.full((filt.shape[0], theta.shape[0]), np.nan, dtype=float)
        cutoffs = np.quantile(finite_filt, theta)
        out = np.full((filt.shape[0], cutoffs.shape[0]), np.nan, dtype=float)
        for i, cutoff in enumerate(cutoffs):
            use = filt >= cutoff
            if np.any(use):
                out[use, i] = _adjust_pvalues(pv[use], method=method)
        return out

    def _apply_pvalue_adjustment(
        self,
        pvalues: np.ndarray,
        independent_filtering: bool,
        alpha: float,
        filter_stat: Optional[np.ndarray],
        theta: Optional[np.ndarray],
        p_adjust_method: str,
    ) -> np.ndarray:
        pvalues = np.asarray(pvalues, dtype=float)
        if not independent_filtering:
            return _adjust_pvalues(pvalues, method=p_adjust_method)

        if filter_stat is None:
            if self.base_mean_ is None:
                self._ensure_base_stats()
            filter_stat = self.base_mean_
        else:
            filter_stat = np.asarray(filter_stat, dtype=float)
            if filter_stat.shape[0] != self.n_genes:
                raise DESeq2Error("filter_stat must have one value per gene.")

        if theta is None:
            lower_quantile = float(np.mean(filter_stat == 0))
            upper_quantile = 0.95 if lower_quantile < 0.95 else 1.0
            theta = np.linspace(lower_quantile, upper_quantile, num=50)
        else:
            theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1 or theta.shape[0] <= 1:
            raise DESeq2Error("theta must be a 1D array with at least two values.")

        filt_padj = self._filtered_p(filter_stat, pvalues, theta, p_adjust_method)
        num_rej = np.nansum((filt_padj < alpha).astype(float), axis=0)

        lo_fit = lowess(num_rej, theta, frac=0.2, it=0, return_sorted=True)
        lo_fit_y = np.interp(theta, lo_fit[:, 0], lo_fit[:, 1], left=lo_fit[0, 1], right=lo_fit[-1, 1])
        if np.max(num_rej) <= 10:
            j = 0
        else:
            if np.all(num_rej == 0):
                residual = np.array([0.0], dtype=float)
            else:
                mask = num_rej > 0
                residual = num_rej[mask] - lo_fit_y[mask]
            max_fit = float(np.max(lo_fit_y))
            rmse = float(np.sqrt(np.mean(residual**2)))
            thresh = max_fit - rmse
            idx = np.where(num_rej > thresh)[0]
            if idx.size > 0:
                j = int(idx[0])
            else:
                idx = np.where(num_rej > 0.9 * max_fit)[0]
                if idx.size > 0:
                    j = int(idx[0])
                else:
                    idx = np.where(num_rej > 0.8 * max_fit)[0]
                    j = int(idx[0]) if idx.size > 0 else 0
        return filt_padj[:, j]

    def _grouping_to_score_vector(
        self,
        grouping: Union[str, np.ndarray, Sequence[float], pd.Series, Sequence[str]],
        numerator: Optional[str],
        denominator: Optional[str],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(grouping, str):
            if numerator is None or denominator is None:
                raise DESeq2Error(
                    "When grouping is a metadata column name, provide numerator and denominator."
                )
            factor = grouping
            if factor not in self.metadata_.columns:
                raise DESeq2Error(f"{factor} should be a factor in metadata.")
            var = self.metadata_[factor]
            levels = [str(x) for x in pd.Series(var).astype(str).unique()]
            if numerator not in levels or denominator not in levels:
                raise DESeq2Error(
                    f"Levels '{numerator}' and '{denominator}' must be present in {factor} values {levels}."
                )
            if numerator == denominator:
                raise DESeq2Error("Numerator and denominator levels should be different.")
            values = var.astype(str).to_numpy()
            z = np.zeros(self.n_samples, dtype=float)
            z[values == numerator] = 1.0
            z[values == denominator] = -1.0
            if np.sum(z == 1.0) == 0 or np.sum(z == -1.0) == 0:
                raise DESeq2Error("Numerator and denominator must each include at least one sample.")
            all_zero = self._contrast_all_zero_character(factor, numerator, denominator)
            return z, all_zero

        if isinstance(grouping, (list, tuple)) and len(grouping) == 3 and all(
            isinstance(x, str) for x in grouping
        ):
            factor = str(grouping[0])
            num = str(grouping[1])
            den = str(grouping[2])
            return self._grouping_to_score_vector(factor, num, den)

        if isinstance(grouping, pd.Series):
            if grouping.index.dtype != "object":
                grouping = grouping.copy()
                grouping.index = grouping.index.astype(str)
            if set(grouping.index) == set(self.sample_names_):
                arr = grouping.loc[self.sample_names_].to_numpy(dtype=float)
            elif grouping.shape[0] == self.n_samples:
                arr = grouping.to_numpy(dtype=float)
            else:
                raise DESeq2Error("grouping series must align to sample names or have one value per sample.")
        else:
            arr = np.asarray(grouping, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != self.n_samples:
            raise DESeq2Error("Numeric grouping vector must have one value per sample.")
        if np.any(~np.isfinite(arr)):
            raise DESeq2Error("Numeric grouping vector must be finite.")
        if np.all(np.isclose(arr, 0.0)):
            raise DESeq2Error("Numeric grouping vector is all zero.")
        return arr, self._active_samples_all_zero(arr)

    def _active_samples_all_zero(self, z: np.ndarray) -> Optional[np.ndarray]:
        active = np.abs(np.asarray(z, dtype=float)) > 0
        if not np.any(active):
            return None
        cts_sub = self.counts_[:, active]
        return np.sum(cts_sub == 0, axis=1) == cts_sub.shape[1]

    def prepare_score_test_null(
        self,
        null_design: str = "~ 1",
        beta_tol: float = 1e-8,
        maxit: int = 100,
        use_optim: bool = False,
        use_qr: bool = True,
        minmu: float = 0.5,
    ) -> "DESeq2":
        """Fit and cache a reduced (null) model used by fast score tests."""
        if self.size_factors_ is None and self.normalization_factors_ is None:
            self.estimate_size_factors()
        if self.dispersions_ is None:
            raise DESeq2Error("Dispersion estimates are missing; run estimate_dispersions first.")
        if self.all_zero_ is None:
            self._ensure_base_stats()

        null_matrix, null_coef_names, _ = _build_design_matrix(self.metadata_, null_design)
        nz_idx = np.where(~self.all_zero_)[0]
        if nz_idx.size == 0:
            raise DESeq2Error("All genes have zero counts.")

        counts_nz = self.counts_[nz_idx, :]
        nf_nz = self._size_or_norm_factors()[nz_idx, :]
        alpha_nz = self.dispersions_[nz_idx]
        weights_nz = self.weights_[nz_idx, :] if self.weights_ is not None else None

        fit = self._fit_nbinom_glms(
            counts=counts_nz,
            model_matrix=null_matrix,
            normalization_factors=nf_nz,
            alpha_hat=alpha_nz,
            beta_tol=beta_tol,
            maxit=maxit,
            use_optim=use_optim,
            use_qr=use_qr,
            minmu=minmu,
            weights=weights_nz,
        )

        beta_full = np.full((self.n_genes, null_matrix.shape[1]), np.nan, dtype=float)
        beta_full[nz_idx, :] = fit["beta_matrix"]
        mu_full = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        mu_full[nz_idx, :] = fit["mu"]
        beta_conv_full = np.full(self.n_genes, np.nan, dtype=float)
        beta_conv_full[nz_idx] = fit["beta_conv"].astype(float)
        beta_iter_full = np.full(self.n_genes, np.nan, dtype=float)
        beta_iter_full[nz_idx] = fit["beta_iter"].astype(float)

        self.score_null_design_ = null_design
        self.score_null_model_matrix_ = np.asarray(null_matrix, dtype=float)
        self.score_null_coef_names_ = list(null_coef_names)
        self.score_null_beta_ = beta_full
        self.score_null_mu_ = mu_full
        self.score_null_beta_conv_ = beta_conv_full
        self.score_null_beta_iter_ = beta_iter_full
        self.score_null_minmu_ = float(minmu)
        return self

    def score_test(
        self,
        grouping: Union[str, np.ndarray, Sequence[float], pd.Series, Sequence[str]],
        numerator: Optional[str] = None,
        denominator: Optional[str] = None,
        null_design: Optional[str] = None,
        refit_null: bool = False,
        beta_tol: float = 1e-8,
        maxit: int = 100,
        use_optim: bool = False,
        use_qr: bool = True,
        minmu: float = 0.5,
        cooks_cutoff: Optional[Union[bool, float]] = None,
        independent_filtering: bool = True,
        alpha: float = 0.1,
        filter_stat: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        p_adjust_method: str = "BH",
    ) -> pd.DataFrame:
        """Approximate fast DE test for arbitrary sample regrouping using Rao score statistics."""
        if self.dispersions_ is None:
            raise DESeq2Error("Dispersion estimates are missing; run estimate_dispersions first.")
        if self.all_zero_ is None:
            self._ensure_base_stats()

        design_to_use = null_design
        if design_to_use is None:
            design_to_use = self.score_null_design_ if self.score_null_design_ is not None else "~ 1"
        need_fit = (
            refit_null
            or self.score_null_model_matrix_ is None
            or self.score_null_mu_ is None
            or self.score_null_design_ != design_to_use
        )
        if need_fit:
            self.prepare_score_test_null(
                null_design=design_to_use,
                beta_tol=beta_tol,
                maxit=maxit,
                use_optim=use_optim,
                use_qr=use_qr,
                minmu=minmu,
            )
        if self.score_null_model_matrix_ is None or self.score_null_mu_ is None:
            raise DESeq2Error("Null score-test cache is missing; run prepare_score_test_null first.")

        z, contrast_all_zero = self._grouping_to_score_vector(grouping, numerator, denominator)
        x0 = self.score_null_model_matrix_

        res = pd.DataFrame(index=self.gene_names_)
        if self.base_mean_ is not None:
            res["baseMean"] = self.base_mean_
        if self.dispersions_ is not None:
            res["dispersion"] = self.dispersions_

        lfc = np.full(self.n_genes, np.nan, dtype=float)
        lfc_se = np.full(self.n_genes, np.nan, dtype=float)
        stat = np.full(self.n_genes, np.nan, dtype=float)
        pvalue = np.full(self.n_genes, np.nan, dtype=float)

        nz_rows = np.where(~self.all_zero_)[0]
        if nz_rows.size > 0:
            alpha_nz = self.dispersions_[nz_rows]
            mu_nz = self.score_null_mu_[nz_rows, :]
            valid = np.isfinite(alpha_nz) & np.all(np.isfinite(mu_nz), axis=1)
            if np.any(valid):
                rows = nz_rows[valid]
                mu_valid = np.maximum(mu_nz[valid, :], 1e-12)
                alpha_valid = alpha_nz[valid]
                y_valid = self.counts_[rows, :]
                if self.weights_ is not None:
                    obs_weights = np.maximum(self.weights_[rows, :], 1e-6)
                else:
                    obs_weights = np.ones_like(y_valid, dtype=float)

                denom = 1.0 + alpha_valid[:, None] * mu_valid
                score_w = obs_weights / denom
                info_w = obs_weights * mu_valid / denom
                resid = y_valid - mu_valid
                score_resid = score_w * resid

                u_gamma = score_resid @ x0
                u_beta = score_resid @ z
                i_gg = np.einsum("gn,nj,nk->gjk", info_w, x0, x0, optimize=True)
                i_gb = (info_w * z[None, :]) @ x0
                i_bb = info_w @ (z**2)

                n_valid = rows.size
                u_eff = np.full(n_valid, np.nan, dtype=float)
                i_eff = np.full(n_valid, np.nan, dtype=float)
                for i in range(n_valid):
                    mat = i_gg[i, :, :]
                    rhs_u = u_gamma[i, :]
                    rhs_i = i_gb[i, :]
                    try:
                        sol_u = np.linalg.solve(mat, rhs_u)
                        sol_i = np.linalg.solve(mat, rhs_i)
                    except np.linalg.LinAlgError:
                        pinv = np.linalg.pinv(mat)
                        sol_u = pinv @ rhs_u
                        sol_i = pinv @ rhs_i
                    u_eff_i = float(u_beta[i] - np.dot(i_gb[i, :], sol_u))
                    i_eff_i = float(i_bb[i] - np.dot(i_gb[i, :], sol_i))
                    if np.isfinite(u_eff_i):
                        u_eff[i] = u_eff_i
                    if np.isfinite(i_eff_i) and i_eff_i > 0.0:
                        i_eff[i] = i_eff_i

                good = np.isfinite(u_eff) & np.isfinite(i_eff) & (i_eff > 0.0)
                if np.any(good):
                    good_rows = rows[good]
                    beta_nat = u_eff[good] / i_eff[good]
                    se = LOG2E / np.sqrt(i_eff[good])
                    lfc[good_rows] = LOG2E * beta_nat
                    lfc_se[good_rows] = se
                    stat[good_rows] = u_eff[good] / np.sqrt(i_eff[good])
                    pvalue[good_rows] = 2.0 * norm.sf(np.abs(stat[good_rows]))

        res["log2FoldChange"] = lfc
        res["lfcSE"] = lfc_se
        res["stat"] = stat
        res["pvalue"] = pvalue

        if contrast_all_zero is not None:
            zero_mask = np.asarray(contrast_all_zero, dtype=bool).copy()
            if self.all_zero_ is not None:
                zero_mask &= ~self.all_zero_
            res.loc[zero_mask, "log2FoldChange"] = 0.0
            res.loc[zero_mask, "stat"] = 0.0
            res.loc[zero_mask, "pvalue"] = 1.0

        # Cook's distance filtering of p-values.
        if self.max_cooks_ is not None:
            m = self.n_samples
            p = self.disp_model_matrix_.shape[1] if self.disp_model_matrix_ is not None else self.n_coefs
            default_cutoff = f_dist.ppf(0.99, p, max(m - p, 1))
            if cooks_cutoff is None:
                cutoff = float(default_cutoff)
                perform_cooks_cutoff = True
            elif isinstance(cooks_cutoff, (bool, np.bool_)):
                perform_cooks_cutoff = bool(cooks_cutoff)
                cutoff = float(default_cutoff)
            else:
                cutoff = float(cooks_cutoff)
                perform_cooks_cutoff = True

            if perform_cooks_cutoff:
                cooks_outlier = np.asarray(self.max_cooks_ > cutoff, dtype=bool)
                cooks_outlier &= np.isfinite(self.max_cooks_)
                pvalue_arr = res["pvalue"].to_numpy(dtype=float, copy=True)
                pvalue_arr[cooks_outlier] = np.nan
                res["pvalue"] = pvalue_arr

        res["padj"] = self._apply_pvalue_adjustment(
            res["pvalue"].to_numpy(dtype=float),
            independent_filtering=independent_filtering,
            alpha=float(alpha),
            filter_stat=filter_stat,
            theta=theta,
            p_adjust_method=p_adjust_method,
        )
        return res

    @staticmethod
    def _log_posterior(
        log_alpha: float,
        y: np.ndarray,
        mu: np.ndarray,
        x: np.ndarray,
        log_alpha_prior_mean: float,
        log_alpha_prior_sigmasq: float,
        use_prior: bool,
        weights: np.ndarray,
        use_weights: bool,
        weight_threshold: float,
        use_cr: bool,
    ) -> float:
        alpha = float(np.exp(log_alpha))
        if alpha <= 0.0 or not np.isfinite(alpha):
            return -np.inf

        cr_term = 0.0
        if use_cr:
            w_diag = 1.0 / (1.0 / mu + alpha)
            x_cr = x
            if use_weights:
                keep = weights > weight_threshold
                if not np.any(keep):
                    return -np.inf
                x_cr = x_cr[keep, :]
                keep_cols = np.sum(np.abs(x_cr), axis=0) > 0.0
                x_cr = x_cr[:, keep_cols]
                w_diag = w_diag[keep]
            b = x_cr.T @ (x_cr * w_diag[:, None])
            sign, logdet = np.linalg.slogdet(b)
            if sign <= 0 or not np.isfinite(logdet):
                return -np.inf
            cr_term = -0.5 * logdet

        alpha_neg1 = 1.0 / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            ll_vec = (
                gammaln(y + alpha_neg1)
                - gammaln(alpha_neg1)
                - y * np.log(mu + alpha_neg1)
                - alpha_neg1 * np.log1p(mu * alpha)
            )
        if use_weights:
            ll_part = np.sum(weights * ll_vec)
        else:
            ll_part = np.sum(ll_vec)

        prior_part = 0.0
        if use_prior:
            prior_part = -0.5 * ((log_alpha - log_alpha_prior_mean) ** 2) / log_alpha_prior_sigmasq

        return float(ll_part + prior_part + cr_term)

    @staticmethod
    def _dlog_posterior(
        log_alpha: float,
        y: np.ndarray,
        mu: np.ndarray,
        x: np.ndarray,
        log_alpha_prior_mean: float,
        log_alpha_prior_sigmasq: float,
        use_prior: bool,
        weights: np.ndarray,
        use_weights: bool,
        weight_threshold: float,
        use_cr: bool,
    ) -> float:
        alpha = float(np.exp(log_alpha))
        if alpha <= 0.0 or not np.isfinite(alpha):
            return 0.0

        cr_term = 0.0
        if use_cr:
            w_diag = 1.0 / (1.0 / mu + alpha)
            dw_diag = -1.0 / ((1.0 / mu + alpha) ** 2)
            x_cr = x
            if use_weights:
                keep = weights > weight_threshold
                if not np.any(keep):
                    return 0.0
                x_cr = x_cr[keep, :]
                keep_cols = np.sum(np.abs(x_cr), axis=0) > 0.0
                x_cr = x_cr[:, keep_cols]
                w_diag = w_diag[keep]
                dw_diag = dw_diag[keep]
            b = x_cr.T @ (x_cr * w_diag[:, None])
            db = x_cr.T @ (x_cr * dw_diag[:, None])
            try:
                tr = float(np.trace(np.linalg.solve(b, db)))
            except np.linalg.LinAlgError:
                tr = 0.0
            cr_term = -0.5 * tr

        alpha_neg1 = 1.0 / alpha
        alpha_neg2 = alpha_neg1**2
        with np.errstate(divide="ignore", invalid="ignore"):
            inner = (
                digamma(alpha_neg1)
                + np.log1p(mu * alpha)
                - (mu * alpha) / (1.0 + mu * alpha)
                - digamma(y + alpha_neg1)
                + y / (mu + alpha_neg1)
            )
        if use_weights:
            ll_part = alpha_neg2 * np.sum(weights * inner)
        else:
            ll_part = alpha_neg2 * np.sum(inner)

        prior_part = 0.0
        if use_prior:
            prior_part = -(log_alpha - log_alpha_prior_mean) / log_alpha_prior_sigmasq

        return float((ll_part + cr_term) * alpha + prior_part)

    @classmethod
    def _fit_disp(
        cls,
        y: np.ndarray,
        x: np.ndarray,
        mu_hat: np.ndarray,
        log_alpha: np.ndarray,
        log_alpha_prior_mean: np.ndarray,
        log_alpha_prior_sigmasq: float,
        min_log_alpha: float,
        kappa_0: float,
        tol: float,
        maxit: int,
        use_prior: bool,
        weights: np.ndarray,
        use_weights: bool,
        weight_threshold: float,
        use_cr: bool,
    ) -> dict:
        if _fit_disp_cy is not None:
            try:
                y_c = np.asarray(y, dtype=float, order="C")
                mu_c = np.asarray(mu_hat, dtype=float, order="C")
                log_alpha_c = np.asarray(log_alpha, dtype=float, order="C")
                prior_mean_c = np.asarray(log_alpha_prior_mean, dtype=float, order="C")
                x_c = np.asarray(x, dtype=float, order="C")
                if use_weights:
                    weights_c = np.asarray(weights, dtype=float, order="C")
                else:
                    weights_c = np.ones_like(y_c, dtype=float)
                return _fit_disp_cy(
                    y_c,
                    mu_c,
                    log_alpha_c,
                    prior_mean_c,
                    float(log_alpha_prior_sigmasq),
                    float(min_log_alpha),
                    float(kappa_0),
                    float(tol),
                    int(maxit),
                    bool(use_prior),
                    weights_c,
                    bool(use_weights),
                    x=x_c,
                    use_cr=bool(use_cr),
                    weight_threshold=float(weight_threshold),
                    n_threads=_cython_num_threads(),
                )
            except Exception as exc:  # pragma: no cover - runtime fallback safety.
                _handle_cython_backend_issue("dispersion", exc)
        else:
            _handle_cython_backend_issue("dispersion")

        y_n = y.shape[0]
        log_alpha_out = np.asarray(log_alpha, dtype=float).copy()
        initial_lp = np.empty(y_n, dtype=float)
        last_lp = np.empty(y_n, dtype=float)
        last_change = np.empty(y_n, dtype=float)
        iterations = np.zeros(y_n, dtype=int)
        iter_accept = np.zeros(y_n, dtype=int)
        epsilon = 1.0e-4

        for i in range(y_n):
            y_row = y[i, :]
            mu_row = mu_hat[i, :]
            w_row = weights[i, :] if use_weights else np.ones_like(mu_row)
            a = float(log_alpha_out[i])
            lp = cls._log_posterior(
                a,
                y_row,
                mu_row,
                x,
                float(log_alpha_prior_mean[i]),
                log_alpha_prior_sigmasq,
                use_prior,
                w_row,
                use_weights,
                weight_threshold,
                use_cr,
            )
            dlp = cls._dlog_posterior(
                a,
                y_row,
                mu_row,
                x,
                float(log_alpha_prior_mean[i]),
                log_alpha_prior_sigmasq,
                use_prior,
                w_row,
                use_weights,
                weight_threshold,
                use_cr,
            )
            kappa = float(kappa_0)
            initial_lp[i] = lp
            change = -1.0
            for _ in range(maxit):
                iterations[i] += 1
                if dlp == 0.0 or not np.isfinite(dlp):
                    break
                a_propose = a + kappa * dlp
                if a_propose < -30.0:
                    kappa = (-30.0 - a) / dlp
                if a_propose > 10.0:
                    kappa = (10.0 - a) / dlp

                theta_kappa = -cls._log_posterior(
                    a + kappa * dlp,
                    y_row,
                    mu_row,
                    x,
                    float(log_alpha_prior_mean[i]),
                    log_alpha_prior_sigmasq,
                    use_prior,
                    w_row,
                    use_weights,
                    weight_threshold,
                    use_cr,
                )
                theta_hat_kappa = -lp - kappa * epsilon * (dlp**2)
                if theta_kappa <= theta_hat_kappa:
                    iter_accept[i] += 1
                    a = a + kappa * dlp
                    lp_new = cls._log_posterior(
                        a,
                        y_row,
                        mu_row,
                        x,
                        float(log_alpha_prior_mean[i]),
                        log_alpha_prior_sigmasq,
                        use_prior,
                        w_row,
                        use_weights,
                        weight_threshold,
                        use_cr,
                    )
                    change = lp_new - lp
                    if change < tol:
                        lp = lp_new
                        break
                    if a < min_log_alpha:
                        break
                    lp = lp_new
                    dlp = cls._dlog_posterior(
                        a,
                        y_row,
                        mu_row,
                        x,
                        float(log_alpha_prior_mean[i]),
                        log_alpha_prior_sigmasq,
                        use_prior,
                        w_row,
                        use_weights,
                        weight_threshold,
                        use_cr,
                    )
                    kappa = min(kappa * 1.1, kappa_0)
                    if iter_accept[i] % 5 == 0:
                        kappa = kappa / 2.0
                else:
                    kappa = kappa / 2.0
                    if kappa < 1e-12:
                        break
            last_lp[i] = lp
            last_change[i] = change
            log_alpha_out[i] = a

        return {
            "log_alpha": log_alpha_out,
            "iter": iterations,
            "iter_accept": iter_accept,
            "last_change": last_change,
            "initial_lp": initial_lp,
            "last_lp": last_lp,
        }

    @classmethod
    def _fit_disp_grid(
        cls,
        y: np.ndarray,
        x: np.ndarray,
        mu_hat: np.ndarray,
        log_alpha_prior_mean: np.ndarray,
        log_alpha_prior_sigmasq: float,
        use_prior: bool,
        weights: np.ndarray,
        use_weights: bool,
        weight_threshold: float,
        use_cr: bool,
    ) -> np.ndarray:
        if _fit_disp_grid_cy is not None:
            try:
                y_c = np.asarray(y, dtype=float, order="C")
                mu_c = np.asarray(mu_hat, dtype=float, order="C")
                prior_mean_c = np.asarray(log_alpha_prior_mean, dtype=float, order="C")
                x_c = np.asarray(x, dtype=float, order="C")
                if use_weights:
                    weights_c = np.asarray(weights, dtype=float, order="C")
                else:
                    weights_c = np.ones_like(y_c, dtype=float)
                return np.asarray(
                    _fit_disp_grid_cy(
                        y_c,
                        mu_c,
                        prior_mean_c,
                        float(log_alpha_prior_sigmasq),
                        bool(use_prior),
                        weights_c,
                        bool(use_weights),
                        x=x_c,
                        use_cr=bool(use_cr),
                        weight_threshold=float(weight_threshold),
                        n_threads=_cython_num_threads(),
                    ),
                    dtype=float,
                )
            except Exception as exc:  # pragma: no cover - runtime fallback safety.
                _handle_cython_backend_issue("dispersion-grid", exc)
        else:
            _handle_cython_backend_issue("dispersion-grid")

        min_log_alpha = np.log(1e-8)
        max_log_alpha = np.log(max(10, y.shape[1]))
        disp_grid = np.linspace(min_log_alpha, max_log_alpha, num=20)
        out = np.empty(y.shape[0], dtype=float)

        for i in range(y.shape[0]):
            y_row = y[i, :]
            mu_row = mu_hat[i, :]
            w_row = weights[i, :] if use_weights else np.ones_like(mu_row)
            logpost = np.array(
                [
                    cls._log_posterior(
                        a,
                        y_row,
                        mu_row,
                        x,
                        float(log_alpha_prior_mean[i]),
                        log_alpha_prior_sigmasq,
                        use_prior,
                        w_row,
                        use_weights,
                        weight_threshold,
                        use_cr,
                    )
                    for a in disp_grid
                ],
                dtype=float,
            )
            idxmax = int(np.nanargmax(logpost))
            a_hat = disp_grid[idxmax]
            delta = disp_grid[1] - disp_grid[0]
            fine = np.linspace(a_hat - delta, a_hat + delta, num=20)
            fine_logpost = np.array(
                [
                    cls._log_posterior(
                        a,
                        y_row,
                        mu_row,
                        x,
                        float(log_alpha_prior_mean[i]),
                        log_alpha_prior_sigmasq,
                        use_prior,
                        w_row,
                        use_weights,
                        weight_threshold,
                        use_cr,
                    )
                    for a in fine
                ],
                dtype=float,
            )
            out[i] = fine[int(np.nanargmax(fine_logpost))]
        return np.exp(out)

    @staticmethod
    def _parametric_dispersion_fit(means: np.ndarray, disps: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        coefs = np.array([0.1, 1.0], dtype=float)
        for _ in range(11):
            residuals = disps / (coefs[0] + coefs[1] / means)
            good = np.where((residuals > 1e-4) & (residuals < 15.0))[0]
            if good.size < 2:
                raise DESeq2Error("Parametric dispersion fit failed: too few usable points.")
            x = np.column_stack([np.ones(good.size), 1.0 / means[good]])
            y = disps[good]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DomainWarning)
                model = sm.GLM(
                    y,
                    x,
                    family=sm.families.Gamma(link=sm.families.links.Identity()),
                )
                fit = model.fit(start_params=coefs, maxiter=100, disp=0)
            old = coefs.copy()
            coefs = np.asarray(fit.params, dtype=float)
            if np.any(coefs <= 0):
                raise DESeq2Error("Parametric dispersion fit failed: non-positive coefficients.")
            if np.sum(np.log(coefs / old) ** 2) < 1e-6 and bool(fit.converged):
                break
        else:
            raise DESeq2Error("Parametric dispersion fit did not converge.")

        def disp_fn(q: np.ndarray) -> np.ndarray:
            return coefs[0] + coefs[1] / np.asarray(q, dtype=float)

        return disp_fn

    @staticmethod
    def _local_dispersion_fit(means: np.ndarray, disps: np.ndarray, min_disp: float) -> Callable[[np.ndarray], np.ndarray]:
        if np.all(disps < min_disp * 10):
            def constant(means_new: np.ndarray) -> np.ndarray:
                return np.full_like(np.asarray(means_new, dtype=float), min_disp)
            return constant
        use = disps >= min_disp * 10
        x = np.log(means[use])
        y = np.log(disps[use])
        fit = lowess(y, x, frac=0.4, it=0, return_sorted=True)
        x_fit = fit[:, 0]
        y_fit = fit[:, 1]

        def disp_fn(means_new: np.ndarray) -> np.ndarray:
            m = np.log(np.asarray(means_new, dtype=float))
            pred = np.interp(m, x_fit, y_fit, left=y_fit[0], right=y_fit[-1])
            return np.exp(pred)

        return disp_fn

    @staticmethod
    def _calculate_cooks_distance(
        counts: np.ndarray,
        mu: np.ndarray,
        hat_diagonals: np.ndarray,
        model_matrix: np.ndarray,
        normalization_factors: np.ndarray,
    ) -> np.ndarray:
        p = model_matrix.shape[1]
        cnts_norm = counts / normalization_factors
        dispersions = _robust_method_of_moments_disp(cnts_norm, model_matrix)
        v = mu + dispersions[:, None] * mu**2
        with np.errstate(divide="ignore", invalid="ignore"):
            pearson_res_sq = (counts - mu) ** 2 / v
            cooks = pearson_res_sq / p * hat_diagonals / np.clip((1.0 - hat_diagonals) ** 2, 1e-12, None)
        return cooks

    @classmethod
    def _fit_nbinom_glms(
        cls,
        counts: np.ndarray,
        model_matrix: np.ndarray,
        normalization_factors: np.ndarray,
        alpha_hat: np.ndarray,
        lambda_: Optional[np.ndarray] = None,
        beta_tol: float = 1e-8,
        maxit: int = 100,
        use_optim: bool = True,
        use_qr: bool = True,
        force_optim: bool = False,
        minmu: float = 0.5,
        weights: Optional[np.ndarray] = None,
        mu_only: bool = False,
        return_optimizer_rows: bool = False,
    ) -> dict:
        counts = _as_float_matrix(counts, "counts")
        x = _as_float_matrix(model_matrix, "model_matrix")
        nf = _as_float_matrix(normalization_factors, "normalization_factors")
        alpha = np.asarray(alpha_hat, dtype=float)
        if counts.shape != nf.shape:
            raise DESeq2Error("counts and normalization_factors must have the same shape.")
        if counts.shape[0] != alpha.shape[0]:
            raise DESeq2Error("alpha_hat must have one value per gene.")
        if counts.shape[1] != x.shape[0]:
            raise DESeq2Error("model_matrix rows must equal the number of samples.")

        n_genes, n_samples = counts.shape
        p = x.shape[1]
        if lambda_ is None:
            lambda_log2 = np.repeat(1e-6, p)
        else:
            lambda_log2 = np.asarray(lambda_, dtype=float)
            if lambda_log2.ndim == 0:
                lambda_log2 = np.repeat(float(lambda_log2), p)
            if lambda_log2.shape[0] != p:
                raise DESeq2Error("lambda must have one value per coefficient.")
        lambda_nat = lambda_log2 / (LN2**2)

        use_weights = weights is not None
        if use_weights:
            wmat = _as_float_matrix(weights, "weights")
            if wmat.shape != counts.shape:
                raise DESeq2Error("weights must have same shape as counts.")
            wmat = np.maximum(wmat, 1e-6)
        else:
            wmat = np.ones_like(counts)

        qrx_rank = np.linalg.matrix_rank(x)
        if qrx_rank == p:
            q, r = np.linalg.qr(x, mode="reduced")
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(counts / nf + 0.1)
            y = np.where(np.isfinite(y), y, 0.0)
            y = np.clip(y, -30.0, 30.0).T
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                rhs = q.T @ y
            rhs = np.where(np.isfinite(rhs), rhs, 0.0)
            beta_nat_init = np.linalg.solve(r, rhs).T
        else:
            beta_nat_init = np.zeros((n_genes, p), dtype=float)
            intercept_cols = np.where(np.all(np.isclose(x, 1.0), axis=0))[0]
            if intercept_cols.size > 0:
                log_base_mean = np.log(np.mean(counts / nf, axis=1))
                beta_nat_init[:, intercept_cols[0]] = log_base_mean
            else:
                beta_nat_init[:] = 1.0

        ridge = np.diag(lambda_nat)
        ridge_sqrt = np.diag(np.sqrt(lambda_nat))
        large = 30.0

        if mu_only:
            used_cython_glm = False
            if _fit_glm_cy is not None:
                try:
                    glm_core = _fit_glm_cy(
                        counts=np.asarray(counts, dtype=float, order="C"),
                        model_matrix=np.asarray(x, dtype=float, order="C"),
                        normalization_factors=np.asarray(nf, dtype=float, order="C"),
                        alpha_hat=np.asarray(alpha, dtype=float, order="C"),
                        beta_nat_init=np.asarray(beta_nat_init, dtype=float, order="C"),
                        lambda_nat=np.asarray(lambda_nat, dtype=float, order="C"),
                        beta_tol=float(beta_tol),
                        maxit=int(maxit),
                        minmu=float(minmu),
                        use_weights=bool(use_weights),
                        weights=np.asarray(wmat, dtype=float, order="C"),
                        use_qr=bool(use_qr),
                        mu_only=True,
                        n_threads=_cython_num_threads(),
                    )
                    mu_cy = glm_core.get("mu")
                    if mu_cy is not None:
                        return {
                            "mu": np.asarray(mu_cy, dtype=float),
                            "beta_iter": np.asarray(
                                glm_core.get("beta_iter", np.zeros(n_genes, dtype=int)),
                                dtype=int,
                            ),
                        }
                    used_cython_glm = True
                    beta_nat_tmp = np.asarray(glm_core["beta_nat"], dtype=float)
                    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                        eta_tmp = (x @ beta_nat_tmp.T).T
                    eta_tmp = np.where(np.isfinite(eta_tmp), eta_tmp, 0.0)
                    eta_tmp = np.clip(eta_tmp, -30.0, 30.0)
                    with np.errstate(over="ignore"):
                        mu_tmp = nf * np.exp(eta_tmp)
                    return {
                        "mu": mu_tmp,
                        "beta_iter": np.asarray(
                            glm_core.get("beta_iter", np.zeros(n_genes, dtype=int)),
                            dtype=int,
                        ),
                    }
                except Exception as exc:  # pragma: no cover - runtime fallback safety.
                    _handle_cython_backend_issue("GLM", exc)
            else:
                _handle_cython_backend_issue("GLM")

            mu = np.full((n_genes, n_samples), np.nan, dtype=float)
            beta_iter = np.zeros(n_genes, dtype=int)
            for i in range(n_genes):
                yrow = counts[i, :]
                nfrow = nf[i, :]
                beta_hat = beta_nat_init[i, :].copy()
                alpha_i = float(alpha[i])
                mu_hat = nfrow * np.exp(x @ beta_hat)
                mu_hat = np.maximum(mu_hat, minmu)
                dev_old = 0.0
                dev = np.nan

                for t in range(maxit):
                    beta_iter[i] += 1
                    if use_weights:
                        w_vec = wmat[i, :] * mu_hat / (1.0 + alpha_i * mu_hat)
                    else:
                        w_vec = mu_hat / (1.0 + alpha_i * mu_hat)
                    w_sqrt = np.sqrt(w_vec)
                    z = np.log(mu_hat / nfrow) + (yrow - mu_hat) / mu_hat

                    try:
                        if use_qr:
                            weighted_x_ridge = np.vstack([x * w_sqrt[:, None], ridge_sqrt])
                            q, r = np.linalg.qr(weighted_x_ridge, mode="reduced")
                            big_z = np.zeros(n_samples + p, dtype=float)
                            big_z[:n_samples] = z * w_sqrt
                            gamma_hat = q.T @ big_z
                            beta_hat = np.linalg.solve(r, gamma_hat)
                        else:
                            a_mat = x.T @ (x * w_vec[:, None]) + ridge
                            b_vec = x.T @ (z * w_vec)
                            beta_hat = np.linalg.solve(a_mat, b_vec)
                    except np.linalg.LinAlgError:
                        beta_iter[i] = maxit
                        break

                    if np.any(np.abs(beta_hat) > large):
                        beta_iter[i] = maxit
                        break

                    mu_hat = nfrow * np.exp(x @ beta_hat)
                    mu_hat = np.maximum(mu_hat, minmu)
                    ll_row = _nbinom_logpmf(yrow, mu_hat, np.array(alpha_i))
                    if use_weights:
                        dev = -2.0 * np.sum(wmat[i, :] * ll_row)
                    else:
                        dev = -2.0 * np.sum(ll_row)
                    conv_test = abs(dev - dev_old) / (abs(dev) + 0.1)
                    if not np.isfinite(conv_test):
                        beta_iter[i] = maxit
                        break
                    if (t > 0) and (conv_test < beta_tol):
                        break
                    dev_old = dev

                mu[i, :] = mu_hat
            return {"mu": mu, "beta_iter": beta_iter}

        beta_nat = np.full((n_genes, p), np.nan, dtype=float)
        beta_var_nat = np.full((n_genes, p), np.nan, dtype=float)
        hat_diagonals = np.full((n_genes, n_samples), np.nan, dtype=float)
        beta_iter = np.zeros(n_genes, dtype=int)

        used_cython_glm = False
        if _fit_glm_cy is not None:
            try:
                glm_core = _fit_glm_cy(
                    counts=np.asarray(counts, dtype=float, order="C"),
                    model_matrix=np.asarray(x, dtype=float, order="C"),
                    normalization_factors=np.asarray(nf, dtype=float, order="C"),
                    alpha_hat=np.asarray(alpha, dtype=float, order="C"),
                    beta_nat_init=np.asarray(beta_nat_init, dtype=float, order="C"),
                    lambda_nat=np.asarray(lambda_nat, dtype=float, order="C"),
                    beta_tol=float(beta_tol),
                    maxit=int(maxit),
                    minmu=float(minmu),
                    use_weights=bool(use_weights),
                    weights=np.asarray(wmat, dtype=float, order="C"),
                    use_qr=bool(use_qr),
                    mu_only=False,
                    n_threads=_cython_num_threads(),
                )
                beta_nat = np.asarray(glm_core["beta_nat"], dtype=float)
                beta_var_nat = np.asarray(glm_core["beta_var_nat"], dtype=float)
                hat_diagonals = np.asarray(glm_core["hat_diagonals"], dtype=float)
                beta_iter = np.asarray(glm_core["beta_iter"], dtype=int)
                used_cython_glm = True
            except Exception as exc:  # pragma: no cover - runtime fallback safety.
                _handle_cython_backend_issue("GLM", exc)
        else:
            _handle_cython_backend_issue("GLM")

        if not used_cython_glm:
            for i in range(n_genes):
                yrow = counts[i, :]
                nfrow = nf[i, :]
                beta_hat = beta_nat_init[i, :].copy()
                alpha_i = float(alpha[i])
                mu_hat = nfrow * np.exp(x @ beta_hat)
                mu_hat = np.maximum(mu_hat, minmu)
                dev_old = 0.0
                dev = np.nan

                for t in range(maxit):
                    beta_iter[i] += 1
                    if use_weights:
                        w_vec = wmat[i, :] * mu_hat / (1.0 + alpha_i * mu_hat)
                    else:
                        w_vec = mu_hat / (1.0 + alpha_i * mu_hat)
                    w_sqrt = np.sqrt(w_vec)
                    z = np.log(mu_hat / nfrow) + (yrow - mu_hat) / mu_hat

                    try:
                        if use_qr:
                            weighted_x_ridge = np.vstack([x * w_sqrt[:, None], ridge_sqrt])
                            q, r = np.linalg.qr(weighted_x_ridge, mode="reduced")
                            big_z = np.zeros(n_samples + p, dtype=float)
                            big_z[:n_samples] = z * w_sqrt
                            gamma_hat = q.T @ big_z
                            beta_hat = np.linalg.solve(r, gamma_hat)
                        else:
                            a_mat = x.T @ (x * w_vec[:, None]) + ridge
                            b_vec = x.T @ (z * w_vec)
                            beta_hat = np.linalg.solve(a_mat, b_vec)
                    except np.linalg.LinAlgError:
                        beta_iter[i] = maxit
                        break

                    if np.any(np.abs(beta_hat) > large):
                        beta_iter[i] = maxit
                        break

                    mu_hat = nfrow * np.exp(x @ beta_hat)
                    mu_hat = np.maximum(mu_hat, minmu)
                    ll_row = _nbinom_logpmf(yrow, mu_hat, np.array(alpha_i))
                    if use_weights:
                        dev = -2.0 * np.sum(wmat[i, :] * ll_row)
                    else:
                        dev = -2.0 * np.sum(ll_row)
                    conv_test = abs(dev - dev_old) / (abs(dev) + 0.1)
                    if not np.isfinite(conv_test):
                        beta_iter[i] = maxit
                        break
                    if (t > 0) and (conv_test < beta_tol):
                        break
                    dev_old = dev

                beta_nat[i, :] = beta_hat

                if use_weights:
                    w_vec = wmat[i, :] * mu_hat / (1.0 + alpha_i * mu_hat)
                else:
                    w_vec = mu_hat / (1.0 + alpha_i * mu_hat)
                xw = x * np.sqrt(w_vec)[:, None]
                xtwx = x.T @ (x * w_vec[:, None])
                try:
                    xtwxr_inv = np.linalg.inv(xtwx + ridge)
                except np.linalg.LinAlgError:
                    xtwxr_inv = np.linalg.pinv(xtwx + ridge)
                hat_diagonals[i, :] = np.einsum("ij,jk,ik->i", xw, xtwxr_inv, xw)
                sigma = xtwxr_inv @ xtwx @ xtwxr_inv
                beta_var_nat[i, :] = np.diag(sigma)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            eta = (x @ beta_nat.T).T
        eta = np.where(np.isfinite(eta), eta, 0.0)
        eta = np.clip(eta, -30.0, 30.0)
        with np.errstate(over="ignore"):
            mu = nf * np.exp(eta)
        ll = _nbinom_logpmf(counts, mu, alpha)
        if use_weights:
            log_like = np.sum(wmat * ll, axis=1)
        else:
            log_like = np.sum(ll, axis=1)

        row_stable = np.all(np.isfinite(beta_nat), axis=1)
        row_var_positive = np.all(beta_var_nat > 0, axis=1)
        beta_conv = beta_iter < maxit
        beta_matrix = LOG2E * beta_nat
        beta_se = LOG2E * np.sqrt(np.clip(beta_var_nat, 0.0, None))

        if use_optim:
            rows_for_optim = np.where(~beta_conv | ~row_stable | ~row_var_positive)[0]
        else:
            rows_for_optim = np.where(~row_stable | ~row_var_positive)[0]
        if force_optim:
            rows_for_optim = np.arange(n_genes)

        for row in rows_for_optim:
            if row_stable[row] and np.all(np.abs(beta_matrix[row, :]) < large):
                beta_start = beta_matrix[row, :].copy()
            else:
                beta_start = LOG2E * beta_nat_init[row, :]
            yrow = counts[row, :]
            nfrow = nf[row, :]
            alpha_row = float(alpha[row])

            def objective(pvec: np.ndarray) -> float:
                mu_row = nfrow * np.power(2.0, x @ pvec)
                ll_vec = _nbinom_logpmf(yrow, mu_row, np.array(alpha_row))
                if use_weights:
                    ll_val = np.sum(wmat[row, :] * ll_vec)
                else:
                    ll_val = np.sum(ll_vec)
                log_prior = -0.5 * np.sum(lambda_log2 * (pvec**2))
                neg_log_post = -1.0 * (ll_val + log_prior)
                if np.isfinite(neg_log_post):
                    return float(neg_log_post)
                return 1e300

            bounds = [(-large, large) for _ in range(p)]
            res = minimize(objective, beta_start, method="L-BFGS-B", bounds=bounds)
            if res.success:
                beta_conv[row] = True
            beta_matrix[row, :] = res.x

            mu_row = nfrow * np.power(2.0, x @ res.x)
            mu[row, :] = mu_row
            mu_clamped = np.maximum(mu_row, minmu)
            if use_weights:
                w_diag = wmat[row, :] / (1.0 / mu_clamped + alpha_row)
            else:
                w_diag = 1.0 / (1.0 / mu_clamped + alpha_row)
            xtwx = x.T @ (x * w_diag[:, None])
            try:
                xtwx_ridge_inv = np.linalg.inv(xtwx + ridge)
            except np.linalg.LinAlgError:
                xtwx_ridge_inv = np.linalg.pinv(xtwx + ridge)
            sigma = xtwx_ridge_inv @ xtwx @ xtwx_ridge_inv
            beta_se[row, :] = LOG2E * np.sqrt(np.clip(np.diag(sigma), 0.0, None))
            ll_vec = _nbinom_logpmf(yrow, mu_clamped, np.array(alpha_row))
            if use_weights:
                log_like[row] = np.sum(wmat[row, :] * ll_vec)
            else:
                log_like[row] = np.sum(ll_vec)

        out = {
            "log_like": log_like,
            "beta_conv": beta_conv,
            "beta_matrix": beta_matrix,
            "beta_se": beta_se,
            "mu": mu,
            "beta_iter": beta_iter,
            "hat_diagonals": hat_diagonals,
        }
        if return_optimizer_rows:
            out["n_rows_for_optim"] = int(rows_for_optim.size)
            out["n_genes"] = int(n_genes)
        return out

    def estimate_dispersions_gene_est(
        self,
        min_disp: float = 1e-8,
        kappa_0: float = 1.0,
        disp_tol: float = 1e-6,
        maxit: int = 100,
        use_cr: bool = True,
        weight_threshold: float = 1e-2,
        niter: int = 1,
        linear_mu: Optional[bool] = None,
        minmu: float = 0.5,
        alpha_init: Optional[Union[float, np.ndarray]] = None,
    ) -> "DESeq2":
        if self.size_factors_ is None and self.normalization_factors_ is None:
            self.estimate_size_factors()
        self._ensure_base_stats()

        model_matrix = self.model_matrix_standard_
        _check_full_rank(model_matrix)

        all_zero = np.asarray(self.all_zero_, dtype=bool)
        nz_idx = np.where(~all_zero)[0]
        if nz_idx.size == 0:
            raise DESeq2Error("All genes have zero counts.")

        counts_nz = self.counts_[nz_idx, :]
        nf_nz = self._size_or_norm_factors()[nz_idx, :]
        base_mean_nz = self.base_mean_[nz_idx]
        base_var_nz = self.base_var_[nz_idx]

        if alpha_init is None:
            rough = self._rough_disp_estimate(counts_nz / nf_nz, model_matrix)
            moments = self._moments_disp_estimate(base_var_nz, base_mean_nz)
            alpha_hat = np.minimum(rough, moments)
        else:
            alpha_arr = np.asarray(alpha_init, dtype=float)
            if alpha_arr.ndim == 0:
                alpha_hat = np.repeat(float(alpha_arr), nz_idx.size)
            else:
                if alpha_arr.shape[0] != nz_idx.size:
                    raise DESeq2Error("alpha_init vector must have one value per non-zero gene.")
                alpha_hat = alpha_arr

        max_disp = max(10.0, float(self.n_samples))
        alpha_hat = np.clip(alpha_hat, min_disp, max_disp)
        alpha_hat_new = alpha_hat.copy()
        alpha_start = alpha_hat.copy()

        if linear_mu is None:
            _, group_sizes = _model_matrix_groups(model_matrix)
            linear_mu = bool(np.unique(group_sizes).size == 1 and np.unique(group_sizes)[0] == model_matrix.shape[1])
            if self.weights_ is not None:
                linear_mu = False

        if self.weights_ is None:
            use_weights = False
            weights_nz = np.ones_like(counts_nz)
        else:
            use_weights = True
            weights_nz = np.maximum(self.weights_[nz_idx, :], 1e-6)

        fitidx = np.ones(nz_idx.size, dtype=bool)
        mu = np.zeros_like(counts_nz)
        disp_iter = np.zeros(nz_idx.size, dtype=float)
        last_lp = np.zeros(nz_idx.size, dtype=float)
        initial_lp = np.zeros(nz_idx.size, dtype=float)

        for _ in range(int(niter)):
            if not np.any(fitidx):
                break
            if not linear_mu:
                fit = self._fit_nbinom_glms(
                    counts=counts_nz[fitidx, :],
                    model_matrix=model_matrix,
                    normalization_factors=nf_nz[fitidx, :],
                    alpha_hat=alpha_hat[fitidx],
                    maxit=maxit,
                    minmu=minmu,
                    use_optim=False,
                    mu_only=True,
                    weights=weights_nz[fitidx, :] if use_weights else None,
                )
                fit_mu = fit["mu"]
            else:
                fit_mu = self._linear_model_mu_normalized(
                    counts_nz[fitidx, :], nf_nz[fitidx, :], model_matrix
                )
            fit_mu = np.maximum(fit_mu, minmu)
            mu[fitidx, :] = fit_mu

            disp_res = self._fit_disp(
                y=counts_nz[fitidx, :],
                x=model_matrix,
                mu_hat=fit_mu,
                log_alpha=np.log(alpha_hat[fitidx]),
                log_alpha_prior_mean=np.log(alpha_hat[fitidx]),
                log_alpha_prior_sigmasq=1.0,
                min_log_alpha=np.log(min_disp / 10.0),
                kappa_0=kappa_0,
                tol=disp_tol,
                maxit=maxit,
                use_prior=False,
                weights=weights_nz[fitidx, :],
                use_weights=use_weights,
                weight_threshold=weight_threshold,
                use_cr=use_cr,
            )
            disp_iter[fitidx] = disp_res["iter"]
            alpha_hat_new[fitidx] = np.minimum(np.exp(disp_res["log_alpha"]), max_disp)
            last_lp[fitidx] = disp_res["last_lp"]
            initial_lp[fitidx] = disp_res["initial_lp"]

            moved = np.abs(np.log(alpha_hat_new) - np.log(alpha_hat)) > 0.05
            moved[~np.isfinite(moved)] = False
            fitidx = moved
            alpha_hat = alpha_hat_new.copy()

        disp_gene_est = alpha_hat.copy()
        if int(niter) == 1:
            no_increase = last_lp < initial_lp + np.abs(initial_lp) / 1e6
            disp_gene_est[no_increase] = alpha_start[no_increase]

        disp_gene_est_conv = (disp_iter < maxit) & ~(disp_iter == 1)
        refit = (~disp_gene_est_conv) & (disp_gene_est > min_disp * 10)
        if np.any(refit):
            disp_grid = self._fit_disp_grid(
                y=counts_nz[refit, :],
                x=model_matrix,
                mu_hat=mu[refit, :],
                log_alpha_prior_mean=np.zeros(np.sum(refit), dtype=float),
                log_alpha_prior_sigmasq=1.0,
                use_prior=False,
                weights=weights_nz[refit, :],
                use_weights=use_weights,
                weight_threshold=weight_threshold,
                use_cr=use_cr,
            )
            disp_gene_est[refit] = disp_grid
        disp_gene_est = np.clip(disp_gene_est, min_disp, max_disp)

        self.disp_gene_est_ = np.full(self.n_genes, np.nan, dtype=float)
        self.disp_gene_est_[nz_idx] = disp_gene_est
        self.disp_gene_iter_ = np.full(self.n_genes, np.nan, dtype=float)
        self.disp_gene_iter_[nz_idx] = disp_iter

        self.mu_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.mu_[nz_idx, :] = mu
        return self

    def estimate_dispersions_fit(
        self, fit_type: str = "parametric", min_disp: float = 1e-8
    ) -> "DESeq2":
        if self.disp_gene_est_ is None:
            self.estimate_dispersions_gene_est(min_disp=min_disp)
        if self.base_mean_ is None or self.all_zero_ is None:
            self._ensure_base_stats()

        fit_type = str(fit_type)
        if fit_type not in {"parametric", "local", "mean"}:
            raise DESeq2Error("fit_type must be one of {'parametric', 'local', 'mean'}.")

        nz_idx = np.where(~self.all_zero_)[0]
        disp_gene = self.disp_gene_est_[nz_idx]
        base_mean = self.base_mean_[nz_idx]

        def _safe_mean_disp(values: np.ndarray) -> float:
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size == 0:
                return float(min_disp)
            vals = np.sort(vals)
            trim = int(0.001 * vals.size)
            if trim >= vals.size:
                trim = vals.size - 1
            return float(max(np.mean(vals[trim:]), min_disp))

        use_for_fit = disp_gene > 100 * min_disp
        chosen_fit_type = fit_type
        if np.sum(use_for_fit) == 0:
            # Match DESeq2's practical behavior on low-information data by falling back
            # to a constant mean trend rather than failing hard.
            chosen_fit_type = "mean"
            mean_disp = _safe_mean_disp(disp_gene)

            def disp_fn(means: np.ndarray) -> np.ndarray:
                return np.full_like(np.asarray(means, dtype=float), float(mean_disp))

        elif fit_type == "parametric":
            try:
                disp_fn = self._parametric_dispersion_fit(
                    base_mean[use_for_fit], disp_gene[use_for_fit]
                )
            except Exception:
                chosen_fit_type = "local"
                disp_fn = self._local_dispersion_fit(
                    base_mean[use_for_fit], disp_gene[use_for_fit], min_disp
                )
        elif fit_type == "local":
            disp_fn = self._local_dispersion_fit(
                base_mean[use_for_fit], disp_gene[use_for_fit], min_disp
            )
        else:
            use_for_mean = disp_gene > 10 * min_disp
            mean_input = disp_gene[use_for_mean] if np.any(use_for_mean) else disp_gene
            mean_disp = _safe_mean_disp(mean_input)

            def disp_fn(means: np.ndarray) -> np.ndarray:
                return np.full_like(np.asarray(means, dtype=float), float(mean_disp))

        disp_fit_nz = np.asarray(disp_fn(base_mean), dtype=float)
        if disp_fit_nz.ndim == 0:
            disp_fit_nz = np.repeat(float(disp_fit_nz), nz_idx.size)
        disp_fit = np.full(self.n_genes, np.nan, dtype=float)
        disp_fit[nz_idx] = disp_fit_nz
        self.disp_fit_ = disp_fit

        above_min_disp = disp_gene >= min_disp * 100
        if np.any(above_min_disp):
            residuals = np.log(disp_gene) - np.log(disp_fit_nz)
            var_log_disp_ests = (
                float(
                    median_abs_deviation(
                        residuals[above_min_disp], scale="normal", nan_policy="omit"
                    )
                )
                ** 2
            )
        else:
            var_log_disp_ests = np.nan

        self.dispersion_fit_ = _DispFit(
            func=lambda means: np.asarray(disp_fn(means), dtype=float),
            fit_type=chosen_fit_type,
            var_log_disp_ests=var_log_disp_ests,
        )
        return self

    def estimate_dispersions_prior_var(
        self,
        min_disp: float = 1e-8,
        model_matrix: Optional[np.ndarray] = None,
    ) -> float:
        if self.disp_fit_ is None or self.disp_gene_est_ is None:
            raise DESeq2Error("Need gene-wise and fitted dispersions before estimating prior variance.")
        if self.all_zero_ is None:
            self._ensure_base_stats()
        nz_idx = np.where(~self.all_zero_)[0]
        disp_gene = self.disp_gene_est_[nz_idx]
        disp_fit = self.disp_fit_[nz_idx]
        above_min_disp = disp_gene >= min_disp * 100
        if np.sum(above_min_disp) == 0:
            raise DESeq2Error("No data found with dispersion estimates above min_disp threshold.")

        disp_residuals = np.log(disp_gene) - np.log(disp_fit)
        if model_matrix is None:
            model_matrix = self.model_matrix_
        m, p = model_matrix.shape

        if self.dispersion_fit_ is None or not np.isfinite(self.dispersion_fit_.var_log_disp_ests):
            var_log_disp_ests = (
                float(
                    median_abs_deviation(
                        disp_residuals[above_min_disp], scale="normal", nan_policy="omit"
                    )
                )
                ** 2
            )
        else:
            var_log_disp_ests = self.dispersion_fit_.var_log_disp_ests

        if ((m - p) <= 3) and (m > p):
            rng = np.random.default_rng(2)
            obs_dist = disp_residuals[above_min_disp]
            brks = np.arange(-10.0, 10.0 + 0.5, 0.5)
            obs_dist = obs_dist[(obs_dist > brks.min()) & (obs_dist < brks.max())]
            obs_var_grid = np.linspace(0.0, 8.0, num=200)
            obs_hist, _ = np.histogram(obs_dist, bins=brks, density=True)
            kl_divs = np.empty_like(obs_var_grid)
            for i, x in enumerate(obs_var_grid):
                rand = (
                    np.log(rng.chisquare(df=(m - p), size=10_000))
                    + rng.normal(loc=0.0, scale=np.sqrt(x), size=10_000)
                    - np.log(m - p)
                )
                rand = rand[(rand > brks.min()) & (rand < brks.max())]
                rand_hist, _ = np.histogram(rand, bins=brks, density=True)
                z = np.concatenate([obs_hist, rand_hist])
                pos = z[z > 0]
                small = np.min(pos) if pos.size else 1e-12
                kl_divs[i] = np.sum(obs_hist * (np.log(obs_hist + small) - np.log(rand_hist + small)))
            lo = lowess(kl_divs, obs_var_grid, frac=0.2, it=0, return_sorted=True)
            fine_grid = np.linspace(0.0, 8.0, num=1000)
            pred = np.interp(fine_grid, lo[:, 0], lo[:, 1], left=lo[0, 1], right=lo[-1, 1])
            argmin_kl = float(fine_grid[int(np.nanargmin(pred))])
            return float(max(argmin_kl, 0.25))

        if m > p:
            exp_var_log_disp = float(polygamma(1, (m - p) / 2.0))
            return float(max(var_log_disp_ests - exp_var_log_disp, 0.25))
        return float(var_log_disp_ests)

    def estimate_dispersions_map(
        self,
        outlier_sd: float = 2.0,
        disp_prior_var: Optional[float] = None,
        min_disp: float = 1e-8,
        kappa_0: float = 1.0,
        disp_tol: float = 1e-6,
        maxit: int = 100,
        use_cr: bool = True,
        weight_threshold: float = 1e-2,
    ) -> "DESeq2":
        if self.disp_fit_ is None:
            self.estimate_dispersions_fit(min_disp=min_disp)
        if self.disp_gene_est_ is None:
            self.estimate_dispersions_gene_est(min_disp=min_disp)
        if self.mu_ is None:
            raise DESeq2Error("Expected means (mu) are missing; run gene-wise dispersion estimation first.")
        if self.all_zero_ is None:
            self._ensure_base_stats()

        nz_idx = np.where(~self.all_zero_)[0]
        counts_nz = self.counts_[nz_idx, :]
        mu_nz = self.mu_[nz_idx, :]
        disp_gene = self.disp_gene_est_[nz_idx]
        disp_fit = self.disp_fit_[nz_idx]
        max_disp = max(10.0, float(self.n_samples))

        if disp_prior_var is None:
            if np.sum(disp_gene >= min_disp * 100) == 0:
                final_disp = np.full(nz_idx.size, min_disp * 10.0, dtype=float)
                self.dispersions_ = np.full(self.n_genes, np.nan, dtype=float)
                self.dispersions_[nz_idx] = final_disp
                self.disp_prior_var_ = 0.25
                return self
            disp_prior_var = self.estimate_dispersions_prior_var(min_disp=min_disp)
        self.disp_prior_var_ = float(disp_prior_var)

        if self.weights_ is None:
            use_weights = False
            weights_nz = np.ones_like(counts_nz)
        else:
            use_weights = True
            weights_nz = np.maximum(self.weights_[nz_idx, :], 1e-6)

        disp_init = np.where(disp_gene > 0.1 * disp_fit, disp_gene, disp_fit)
        disp_init = np.where(np.isfinite(disp_init), disp_init, disp_fit)

        disp_res_map = self._fit_disp(
            y=counts_nz,
            x=self.model_matrix_,
            mu_hat=mu_nz,
            log_alpha=np.log(disp_init),
            log_alpha_prior_mean=np.log(disp_fit),
            log_alpha_prior_sigmasq=float(disp_prior_var),
            min_log_alpha=np.log(min_disp / 10.0),
            kappa_0=kappa_0,
            tol=disp_tol,
            maxit=maxit,
            use_prior=True,
            weights=weights_nz,
            use_weights=use_weights,
            weight_threshold=weight_threshold,
            use_cr=use_cr,
        )
        disp_map = np.exp(disp_res_map["log_alpha"])
        disp_iter = disp_res_map["iter"].astype(float)
        disp_conv = disp_res_map["iter"] < maxit
        refit = ~disp_conv
        if np.any(refit):
            disp_grid = self._fit_disp_grid(
                y=counts_nz[refit, :],
                x=self.model_matrix_,
                mu_hat=mu_nz[refit, :],
                log_alpha_prior_mean=np.log(disp_fit[refit]),
                log_alpha_prior_sigmasq=float(disp_prior_var),
                use_prior=True,
                weights=weights_nz[refit, :],
                use_weights=use_weights,
                weight_threshold=weight_threshold,
                use_cr=True,
            )
            disp_map[refit] = disp_grid

        disp_map = np.clip(disp_map, min_disp, max_disp)
        dispersion_final = disp_map.copy()

        if self.dispersion_fit_ is not None and np.isfinite(self.dispersion_fit_.var_log_disp_ests):
            var_log_disp_ests = self.dispersion_fit_.var_log_disp_ests
        else:
            residuals = np.log(disp_gene) - np.log(disp_fit)
            var_log_disp_ests = (
                float(
                    median_abs_deviation(
                        residuals[disp_gene >= min_disp * 100], scale="normal", nan_policy="omit"
                    )
                )
                ** 2
            )
        disp_outlier = np.log(disp_gene) > np.log(disp_fit) + outlier_sd * np.sqrt(var_log_disp_ests)
        disp_outlier = np.where(np.isnan(disp_outlier), False, disp_outlier)
        dispersion_final[disp_outlier] = disp_gene[disp_outlier]

        self.disp_map_ = np.full(self.n_genes, np.nan, dtype=float)
        self.disp_map_[nz_idx] = disp_map
        self.disp_iter_ = np.full(self.n_genes, np.nan, dtype=float)
        self.disp_iter_[nz_idx] = disp_iter
        self.disp_outlier_ = np.full(self.n_genes, np.nan, dtype=float)
        self.disp_outlier_[nz_idx] = disp_outlier.astype(float)
        self.dispersions_ = np.full(self.n_genes, np.nan, dtype=float)
        self.dispersions_[nz_idx] = dispersion_final
        return self

    def estimate_dispersions(
        self,
        fit_type: str = "parametric",
        min_disp: float = 1e-8,
        minmu: float = 0.5,
        use_cr: Optional[bool] = None,
    ) -> "DESeq2":
        if use_cr is None:
            # Cox-Reid is most useful for smaller n; skip it by default for larger cohorts.
            use_cr = self.n_samples < 100
        self.model_matrix_ = self.model_matrix_standard_.copy()
        self.coef_names_ = list(self.coef_names_standard_)
        self.model_matrix_type_ = "standard"
        self.disp_model_matrix_ = self.model_matrix_standard_.copy()
        self._refresh_result_name_map()
        self.estimate_dispersions_gene_est(min_disp=min_disp, minmu=minmu, use_cr=bool(use_cr))
        self.estimate_dispersions_fit(fit_type=fit_type, min_disp=min_disp)
        self.estimate_dispersions_map(min_disp=min_disp, use_cr=bool(use_cr))
        return self

    def nbinom_wald_test(
        self,
        beta_prior: bool = False,
        beta_prior_var: Optional[Union[np.ndarray, Sequence[float]]] = None,
        beta_prior_method: str = "weighted",
        model_matrix_type: Optional[str] = None,
        beta_tol: float = 1e-8,
        maxit: int = 100,
        use_optim: bool = True,
        use_qr: bool = True,
        use_t: bool = False,
        df: Optional[Union[float, np.ndarray]] = None,
        minmu: float = 0.5,
    ) -> "DESeq2":
        if self.dispersions_ is None:
            raise DESeq2Error("Dispersion estimates are missing; run estimate_dispersions first.")
        if not self.is_integer_counts_:
            raise DESeq2Error("Count data must be integer for DESeq2-style Wald test.")
        if self.all_zero_ is None:
            self._ensure_base_stats()

        nz_idx = np.where(~self.all_zero_)[0]
        counts_nz = self.counts_[nz_idx, :]
        nf_nz = self._size_or_norm_factors()[nz_idx, :]
        alpha_nz = self.dispersions_[nz_idx]
        weights_nz = self.weights_[nz_idx, :] if self.weights_ is not None else None

        mm_standard = self.model_matrix_standard_
        coef_standard = list(self.coef_names_standard_)
        if model_matrix_type is None:
            model_matrix_type = "expanded" if beta_prior else "standard"
        if model_matrix_type not in {"standard", "expanded"}:
            raise DESeq2Error("model_matrix_type must be one of {'standard', 'expanded'}.")
        if model_matrix_type == "expanded" and not beta_prior:
            raise DESeq2Error("expanded model matrices require beta_prior=True.")

        if beta_prior:
            if self._design_has_interactions():
                raise DESeq2Error("beta_prior=True is not currently supported for designs with interactions.")
            has_intercept = bool(np.any(np.all(np.isclose(mm_standard, 1.0), axis=0)))
            if not has_intercept:
                raise DESeq2Error("beta_prior=True requires an intercept in the design.")

        mle_fit = self._fit_nbinom_glms(
            counts=counts_nz,
            model_matrix=mm_standard,
            normalization_factors=nf_nz,
            alpha_hat=alpha_nz,
            beta_tol=beta_tol,
            maxit=maxit,
            use_optim=use_optim,
            use_qr=use_qr,
            minmu=minmu,
            weights=weights_nz,
            return_optimizer_rows=True,
        )
        full_mle = np.full((self.n_genes, len(coef_standard)), np.nan, dtype=float)
        full_mle[nz_idx, :] = mle_fit["beta_matrix"]
        self.mle_beta_ = full_mle
        self.mle_coef_names_ = coef_standard

        self.beta_prior_ = bool(beta_prior)
        if beta_prior:
            if model_matrix_type == "expanded":
                fit_model_matrix, fit_coef_names = self._make_expanded_model_matrix()
            else:
                fit_model_matrix, fit_coef_names = mm_standard, coef_standard

            if beta_prior_var is None:
                standard_prior_var = self._estimate_beta_prior_var(
                    full_mle,
                    beta_prior_method=beta_prior_method,
                )
                if model_matrix_type == "expanded":
                    beta_prior_var = self._expand_beta_prior_var(
                        standard_prior_var, fit_coef_names
                    )
                else:
                    beta_prior_var = standard_prior_var
            else:
                beta_prior_var = np.asarray(beta_prior_var, dtype=float)
                if beta_prior_var.ndim != 1 or beta_prior_var.shape[0] != len(fit_coef_names):
                    raise DESeq2Error("beta_prior_var must be a vector with one value per coefficient.")

            self.model_matrix_ = np.asarray(fit_model_matrix, dtype=float)
            self.coef_names_ = list(fit_coef_names)
            self.model_matrix_type_ = model_matrix_type
            self._refresh_result_name_map()

            self.beta_prior_var_ = np.asarray(beta_prior_var, dtype=float)
            lambda_prior = 1.0 / np.maximum(self.beta_prior_var_, 1e-12)
            fit = self._fit_nbinom_glms(
                counts=counts_nz,
                model_matrix=self.model_matrix_,
                normalization_factors=nf_nz,
                alpha_hat=alpha_nz,
                lambda_=lambda_prior,
                beta_tol=beta_tol,
                maxit=maxit,
                use_optim=use_optim,
                use_qr=use_qr,
                minmu=minmu,
                weights=weights_nz,
                return_optimizer_rows=True,
            )
        else:
            self.model_matrix_ = mm_standard.copy()
            self.coef_names_ = list(coef_standard)
            self.model_matrix_type_ = "standard"
            self._refresh_result_name_map()
            fit = mle_fit
            self.beta_prior_var_ = np.repeat(1e6, self.n_coefs)

        n_nz_genes = int(nz_idx.size)
        mle_rows_for_optim = int(mle_fit.get("n_rows_for_optim", 0))
        final_rows_for_optim = int(fit.get("n_rows_for_optim", mle_rows_for_optim))
        if n_nz_genes > 0:
            mle_frac_for_optim = float(mle_rows_for_optim / n_nz_genes)
            final_frac_for_optim = float(final_rows_for_optim / n_nz_genes)
        else:
            mle_frac_for_optim = 0.0
            final_frac_for_optim = 0.0
        self.optimizer_stats_ = {
            "test": "wald",
            "n_nonzero_genes": n_nz_genes,
            "wald_mle_rows_for_optim": mle_rows_for_optim,
            "wald_mle_fraction_for_optim": mle_frac_for_optim,
            "wald_final_rows_for_optim": final_rows_for_optim,
            "wald_final_fraction_for_optim": final_frac_for_optim,
        }

        # Cook's distance follows DESeq2 behavior: based on standard model matrix fit.
        self.disp_model_matrix_ = mm_standard.copy()
        if beta_prior:
            mu_nz = mle_fit["mu"]
            hat_nz = mle_fit["hat_diagonals"]
        else:
            mu_nz = fit["mu"]
            hat_nz = fit["hat_diagonals"]
        cooks_nz = self._calculate_cooks_distance(
            counts=counts_nz,
            mu=mu_nz,
            hat_diagonals=hat_nz,
            model_matrix=self.disp_model_matrix_,
            normalization_factors=nf_nz,
        )
        max_cooks_nz = _record_max_cooks(self.disp_model_matrix_, cooks_nz, nz_idx.size)

        beta_nz = fit["beta_matrix"]
        beta_se_nz = fit["beta_se"]
        with np.errstate(divide="ignore", invalid="ignore"):
            wald_stat_nz = beta_nz / beta_se_nz

        if use_t:
            if df is None:
                if self.weights_ is not None:
                    num_samps = np.sum(np.maximum(weights_nz, 1e-6), axis=1)
                else:
                    num_samps = np.repeat(self.n_samples, nz_idx.size)
                df_nz = num_samps - self.disp_model_matrix_.shape[1]
            else:
                df_arr = np.asarray(df, dtype=float)
                if df_arr.ndim == 0:
                    df_nz = np.repeat(float(df_arr), nz_idx.size)
                else:
                    if df_arr.shape[0] != self.n_genes:
                        raise DESeq2Error("df vector must have one value per gene.")
                    df_nz = df_arr[nz_idx]
            df_nz = np.where(df_nz > 0, df_nz, np.nan)
            wald_pval_nz = 2.0 * t_dist.sf(np.abs(wald_stat_nz), df=df_nz[:, None])
        else:
            wald_pval_nz = 2.0 * norm.sf(np.abs(wald_stat_nz))

        self.mu_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.mu_[nz_idx, :] = mu_nz
        self.hat_diagonals_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.hat_diagonals_[nz_idx, :] = hat_nz
        self.cooks_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.cooks_[nz_idx, :] = cooks_nz
        self.max_cooks_ = np.full(self.n_genes, np.nan, dtype=float)
        self.max_cooks_[nz_idx] = max_cooks_nz

        self.beta_ = np.full((self.n_genes, self.n_coefs), np.nan, dtype=float)
        self.beta_[nz_idx, :] = beta_nz
        self.beta_se_ = np.full((self.n_genes, self.n_coefs), np.nan, dtype=float)
        self.beta_se_[nz_idx, :] = beta_se_nz
        self.wald_stat_ = np.full((self.n_genes, self.n_coefs), np.nan, dtype=float)
        self.wald_stat_[nz_idx, :] = wald_stat_nz
        self.wald_pvalue_ = np.full((self.n_genes, self.n_coefs), np.nan, dtype=float)
        self.wald_pvalue_[nz_idx, :] = wald_pval_nz
        self.beta_conv_ = np.full(self.n_genes, np.nan, dtype=float)
        self.beta_conv_[nz_idx] = fit["beta_conv"].astype(float)
        self.beta_iter_ = np.full(self.n_genes, np.nan, dtype=float)
        self.beta_iter_[nz_idx] = fit["beta_iter"].astype(float)
        self.deviance_ = np.full(self.n_genes, np.nan, dtype=float)
        self.deviance_[nz_idx] = -2.0 * fit["log_like"]
        self.test_ = "wald"
        return self

    def nbinom_lrt(
        self,
        reduced: str,
        beta_tol: float = 1e-8,
        maxit: int = 100,
        use_optim: bool = True,
        use_qr: bool = True,
        minmu: float = 0.5,
    ) -> "DESeq2":
        if self.dispersions_ is None:
            raise DESeq2Error("Dispersion estimates are missing; run estimate_dispersions first.")
        if not self.is_integer_counts_:
            raise DESeq2Error("Count data must be integer for DESeq2-style LRT.")
        if self.all_zero_ is None:
            self._ensure_base_stats()

        self.model_matrix_ = self.model_matrix_standard_.copy()
        self.coef_names_ = list(self.coef_names_standard_)
        self.model_matrix_type_ = "standard"
        self._refresh_result_name_map()
        self.disp_model_matrix_ = self.model_matrix_standard_.copy()

        reduced_matrix, _, _ = _build_design_matrix(self.metadata_, reduced)
        df = self.model_matrix_standard_.shape[1] - reduced_matrix.shape[1]
        if df < 1:
            raise DESeq2Error("LRT requires at least one degree of freedom.")

        nz_idx = np.where(~self.all_zero_)[0]
        counts_nz = self.counts_[nz_idx, :]
        nf_nz = self._size_or_norm_factors()[nz_idx, :]
        alpha_nz = self.dispersions_[nz_idx]
        weights_nz = self.weights_[nz_idx, :] if self.weights_ is not None else None

        full_fit = self._fit_nbinom_glms(
            counts=counts_nz,
            model_matrix=self.model_matrix_standard_,
            normalization_factors=nf_nz,
            alpha_hat=alpha_nz,
            beta_tol=beta_tol,
            maxit=maxit,
            use_optim=use_optim,
            use_qr=use_qr,
            minmu=minmu,
            weights=weights_nz,
            return_optimizer_rows=True,
        )
        reduced_fit = self._fit_nbinom_glms(
            counts=counts_nz,
            model_matrix=reduced_matrix,
            normalization_factors=nf_nz,
            alpha_hat=alpha_nz,
            beta_tol=beta_tol,
            maxit=maxit,
            use_optim=use_optim,
            use_qr=use_qr,
            minmu=minmu,
            weights=weights_nz,
            return_optimizer_rows=True,
        )
        n_nz_genes = int(nz_idx.size)
        full_rows_for_optim = int(full_fit.get("n_rows_for_optim", 0))
        reduced_rows_for_optim = int(reduced_fit.get("n_rows_for_optim", 0))
        if n_nz_genes > 0:
            full_frac_for_optim = float(full_rows_for_optim / n_nz_genes)
            reduced_frac_for_optim = float(reduced_rows_for_optim / n_nz_genes)
        else:
            full_frac_for_optim = 0.0
            reduced_frac_for_optim = 0.0
        self.optimizer_stats_ = {
            "test": "lrt",
            "n_nonzero_genes": n_nz_genes,
            "lrt_full_rows_for_optim": full_rows_for_optim,
            "lrt_full_fraction_for_optim": full_frac_for_optim,
            "lrt_reduced_rows_for_optim": reduced_rows_for_optim,
            "lrt_reduced_fraction_for_optim": reduced_frac_for_optim,
        }

        lrt_stat_nz = 2.0 * (full_fit["log_like"] - reduced_fit["log_like"])
        lrt_pval_nz = chi2.sf(lrt_stat_nz, df=df)

        mu_nz = full_fit["mu"]
        hat_nz = full_fit["hat_diagonals"]
        cooks_nz = self._calculate_cooks_distance(
            counts=counts_nz,
            mu=mu_nz,
            hat_diagonals=hat_nz,
            model_matrix=self.disp_model_matrix_,
            normalization_factors=nf_nz,
        )
        max_cooks_nz = _record_max_cooks(self.disp_model_matrix_, cooks_nz, nz_idx.size)

        self.mu_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.mu_[nz_idx, :] = mu_nz
        self.hat_diagonals_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.hat_diagonals_[nz_idx, :] = hat_nz
        self.cooks_ = np.full((self.n_genes, self.n_samples), np.nan, dtype=float)
        self.cooks_[nz_idx, :] = cooks_nz
        self.max_cooks_ = np.full(self.n_genes, np.nan, dtype=float)
        self.max_cooks_[nz_idx] = max_cooks_nz

        self.beta_ = np.full((self.n_genes, self.n_coefs), np.nan, dtype=float)
        self.beta_[nz_idx, :] = full_fit["beta_matrix"]
        self.beta_se_ = np.full((self.n_genes, self.n_coefs), np.nan, dtype=float)
        self.beta_se_[nz_idx, :] = full_fit["beta_se"]
        self.beta_conv_ = np.full(self.n_genes, np.nan, dtype=float)
        self.beta_conv_[nz_idx] = full_fit["beta_conv"].astype(float)
        self.beta_iter_ = np.full(self.n_genes, np.nan, dtype=float)
        self.beta_iter_[nz_idx] = full_fit["beta_iter"].astype(float)
        self.deviance_ = np.full(self.n_genes, np.nan, dtype=float)
        self.deviance_[nz_idx] = -2.0 * full_fit["log_like"]
        self.lrt_stat_ = np.full(self.n_genes, np.nan, dtype=float)
        self.lrt_stat_[nz_idx] = lrt_stat_nz
        self.lrt_pvalue_ = np.full(self.n_genes, np.nan, dtype=float)
        self.lrt_pvalue_[nz_idx] = lrt_pval_nz
        self.beta_prior_ = False
        self.beta_prior_var_ = np.repeat(1e6, self.n_coefs)
        self.test_ = "lrt"
        return self

    def _replace_outliers_counts(
        self,
        trim: float = 0.2,
        cooks_cutoff: Optional[float] = None,
        min_replicates: int = 7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cooks_ is None or self.disp_model_matrix_ is None:
            raise DESeq2Error("Outlier replacement requires an existing fit with Cook's distances.")
        if min_replicates < 3:
            raise DESeq2Error("At least 3 replicates are required for outlier replacement.")
        p = self.disp_model_matrix_.shape[1]
        m = self.n_samples
        if m <= p:
            return self.counts_.copy(), np.zeros(self.n_genes, dtype=bool), np.zeros(self.n_samples, dtype=bool)

        if cooks_cutoff is None:
            cooks_cutoff = float(f_dist.ppf(0.99, p, max(m - p, 1)))
        cooks_outlier = np.asarray(self.cooks_ > cooks_cutoff, dtype=bool)
        cooks_outlier &= np.isfinite(self.cooks_)
        replace_mask = np.any(cooks_outlier, axis=1)

        cts_norm = self._normalized_counts()
        trim_base_mean = _trimmed_mean_rows(cts_norm, trim)
        nf = self._size_or_norm_factors()
        replacement_counts = np.asarray(np.rint(trim_base_mean[:, None] * nf), dtype=float)
        new_counts = self.counts_.copy()

        replaceable_samples = _n_or_more_in_cell(self.disp_model_matrix_, n=min_replicates)
        if np.any(replaceable_samples):
            outlier_mask = cooks_outlier.copy()
            outlier_mask[:, ~replaceable_samples] = False
            new_counts[outlier_mask] = replacement_counts[outlier_mask]
            replace_mask = np.any(outlier_mask, axis=1)
            self.original_counts_ = self.counts_.copy()
            self.replace_mask_ = replace_mask.copy()
            self.replaceable_samples_ = replaceable_samples.copy()
        else:
            replace_mask[:] = False
        return new_counts, replace_mask, replaceable_samples

    def _refit_with_replaced_outliers(
        self,
        test: str,
        reduced: Optional[str],
        beta_prior: bool,
        beta_prior_var: Optional[Union[np.ndarray, Sequence[float]]],
        beta_prior_method: str,
        model_matrix_type: Optional[str],
        fit_type: str,
        use_cr: Optional[bool],
        minmu: float,
        min_replicates_for_replace: int,
        cooks_cutoff_replace: Optional[float],
        replace_trim: float,
    ) -> None:
        if self.cooks_ is None:
            return
        new_counts, replace_mask, replaceable_samples = self._replace_outliers_counts(
            trim=replace_trim,
            cooks_cutoff=cooks_cutoff_replace,
            min_replicates=min_replicates_for_replace,
        )
        nrefit = int(np.sum(replace_mask))
        if nrefit == 0:
            return

        original_counts = self.counts_.copy()
        original_cooks = None if self.cooks_ is None else self.cooks_.copy()
        self.counts_ = new_counts
        self.counts_df_ = pd.DataFrame(new_counts, index=self.gene_names_, columns=self.sample_names_)

        self.estimate_dispersions(fit_type=fit_type, minmu=minmu, use_cr=use_cr)
        if test == "wald":
            self.nbinom_wald_test(
                beta_prior=beta_prior,
                beta_prior_var=beta_prior_var,
                beta_prior_method=beta_prior_method,
                model_matrix_type=model_matrix_type,
                minmu=minmu,
            )
        else:
            if reduced is None:
                raise DESeq2Error("reduced formula must be provided for LRT.")
            self.nbinom_lrt(reduced=reduced, minmu=minmu)

        self.replace_counts_ = self.counts_.copy()
        self.replace_cooks_ = None if self.cooks_ is None else self.cooks_.copy()
        if self.replace_cooks_ is not None and self.max_cooks_ is not None:
            if np.all(replaceable_samples):
                self.max_cooks_[:] = np.nan
            else:
                rc = self.replace_cooks_.copy()
                rc[:, replaceable_samples] = 0.0
                self.max_cooks_ = _record_max_cooks(self.disp_model_matrix_, rc, self.n_genes)

        self.counts_ = original_counts
        self.counts_df_ = pd.DataFrame(original_counts, index=self.gene_names_, columns=self.sample_names_)
        if original_cooks is not None:
            self.cooks_ = original_cooks

    def deseq(
        self,
        test: str = "wald",
        reduced: Optional[str] = None,
        fit_type: str = "parametric",
        beta_prior: bool = False,
        beta_prior_var: Optional[Union[np.ndarray, Sequence[float]]] = None,
        beta_prior_method: str = "weighted",
        model_matrix_type: Optional[str] = None,
        min_replicates_for_replace: float = 7,
        cooks_cutoff_replace: Optional[float] = None,
        replace_trim: float = 0.2,
        minmu: float = 0.5,
        use_cr: Optional[bool] = None,
    ) -> "DESeq2":
        if self.size_factors_ is None and self.normalization_factors_ is None:
            self.estimate_size_factors()
        self.estimate_dispersions(fit_type=fit_type, minmu=minmu, use_cr=use_cr)
        test_lower = test.lower()
        if test_lower == "wald":
            self.nbinom_wald_test(
                beta_prior=beta_prior,
                beta_prior_var=beta_prior_var,
                beta_prior_method=beta_prior_method,
                model_matrix_type=model_matrix_type,
                minmu=minmu,
            )
            if np.isfinite(min_replicates_for_replace):
                sufficient_reps = np.any(
                    _n_or_more_in_cell(
                        self.disp_model_matrix_,
                        int(min_replicates_for_replace),
                    )
                )
                if sufficient_reps:
                    self._refit_with_replaced_outliers(
                        test="wald",
                        reduced=reduced,
                        beta_prior=beta_prior,
                        beta_prior_var=beta_prior_var,
                        beta_prior_method=beta_prior_method,
                        model_matrix_type=model_matrix_type,
                        fit_type=fit_type,
                        use_cr=use_cr,
                        minmu=minmu,
                        min_replicates_for_replace=int(min_replicates_for_replace),
                        cooks_cutoff_replace=cooks_cutoff_replace,
                        replace_trim=replace_trim,
                    )
            return self
        if test_lower == "lrt":
            if reduced is None:
                raise DESeq2Error("reduced formula must be provided for LRT.")
            self.nbinom_lrt(reduced=reduced, minmu=minmu)
            if np.isfinite(min_replicates_for_replace):
                sufficient_reps = np.any(
                    _n_or_more_in_cell(
                        self.disp_model_matrix_,
                        int(min_replicates_for_replace),
                    )
                )
                if sufficient_reps:
                    self._refit_with_replaced_outliers(
                        test="lrt",
                        reduced=reduced,
                        beta_prior=beta_prior,
                        beta_prior_var=beta_prior_var,
                        beta_prior_method=beta_prior_method,
                        model_matrix_type=model_matrix_type,
                        fit_type=fit_type,
                        use_cr=use_cr,
                        minmu=minmu,
                        min_replicates_for_replace=int(min_replicates_for_replace),
                        cooks_cutoff_replace=cooks_cutoff_replace,
                        replace_trim=replace_trim,
                    )
            return self
        raise DESeq2Error("test must be either 'wald' or 'lrt'.")

    def _resolve_coef_index(self, coef: Optional[Union[int, str]]) -> int:
        if coef is None:
            return self.n_coefs - 1
        if isinstance(coef, int):
            idx = coef
        else:
            internal = self._internal_coef_name(coef)
            idx = self.coef_names_.index(internal)
        if idx < 0 or idx >= self.n_coefs:
            raise DESeq2Error(f"Coefficient index out of range: {idx}.")
        return idx

    def results(
        self,
        coef: Optional[Union[int, str]] = None,
        contrast: Optional[
            Union[
                np.ndarray,
                Sequence[float],
                dict[str, float],
                Sequence[str],
                Sequence[Sequence[str]],
            ]
        ] = None,
        list_values: tuple[float, float] = (1.0, -1.0),
        cooks_cutoff: Optional[Union[bool, float]] = None,
        independent_filtering: bool = True,
        alpha: float = 0.1,
        filter_stat: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
        p_adjust_method: str = "BH",
    ) -> pd.DataFrame:
        if self.test_ is None:
            raise DESeq2Error("No fitted test found. Run nbinom_wald_test() or nbinom_lrt() first.")

        res = pd.DataFrame(index=self.gene_names_)
        if self.base_mean_ is not None:
            res["baseMean"] = self.base_mean_
        if self.dispersions_ is not None:
            res["dispersion"] = self.dispersions_

        contrast_vec, contrast_all_zero = self._parse_contrast_input(
            contrast,
            list_values=list_values,
        )
        if contrast_vec is None:
            idx = self._resolve_coef_index(coef)
            if self.beta_ is not None:
                res["log2FoldChange"] = self.beta_[:, idx]
            if self.beta_se_ is not None:
                res["lfcSE"] = self.beta_se_[:, idx]

            if self.test_ == "wald":
                if self.wald_stat_ is None or self.wald_pvalue_ is None:
                    raise DESeq2Error("Wald statistics are missing.")
                res["stat"] = self.wald_stat_[:, idx]
                res["pvalue"] = self.wald_pvalue_[:, idx]
            else:
                if self.lrt_stat_ is None or self.lrt_pvalue_ is None:
                    raise DESeq2Error("LRT statistics are missing.")
                res["stat"] = self.lrt_stat_
                res["pvalue"] = self.lrt_pvalue_
        else:
            if self.beta_ is None:
                raise DESeq2Error("Missing fitted coefficients for contrast.")
            cvec = np.asarray(contrast_vec, dtype=float)
            lfc = self.beta_ @ cvec
            if self.beta_prior_var_ is None:
                ridge_log2 = np.repeat(1e-6, self.n_coefs)
            else:
                ridge_log2 = 1.0 / np.maximum(self.beta_prior_var_, 1e-12)
            lfc_se = np.full(self.n_genes, np.nan, dtype=float)
            stat = np.full(self.n_genes, np.nan, dtype=float)
            pvalue = np.full(self.n_genes, np.nan, dtype=float)
            nz_rows = np.where(~self.all_zero_)[0] if self.all_zero_ is not None else np.arange(self.n_genes)
            nonzero = np.flatnonzero(np.abs(cvec) > 1e-12)
            if nonzero.size == 1 and self.beta_se_ is not None:
                coef_idx = int(nonzero[0])
                scale = float(cvec[coef_idx])
                lfc_se = np.abs(scale) * self.beta_se_[:, coef_idx]
                if self.test_ == "wald":
                    if self.wald_stat_ is None or self.wald_pvalue_ is None:
                        raise DESeq2Error("Wald statistics are missing.")
                    stat = np.sign(scale) * self.wald_stat_[:, coef_idx]
                    pvalue = self.wald_pvalue_[:, coef_idx]
            elif nz_rows.size > 0:
                if self.dispersions_ is None:
                    raise DESeq2Error("Dispersion estimates are missing; run estimate_dispersions first.")

                x = self.model_matrix_
                ridge_nat = np.diag(ridge_log2 / (LN2**2))
                alpha_nz = self.dispersions_[nz_rows]
                nf = self._size_or_norm_factors()

                if self.mu_ is not None and not self.beta_prior_:
                    mu_nz = self.mu_[nz_rows, :]
                else:
                    eta = (x @ (self.beta_[nz_rows, :] * LN2).T).T
                    eta = np.clip(np.where(np.isfinite(eta), eta, 0.0), -30.0, 30.0)
                    mu_nz = nf[nz_rows, :] * np.exp(eta)

                valid = np.isfinite(alpha_nz) & np.all(np.isfinite(mu_nz), axis=1)
                if np.any(valid):
                    rows = nz_rows[valid]
                    mu_valid = np.maximum(mu_nz[valid, :], 1e-12)
                    alpha_valid = alpha_nz[valid]
                    if self.weights_ is not None:
                        w = self.weights_[rows, :] / (1.0 / mu_valid + alpha_valid[:, None])
                    else:
                        w = 1.0 / (1.0 / mu_valid + alpha_valid[:, None])

                    xtwx = np.einsum("gn,nj,nk->gjk", w, x, x, optimize=True)
                    xtwx_ridge = xtwx + ridge_nat[None, :, :]
                    try:
                        xtwx_ridge_inv = np.linalg.inv(xtwx_ridge)
                    except np.linalg.LinAlgError:
                        xtwx_ridge_inv = np.linalg.pinv(xtwx_ridge)
                    sigma_nat = xtwx_ridge_inv @ xtwx @ xtwx_ridge_inv
                    var_nat = np.einsum("j,gjk,k->g", cvec, sigma_nat, cvec, optimize=True)
                    var_nat = np.where(np.isfinite(var_nat), np.clip(var_nat, 0.0, None), np.nan)
                    se = LOG2E * np.sqrt(var_nat)
                    lfc_se[rows] = se

                    if self.test_ == "wald":
                        good = se > 0
                        if np.any(good):
                            good_rows = rows[good]
                            stat[good_rows] = lfc[good_rows] / se[good]
                            pvalue[good_rows] = 2.0 * norm.sf(np.abs(stat[good_rows]))
            res["log2FoldChange"] = lfc
            res["lfcSE"] = lfc_se
            if self.test_ == "wald":
                res["stat"] = stat
                res["pvalue"] = pvalue
            else:
                if self.lrt_stat_ is None or self.lrt_pvalue_ is None:
                    raise DESeq2Error("LRT statistics are missing.")
                res["stat"] = self.lrt_stat_
                res["pvalue"] = self.lrt_pvalue_

            if contrast_all_zero is not None:
                zero_mask = contrast_all_zero.copy()
                if self.all_zero_ is not None:
                    zero_mask &= ~self.all_zero_
                res.loc[zero_mask, "log2FoldChange"] = 0.0
                if "stat" in res.columns and self.test_ == "wald":
                    res.loc[zero_mask, "stat"] = 0.0
                res.loc[zero_mask, "pvalue"] = 1.0

        # Cook's distance filtering of p-values
        if self.max_cooks_ is not None:
            m = self.n_samples
            p = self.disp_model_matrix_.shape[1] if self.disp_model_matrix_ is not None else self.n_coefs
            default_cutoff = f_dist.ppf(0.99, p, max(m - p, 1))
            if cooks_cutoff is None:
                cutoff = float(default_cutoff)
                perform_cooks_cutoff = True
            elif isinstance(cooks_cutoff, (bool, np.bool_)):
                perform_cooks_cutoff = bool(cooks_cutoff)
                cutoff = float(default_cutoff)
            else:
                cutoff = float(cooks_cutoff)
                perform_cooks_cutoff = True

            if perform_cooks_cutoff:
                cooks_outlier = np.asarray(self.max_cooks_ > cutoff, dtype=bool)
                cooks_outlier &= np.isfinite(self.max_cooks_)

                # Two-group heuristic from DESeq2 to avoid over-filtering low-count outliers.
                design_vars = self._design_vars()
                if (
                    np.any(cooks_outlier)
                    and len(design_vars) == 1
                    and design_vars[0] in self.metadata_.columns
                ):
                    var = self.metadata_[design_vars[0]]
                    if isinstance(var.dtype, pd.CategoricalDtype) and len(var.cat.categories) == 2:
                        outlier_rows = np.where(cooks_outlier)[0]
                        for ii in outlier_rows:
                            cooks_row = self.cooks_[ii, :] if self.cooks_ is not None else None
                            if cooks_row is None or np.all(~np.isfinite(cooks_row)):
                                continue
                            jj = int(np.nanargmax(cooks_row))
                            out_count = self.counts_[ii, jj]
                            if np.sum(self.counts_[ii, :] > out_count) >= 3:
                                cooks_outlier[ii] = False

                pvalue_arr = res["pvalue"].to_numpy(dtype=float, copy=True)
                pvalue_arr[cooks_outlier] = np.nan
                res["pvalue"] = pvalue_arr

        res["padj"] = self._apply_pvalue_adjustment(
            res["pvalue"].to_numpy(dtype=float),
            independent_filtering=independent_filtering,
            alpha=float(alpha),
            filter_stat=filter_stat,
            theta=theta,
            p_adjust_method=p_adjust_method,
        )
        return res
