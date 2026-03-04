import numpy as np
import pytest
import warnings

import pydeseq2.deseq2 as deseq2_mod
from pydeseq2.deseq2 import DESeq2


def _fit_disp_inputs() -> dict:
    y = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    x = np.column_stack([np.ones(3), np.array([0.0, 1.0, 0.0])])
    mu_hat = np.array([[11.0, 11.5, 12.0], [8.5, 8.0, 7.5]], dtype=float)
    log_alpha = np.log(np.array([0.2, 0.3], dtype=float))
    log_alpha_prior_mean = log_alpha.copy()
    weights = np.ones_like(y)
    return {
        "y": y,
        "x": x,
        "mu_hat": mu_hat,
        "log_alpha": log_alpha,
        "log_alpha_prior_mean": log_alpha_prior_mean,
        "log_alpha_prior_sigmasq": 1.0,
        "min_log_alpha": np.log(1e-9),
        "kappa_0": 1.0,
        "tol": 1e-6,
        "maxit": 10,
        "use_prior": False,
        "weights": weights,
        "use_weights": False,
        "weight_threshold": 1e-2,
    }


def _fake_fit_disp_output(n_rows: int) -> dict:
    return {
        "log_alpha": np.zeros(n_rows, dtype=float),
        "iter": np.zeros(n_rows, dtype=int),
        "iter_accept": np.zeros(n_rows, dtype=int),
        "last_change": np.zeros(n_rows, dtype=float),
        "initial_lp": np.zeros(n_rows, dtype=float),
        "last_lp": np.zeros(n_rows, dtype=float),
    }


def test_fit_disp_uses_cython_backend_when_available(monkeypatch) -> None:
    called = {"value": False}

    def fake_fit_disp_cy(
        y,
        mu_hat,
        log_alpha,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        min_log_alpha,
        kappa_0,
        tol,
        maxit,
        use_prior,
        weights,
        use_weights,
        x=None,
        use_cr=False,
        weight_threshold=1e-2,
        n_threads=0,
    ):
        called["value"] = True
        assert n_threads == 3
        assert x is not None
        assert use_cr is False
        assert weight_threshold == pytest.approx(1e-2)
        return _fake_fit_disp_output(y.shape[0])

    monkeypatch.setattr(deseq2_mod, "_fit_disp_cy", fake_fit_disp_cy)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 3)

    args = _fit_disp_inputs()
    out = DESeq2._fit_disp(use_cr=False, **args)

    assert called["value"]
    assert set(out.keys()) == {"log_alpha", "iter", "iter_accept", "last_change", "initial_lp", "last_lp"}


def test_fit_disp_uses_cython_backend_when_cox_reid_enabled(monkeypatch) -> None:
    called = {"value": False}

    def fake_fit_disp_cy(
        y,
        mu_hat,
        log_alpha,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        min_log_alpha,
        kappa_0,
        tol,
        maxit,
        use_prior,
        weights,
        use_weights,
        x=None,
        use_cr=False,
        weight_threshold=1e-2,
        n_threads=0,
    ):
        called["value"] = True
        assert x is not None
        assert use_cr is True
        return _fake_fit_disp_output(y.shape[0])

    monkeypatch.setattr(deseq2_mod, "_fit_disp_cy", fake_fit_disp_cy)
    args = _fit_disp_inputs()
    out = DESeq2._fit_disp(use_cr=True, **args)
    assert called["value"]
    assert out["log_alpha"].shape[0] == args["y"].shape[0]


def test_fit_disp_grid_uses_cython_backend_when_available(monkeypatch) -> None:
    called = {"value": False}

    def fake_fit_disp_grid_cy(
        y,
        mu_hat,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        use_prior,
        weights,
        use_weights,
        x=None,
        use_cr=False,
        weight_threshold=1e-2,
        n_threads=0,
    ):
        called["value"] = True
        assert n_threads == 5
        assert x is not None
        assert use_cr is True
        assert weight_threshold == pytest.approx(1e-2)
        return np.full(y.shape[0], 0.2, dtype=float)

    monkeypatch.setattr(deseq2_mod, "_fit_disp_grid_cy", fake_fit_disp_grid_cy)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 5)

    args = _fit_disp_inputs()
    out = DESeq2._fit_disp_grid(
        y=args["y"],
        x=args["x"],
        mu_hat=args["mu_hat"],
        log_alpha_prior_mean=args["log_alpha_prior_mean"],
        log_alpha_prior_sigmasq=args["log_alpha_prior_sigmasq"],
        use_prior=args["use_prior"],
        weights=args["weights"],
        use_weights=args["use_weights"],
        weight_threshold=args["weight_threshold"],
        use_cr=True,
    )

    assert called["value"]
    np.testing.assert_allclose(out, np.full(args["y"].shape[0], 0.2, dtype=float))


def test_invalid_cython_num_threads_env_warns(monkeypatch) -> None:
    monkeypatch.setenv("PYDESEQ2_NUM_THREADS", "abc")
    with pytest.warns(RuntimeWarning, match="PYDESEQ2_NUM_THREADS"):
        n = deseq2_mod._cython_num_threads()
    assert n == 0


def test_fit_nbinom_glms_uses_cython_backend_when_available(monkeypatch) -> None:
    called = {"value": False}

    def fake_fit_glm_core(
        counts,
        model_matrix,
        normalization_factors,
        alpha_hat,
        beta_nat_init,
        lambda_nat,
        beta_tol,
        maxit,
        minmu,
        use_weights,
        weights,
        use_qr,
        mu_only=False,
        n_threads=0,
    ):
        called["value"] = True
        assert n_threads == 4
        assert not mu_only
        n_genes, n_samples = counts.shape
        p = model_matrix.shape[1]
        return {
            "beta_nat": np.zeros((n_genes, p), dtype=float),
            "beta_var_nat": np.ones((n_genes, p), dtype=float),
            "hat_diagonals": np.zeros((n_genes, n_samples), dtype=float),
            "beta_iter": np.zeros(n_genes, dtype=int),
        }

    monkeypatch.setattr(deseq2_mod, "_fit_glm_cy", fake_fit_glm_core)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 4)

    counts = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    model_matrix = np.column_stack([np.ones(3, dtype=float), np.array([0.0, 1.0, 0.0])])
    normalization_factors = np.ones_like(counts)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    out = DESeq2._fit_nbinom_glms(
        counts=counts,
        model_matrix=model_matrix,
        normalization_factors=normalization_factors,
        alpha_hat=alpha_hat,
        use_optim=False,
    )

    assert called["value"]
    assert set(out.keys()) == {
        "log_like",
        "beta_conv",
        "beta_matrix",
        "beta_se",
        "mu",
        "beta_iter",
        "hat_diagonals",
    }


def test_fit_nbinom_glms_mu_only_uses_cython_backend_when_available(monkeypatch) -> None:
    called = {"value": False}

    def fake_fit_glm_core(
        counts,
        model_matrix,
        normalization_factors,
        alpha_hat,
        beta_nat_init,
        lambda_nat,
        beta_tol,
        maxit,
        minmu,
        use_weights,
        weights,
        use_qr,
        mu_only=False,
        n_threads=0,
    ):
        called["value"] = True
        assert mu_only
        n_genes, n_samples = counts.shape
        return {
            "mu": np.ones((n_genes, n_samples), dtype=float),
            "beta_iter": np.zeros(n_genes, dtype=int),
        }

    monkeypatch.setattr(deseq2_mod, "_fit_glm_cy", fake_fit_glm_core)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 2)

    counts = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    model_matrix = np.column_stack([np.ones(3, dtype=float), np.array([0.0, 1.0, 0.0])])
    normalization_factors = np.ones_like(counts)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    out = DESeq2._fit_nbinom_glms(
        counts=counts,
        model_matrix=model_matrix,
        normalization_factors=normalization_factors,
        alpha_hat=alpha_hat,
        use_optim=False,
        mu_only=True,
    )

    assert called["value"]
    assert set(out.keys()) == {"mu", "beta_iter"}


def test_matmul_fallback_warning_is_suppressed(monkeypatch) -> None:
    monkeypatch.setenv("PYDESEQ2_ALLOW_PYTHON_FALLBACK", "1")

    def fake_fit_glm_core(*args, **kwargs):
        raise ValueError(
            "matmul: Input operand 1 has a mismatch in its core dimension 0"
        )

    monkeypatch.setattr(deseq2_mod, "_fit_glm_cy", fake_fit_glm_core)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 2)
    monkeypatch.setattr(deseq2_mod, "_CYTHON_FALLBACK_WARNED", set())

    counts = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    model_matrix = np.column_stack(
        [np.ones(3, dtype=float), np.array([0.0, 1.0, 0.0])]
    )
    normalization_factors = np.ones_like(counts)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = DESeq2._fit_nbinom_glms(
            counts=counts,
            model_matrix=model_matrix,
            normalization_factors=normalization_factors,
            alpha_hat=alpha_hat,
            use_optim=False,
            mu_only=True,
        )
    assert out["mu"].shape == counts.shape
    assert not any("matmul" in str(w.message).lower() for w in rec)


def test_non_matmul_fallback_warning_is_generic_and_once(monkeypatch) -> None:
    monkeypatch.setenv("PYDESEQ2_ALLOW_PYTHON_FALLBACK", "1")

    def fake_fit_glm_core(*args, **kwargs):
        raise ValueError("synthetic backend failure")

    monkeypatch.setattr(deseq2_mod, "_fit_glm_cy", fake_fit_glm_core)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 2)
    monkeypatch.setattr(deseq2_mod, "_CYTHON_FALLBACK_WARNED", set())
    monkeypatch.delenv("PYDESEQ2_SHOW_FALLBACK_EXCEPTIONS", raising=False)

    counts = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    model_matrix = np.column_stack(
        [np.ones(3, dtype=float), np.array([0.0, 1.0, 0.0])]
    )
    normalization_factors = np.ones_like(counts)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        DESeq2._fit_nbinom_glms(
            counts=counts,
            model_matrix=model_matrix,
            normalization_factors=normalization_factors,
            alpha_hat=alpha_hat,
            use_optim=False,
            mu_only=True,
        )
        DESeq2._fit_nbinom_glms(
            counts=counts,
            model_matrix=model_matrix,
            normalization_factors=normalization_factors,
            alpha_hat=alpha_hat,
            use_optim=False,
            mu_only=True,
        )
    runtime = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1
    msg = str(runtime[0].message)
    assert "falling back to Python implementation" in msg
    assert "synthetic backend failure" not in msg


def test_backend_failure_raises_when_python_fallback_not_enabled(monkeypatch) -> None:
    monkeypatch.delenv("PYDESEQ2_ALLOW_PYTHON_FALLBACK", raising=False)
    monkeypatch.delenv("PYDESEQ2_DISABLE_CYTHON", raising=False)

    def fake_fit_glm_core(*args, **kwargs):
        raise ValueError("synthetic backend failure")

    monkeypatch.setattr(deseq2_mod, "_fit_glm_cy", fake_fit_glm_core)
    monkeypatch.setattr(deseq2_mod, "_cython_num_threads", lambda: 2)

    counts = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    model_matrix = np.column_stack(
        [np.ones(3, dtype=float), np.array([0.0, 1.0, 0.0])]
    )
    normalization_factors = np.ones_like(counts)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    with pytest.raises(RuntimeError, match="PYDESEQ2_ALLOW_PYTHON_FALLBACK=1"):
        DESeq2._fit_nbinom_glms(
            counts=counts,
            model_matrix=model_matrix,
            normalization_factors=normalization_factors,
            alpha_hat=alpha_hat,
            use_optim=False,
            mu_only=True,
        )


def test_fit_nbinom_glms_can_report_optimizer_rows() -> None:
    counts = np.array([[10.0, 12.0, 14.0], [9.0, 8.0, 7.0]], dtype=float)
    model_matrix = np.column_stack([np.ones(3, dtype=float), np.array([0.0, 1.0, 0.0])])
    normalization_factors = np.ones_like(counts)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    out = DESeq2._fit_nbinom_glms(
        counts=counts,
        model_matrix=model_matrix,
        normalization_factors=normalization_factors,
        alpha_hat=alpha_hat,
        use_optim=False,
        force_optim=True,
        return_optimizer_rows=True,
    )

    assert out["n_rows_for_optim"] == counts.shape[0]
    assert out["n_genes"] == counts.shape[0]
