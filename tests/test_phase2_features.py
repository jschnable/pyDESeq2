import numpy as np
import pandas as pd
import pytest

from pydeseq2 import DESeq2
from pydeseq2.deseq2 import DESeq2Error


def _simulate_dataset(seed: int = 77, n_genes: int = 300, n_samples: int = 10):
    rng = np.random.default_rng(seed)
    n_per_group = n_samples // 2
    condition = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    size_factors = np.exp(rng.normal(0.0, 0.3, size=n_samples))
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))

    base_means = np.exp(rng.normal(4.1, 0.9, size=n_genes))
    dispersions = np.exp(rng.normal(-1.1, 0.4, size=n_genes))
    de_idx = rng.choice(n_genes, size=max(10, n_genes // 8), replace=False)
    lfc = np.zeros(n_genes, dtype=float)
    lfc[de_idx] = rng.normal(1.4, 0.35, size=de_idx.size) * rng.choice([-1.0, 1.0], size=de_idx.size)

    counts = np.zeros((n_genes, n_samples), dtype=int)
    for g in range(n_genes):
        for j in range(n_samples):
            fold = 2.0 ** lfc[g] if condition[j] == "B" else 1.0
            mu = base_means[g] * size_factors[j] * fold
            alpha = dispersions[g]
            size = 1.0 / alpha
            p = size / (size + mu)
            counts[g, j] = rng.negative_binomial(size, p)

    genes = pd.Index([f"gene_{i}" for i in range(n_genes)], dtype=str)
    samples = pd.Index([f"sample_{j}" for j in range(n_samples)], dtype=str)
    counts_df = pd.DataFrame(counts, index=genes, columns=samples)
    metadata = pd.DataFrame(
        {"condition": pd.Categorical(condition, categories=["A", "B"])},
        index=samples,
    )
    return counts_df, metadata


def test_rank_deficient_design_raises() -> None:
    counts = pd.DataFrame(
        [[10, 20, 10, 20], [5, 6, 5, 6]],
        index=["g1", "g2"],
        columns=["s1", "s2", "s3", "s4"],
    )
    metadata = pd.DataFrame(
        {
            "condition": pd.Categorical(["A", "A", "B", "B"]),
            "batch": pd.Categorical(["A", "A", "B", "B"]),
        },
        index=counts.columns,
    )
    with pytest.raises(DESeq2Error):
        DESeq2(counts=counts, metadata=metadata, design="~ condition + batch")


def test_non_integer_counts_raise_for_wald() -> None:
    counts, metadata = _simulate_dataset(seed=81, n_genes=60, n_samples=8)
    counts = counts.astype(float)
    counts.iloc[0, 0] += 0.25
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.estimate_size_factors().estimate_dispersions()
    with pytest.raises(DESeq2Error):
        dds.nbinom_wald_test()


def test_weighted_pipeline_runs_and_returns_finite_outputs() -> None:
    counts, metadata = _simulate_dataset(seed=82, n_genes=120, n_samples=10)
    rng = np.random.default_rng(82)
    weights = rng.uniform(0.2, 1.0, size=counts.shape)
    weights[rng.random(size=weights.shape) < 0.05] = 0.0

    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition", weights=weights)
    dds.deseq(test="wald")
    res = dds.results(coef="condition[T.B]", cooks_cutoff=False, independent_filtering=False)
    finite_mask = np.isfinite(res["pvalue"].to_numpy())
    assert finite_mask.sum() > 0
    assert np.all(np.isfinite(res.loc[finite_mask, "log2FoldChange"]))


def test_cooks_cutoff_toggle_changes_number_of_na_pvalues() -> None:
    counts, metadata = _simulate_dataset(seed=90, n_genes=140, n_samples=10)
    counts.iloc[0, 0] = int(counts.iloc[0, 0] * 80 + 1000)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(test="wald")
    res_no_cooks = dds.results(cooks_cutoff=False, independent_filtering=False)
    res_cooks = dds.results(cooks_cutoff=True, independent_filtering=False)
    na_no = int(np.sum(~np.isfinite(res_no_cooks["pvalue"].to_numpy())))
    na_yes = int(np.sum(~np.isfinite(res_cooks["pvalue"].to_numpy())))
    assert na_yes >= na_no


def test_independent_filtering_adds_or_preserves_na_padj() -> None:
    counts, metadata = _simulate_dataset(seed=91, n_genes=500, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(test="wald")
    res_no_filter = dds.results(cooks_cutoff=False, independent_filtering=False)
    res_filter = dds.results(cooks_cutoff=False, independent_filtering=True)
    na_no = int(np.sum(~np.isfinite(res_no_filter["padj"].to_numpy())))
    na_yes = int(np.sum(~np.isfinite(res_filter["padj"].to_numpy())))
    assert na_yes >= na_no


def test_numeric_contrast_matches_named_coefficient() -> None:
    counts, metadata = _simulate_dataset(seed=92, n_genes=160, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(test="wald")
    coef_res = dds.results(coef="condition[T.B]", cooks_cutoff=False, independent_filtering=False)
    cvec = np.zeros(dds.n_coefs, dtype=float)
    cvec[dds.coef_names_.index("condition[T.B]")] = 1.0
    contrast_res = dds.results(contrast=cvec, cooks_cutoff=False, independent_filtering=False)
    np.testing.assert_allclose(
        coef_res["log2FoldChange"].to_numpy(),
        contrast_res["log2FoldChange"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        coef_res["pvalue"].to_numpy(),
        contrast_res["pvalue"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_beta_prior_shrinks_lfc_magnitude() -> None:
    counts, metadata = _simulate_dataset(seed=93, n_genes=280, n_samples=10)
    no_prior = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(
        test="wald",
        beta_prior=False,
    )
    with_prior = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(
        test="wald",
        beta_prior=True,
    )
    lfc_no = no_prior.results(
        contrast=("condition", "B", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )[
        "log2FoldChange"
    ].to_numpy()
    lfc_pr = with_prior.results(
        contrast=("condition", "B", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )[
        "log2FoldChange"
    ].to_numpy()
    mask = np.isfinite(lfc_no) & np.isfinite(lfc_pr)
    assert np.median(np.abs(lfc_pr[mask])) <= np.median(np.abs(lfc_no[mask]))


def test_low_maxit_flags_non_convergence() -> None:
    counts, metadata = _simulate_dataset(seed=94, n_genes=120, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.estimate_size_factors().estimate_dispersions().nbinom_wald_test(maxit=1, use_optim=False)
    assert dds.beta_conv_ is not None
    conv = dds.beta_conv_[np.isfinite(dds.beta_conv_)]
    assert np.any(conv == 0.0)


def test_wald_test_records_optimizer_stats() -> None:
    counts, metadata = _simulate_dataset(seed=941, n_genes=80, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.estimate_size_factors().estimate_dispersions().nbinom_wald_test(maxit=1, use_optim=True)

    assert dds.optimizer_stats_ is not None
    stats = dds.optimizer_stats_
    assert stats["test"] == "wald"
    n_nonzero = int(stats["n_nonzero_genes"])
    assert n_nonzero > 0
    assert 0 <= int(stats["wald_mle_rows_for_optim"]) <= n_nonzero
    assert 0.0 <= float(stats["wald_mle_fraction_for_optim"]) <= 1.0
    assert 0 <= int(stats["wald_final_rows_for_optim"]) <= n_nonzero
    assert 0.0 <= float(stats["wald_final_fraction_for_optim"]) <= 1.0


def test_dispersion_fit_falls_back_to_mean_when_all_gene_estimates_near_min_disp() -> None:
    counts, metadata = _simulate_dataset(seed=95, n_genes=80, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.estimate_size_factors()
    dds._ensure_base_stats()

    min_disp = 1e-8
    nz = ~dds.all_zero_
    dds.disp_gene_est_ = np.full(dds.n_genes, np.nan, dtype=float)
    dds.disp_gene_est_[nz] = min_disp

    dds.estimate_dispersions_fit(fit_type="parametric", min_disp=min_disp)
    assert dds.dispersion_fit_ is not None
    assert dds.dispersion_fit_.fit_type == "mean"
    assert np.all(np.isfinite(dds.disp_fit_[nz]))
    assert np.all(dds.disp_fit_[nz] >= min_disp)


def test_glm_fit_is_invariant_to_sample_order() -> None:
    counts = np.array(
        [
            [100, 120, 80, 300, 320, 290],
            [50, 40, 45, 30, 25, 35],
        ],
        dtype=float,
    )
    model_matrix = np.column_stack([np.ones(6, dtype=float), np.array([0, 0, 0, 1, 1, 1], dtype=float)])
    normalization_factors = np.ones_like(counts, dtype=float)
    alpha_hat = np.array([0.2, 0.3], dtype=float)

    fit_a = DESeq2._fit_nbinom_glms(
        counts=counts,
        model_matrix=model_matrix,
        normalization_factors=normalization_factors,
        alpha_hat=alpha_hat,
        use_optim=False,
    )

    perm = np.array([3, 0, 4, 1, 5, 2], dtype=int)
    fit_b = DESeq2._fit_nbinom_glms(
        counts=counts[:, perm],
        model_matrix=model_matrix[perm, :],
        normalization_factors=normalization_factors[:, perm],
        alpha_hat=alpha_hat,
        use_optim=False,
    )

    np.testing.assert_allclose(fit_a["beta_matrix"], fit_b["beta_matrix"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_a["log_like"], fit_b["log_like"], rtol=1e-10, atol=1e-10)
