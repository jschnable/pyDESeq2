import numpy as np
import pandas as pd

from pydeseq2 import DESeq2


def _simulate(seed: int = 202, n_genes: int = 220, n_samples: int = 10):
    rng = np.random.default_rng(seed)
    n_per_group = n_samples // 2
    condition = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    sf = np.exp(rng.normal(0.0, 0.35, size=n_samples))
    sf = sf / np.exp(np.mean(np.log(sf)))
    base = np.exp(rng.normal(4.0, 0.8, size=n_genes))
    alpha = np.exp(rng.normal(-1.2, 0.4, size=n_genes))
    lfc = np.zeros(n_genes)
    de = rng.choice(n_genes, size=max(10, n_genes // 10), replace=False)
    lfc[de] = rng.normal(1.3, 0.3, size=de.size) * rng.choice([-1.0, 1.0], size=de.size)

    counts = np.zeros((n_genes, n_samples), dtype=int)
    for g in range(n_genes):
        for j in range(n_samples):
            mu = base[g] * sf[j] * (2.0 ** lfc[g] if condition[j] == "B" else 1.0)
            size = 1.0 / alpha[g]
            p = size / (size + mu)
            counts[g, j] = rng.negative_binomial(size, p)

    genes = [f"gene_{i}" for i in range(n_genes)]
    samples = [f"sample_{j}" for j in range(n_samples)]
    counts_df = pd.DataFrame(counts, index=genes, columns=samples)
    metadata = pd.DataFrame(
        {"condition": pd.Categorical(condition, categories=["A", "B"])},
        index=samples,
    )
    return counts_df, metadata


def _simulate_three_group(seed: int = 206, n_genes: int = 240, n_samples: int = 12):
    rng = np.random.default_rng(seed)
    n_a = n_samples // 3
    n_b = n_samples // 3
    n_c = n_samples - n_a - n_b
    condition = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c)
    sf = np.exp(rng.normal(0.0, 0.35, size=n_samples))
    sf = sf / np.exp(np.mean(np.log(sf)))
    base = np.exp(rng.normal(4.0, 0.8, size=n_genes))
    alpha = np.exp(rng.normal(-1.2, 0.4, size=n_genes))
    lfc_b = rng.normal(0.7, 0.25, size=n_genes)
    lfc_c = rng.normal(-0.6, 0.25, size=n_genes)

    counts = np.zeros((n_genes, n_samples), dtype=int)
    for g in range(n_genes):
        for j in range(n_samples):
            if condition[j] == "B":
                fold = 2.0 ** lfc_b[g]
            elif condition[j] == "C":
                fold = 2.0 ** lfc_c[g]
            else:
                fold = 1.0
            mu = base[g] * sf[j] * fold
            size = 1.0 / alpha[g]
            p = size / (size + mu)
            counts[g, j] = rng.negative_binomial(size, p)

    genes = [f"gene_{i}" for i in range(n_genes)]
    samples = [f"sample_{j}" for j in range(n_samples)]
    counts_df = pd.DataFrame(counts, index=genes, columns=samples)
    metadata = pd.DataFrame(
        {"condition": pd.Categorical(condition, categories=["A", "B", "C"])},
        index=samples,
    )
    return counts_df, metadata


def test_character_and_list_contrasts_match_named_coef() -> None:
    counts, metadata = _simulate(seed=203, n_genes=180, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(test="wald")

    assert "condition_B_vs_A" in dds.results_names()

    by_name = dds.results(
        coef="condition_B_vs_A",
        cooks_cutoff=False,
        independent_filtering=False,
    )
    by_char = dds.results(
        contrast=("condition", "B", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )
    by_list = dds.results(
        contrast=(["condition_B_vs_A"], []),
        cooks_cutoff=False,
        independent_filtering=False,
    )

    np.testing.assert_allclose(
        by_name["log2FoldChange"].to_numpy(),
        by_char["log2FoldChange"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        by_char["log2FoldChange"].to_numpy(),
        by_list["log2FoldChange"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        by_name["pvalue"].to_numpy(),
        by_char["pvalue"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_expanded_model_matrix_beta_prior_runs() -> None:
    counts, metadata = _simulate(seed=204, n_genes=220, n_samples=10)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(
        test="wald",
        beta_prior=True,
        model_matrix_type="expanded",
        min_replicates_for_replace=np.inf,
    )
    assert dds.model_matrix_type_ == "expanded"
    assert any(name.startswith("conditionA") for name in dds.coef_names_)
    assert any(name.startswith("conditionB") for name in dds.coef_names_)

    res = dds.results(
        contrast=("condition", "B", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )
    finite = np.isfinite(res["pvalue"].to_numpy())
    assert finite.sum() > 0


def test_three_level_character_contrast_matches_numeric_contrast() -> None:
    counts, metadata = _simulate_three_group(seed=207, n_genes=240, n_samples=12)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(test="wald")
    by_char = dds.results(
        contrast=("condition", "C", "B"),
        cooks_cutoff=False,
        independent_filtering=False,
    )
    cvec = np.zeros(dds.n_coefs, dtype=float)
    cvec[dds.coef_names_.index("condition[T.C]")] = 1.0
    cvec[dds.coef_names_.index("condition[T.B]")] = -1.0
    by_numeric = dds.results(
        contrast=cvec,
        cooks_cutoff=False,
        independent_filtering=False,
    )
    np.testing.assert_allclose(
        by_char["log2FoldChange"].to_numpy(),
        by_numeric["log2FoldChange"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        by_char["pvalue"].to_numpy(),
        by_numeric["pvalue"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_outlier_replacement_refit_path_runs_and_preserves_original_counts() -> None:
    counts, metadata = _simulate(seed=205, n_genes=160, n_samples=10)
    # Inject a very large outlier count.
    counts.iloc[0, 0] = int(counts.iloc[0, 0] * 200 + 5000)

    original_counts = counts.to_numpy(copy=True)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(
        test="wald",
        min_replicates_for_replace=3,
        cooks_cutoff_replace=1.0,
    )

    assert dds.replace_counts_ is not None
    assert dds.replace_cooks_ is not None
    assert dds.replace_mask_ is not None
    assert dds.replaceable_samples_ is not None
    assert np.any(dds.replace_mask_)
    np.testing.assert_array_equal(dds.counts_, original_counts)

    res = dds.results(coef="condition_B_vs_A")
    assert "padj" in res.columns


def test_score_test_reuses_cached_null_model_across_multiple_groupings() -> None:
    counts, metadata = _simulate_three_group(seed=208, n_genes=220, n_samples=12)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.estimate_size_factors().estimate_dispersions()
    dds.prepare_score_test_null(null_design="~ 1")
    cache_id = id(dds.score_null_mu_)

    res_ba = dds.score_test(
        grouping=("condition", "B", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )
    res_ca = dds.score_test(
        grouping=("condition", "C", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )

    assert id(dds.score_null_mu_) == cache_id
    assert np.sum(np.isfinite(res_ba["pvalue"].to_numpy())) > 0
    assert np.sum(np.isfinite(res_ca["pvalue"].to_numpy())) > 0


def test_score_test_tracks_wald_effect_direction() -> None:
    counts, metadata = _simulate(seed=209, n_genes=240, n_samples=12)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition").deseq(test="wald")
    wald = dds.results(
        contrast=("condition", "B", "A"),
        cooks_cutoff=False,
        independent_filtering=False,
    )
    score = dds.score_test(
        grouping=("condition", "B", "A"),
        null_design="~ 1",
        cooks_cutoff=False,
        independent_filtering=False,
    )

    wald_stat = wald["stat"].to_numpy(dtype=float)
    score_stat = score["stat"].to_numpy(dtype=float)
    wald_lfc = wald["log2FoldChange"].to_numpy(dtype=float)
    score_lfc = score["log2FoldChange"].to_numpy(dtype=float)
    finite = (
        np.isfinite(wald_stat)
        & np.isfinite(score_stat)
        & np.isfinite(wald_lfc)
        & np.isfinite(score_lfc)
    )
    assert np.sum(finite) > 100
    corr = float(np.corrcoef(wald_stat[finite], score_stat[finite])[0, 1])
    assert corr > 0.6
    sign_match = np.mean(np.sign(wald_lfc[finite]) == np.sign(score_lfc[finite]))
    assert sign_match > 0.8


def test_score_test_accepts_numeric_grouping_vector_with_excluded_samples() -> None:
    counts, metadata = _simulate_three_group(seed=210, n_genes=220, n_samples=12)
    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.estimate_size_factors().estimate_dispersions()

    values = metadata["condition"].astype(str).to_numpy()
    grouping = np.zeros(dds.n_samples, dtype=float)
    grouping[values == "B"] = 1.0
    grouping[values == "C"] = -1.0
    res = dds.score_test(
        grouping=grouping,
        null_design="~ 1",
        cooks_cutoff=False,
        independent_filtering=False,
    )

    assert np.sum(np.isfinite(res["pvalue"].to_numpy())) > 0
    assert "padj" in res.columns
