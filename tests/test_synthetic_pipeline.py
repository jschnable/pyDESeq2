import numpy as np
import pandas as pd

from pydeseq2 import DESeq2


def simulate_counts(seed: int = 11):
    rng = np.random.default_rng(seed)
    n_genes = 500
    n_samples = 10
    n_per_group = n_samples // 2

    condition = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    size_factors = np.exp(rng.normal(0.0, 0.35, size=n_samples))
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))

    base_means = np.exp(rng.normal(4.0, 0.9, size=n_genes))
    dispersions = np.exp(rng.normal(-1.2, 0.5, size=n_genes))

    de_idx = rng.choice(n_genes, size=70, replace=False)
    true_lfc = np.zeros(n_genes, dtype=float)
    true_lfc[de_idx] = rng.normal(1.6, 0.3, size=de_idx.size) * rng.choice(
        [-1.0, 1.0], size=de_idx.size
    )

    counts = np.zeros((n_genes, n_samples), dtype=int)
    for g in range(n_genes):
        for j in range(n_samples):
            fold = 2.0 ** true_lfc[g] if condition[j] == "B" else 1.0
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
    return counts_df, metadata, de_idx, true_lfc, size_factors


def test_wald_pipeline_recovers_signal_on_synthetic_data() -> None:
    counts, metadata, de_idx, true_lfc, true_size_factors = simulate_counts()

    dds = DESeq2(counts=counts, metadata=metadata, design="~ condition")
    dds.deseq(test="wald")
    res = dds.results(coef="condition[T.B]", cooks_cutoff=False, independent_filtering=False)

    assert dds.size_factors_ is not None
    assert dds.dispersions_ is not None
    assert dds.wald_pvalue_ is not None

    nz = ~dds.all_zero_
    disp = dds.dispersions_[nz]
    assert np.all(np.isfinite(disp))
    assert np.all(disp > 0)

    true_sf = true_size_factors / np.exp(np.mean(np.log(true_size_factors)))
    corr = np.corrcoef(np.log(true_sf), np.log(dds.size_factors_))[0, 1]
    assert corr > 0.7

    ranked = res[np.isfinite(res["pvalue"])].sort_values("pvalue")
    top_genes = set(ranked.index[:80])
    true_de_genes = {counts.index[i] for i in de_idx}
    hits = len(top_genes.intersection(true_de_genes))
    assert hits >= 20

    lfc = res["log2FoldChange"].to_numpy()
    sign_match = np.mean(np.sign(lfc[de_idx]) == np.sign(true_lfc[de_idx]))
    assert sign_match > 0.6
