import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from pydeseq2 import DESeq2


def _r_has_deseq2() -> bool:
    if shutil.which("Rscript") is None:
        return False
    check = subprocess.run(
        ["Rscript", "-e", "quit(status=ifelse(requireNamespace('DESeq2', quietly=TRUE),0,1))"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return check.returncode == 0


def _simulate_counts(
    *,
    seed: int,
    n_genes: int,
    n_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Index]:
    rng = np.random.default_rng(seed)
    cond = np.array(["A"] * (n_samples // 2) + ["B"] * (n_samples - (n_samples // 2)))
    sf = np.exp(rng.normal(0.0, 0.3, size=n_samples))
    sf = sf / np.exp(np.mean(np.log(sf)))
    base_means = np.exp(rng.normal(4.1, 0.8, size=n_genes))
    alpha = np.exp(rng.normal(-1.0, 0.45, size=n_genes))
    lfc = np.zeros(n_genes)
    de = rng.choice(n_genes, size=max(1, int(0.15 * n_genes)), replace=False)
    lfc[de] = rng.normal(1.4, 0.25, size=de.size)

    counts = np.zeros((n_genes, n_samples), dtype=int)
    for g in range(n_genes):
        for j in range(n_samples):
            mu = base_means[g] * sf[j] * (2.0 ** lfc[g] if cond[j] == "B" else 1.0)
            size = 1.0 / alpha[g]
            p = size / (size + mu)
            counts[g, j] = rng.negative_binomial(size, p)

    genes = pd.Index([f"gene_{i}" for i in range(n_genes)], dtype=str)
    samples = pd.Index([f"sample_{j}" for j in range(n_samples)], dtype=str)
    counts_df = pd.DataFrame(counts, index=genes, columns=samples)
    metadata = pd.DataFrame(
        {"condition": pd.Categorical(cond, categories=["A", "B"])},
        index=samples,
    )
    return counts_df, metadata, genes, samples


@pytest.mark.skipif(
    os.environ.get("RUN_R_PARITY", "0") != "1",
    reason="Set RUN_R_PARITY=1 to run DESeq2 parity test.",
)
def test_parity_against_r_deseq2_on_small_dataset() -> None:
    if not _r_has_deseq2():
        pytest.skip("Rscript or R DESeq2 package not available.")

    counts_df, metadata, genes, samples = _simulate_counts(
        seed=123,
        n_genes=120,
        n_samples=8,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        counts_path = tmp / "counts.csv"
        meta_path = tmp / "metadata.csv"
        out_rows = tmp / "r_rows.csv"
        out_sf = tmp / "r_sf.csv"
        counts_df.to_csv(counts_path)
        metadata.to_csv(meta_path)

        r_code = f"""
        suppressPackageStartupMessages(library(DESeq2))
        counts <- as.matrix(read.csv("{counts_path}", row.names=1, check.names=FALSE))
        meta <- read.csv("{meta_path}", row.names=1, check.names=FALSE)
        meta$condition <- factor(meta$condition, levels=c("A","B"))
        dds <- DESeqDataSetFromMatrix(countData=counts, colData=meta, design=~ condition)
        dds <- DESeq(dds, fitType="parametric", betaPrior=FALSE, quiet=TRUE, minReplicatesForReplace=Inf)
        rn <- resultsNames(dds)
        coef_name <- rn[grepl("^condition_", rn)][1]
        res <- results(dds, name=coef_name, cooksCutoff=FALSE, independentFiltering=FALSE)
        write.csv(data.frame(sample=colnames(dds), sizeFactor=sizeFactors(dds)), "{out_sf}", row.names=FALSE)
        write.csv(data.frame(gene=rownames(dds), disp=dispersions(dds), lfc=res$log2FoldChange, pvalue=res$pvalue),
                  "{out_rows}", row.names=FALSE)
        """
        subprocess.run(["Rscript", "-e", r_code], check=True)

        r_sf = pd.read_csv(out_sf).set_index("sample").loc[samples, "sizeFactor"].to_numpy(dtype=float)
        r_rows = pd.read_csv(out_rows).set_index("gene").loc[genes]

    py = DESeq2(counts=counts_df, metadata=metadata, design="~ condition").deseq(test="wald")
    py_res = py.results(
        coef="condition[T.B]",
        cooks_cutoff=False,
        independent_filtering=False,
    ).loc[genes]

    sf_corr = np.corrcoef(np.log(r_sf), np.log(py.size_factors_))[0, 1]
    assert sf_corr > 0.9

    disp_corr = spearmanr(r_rows["disp"].to_numpy(), py.dispersions_)[0]
    assert disp_corr > 0.6

    lfc_corr = spearmanr(r_rows["lfc"].to_numpy(), py_res["log2FoldChange"].to_numpy())[0]
    assert lfc_corr > 0.7

    pval_corr = spearmanr(r_rows["pvalue"].to_numpy(), py_res["pvalue"].to_numpy())[0]
    assert pval_corr > 0.5


@pytest.mark.skipif(
    os.environ.get("RUN_R_RUNTIME", "0") != "1",
    reason="Set RUN_R_RUNTIME=1 to run Python vs R runtime comparison.",
)
def test_runtime_py_vs_r_with_use_cr_enabled() -> None:
    if not _r_has_deseq2():
        pytest.skip("Rscript or R DESeq2 package not available.")

    n_genes = int(os.environ.get("RUNTIME_N_GENES", "800"))
    n_samples = int(os.environ.get("RUNTIME_N_SAMPLES", "96"))
    repeats = int(os.environ.get("RUNTIME_REPEATS", "3"))
    seed = int(os.environ.get("RUNTIME_SEED", "20260303"))
    use_cr = True

    counts_df, metadata, _, _ = _simulate_counts(
        seed=seed,
        n_genes=n_genes,
        n_samples=n_samples,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        counts_path = tmp / "counts.csv"
        meta_path = tmp / "metadata.csv"
        out_r = tmp / "r_runtime.csv"
        counts_df.to_csv(counts_path)
        metadata.to_csv(meta_path)

        r_code = f"""
        suppressPackageStartupMessages(library(DESeq2))
        counts <- as.matrix(read.csv("{counts_path}", row.names=1, check.names=FALSE))
        meta <- read.csv("{meta_path}", row.names=1, check.names=FALSE)
        meta$condition <- factor(meta$condition, levels=c("A","B"))
        times <- numeric({repeats})
        for (i in seq_len({repeats})) {{
            dds <- DESeqDataSetFromMatrix(countData=counts, colData=meta, design=~ condition)
            times[i] <- as.numeric(system.time({{
                dds <- DESeq(dds, fitType="parametric", betaPrior=FALSE, quiet=TRUE, minReplicatesForReplace=Inf)
            }})["elapsed"])
        }}
        write.csv(data.frame(run=seq_along(times), seconds=times), "{out_r}", row.names=FALSE)
        """
        subprocess.run(["Rscript", "-e", r_code], check=True)
        r_times = pd.read_csv(out_r)["seconds"].to_numpy(dtype=float)

    py_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        DESeq2(counts=counts_df, metadata=metadata, design="~ condition").deseq(
            test="wald",
            use_cr=use_cr,
        )
        py_times.append(time.perf_counter() - t0)
    py_times = np.asarray(py_times, dtype=float)

    assert py_times.size == repeats
    assert r_times.size == repeats
    assert np.all(py_times > 0.0)
    assert np.all(r_times > 0.0)

    print(
        "\nRUNTIME_COMPARISON "
        f"n_genes={n_genes} n_samples={n_samples} repeats={repeats} use_cr={use_cr} "
        f"py_mean_s={float(np.mean(py_times)):.3f} r_mean_s={float(np.mean(r_times)):.3f} "
        f"py_over_r={float(np.mean(py_times) / np.mean(r_times)):.3f}"
    )
