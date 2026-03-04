# pyDESeq2

Python reimplementation of the core DESeq2 statistical workflow, with a focus on:

- Statistical behavior parity with DESeq2.
- Pythonic data interfaces (`pandas`, formula design via `patsy`).
- Performance-conscious numerical routines for NB GLMs and dispersion fitting.

## Current Scope

Implemented in this phase:

- Median-ratio and `poscounts` size-factor estimation.
- Gene-wise dispersion estimation with DESeq2-style line search.
- Parametric/local/mean dispersion trend fitting.
- MAP dispersion shrinkage with outlier handling.
- Negative-binomial GLM fitting with IRLS and fallback optimization.
- Wald testing and likelihood-ratio testing.
- Wald beta-prior support (estimated or user-specified prior variances).
- Expanded model matrix support for beta-prior Wald fits.
- Numeric and dictionary-based contrasts for Wald results.
- Character and list-style contrasts in `results()`.
- Cook's distance p-value filtering controls in `results()`.
- Independent filtering controls in `results()`.
- Multiple p-value adjustment methods via `p_adjust_method`.
- Outlier replacement and automatic refit path in `deseq()`.

Planned for later phases:

- Outlier replacement/refit workflow.
- Beta prior workflows and expanded model matrix behavior.
- `lfcShrink`, `rlog`, `vst`, and plotting helpers.
- Full testthat-to-pytest parity coverage.

## Quickstart

```python
from pydeseq2 import DESeq2

dds = DESeq2(counts=counts_df, metadata=metadata_df, design="~ condition")
dds.deseq(test="wald")
res = dds.results(coef="condition[T.B]")

# Optional controls:
# res = dds.results(
#     coef="condition[T.B]",
#     cooks_cutoff=False,
#     independent_filtering=False,
# )

# DESeq2-style contrasts:
# res2 = dds.results(contrast=("condition", "B", "A"))
# res3 = dds.results(contrast=(["condition_B_vs_A"], []))

# Expanded matrix with beta prior:
# dds.deseq(test="wald", beta_prior=True, model_matrix_type="expanded")

# Outlier replacement + refit:
# dds.deseq(test="wald", min_replicates_for_replace=7, cooks_cutoff_replace=None)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

Optional Cython + OpenMP acceleration for dispersion fitting is built at install time when possible.

- Set `PYDESEQ2_OPENMP=1` to force OpenMP build attempt.
- Set `PYDESEQ2_OPENMP=0` to disable OpenMP flags.
- Set `PYDESEQ2_NUM_THREADS=<n>` to control runtime threads used by the Cython dispersion kernel.
- Set `PYDESEQ2_DISABLE_CYTHON=1` to force Python fallback.
- Set `PYDESEQ2_SHOW_FALLBACK_EXCEPTIONS=1` to include full exception details in Cython fallback warnings.

Optional parity test against installed R `DESeq2`:

```bash
RUN_R_PARITY=1 pytest -k parity
```
