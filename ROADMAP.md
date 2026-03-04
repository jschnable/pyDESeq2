# Reimplementation Roadmap

## Phase 1: Core Differential Expression Engine

Status: complete in this iteration.

- Package scaffold and model object.
- Size-factor estimation (`ratio`, `poscounts`).
- Dispersion estimation pipeline (gene-wise, trend, MAP).
- NB GLM fitting with IRLS + optimization fallback.
- Wald and LRT significance testing.
- Core unit/integration tests.

## Phase 2: Statistical Parity Hardening

Status: complete in this iteration.

- Added parity-focused behavior checks for rank-deficient designs and non-integer counts.
- Added convergence regression test coverage (`maxit` and optimization fallback behavior).
- Implemented Cook's distance p-value filtering controls in `results()`.
- Implemented DESeq2-style independent filtering controls in `results()`.
- Added weighted-fitting coverage and tests.

## Phase 3: Advanced DESeq2 Functionality

Status: complete in this iteration.

- Implemented beta prior workflow for Wald tests (estimated or user-supplied prior variances).
- Implemented expanded model matrices for beta-prior Wald fitting.
- Implemented outlier replacement and refit path in `deseq()`.
- Implemented character/list/numeric/dictionary contrasts in `results()`.
- Added tests for each feature area.

## Phase 4: Transformations and Shrinkage

Status: planned.

- `lfcShrink` methods.
- Variance-stabilizing transformation (`vst`).
- Regularized log transform (`rlog`).

## Phase 5: Performance and Ergonomics

Status: planned.

- Hot-loop acceleration (numba/Cython) for large datasets.
- Profiling-driven optimization and memory reductions.
- Optional parallel fitting APIs.
