"""Core public API for the Python DESeq2 reimplementation."""

from .deseq2 import DESeq2, DESeq2Error, estimate_size_factors_for_matrix

__all__ = ["DESeq2", "DESeq2Error", "estimate_size_factors_for_matrix"]
