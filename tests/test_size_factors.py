import numpy as np
import pytest

from pydeseq2 import estimate_size_factors_for_matrix


def manual_ratio_size_factors(counts: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        log_geomeans = np.mean(np.log(counts), axis=1)
    if np.all(np.isinf(log_geomeans)):
        raise ValueError("Cannot compute ratio size factors.")
    out = np.empty(counts.shape[1], dtype=float)
    for j in range(counts.shape[1]):
        cnts = counts[:, j]
        keep = np.isfinite(log_geomeans) & (cnts > 0)
        out[j] = np.exp(np.median(np.log(cnts[keep]) - log_geomeans[keep]))
    return out


def test_ratio_size_factors_matches_reference_formula() -> None:
    counts = np.array(
        [
            [10, 20, 30, 60],
            [5, 0, 10, 5],
            [100, 120, 80, 90],
            [40, 35, 20, 10],
        ],
        dtype=float,
    )
    expected = manual_ratio_size_factors(counts)
    observed = estimate_size_factors_for_matrix(counts, type="ratio")
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_poscounts_succeeds_when_ratio_fails() -> None:
    counts = np.array(
        [
            [0, 8, 12],
            [5, 0, 10],
            [7, 9, 0],
        ],
        dtype=float,
    )
    with pytest.raises(Exception):
        estimate_size_factors_for_matrix(counts, type="ratio")
    sf = estimate_size_factors_for_matrix(counts, type="poscounts")
    assert np.all(np.isfinite(sf))
    assert np.all(sf > 0)


def test_incoming_geo_means_are_recentered_to_geometric_mean_one() -> None:
    counts = np.array(
        [
            [10, 20, 30, 40],
            [15, 18, 17, 16],
            [100, 90, 120, 110],
            [30, 31, 32, 33],
        ],
        dtype=float,
    )
    with np.errstate(divide="ignore"):
        geo_means = np.exp(np.mean(np.log(counts), axis=1))
    sf = estimate_size_factors_for_matrix(counts, geo_means=geo_means)
    gm = np.exp(np.mean(np.log(sf)))
    np.testing.assert_allclose(gm, 1.0, rtol=1e-12, atol=1e-12)
