"""Tests for pipeline training wrapper."""

import numpy as np
import pytest
import scipy.io as sio
from pathlib import Path

from pymshbm.pipeline.training import params_training
from pymshbm.types import MSHBMParams


@pytest.mark.parametrize("save_all", [False, True])
def test_saved_costs_preserve_values_returned_by_actual_fit(tmp_path, save_all):
    """Retain all objective levels even when latent arrays are not saved."""
    fixture_path = Path(__file__).resolve().parents[2] / "validation/fixtures/cbig_synthetic_single.npz"
    with np.load(fixture_path) as fixture:
        params = params_training(fixture["data"], fixture["g_mu"], 3, max_iter=5,
                                 output_dir=tmp_path, save_all=save_all)
    saved = sio.loadmat(tmp_path / "priors/Params_Final.mat", simplify_cells=True)["Params"]
    for name in ("cost_em", "cost_intra", "cost_inter"):
        actual = np.asarray(getattr(params, name))
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(np.asarray(saved[name]).reshape(actual.shape), actual)
    np.testing.assert_array_equal(saved["Record"], params.record)


def test_params_training_basic(tmp_path, rng):
    """params_training should run the full pipeline and return MSHBMParams."""
    N, D, S, T, L = 20, 10, 2, 2, 3

    # Create synthetic data
    data = rng.standard_normal((N, D, S, T)).astype(np.float64)
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data[:, :, s, t] /= norms

    # Create synthetic group centroids
    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    params = params_training(
        data=data,
        g_mu=g_mu,
        num_clusters=L,
        max_iter=2,
        output_dir=tmp_path,
    )
    assert isinstance(params, MSHBMParams)
    assert params.mu.shape == (D, L)
    assert params.iter_inter > 0

    # Check that Params_Final.mat was saved
    final_path = tmp_path / "priors" / "Params_Final.mat"
    assert final_path.exists()


def test_params_training_with_subject_ids(tmp_path, rng):
    """Training with subject IDs should produce named parcellation files."""
    N, D, S, T, L = 20, 10, 2, 2, 3

    data = rng.standard_normal((N, D, S, T)).astype(np.float64)
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data[:, :, s, t] /= norms

    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    params = params_training(
        data=data,
        g_mu=g_mu,
        num_clusters=L,
        max_iter=2,
        output_dir=tmp_path,
        subject_ids=["sub01", "sub02"],
        save_all=True,
    )
    assert isinstance(params, MSHBMParams)
    # With save_all, s_lambda should be non-None
    assert params.s_lambda is not None
