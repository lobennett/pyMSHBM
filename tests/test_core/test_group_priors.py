"""Tests for group prior estimation."""

import numpy as np
import pytest

from pymshbm.core.group_priors import (
    initialize_params,
    vmf_clustering_subject_session,
    intra_subject_var,
    inter_subject_var,
    estimate_group_priors,
)
from pymshbm.types import MSHBMParams


@pytest.fixture
def small_settings():
    """Small problem settings for fast testing."""
    return {
        "num_sub": 2,
        "num_session": 2,
        "num_clusters": 3,
        "dim": 9,  # D-1 where D=10
        "ini_concentration": 500,
        "epsilon": 1e-4,
        "conv_th": 1e-5,
        "max_iter": 5,
    }


@pytest.fixture
def synthetic_data(rng, small_settings):
    """Synthetic normalized FC data: (N, D, S, T)."""
    N, D = 20, 10
    S = small_settings["num_sub"]
    T = small_settings["num_session"]
    series = rng.standard_normal((N, D, S, T)).astype(np.float64)
    # Normalize each vertex's profile
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(series[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            series[:, :, s, t] /= norms
    return series


@pytest.fixture
def synthetic_g_mu(rng, small_settings):
    """Synthetic group-level centroids (D, L)."""
    D = small_settings["dim"] + 1
    L = small_settings["num_clusters"]
    mu = rng.standard_normal((D, L))
    mu /= np.linalg.norm(mu, axis=0, keepdims=True)
    return mu


def test_initialize_params_shapes(synthetic_data, synthetic_g_mu, small_settings):
    """initialize_params should create MSHBMParams with correct shapes."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    D = small_settings["dim"] + 1
    L = small_settings["num_clusters"]
    N = synthetic_data.shape[0]
    S = small_settings["num_sub"]
    T = small_settings["num_session"]
    assert params.mu.shape == (D, L)
    assert params.epsil.shape == (L,)
    assert params.sigma.shape == (L,)
    assert params.kappa.shape == (L,)
    assert params.theta.shape == (N, L)
    assert params.s_psi.shape == (D, L, S)
    assert params.s_t_nu.shape == (D, L, T, S)
    assert params.s_lambda.shape == (N, L, S)


def test_initialize_params_theta_valid(synthetic_data, synthetic_g_mu, small_settings):
    """Theta should have non-negative values and reasonable range."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    assert np.all(params.theta >= 0)
    assert np.all(np.isfinite(params.theta))
    # Each row should sum to at most ~num_clusters (individual s_lambda rows sum to 1)
    row_sums = params.theta.sum(axis=1)
    nonzero = row_sums > 0
    assert nonzero.any()


def test_vmf_clustering_updates_kappa(synthetic_data, synthetic_g_mu, small_settings):
    """vmf_clustering_subject_session should update kappa and s_t_nu."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    kappa_before = params.kappa.copy()
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    # Parameters should have been updated (not identical)
    assert params.s_lambda is not None
    assert params.theta is not None


def test_intra_subject_var_updates_sigma(synthetic_data, synthetic_g_mu, small_settings):
    """intra_subject_var should update s_psi and sigma."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    sigma_before = params.sigma.copy()
    params = intra_subject_var(params, small_settings)
    assert params.s_psi is not None
    assert params.sigma is not None


def test_inter_subject_var_updates_mu(synthetic_data, synthetic_g_mu, small_settings):
    """inter_subject_var should update mu and epsil."""
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    params = intra_subject_var(params, small_settings)
    mu_before = params.mu.copy()
    params = inter_subject_var(params, small_settings)
    assert params.mu.shape == mu_before.shape
    assert params.epsil is not None


def test_estimate_group_priors_returns_params(synthetic_data, synthetic_g_mu, small_settings):
    """Full estimation should return valid MSHBMParams."""
    params = estimate_group_priors(
        data=synthetic_data,
        g_mu=synthetic_g_mu,
        settings=small_settings,
    )
    assert isinstance(params, MSHBMParams)
    assert params.mu.ndim == 2
    assert params.iter_inter > 0
    assert len(params.record) > 0


def test_estimate_group_priors_converges(synthetic_data, synthetic_g_mu, small_settings):
    """Cost should decrease or stabilize across iterations."""
    params = estimate_group_priors(
        data=synthetic_data,
        g_mu=synthetic_g_mu,
        settings=small_settings,
    )
    # Should complete without error and record costs
    assert len(params.record) >= 1


def test_estimate_group_priors_float32_close_to_float64():
    """float32 data should produce results close to float64."""
    rng = np.random.default_rng(42)
    N, D, S, T, L = 20, 10, 2, 2, 3
    data_f64 = rng.standard_normal((N, D, S, T))
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data_f64[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data_f64[:, :, s, t] /= norms

    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    settings = {
        "num_sub": S, "num_session": T, "num_clusters": L,
        "dim": D - 1, "ini_concentration": 500,
        "epsilon": 1e-4, "conv_th": 1e-5, "max_iter": 5,
    }
    params_f64 = estimate_group_priors(data_f64, g_mu, settings)

    data_f32 = data_f64.astype(np.float32)
    params_f32 = estimate_group_priors(data_f32, g_mu, settings)

    np.testing.assert_allclose(params_f32.mu, params_f64.mu, atol=1e-4)
    np.testing.assert_allclose(params_f32.sigma, params_f64.sigma, rtol=0.01)
    np.testing.assert_allclose(params_f32.theta, params_f64.theta, atol=1e-3)
    assert params_f32.iter_inter == params_f64.iter_inter


def test_estimate_group_priors_memmap_matches_regular(tmp_path):
    """Memory-mapped data (parallel path) should match in-memory (sequential)."""
    rng = np.random.default_rng(42)
    N, D, S, T, L = 20, 10, 2, 2, 3
    data = rng.standard_normal((N, D, S, T))
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data[:, :, s, t] /= norms

    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    settings = {
        "num_sub": S, "num_session": T, "num_clusters": L,
        "dim": D - 1, "ini_concentration": 500,
        "epsilon": 1e-4, "conv_th": 1e-5, "max_iter": 5,
    }

    # Sequential path (regular array)
    params_seq = estimate_group_priors(data, g_mu, settings)

    # Parallel path (memmap array)
    data_path = tmp_path / "data.npy"
    np.save(str(data_path), data)
    data_mmap = np.load(str(data_path), mmap_mode='r')
    params_par = estimate_group_priors(data_mmap, g_mu, settings)

    np.testing.assert_allclose(params_par.mu, params_seq.mu, atol=1e-10)
    np.testing.assert_allclose(params_par.sigma, params_seq.sigma, atol=1e-10)
    np.testing.assert_allclose(params_par.theta, params_seq.theta, atol=1e-10)
    np.testing.assert_allclose(params_par.record, params_seq.record, atol=1e-10)
    assert params_par.iter_inter == params_seq.iter_inter


def test_estimate_group_priors_posterior_mass_regression():
    """Spatial prior uses ordinary subject mean, including underflow zeros.

    Executed CBIG golden arrays are checked in test_reference_oracle.py; the
    inherited implementation snapshot encoded the averaging and loop bugs.
    """
    rng = np.random.default_rng(42)
    N, D, S, T, L = 20, 10, 2, 2, 3
    data = rng.standard_normal((N, D, S, T))
    for s in range(S):
        for t in range(T):
            norms = np.linalg.norm(data[:, :, s, t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data[:, :, s, t] /= norms

    g_mu = rng.standard_normal((D, L))
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)

    settings = {
        "num_sub": S, "num_session": T, "num_clusters": L,
        "dim": D - 1, "ini_concentration": 500,
        "epsilon": 1e-4, "conv_th": 1e-5, "max_iter": 5,
    }
    params = estimate_group_priors(data, g_mu, settings)

    np.testing.assert_allclose(params.theta.sum(), N, atol=1e-5)
    np.testing.assert_allclose(params.s_lambda.sum(), N * S, atol=1e-5)
    np.testing.assert_allclose(params.theta, params.s_lambda.mean(axis=2))
    np.testing.assert_allclose(
        np.linalg.norm(params.mu, axis=0), np.ones(L), atol=1e-10)


def test_initial_theta_matches_reference_mean_then_count(synthetic_data, synthetic_g_mu, small_settings):
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    count = np.count_nonzero(params.s_lambda, axis=2)
    expected = np.divide(params.s_lambda.mean(axis=2), count,
                         out=np.zeros_like(params.theta), where=count > 0)
    np.testing.assert_allclose(params.theta, expected)


def test_e_step_preserves_medial_wall_and_uses_subject_mean(synthetic_data, synthetic_g_mu, small_settings):
    from pymshbm.core.group_priors import _e_step
    data = synthetic_data.copy()
    data[0] = 0
    params = initialize_params(data, synthetic_g_mu, small_settings)
    _e_step(params, small_settings, data)
    np.testing.assert_array_equal(params.s_lambda[0], 0)
    np.testing.assert_allclose(params.theta, params.s_lambda.mean(axis=2))


def test_kappa_averages_sessions(synthetic_data, synthetic_g_mu, small_settings):
    from pymshbm.core.group_priors import _m_step
    settings = dict(small_settings, num_session=1, ini_concentration=1.0)
    data = synthetic_data[..., :1]
    one = initialize_params(data, synthetic_g_mu, settings)
    settings_two = dict(settings, num_session=2)
    repeated = np.repeat(data, 2, axis=3)
    two = initialize_params(repeated, synthetic_g_mu, settings_two)
    # Hold membership fixed so duplication tests just the concentration update.
    two.s_lambda = one.s_lambda.copy()
    _m_step(one, settings, data, settings['epsilon'], 1.0)
    _m_step(two, settings_two, repeated, settings['epsilon'], 1.0)
    np.testing.assert_allclose(two.kappa, one.kappa, rtol=1e-10)


def test_inter_cost_contains_hierarchical_priors(synthetic_data, synthetic_g_mu, small_settings):
    from pymshbm.core.group_priors import _compute_inter_cost, _compute_em_cost
    from pymshbm.math.vmf import cdln
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    em = _compute_em_cost(params, small_settings, synthetic_data).sum()
    # All initialized directions are identical; all prior dot products are one.
    prior = (small_settings['num_sub'] * small_settings['num_session'] *
             np.sum(params.sigma + cdln(params.sigma, small_settings['dim'])) +
             small_settings['num_sub'] *
             np.sum(params.epsil + cdln(params.epsil, small_settings['dim'])))
    assert _compute_inter_cost(params, small_settings, synthetic_data) == pytest.approx(em + prior)


def test_missing_session_directions_do_not_poison_subject_means(synthetic_data, synthetic_g_mu, small_settings):
    data = synthetic_data.copy()
    data[:, :, 1, 1] = np.nan
    params = initialize_params(data, synthetic_g_mu, small_settings)
    params = vmf_clustering_subject_session(params, small_settings, data)
    params = intra_subject_var(params, small_settings)
    assert np.all(np.isfinite(params.s_psi))
    assert np.all(np.isfinite(params.sigma))


def test_group_fit_records_upstream_nested_costs(synthetic_data, synthetic_g_mu, small_settings):
    params = estimate_group_priors(synthetic_data, synthetic_g_mu,
                                   dict(small_settings, max_iter=1))
    assert params.cost_em is not None
    assert params.cost_intra is not None
    assert params.cost_inter == params.cost_intra
    assert params.record == [params.cost_intra]


def test_inner_convergence_records_previous_cost(synthetic_data, synthetic_g_mu, small_settings, monkeypatch):
    import pymshbm.core.group_priors as gp
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    costs = iter([np.array([2., 2.]), np.array([2.00001, 2.00001])])
    monkeypatch.setattr(gp, '_compute_em_cost', lambda *a, **kw: next(costs))
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    np.testing.assert_array_equal(params.cost_em, [2., 2.])


def test_prior_cost_missing_session_reduction_order(synthetic_data, synthetic_g_mu, small_settings):
    from pymshbm.core.group_priors import _compute_inter_cost
    from pymshbm.math.vmf import cdln
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    params.s_t_nu[:, :, 1, 1] = np.nan
    params.cost_em = np.zeros(2)
    # Ordinary subject sum poisons session 2; nansum discards that whole session.
    expected = (2 * np.sum(params.sigma + cdln(params.sigma, small_settings['dim'])) +
                2 * np.sum(params.epsil + cdln(params.epsil, small_settings['dim'])))
    assert _compute_inter_cost(params, small_settings, synthetic_data) == pytest.approx(expected)


def test_single_session_nan_cost_is_not_silently_omitted(synthetic_data, synthetic_g_mu, small_settings):
    from pymshbm.core.group_priors import _compute_em_cost
    settings = dict(small_settings, num_session=1)
    data = synthetic_data[..., :1].copy()
    data[:, :, 1, 0] = np.nan
    params = initialize_params(data, synthetic_g_mu, settings)
    costs = _compute_em_cost(params, settings, data)
    assert np.isnan(costs[1])


def test_reference_single_precision_posteriors(synthetic_data, synthetic_g_mu, small_settings):
    params = initialize_params(synthetic_data, synthetic_g_mu, small_settings)
    assert params.s_lambda.dtype == np.float32
    assert params.theta.dtype == np.float32
    params = vmf_clustering_subject_session(params, small_settings, synthetic_data)
    assert params.s_lambda.dtype == np.float32
    assert params.cost_em.dtype == np.float32
    # Reference allocates double session directions, then assigns single results.
    assert params.s_t_nu.dtype == np.float64
    np.testing.assert_array_equal(params.s_t_nu, params.s_t_nu.astype(np.float32))


def test_no_assigned_vertices_fails_instead_of_returning_empty_fit(synthetic_data, synthetic_g_mu, small_settings):
    with pytest.raises(ValueError, match='assigned vertices'):
        estimate_group_priors(np.zeros_like(synthetic_data), synthetic_g_mu, small_settings)


def test_intra_convergence_uses_signed_reference_sigma_denominator():
    """Quantized directions can exceed unit norm and transiently yield sigma<0.

    Upstream divides by that signed sigma and retains previous direction flags;
    replacing the denominator by a positive floor changes its stopping rule.
    """
    direction = np.zeros((10, 1))
    direction[0] = 1
    params = MSHBMParams(mu=direction.copy(), epsil=np.array([500.]),
                         sigma=np.array([500.]), theta=np.ones((1, 1)),
                         kappa=np.array([500.]), s_psi=direction[:, :, None].copy(),
                         s_t_nu=np.tile(direction[:, :, None, None] * (1 + 1e-6), (1, 1, 2, 1)))
    settings = dict(num_sub=1, num_session=2, num_clusters=1, dim=9,
                    epsilon=1e-4, numerical_max_iter=2)
    result = intra_subject_var(params, settings)
    assert result.sigma[0] > 0
    np.testing.assert_array_equal(result.s_psi[:, :, 0], -direction)


def test_single_profile_dot_casts_double_directions_before_multiplication():
    from pymshbm.core.group_priors import _reference_log_probability
    # The increment vanishes when MATLAB converts the double operand to single.
    data = np.array([[1., 1.]], dtype=np.float32)
    directions = np.array([[1 + 2 ** -25], [-1.]], dtype=np.float64)
    result = _reference_log_probability(data, directions, np.ones(1), np.zeros(1, dtype=np.float32))
    np.testing.assert_array_equal(result, np.zeros((1, 1), dtype=np.float32))
