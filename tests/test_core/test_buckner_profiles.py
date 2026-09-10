"""Scientific regressions for the executable Buckner/CBIG profile workflow."""

import numpy as np
import pytest

from pymshbm.core.buckner import (
    binary_profiles, centroids_from_labels, normalize_profiles,
)


def test_global_threshold_samples_vertices_and_preserves_ties():
    # Two identical seed series; only positively correlated target vertices
    # survive the joint top-10% threshold, including all ties at 1.
    x = np.array([-2., -1., 0., 1., 2.])
    lh = np.column_stack([x, -x, x])
    rh = np.column_stack([x, -x, -x])
    p = binary_profiles(lh, rh, np.ones(3, bool), np.ones(3, bool), seed_vertices=1)
    np.testing.assert_array_equal(p, [[1, 1], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])


def test_seed_medial_wall_excluded_and_no_roi_averaging():
    rng = np.random.default_rng(9)
    lh, rh = rng.normal(size=(30, 8)), rng.normal(size=(30, 8))
    cortex = np.array([True, False, True, True, True, True, True, True])
    p = binary_profiles(lh, rh, cortex, cortex, seed_vertices=3)
    assert p.shape == (16, 4)
    assert set(np.unique(p)) <= {0, 1}


def test_partial_nan_medial_column_is_not_imputed_into_threshold():
    rng = np.random.default_rng(5)
    lh, rh = rng.normal(size=(10, 10)), rng.normal(size=(10, 10))
    cortex = np.ones(10, bool)
    cortex[-1] = False
    lh[0, -1] = np.nan
    actual = binary_profiles(lh, rh, cortex, cortex, seed_vertices=5)
    lh[:, -1] = 0  # CBIG_corr yields NaN for this whole column, then replaces it with 0.
    expected = binary_profiles(lh, rh, cortex, cortex, seed_vertices=5)
    np.testing.assert_array_equal(actual, expected)


def test_mean_center_and_literal_cbig_normalization():
    p = np.array([[1., 1., 0., 0.], [0., 1., 2., 1.], [1., 0., 0., 1.]])
    result = normalize_profiles(p, np.array([True, True, False]))
    np.testing.assert_allclose(result[0], [.5, .5, -.5, -.5])
    # CBIG only normalizes rows for which ALL centered entries are nonzero.
    np.testing.assert_array_equal(result[1], [-1, 0, 1, 0])
    np.testing.assert_array_equal(result[2], 0)


def test_centroids_use_template_labels_without_relabeling():
    p = np.array([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    labels = np.array([2, 2, 1, 0])
    mu = centroids_from_labels(p, labels, num_clusters=2)
    expected = np.array([[-1, 3, -1, -1], [3, -1, -1, -1]]).T / np.sqrt(12)
    np.testing.assert_allclose(mu, expected)


def test_empty_network_fails_instead_of_nan_centroid():
    with pytest.raises(ValueError, match='network 2'):
        centroids_from_labels(np.eye(3), np.ones(3, int), num_clusters=2)


@pytest.mark.parametrize('problem', ['nonfinite', 'timepoints', 'constant'])
def test_bad_time_series_rejected(problem):
    rng = np.random.default_rng(0)
    lh, rh = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
    if problem == 'nonfinite':
        lh[0, 0] = np.nan
    elif problem == 'timepoints':
        rh = rh[:-1]
    else:
        lh[:, 0] = 0
    with pytest.raises(ValueError):
        binary_profiles(lh, rh, np.ones(4, bool), np.ones(4, bool), seed_vertices=2)
