"""Exact high-dimensional float32 normalization against executed CBIG source."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pymshbm.core.buckner import normalize_profiles

ROOT = Path(__file__).resolve().parents[1] / 'validation'


@pytest.fixture(scope='module')
def fixture():
    with np.load(ROOT / 'fixtures/cbig_normalization_high_dim.npz') as arrays:
        return dict(arrays)


@pytest.mark.parametrize('order', ['C', 'F'])
def test_full_seed_dimension_matches_upstream_exactly(fixture, order):
    profiles = np.array(fixture['profiles'], order=order)
    assert profiles.shape == (16, 1175)
    actual = normalize_profiles(profiles, fixture['cortex'])
    np.testing.assert_array_equal(actual, fixture['normalized'])
    assert actual.dtype == np.float32
    # Zero input and an excluded cortex row both remain zero.
    np.testing.assert_array_equal(actual[:2], 0)
    # A centered zero prevents scaling, exactly as CBIG's all() predicate does.
    np.testing.assert_array_equal(actual[2], np.arange(-587, 588))


@pytest.mark.parametrize('row', [0, 1, 2, 3, 7, 15])
def test_single_row_matches_separately_executed_upstream(fixture, row):
    actual = normalize_profiles(fixture['profiles'][row:row + 1], fixture['cortex'][row:row + 1])
    np.testing.assert_array_equal(actual, fixture['single_normalized'][row:row + 1])


def test_high_dimension_oracle_provenance_is_pinned():
    provenance = json.loads((ROOT / 'fixtures/normalization_high_dim_provenance.json').read_text())
    assert provenance['runtime']['version'] == '6.4.0'
    for name, expected in provenance['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
