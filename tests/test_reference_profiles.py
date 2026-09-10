"""Upstream Octave comparisons for Buckner's profile/centroid array kernels."""

import hashlib
import json
from pathlib import Path

import numpy as np

from pymshbm.core.buckner import binary_profiles, centroids_from_labels, normalize_profiles

ROOT = Path(__file__).resolve().parents[1] / "validation"


def test_global_binary_profile_matches_executed_cbig():
    with np.load(ROOT / "fixtures/cbig_profiles.npz") as fixture:
        actual = binary_profiles(fixture["lh"], fixture["rh"], fixture["lh_cortex"], fixture["rh_cortex"], seed_vertices=int(fixture["seed_vertices"]))
        np.testing.assert_array_equal(actual, fixture["binary"])


def test_profile_normalization_matches_executed_cbig():
    with np.load(ROOT / "fixtures/cbig_profiles.npz") as fixture:
        actual = normalize_profiles(fixture["profiles"], fixture["profile_cortex"])
        np.testing.assert_allclose(actual, fixture["normalized"], atol=2e-7, rtol=2e-6)
        # A centered zero in a nonconstant row exercises CBIG's all() predicate.
        np.testing.assert_array_equal(actual[0], np.arange(-3, 4))


def test_label_centroids_match_executed_cbig():
    with np.load(ROOT / "fixtures/cbig_profiles.npz") as fixture:
        actual = centroids_from_labels(fixture["avg_profile"], fixture["labels"], num_clusters=3)
        np.testing.assert_allclose(actual, fixture["centroids"], atol=1e-12, rtol=1e-12)


def test_profile_oracle_source_and_fixture_checksums():
    provenance = json.loads((ROOT / "fixtures/profile-provenance.json").read_text())
    for name, expected in provenance["sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
