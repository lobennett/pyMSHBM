"""Censoring must never silently double-remove or keep rejected MSC frames."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'prepare_msc', Path(__file__).parents[1] / 'validation' / 'prepare_msc.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_and_already_censored_series_keep_the_same_observations():
    series = np.arange(24).reshape(6, 4)
    mask = np.array([0, 1, 1, 0, 1, 1])
    selected, action = module.censor(series, mask)
    assert action == 'applied_supplied_mask'
    np.testing.assert_array_equal(selected, series[[1, 2, 4, 5]])
    selected_again, action = module.censor(selected, mask)
    assert action == 'already_censored_by_source'
    np.testing.assert_array_equal(selected_again, selected)


@pytest.mark.parametrize('mask', [[0, 1, 2, 1], [0, np.nan, 1, 1], [0, 0, 1, 1]])
def test_invalid_or_too_short_masks_fail(mask):
    with pytest.raises(ValueError):
        module.censor(np.ones((4, 2)), np.array(mask))


def test_incompatible_time_axis_fails():
    with pytest.raises(ValueError, match='length'):
        module.censor(np.ones((5, 2)), np.array([1, 1, 0, 1]))


def test_boundary_fill_is_bounded_on_surface_edges_and_preserves_observations():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    series = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    valid = np.array([True, False, False, False])
    cortex = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match='beyond'):
        module.fill_boundary(series, valid, cortex, coords, faces, 1.5)
    filled, distance, sources = module.fill_boundary(series, valid, cortex, coords, faces, 2.)
    np.testing.assert_array_equal(filled, np.tile(series[0], (4, 1)))
    np.testing.assert_array_equal(distance, [0, 1, 1, 2])
    np.testing.assert_array_equal(sources, [0, 0, 0, 0])


def test_triangle_refinement_obeys_same_limit():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    series = np.arange(12).reshape(4, 3)
    valid = np.array([True, False, False, False])
    cortex = np.ones(4, dtype=bool)
    def refined(vertex):
        return np.array([np.sqrt(2), 1, 1, 0])
    filled, distances, sources = module.fill_boundary(
        series, valid, cortex, coords, faces, 1.5, refined)
    assert distances[3] == np.sqrt(2)
    assert sources[3] == 0
    np.testing.assert_array_equal(filled[3], series[0])
    with pytest.raises(ValueError, match='beyond'):
        module.fill_boundary(series, valid, cortex, coords, faces, 1.2, refined)


def test_unresolved_zero_mode_never_extrapolates_beyond_distance_limit():
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    series = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0], [99, 99, 99]], dtype=float)
    valid = np.array([True, False, False, False])
    filled, distances, sources = module.fill_boundary(
        series, valid, np.ones(4, bool), coords, faces, 1.5, allow_zero_cortex=True)
    np.testing.assert_array_equal(filled[:3], np.tile(series[0], (3, 1)))
    np.testing.assert_array_equal(filled[3], 0)
    assert distances[3] == 2
    assert sources[3] == -1


def test_zero_coverage_requires_explicit_mode_observed_seeds_and_valid_measured_data():
    series = np.array([[1, 2, 3], [1, 2, 3], [0, 0, 0]], dtype=float)
    cortex = np.ones(3, bool)
    unresolved = np.array([False, False, True])
    with pytest.raises(ValueError, match='coverage'):
        module.validate_coverage(series, cortex, unresolved, seed_vertices=1)
    module.validate_coverage(series, cortex, unresolved, allow_zero_cortex=True, seed_vertices=1)
    with pytest.raises(ValueError, match='seed'):
        module.validate_coverage(series, cortex, unresolved, allow_zero_cortex=True, seed_vertices=3)
    for value in (0., 2., np.nan):
        bad = series.copy()
        bad[1] = value
        with pytest.raises(ValueError, match='coverage'):
            module.validate_coverage(bad, cortex, unresolved, allow_zero_cortex=True, seed_vertices=1)


def test_retry_cannot_publish_stale_participant_from_failed_preparation(tmp_path, monkeypatch):
    work = tmp_path / 'failed-work'
    stale = work / 'prepared-bids/sub-MSC02/func/stale.func.gii'
    stale.parent.mkdir(parents=True)
    stale.write_text('stale hemisphere from another cohort')
    monkeypatch.setattr(module, 'git_head', lambda p: module.DATASET_COMMIT
                        if p.name == 'dataset' else module.CBIG_COMMIT)
    args = SimpleNamespace(dataset=tmp_path / 'dataset', cbig=tmp_path / 'cbig',
                           workdir=work, output=tmp_path / 'output', participant_label=['MSC01'])
    with pytest.raises(FileExistsError):
        module.run(args)
    assert not args.output.exists()
    assert stale.exists()  # Preserve the failed attempt's evidence.
