from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pymshbm.io.bids import SurfaceRun
from pymshbm.pipeline.buckner import read_surface_bold, run_buckner_workflow, validate_fit
from pymshbm.types import MSHBMParams


def test_read_gifti_timepoint_arrays_preserves_vertex_order(tmp_path):
    data = np.arange(15, dtype=np.float32).reshape(3, 5)
    image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(x, intent='NIFTI_INTENT_TIME_SERIES') for x in data])
    path = tmp_path / 'bold.func.gii'
    nib.save(image, path)
    np.testing.assert_array_equal(read_surface_bold(path, expected_vertices=5), data)


def test_installed_source_hashes_are_portable_and_cover_fit_and_profiles():
    import hashlib
    import pymshbm
    from pymshbm.pipeline.buckner import python_source_hashes

    hashes = python_source_hashes()
    root = Path(pymshbm.__file__).resolve().parent
    for name in ("core/buckner.py", "core/group_priors.py", "pipeline/training.py", "pipeline/buckner.py"):
        assert hashes[name] == hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert all(not Path(name).is_absolute() for name in hashes)


def test_read_single_matrix_gifti_uses_vertices_by_time(tmp_path):
    data = np.arange(15, dtype=np.float32).reshape(5, 3)
    path = tmp_path / 'bold.func.gii'
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data, intent='NIFTI_INTENT_TIME_SERIES')]), path)
    np.testing.assert_array_equal(read_surface_bold(path, expected_vertices=5), data.T)


def test_wrong_mesh_fails_clearly(tmp_path):
    path = tmp_path / 'bold.func.gii'
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.zeros(5, np.float32))]), path)
    with pytest.raises(ValueError, match='40962'):
        read_surface_bold(path)


def test_empty_run_list_and_existing_output_fail_before_fitting(tmp_path):
    with pytest.raises(ValueError, match='No runs'):
        run_buckner_workflow([], tmp_path / 'out', tmp_path)
    run = SurfaceRun('01', None, 'rest', None, Path('lh'), Path('rh'))
    (tmp_path / 'out').mkdir()
    with pytest.raises(FileExistsError):
        run_buckner_workflow([run], tmp_path / 'out', tmp_path)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf])
def test_invalid_concentration_cannot_be_published(bad):
    params = MSHBMParams(mu=np.ones((3, 2)), epsil=np.ones(2), sigma=np.array([1., bad]),
                         theta=np.ones((4, 2)) / 2, kappa=np.ones(2),
                         s_lambda=np.ones((4, 2, 1)) / 2)
    with pytest.raises(ValueError, match='concentration'):
        validate_fit(params)


@pytest.fixture
def coverage_inputs(tmp_path, monkeypatch):
    """Small surface workflow with real profile/I/O code and a stubbed optimizer."""
    from types import SimpleNamespace
    import pymshbm.pipeline.buckner as pipeline
    rng = np.random.default_rng(721)
    size = 650  # Full first-642 seed range plus several nonseed targets.
    cortex = np.ones(size, bool)
    cortex[648] = False
    labels = np.arange(2 * size) % 15 + 1
    labels[[648, size + 648]] = 0
    assets = SimpleNamespace(lh_cortex=cortex, rh_cortex=cortex,
                             labels=labels, colors=(), provenance={})
    monkeypatch.setattr(pipeline, 'N_VERTICES', size)
    monkeypatch.setattr(pipeline, 'load_assets', lambda path: assets)
    series = {}
    runs = []
    for subject, run_id in (('01', '1'), ('01', '2'), ('02', '1')):
        paths = []
        for hemi in ('L', 'R'):
            path = tmp_path / f'{subject}-{run_id}-{hemi}.gii'
            path.write_text('Input hash placeholder; arrays provided by the reader fixture.')
            values = rng.normal(size=(40, size))
            values[:, 648] = 0
            series[path] = values
            paths.append(path)
        runs.append(SurfaceRun(subject, None, 'rest', run_id, *paths))
    monkeypatch.setattr(pipeline, 'read_surface_bold', lambda path: series[Path(path)])
    captured = {}
    def fit(data, g_mu, clusters, **kwargs):
        captured['data'] = np.asarray(data).copy()
        # Only optimizer work is stubbed; verify output label masking separately.
        usable = np.any(np.nan_to_num(data) != 0, axis=(1, 3))
        posterior = np.zeros((data.shape[0], clusters, data.shape[2]), np.float32)
        posterior[:, 0, :] = usable
        return MSHBMParams(mu=g_mu, epsil=np.ones(clusters), sigma=np.ones(clusters),
                           kappa=np.ones(clusters), theta=posterior.mean(axis=2),
                           s_lambda=posterior, iter_inter=1, record=[1.])
    monkeypatch.setattr(pipeline, 'params_training', fit)
    return runs, series, captured


def _insert_cortical_zeros(runs, series):
    for run in runs[:2]:
        series[run.lh][:, 649] = 0  # Never usable for subject 01.
    series[runs[0].lh][:, 643] = 0  # Usable in one of two sessions.
    series[runs[1].rh][:, 647] = 0


@pytest.mark.parametrize('enabled', [False, True])
def test_workflow_records_zero_policy_and_subject_coverage(tmp_path, coverage_inputs, enabled):
    import json
    runs, series, captured = coverage_inputs
    if enabled:
        _insert_cortical_zeros(runs, series)
    output = run_buckner_workflow(runs, tmp_path / 'output', tmp_path / 'assets',
                                  max_iter=1, allow_zero_cortex=enabled)
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['settings']['allow_zero_cortex'] is enabled
    assert provenance['runs'][0]['lh']['zero_cortex_indices'] == ([643, 649] if enabled else [])
    assert provenance['runs'][0]['lh']['zero_cortex_count'] == (2 if enabled else 0)
    assert provenance['coverage']['index_base'] == 0
    for subject, total in (('01', 2), ('02', 1)):
        coverage_path = output / provenance['coverage']['subject_files'][subject]
        with np.load(coverage_path) as coverage:
            counts = coverage['usable_session_count']
            assert counts.dtype == np.uint32
            np.testing.assert_array_equal(coverage['usable_in_any_session'], counts > 0)
            np.testing.assert_array_equal(coverage['full_session_coverage'], (counts == total) & coverage['cortex_mask'])
            assert counts.shape == (1300,)
            assert counts[648] == counts[1298] == 0
            assert not coverage['cortex_mask'][648]
            assert counts[0] == total
            if enabled and subject == '01':
                assert counts[649] == 0
                assert counts[643] == counts[650 + 647] == 1
    # Missing subject sessions remain NaN. Vertex zeros remain literal zeros.
    assert np.isnan(captured['data'][:, :, 1, 1]).all()
    if enabled:
        np.testing.assert_array_equal(captured['data'][649, :, 0, :], 0)
        labels = nib.load(output / 'sub-01/func/sub-01_space-fsaverage6_atlas-DU15NET_hemi-L_dseg.label.gii').darrays[0].data
        assert labels[649] == 0
        assert labels[643] != 0


def test_workflow_strict_default_rejects_zero_cortex_before_fitting(tmp_path, coverage_inputs):
    runs, series, captured = coverage_inputs
    _insert_cortical_zeros(runs, series)
    with pytest.raises(ValueError, match='(?i)constant cortical'):
        run_buckner_workflow(runs, tmp_path / 'output', tmp_path / 'assets')
    assert not captured
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('problem', ['seed', 'constant', 'nonfinite'])
def test_workflow_zero_opt_in_rejects_unusable_inputs_before_fitting(tmp_path, coverage_inputs, problem):
    runs, series, captured = coverage_inputs
    if problem == 'seed':
        series[runs[0].lh][:, 0] = 0
    elif problem == 'constant':
        series[runs[0].lh][:, 649] = .1
    else:
        series[runs[0].lh][0, 649] = np.nan
    with pytest.raises(ValueError):
        run_buckner_workflow(runs, tmp_path / 'output', tmp_path / 'assets', allow_zero_cortex=True)
    assert not captured
    assert not (tmp_path / 'output').exists()
