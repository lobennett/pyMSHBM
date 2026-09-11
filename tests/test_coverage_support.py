"""Coverage subsets must preserve mismatches and reject changed artifacts."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('coverage_support', Path(__file__).resolve().parents[1] / 'validation/compare_coverage_support.py')
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)


def test_support_statistics_separates_empty_and_unassigned():
    actual = np.array([1, 2, 0, 0])
    expected = np.array([1, 1, 0, 2])
    stats = coverage.support_statistics(actual, expected, np.ones(4, bool))
    dice = stats.pop('per_network_dice')
    assert stats == dict(vertices=4, matches=2, mismatches=2, agreement=.5,
                        python_unassigned=2, reference_unassigned=1, both_unassigned=1)
    assert dice == {str(k): (2 / 3 if k == 1 else 0.0 if k == 2 else None)
                    for k in range(1, 16)}
    empty = coverage.support_statistics(actual, expected, np.zeros(4, bool))
    assert empty['vertices'] == 0
    assert empty['agreement'] is None
    assert empty['per_network_dice'] == {str(k): None for k in range(1, 16)}


def test_network_dice_uses_only_selected_support_and_native_ids():
    actual = np.array([1, 1, 2, 2, 3, 15])
    expected = np.array([1, 2, 2, 0, 3, 0])
    mask = np.array([True, True, True, True, False, True])
    dice = coverage.support_statistics(actual, expected, mask)['per_network_dice']
    assert dice['1'] == pytest.approx(2 / 3)
    assert dice['2'] == .5
    assert dice['15'] == 0.0  # Present in only one engine.
    assert all(dice[str(k)] is None for k in range(3, 15))
    assert set(dice) == {str(k) for k in range(1, 16)}


def test_subject_masks_distinguish_original_observation_from_imputation():
    cortex = np.array([True, True, True, True, False])
    actual = np.array([1, 2, 0, 1, 0])
    expected = np.array([1, 1, 0, 1, 0])
    runs = [
        dict(observed=np.array([1, 1, 0, 0, 0], bool), imputed=np.array([0, 0, 0, 1, 0], bool), unresolved=np.array([0, 0, 1, 0, 0], bool)),
        dict(observed=np.array([1, 0, 0, 1, 0], bool), imputed=np.array([0, 1, 0, 0, 0], bool), unresolved=np.array([0, 0, 1, 0, 0], bool)),
    ]
    report, masks = coverage.summarize_subject_support(actual, expected, cortex, runs)
    assert report['complete_cortex']['vertices'] == 4
    assert report['complete_cortex']['mismatches'] == 1
    assert report['usable_all_sessions']['vertices'] == 3
    assert report['usable_all_sessions']['mismatches'] == 1
    assert report['originally_observed_all_sessions']['vertices'] == 1
    assert report['originally_observed_all_sessions']['agreement'] == 1
    assert report['unresolved_all_sessions']['both_unassigned'] == 1
    np.testing.assert_array_equal(masks['originally_observed_all_sessions'], [1, 0, 0, 0, 0])


def test_hash_check_rejects_mutated_sidecar(tmp_path):
    import hashlib
    path = tmp_path / 'coverage.npz'
    path.write_bytes(b'bound coverage')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert coverage.require_hash(path, digest) == digest
    path.write_bytes(b'changed coverage')
    with pytest.raises(ValueError, match='hash'):
        coverage.require_hash(path, digest)


@pytest.fixture
def bound_case(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import nibabel as nib
    from scipy.io import savemat
    from pymshbm.pipeline.buckner import python_source_hashes
    prepared, work, production = (tmp_path / name for name in ('input', 'reference', 'python'))
    for path in (prepared, work, production):
        path.mkdir()
    half, subjects = 646, ['01', '02']
    hemi_mask = np.zeros(half, bool)
    hemi_mask[[0, 1, 2, 643, 644, 645]] = True
    cortex = np.r_[hemi_mask, hemi_mask]
    assets = SimpleNamespace(lh_cortex=hemi_mask, rh_cortex=hemi_mask, provenance={})
    monkeypatch.setattr(coverage, 'load_assets', lambda path: assets)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    prep_runs, runs, coverage_paths, bold_paths = [], [], [], []
    for subject in subjects:
        for session in ('func01', 'func02'):
            prep_record = dict(subject=subject, session=session, hemispheres={})
            record = dict(stem=f'run-{len(runs)+1:03d}', subject=subject, session=session,
                          run=None, task='rest', model_session=int(session[-1]))
            for hemi, name in (('L', 'lh'), ('R', 'rh')):
                folder = prepared / f'sub-{subject}' / f'ses-{session}' / 'func'
                folder.mkdir(parents=True, exist_ok=True)
                stem = f'sub-{subject}_ses-{session}_task-rest_hemi-{hemi}_space-fsaverage6_desc-preproc_bold'
                bold = folder / f'{stem}.func.gii'
                bold.write_bytes(f'bound prepared BOLD {subject} {session} {hemi}'.encode())
                observed = hemi_mask.copy()
                imputed, unresolved = np.zeros(half, bool), np.zeros(half, bool)
                if subject == '02' and hemi == 'L':
                    unresolved[645] = True
                    observed[645] = False
                    if session == 'func01':
                        observed[644] = False
                        imputed[644] = True
                sidecar = folder / f'{stem}_coverage.npz'
                np.savez(sidecar, observed_mask=observed, imputed_mask=imputed, unresolved_zero_mask=unresolved)
                qc = dict(coverage_file=str(sidecar.relative_to(prepared)), coverage_sha256=coverage.sha256_file(sidecar),
                          unresolved_zero_vertices=np.flatnonzero(unresolved).tolist())
                write(folder / f'{stem}.json', {'BoundaryImputation': qc})
                qc['prepared_bold_sha256'] = coverage.sha256_file(bold)
                prep_record['hemispheres'][hemi] = qc
                record[name] = dict(path=str(bold), sha256=coverage.sha256_file(bold),
                                    zero_cortex_count=int(unresolved.sum()), zero_cortex_indices=np.flatnonzero(unresolved).tolist())
                coverage_paths.append(sidecar)
                bold_paths.append(bold)
            runs.append(record)
            prep_runs.append(prep_record)
    write(prepared / 'preparation.json', dict(selected_subjects=subjects, allow_zero_cortex=True, assets={}, runs=prep_runs))
    (work / 'config.mat').write_bytes(b'configuration')
    manifest = dict(subjects=subjects, runs=runs, shape_N_D_S_T=[1292, 5, 2, 2], clusters=2,
                    bids_input=str(prepared), assets={}, assets_dir=str(tmp_path / 'assets'), max_iter=5,
                    allow_zero_cortex=True, python_source_sha256=python_source_hashes(),
                    config_sha256=coverage.sha256_file(work / 'config.mat'), source_runtime_sha256={},
                    tolerances={'s_lambda': [1e-4, 1e-4]})
    write(work / 'manifest.json', manifest)
    reference_labels = np.tile(cortex[:, None].astype(int), (1, 2))
    reference_labels[645, 1] = 0
    actual = reference_labels.copy()
    actual[644, 1] = 2  # Mismatch only outside originally-observed-all-sessions support.
    def posterior(labels):
        return np.stack([labels == 1, labels == 2], axis=1).astype(np.float32)
    savemat(work / 'reference-params.mat', {'Params': {'s_lambda': posterior(reference_labels)}})
    params_path = production / 'model/priors/Params_Final.mat'
    params_path.parent.mkdir(parents=True)
    savemat(params_path, {'Params': {'s_lambda': posterior(actual)}})
    files = {}
    for index, subject in enumerate(subjects):
        for hemi, slc in (('L', slice(0, half)), ('R', slice(half, None))):
            path = production / f'sub-{subject}/func/sub-{subject}_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii'
            path.parent.mkdir(parents=True, exist_ok=True)
            image = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(actual[slc, index].astype(np.int32), intent='NIFTI_INTENT_LABEL')],
                                         meta=nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary': 'CortexLeft' if hemi == 'L' else 'CortexRight'}))
            nib.save(image, path)
            files[f'sub-{subject}_hemi-{hemi}'] = dict(path=str(path), sha256=coverage.sha256_file(path))
    provenance = dict(subjects_in_model_order=subjects, runs=runs, assets={},
                      python_source_sha256=manifest['python_source_sha256'], settings=dict(max_iter=5, allow_zero_cortex=True))
    write(production / 'provenance.json', provenance)
    profile_runs = []
    for record in runs:
        path = work / f"{record['stem']}-reference.mat"
        path.write_bytes(b'bound profile')
        profile_runs.append(dict(stem=record['stem'], reference_mat_sha256=coverage.sha256_file(path)))
    (work / 'reference-centroids.mat').write_bytes(b'centroids')
    write(work / 'profile-comparison.json', dict(passed=True, manifest_sha256=coverage.sha256_file(work / 'manifest.json'),
          python_source_sha256=manifest['python_source_sha256'], runs=profile_runs,
          reference_centroids_mat_sha256=coverage.sha256_file(work / 'reference-centroids.mat')))
    comparison = dict(passed=False, manifest_sha256=coverage.sha256_file(work / 'manifest.json'),
                      python_params=dict(path=str(params_path), sha256=coverage.sha256_file(params_path)),
                      reference_params_sha256=coverage.sha256_file(work / 'reference-params.mat'),
                      python_capture_matches_production_saved_fields=True,
                      same_production_inputs_assets_and_settings=True, same_production_python_source=True,
                      published_labels={'files': files}, parameters={'s_lambda': dict(atol=1e-4, rtol=1e-4)})
    write(work / 'comparison.json', comparison)
    args = SimpleNamespace(input=prepared, workdir=work, python_output=production, assets=tmp_path / 'assets', output=tmp_path / 'coverage.json')
    return args, coverage_paths, bold_paths


def test_bound_report_preserves_overall_failure_and_reports_common_support(bound_case):
    args, _, _ = bound_case
    report = coverage.compare_coverage(args)
    assert report['overall_comparison_passed'] is False
    assert report['subjects']['02']['complete_cortex']['mismatches'] == 1
    assert report['subjects']['02']['originally_observed_all_sessions']['agreement'] == 1
    assert report['subjects']['02']['originally_observed_all_sessions']['vertices'] == 10
    assert report['subjects']['02']['unresolved_all_sessions']['both_unassigned'] == 1
    assert report['all_subjects_all_sessions_originally_observed_intersection']['vertices'] == 10
    assert report['runs'][2]['hemispheres']['L']['counts']['imputed'] == 1
    assert report['runs'][2]['hemispheres']['L']['seed_counts']['unresolved'] == 0


@pytest.mark.parametrize('corruption', ['coverage', 'prepared_bold', 'comparison_binding', 'published_label'])
def test_bound_report_rejects_changed_artifacts(bound_case, corruption):
    import json
    args, sidecars, bold = bound_case
    if corruption == 'coverage':
        sidecars[-1].write_bytes(b'changed coverage')
    elif corruption == 'prepared_bold':
        bold[-1].write_bytes(b'changed BOLD')
    elif corruption == 'published_label':
        path = next(args.python_output.rglob('*.label.gii'))
        path.write_bytes(b'changed label')
    else:
        path = args.workdir / 'comparison.json'
        report = json.loads(path.read_text())
        report['manifest_sha256'] = 'different experiment'
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='hash|binding'):
        coverage.compare_coverage(args)


@pytest.mark.parametrize('problem', ['overlapping_masks', 'unresolved_seed'])
def test_semantically_invalid_coverage_is_rejected_even_with_updated_hash(bound_case, problem):
    import json
    args, sidecars, _ = bound_case
    path = sidecars[0]
    with np.load(path) as loaded:
        arrays = dict(loaded)
    if problem == 'overlapping_masks':
        arrays['imputed_mask'][0] = True
    else:
        arrays['observed_mask'][0] = False
        arrays['unresolved_zero_mask'][0] = True
    np.savez(path, **arrays)
    prep_path = args.input / 'preparation.json'
    preparation = json.loads(prep_path.read_text())
    qc = preparation['runs'][0]['hemispheres']['L']
    qc['coverage_sha256'] = coverage.sha256_file(path)
    metadata_path = path.with_name(path.name.removesuffix('_coverage.npz') + '.json')
    metadata_path.write_text(json.dumps({'BoundaryImputation': {k: v for k, v in qc.items() if k != 'prepared_bold_sha256'}}))
    prep_path.write_text(json.dumps(preparation))
    with pytest.raises(ValueError, match='partition|seed'):
        coverage.compare_coverage(args)


def test_cli_refuses_existing_report_without_changing_it(tmp_path):
    path = tmp_path / 'report.json'
    path.write_text('retained baseline')
    with pytest.raises(SystemExit):
        coverage.main(['--input', str(tmp_path), '--workdir', str(tmp_path),
                       '--python-output', str(tmp_path), '--assets', str(tmp_path), '--output', str(path)])
    assert path.read_text() == 'retained baseline'
