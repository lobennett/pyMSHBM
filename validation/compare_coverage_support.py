"""Supplement a bound real-data comparison with coverage-specific label counts.

This report does not replace the overall comparison or change its tolerances.
Inputs must include the preparation's observed/imputed/unresolved coverage masks.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat

from pymshbm.io.assets import load_assets, sha256_file

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('coverage_real_reference', ROOT / 'real_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def support_statistics(python_labels, reference_labels, mask):
    actual, expected, mask = map(np.asarray, (python_labels, reference_labels, mask))
    if actual.shape != expected.shape or mask.shape != actual.shape or mask.dtype != bool:
        raise ValueError('Label and boolean support-mask shapes must match')
    a, b = actual[mask], expected[mask]
    count = int(mask.sum())
    matches = int(np.count_nonzero(a == b))
    dice = {}
    for network in range(1, 16):
        a_network, b_network = a == network, b == network
        denominator = int(a_network.sum()) + int(b_network.sum())
        dice[str(network)] = (2 * int(np.count_nonzero(a_network & b_network)) / denominator
                              if denominator else None)
    return dict(vertices=count, matches=matches, mismatches=count - matches,
                agreement=matches / count if count else None,
                per_network_dice=dice,
                python_unassigned=int(np.count_nonzero(a == 0)),
                reference_unassigned=int(np.count_nonzero(b == 0)),
                both_unassigned=int(np.count_nonzero((a == 0) & (b == 0))))


def summarize_subject_support(python_labels, reference_labels, cortex, runs):
    if not runs:
        raise ValueError('Every subject must have at least one coverage run')
    observed = np.stack([r['observed'] for r in runs])
    usable = np.stack([r['observed'] | r['imputed'] for r in runs])
    unresolved = np.stack([r['unresolved'] for r in runs])
    masks = dict(complete_cortex=cortex, usable_any_session=cortex & usable.any(axis=0),
                 usable_all_sessions=cortex & usable.all(axis=0),
                 originally_observed_all_sessions=cortex & observed.all(axis=0),
                 unresolved_any_session=cortex & unresolved.any(axis=0),
                 unresolved_all_sessions=cortex & unresolved.all(axis=0))
    return {name: support_statistics(python_labels, reference_labels, mask)
            for name, mask in masks.items()}, masks


def require_hash(path, expected):
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return actual


def _json(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(f'Comparison binding invalid: {message}')


def _safe_child(root, relative):
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), 'coverage path is outside prepared input')
    return path


def _parameter_labels(path, shape, clusters):
    params = reference.canonical_params(loadmat(path, simplify_cells=True)['Params'], shape, clusters)
    posterior = params['s_lambda']
    _require(np.isfinite(posterior).all() and np.all(posterior >= 0), 'invalid posterior probabilities')
    labels = posterior.argmax(axis=1) + 1
    labels[np.all(posterior == 0, axis=1)] = 0
    return labels, params


def compare_coverage(args):
    """Verify artifact bindings and return supplemental support statistics."""
    prepared, work, production = (Path(p).resolve() for p in (args.input, args.workdir, args.python_output))
    manifest = reference.load_manifest(work)  # Config, source, runtime and installed-source hashes.
    comparison = _json(work / 'comparison.json')
    require_hash(work / 'manifest.json', comparison.get('manifest_sha256'))
    provenance = _json(production / 'provenance.json')
    preparation = _json(prepared / 'preparation.json')
    assets = load_assets(args.assets)
    subjects = manifest['subjects']
    cortex = np.r_[assets.lh_cortex, assets.rh_cortex].astype(bool)
    half = len(assets.lh_cortex)
    _require(len(assets.rh_cortex) == half and manifest['shape_N_D_S_T'][0] == len(cortex), 'cortical shape differs')
    _require(Path(manifest['bids_input']).resolve() == prepared, 'prepared input directory differs')
    _require(provenance['subjects_in_model_order'] == subjects and
             set(preparation['selected_subjects']) == set(subjects), 'selected subjects differ')
    _require(preparation['assets'] == manifest['assets'] == provenance['assets'] == assets.provenance, 'assets differ')
    _require(provenance.get('python_source_sha256') == manifest['python_source_sha256'], 'production source differs')
    policy = manifest.get('allow_zero_cortex', False)
    _require(provenance['settings'].get('allow_zero_cortex', False) == policy ==
             preparation.get('allow_zero_cortex', False), 'zero-cortex policy differs')
    _require(provenance['settings']['max_iter'] == manifest['max_iter'], 'iteration settings differ')
    for key in ('same_production_inputs_assets_and_settings', 'same_production_python_source',
                'python_capture_matches_production_saved_fields'):
        _require(comparison.get(key) is True, f'original comparison lacks valid {key}')
    profiles = reference.profile_evidence(work, manifest)
    _require(profiles['binding_passed'], 'profile comparison artifacts differ')
    for name, tolerance in manifest['tolerances'].items():
        result = comparison['parameters'][name]
        if 'atol' in result:
            _require([result['atol'], result['rtol']] == list(tolerance), f'{name} tolerances differ')

    reference_file = work / 'reference-params.mat'
    require_hash(reference_file, comparison.get('reference_params_sha256'))
    reported_python = Path(comparison['python_params']['path'])
    require_hash(reported_python, comparison['python_params']['sha256'])
    shape, clusters = manifest['shape_N_D_S_T'], manifest['clusters']
    expected, _ = _parameter_labels(reference_file, shape, clusters)
    reported_labels, reported_params = _parameter_labels(reported_python, shape, clusters)
    production_file = production / 'model/priors/Params_Final.mat'
    production_labels, production_params = _parameter_labels(production_file, shape, clusters)
    _require(all(key in reported_params and np.array_equal(value, reported_params[key], equal_nan=True)
                 for key, value in production_params.items()), 'saved production parameters differ from compared parameters')
    actual = np.empty_like(expected)
    bound_labels = {}
    for index, subject in enumerate(subjects):
        for hemi, slc, structure in (('L', slice(0, half), 'CortexLeft'), ('R', slice(half, None), 'CortexRight')):
            path = production / f'sub-{subject}/func/sub-{subject}_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii'
            entry = comparison['published_labels']['files'][f'sub-{subject}_hemi-{hemi}']
            _require(Path(entry['path']).resolve() == path, 'published label path differs')
            bound_labels[f'{subject}:{hemi}'] = require_hash(path, entry.get('sha256'))
            image = nib.load(path)
            _require(isinstance(image, nib.gifti.GiftiImage) and len(image.darrays) == 1, 'invalid label image')
            _require(image.meta.get('AnatomicalStructurePrimary') == structure and
                     image.darrays[0].intent == nib.nifti1.intent_codes['NIFTI_INTENT_LABEL'], 'label hemisphere or intent differs')
            labels = image.darrays[0].data
            _require(labels.shape == (half,) and np.isfinite(labels).all() and
                     np.all(labels == np.floor(labels)) and np.all((labels >= 0) & (labels <= clusters)), 'invalid label values')
            actual[slc, index] = labels
    _require(np.array_equal(actual, production_labels) and np.array_equal(actual, reported_labels), 'published labels differ from saved parameter labels')

    prep_runs = {(r['subject'], r['session']): r for r in preparation['runs']}
    _require(len(prep_runs) == len(preparation['runs']), 'duplicate preparation runs')
    _require(set(prep_runs) == {(r['subject'], r['session']) for r in manifest['runs']}, 'preparation run selection differs')
    _require(len(provenance['runs']) == len(manifest['runs']), 'production run count differs')
    per_subject = {subject: [] for subject in subjects}
    run_reports = []
    for run, python_run in zip(manifest['runs'], provenance['runs'], strict=True):
        _require(all(run.get(k) == python_run.get(k) for k in ('subject', 'session', 'run', 'task', 'model_session')), 'run identities differ')
        prepared_run = prep_runs[run['subject'], run['session']]
        report = {key: run.get(key) for key in ('stem', 'subject', 'session', 'run', 'task', 'model_session')}
        report['hemispheres'] = {}
        combined = {key: [] for key in ('observed', 'imputed', 'unresolved')}
        for hemi, name, mask in (('L', 'lh', assets.lh_cortex), ('R', 'rh', assets.rh_cortex)):
            qc = prepared_run['hemispheres'][hemi]
            path = _safe_child(prepared, qc['coverage_file'])
            coverage_hash = require_hash(path, qc['coverage_sha256'])
            bold = path.with_name(path.name.removesuffix('_coverage.npz') + '.func.gii')
            _require(Path(run[name]['path']).resolve() == bold == Path(python_run[name]['path']).resolve(), 'prepared BOLD path differs')
            bold_hash = require_hash(bold, qc['prepared_bold_sha256'])
            _require(bold_hash == run[name]['sha256'] == python_run[name]['sha256'], 'prepared BOLD hash differs between engines')
            metadata_path = bold.with_suffix('').with_suffix('.json')
            metadata = _json(metadata_path)
            _require(metadata['BoundaryImputation'] == {k: v for k, v in qc.items() if k != 'prepared_bold_sha256'}, 'coverage JSON metadata differs from preparation')
            with np.load(path) as arrays:
                masks = {short: np.asarray(arrays[key]) for short, key in (
                    ('observed', 'observed_mask'), ('imputed', 'imputed_mask'), ('unresolved', 'unresolved_zero_mask'))}
            _require(all(value.shape == mask.shape and value.dtype == bool for value in masks.values()), 'coverage masks have wrong shape or type')
            partition = sum(value.astype(np.uint8) for value in masks.values())
            _require(np.array_equal(partition, mask.astype(np.uint8)), 'coverage masks do not partition canonical cortex')
            _require(not masks['unresolved'][:642].any(), 'unresolved cortical seed support')
            unresolved_indices = np.flatnonzero(masks['unresolved']).tolist()
            _require(unresolved_indices == qc.get('unresolved_zero_vertices', []), 'preparation unresolved indices differ')
            for engine_run in (run, python_run):
                _require(engine_run[name].get('zero_cortex_indices', []) == unresolved_indices and
                         engine_run[name].get('zero_cortex_count', 0) == len(unresolved_indices), 'zero-cortex indices differ between engines and preparation')
            report['hemispheres'][hemi] = {
                'counts': {key: int(value.sum()) for key, value in masks.items()},
                'seed_counts': {key: int(value[:642].sum()) for key, value in masks.items()},
                'imputed_indices': np.flatnonzero(masks['imputed']).tolist(),
                'unresolved_zero_indices': unresolved_indices,
                'coverage_file': str(path), 'coverage_sha256': coverage_hash,
                'prepared_bold_file': str(bold), 'prepared_bold_sha256': bold_hash,
                'metadata_sha256': sha256_file(metadata_path),
            }
            for key in combined:
                combined[key].append(masks[key])
        per_subject[run['subject']].append({key: np.concatenate(values) for key, values in combined.items()})
        run_reports.append(report)

    subject_reports, originally_observed = {}, []
    for index, subject in enumerate(subjects):
        subject_reports[subject], masks = summarize_subject_support(actual[:, index], expected[:, index], cortex, per_subject[subject])
        originally_observed.append(masks['originally_observed_all_sessions'])
    intersection = np.all(np.stack(originally_observed), axis=0)
    intersection_reports = {subject: support_statistics(actual[:, i], expected[:, i], intersection)
                            for i, subject in enumerate(subjects)}
    return {
        'scope': 'Supplemental coverage-specific exact label agreement; shared prepared inputs, no label permutation',
        'bindings_verified': True,
        'overall_comparison_passed': comparison['passed'],
        'acceptance_rule': 'This supplement does not replace overall acceptance or modify numerical tolerances',
        'parameter_tolerances_unchanged': manifest['tolerances'],
        'support_definition': 'Originally observed means valid resampling support before boundary imputation; usable means observed or bounded-imputed; zeros are literal observations, not NaN sessions',
        'dice_definition': 'Native network IDs 1 through 15; twice the intersection divided by the sum of network counts within the selected support; null when absent in both engines, including empty support',
        'index_base': 0, 'hemisphere_indices': 'local to hemisphere',
        'subjects_in_model_order': subjects, 'subjects': subject_reports, 'runs': run_reports,
        'all_subjects_all_sessions_originally_observed_intersection': {
            'vertices': int(intersection.sum()), 'subjects': intersection_reports},
        'bindings': {
            'preparation_sha256': sha256_file(prepared / 'preparation.json'),
            'manifest_sha256': sha256_file(work / 'manifest.json'),
            'comparison_sha256': sha256_file(work / 'comparison.json'),
            'production_provenance_sha256': sha256_file(production / 'provenance.json'),
            'profile_comparison_sha256': sha256_file(work / 'profile-comparison.json'),
            'reference_params_sha256': sha256_file(reference_file),
            'production_params_sha256': sha256_file(production_file),
            'published_label_sha256': bound_labels,
            'script_sha256': sha256_file(Path(__file__)),
            'python_source_sha256': manifest['python_source_sha256'],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Prepared BIDS dataset with preparation.json')
    parser.add_argument('--workdir', type=Path, required=True, help='Completed real_reference comparison directory')
    parser.add_argument('--python-output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output already exists; choose a new supplement path')
    try:
        report = compare_coverage(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f'Coverage supplement: {args.output}')


if __name__ == '__main__':
    main()
