"""BIDS orchestration of the pinned Buckner group-estimation workflow."""

from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import shutil
import tempfile

import nibabel as nib
import numpy as np

from pymshbm.core.buckner import (
    binary_profiles, centroids_from_labels, normalize_profiles, validate_cortical_bold,
)
from pymshbm.io.assets import N_VERTICES, load_assets, sha256_file
from pymshbm.pipeline.training import params_training

logger = logging.getLogger(__name__)


def python_source_hashes():
    """Identify installed Python implementation independent of checkout paths."""
    package = Path(__file__).resolve().parents[1]
    return {path.relative_to(package).as_posix(): sha256_file(path)
            for path in sorted(package.rglob('*.py'))}


def read_surface_bold(path, *, expected_vertices=N_VERTICES):
    """Read fMRIPrep GIFTI into (time, vertices), with explicit orientation."""
    path = Path(path)
    image = nib.load(path)
    if not isinstance(image, nib.gifti.GiftiImage) or not image.darrays:
        raise ValueError(f'Expected a nonempty surface GIFTI: {path}')
    arrays = [np.asarray(a.data) for a in image.darrays]
    if all(a.ndim == 1 and a.size == expected_vertices for a in arrays):
        data = np.stack(arrays)
    elif len(arrays) == 1 and arrays[0].ndim == 2 and arrays[0].shape[0] == expected_vertices:
        data = arrays[0].T
    else:
        raise ValueError(f'Expected {expected_vertices} fsaverage6 vertices in {path}; '
                         f'found array shapes {[a.shape for a in arrays]}')
    if data.shape[0] < 3:
        raise ValueError(f'At least three BOLD timepoints required: {path}')
    return np.asarray(data, dtype=np.float64)


def _write_labels(path, labels, hemi, colors):
    table = nib.gifti.GiftiLabelTable()
    for name, index, red, green, blue, alpha in (('Medial wall / unassigned', 0, 0, 0, 0, 0), *colors):
        entry = nib.gifti.GiftiLabel(index, red / 255, green / 255, blue / 255, alpha / 255)
        entry.label = name
        table.labels.append(entry)
    image = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(labels.astype(np.int32), intent='NIFTI_INTENT_LABEL')],
        labeltable=table,
        meta=nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary': 'CortexLeft' if hemi == 'L' else 'CortexRight'}),
    )
    nib.save(image, path)


def validate_fit(params):
    """Reject invalid results even when literal upstream arithmetic returns them."""
    if any(not np.isfinite(x).all() or np.any(x <= 0)
           for x in (params.sigma, params.epsil, params.kappa)):
        raise ValueError('Model concentration estimates are nonpositive or nonfinite; '
                         'variability may be unidentifiable. No completed dataset was published.')
    if params.s_lambda is None or not all(np.isfinite(x).all() for x in (
            params.mu, params.theta, params.s_lambda)):
        raise ValueError('Model returned nonfinite parameters; no completed dataset was published')


def run_buckner_workflow(runs, output_dir, assets_dir, *, max_iter=5, allow_zero_cortex=False):
    """Fit the joint cohort model and atomically write one derivative dataset.

    Each paired run contributes one model session. A disk-backed single-
    precision profile tensor avoids retaining every full correlation matrix
    in RAM. Outputs are conditional on the complete selected cohort.
    allow_zero_cortex explicitly permits zero nonseed targets and records their
    coverage; they remain zero observations rather than NaN missing sessions.
    """
    runs = list(runs)
    if not runs:
        raise ValueError('No runs supplied')
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output_dir}')
    source_hashes = python_source_hashes()
    assets = load_assets(assets_dir)
    subjects = sorted({r.subject for r in runs})
    if any(not sub.isascii() or not sub.isalnum() for sub in subjects):
        raise ValueError('Subject labels must be BIDS alphanumeric labels')
    seen = set()
    counts = defaultdict(int)
    input_records = []
    usable_session_counts = {subject: np.zeros(2 * N_VERTICES, dtype=np.uint32)
                             for subject in subjects}
    for run in runs:
        pair = (Path(run.lh).resolve(), Path(run.rh).resolve())
        if any(p in seen for p in pair) or pair[0] == pair[1]:
            raise ValueError('An input hemisphere is repeated across model sessions')
        seen.update(pair)
        lh, rh = read_surface_bold(pair[0]), read_surface_bold(pair[1])
        if lh.shape[0] != rh.shape[0]:
            raise ValueError(f'Hemisphere timepoints disagree for sub-{run.subject}')
        hemisphere_records = {}
        for hemi, data, mask, path, offset in (
                ('lh', lh, assets.lh_cortex, pair[0], 0),
                ('rh', rh, assets.rh_cortex, pair[1], N_VERTICES)):
            try:
                zero = validate_cortical_bold(data, mask, allow_zero_cortex=allow_zero_cortex)
            except ValueError as exc:
                raise ValueError(f'{exc} (sub-{run.subject}, {hemi})') from exc
            indices = np.flatnonzero(zero)
            hemisphere_records[hemi] = {
                'path': str(path), 'sha256': sha256_file(path),
                'zero_cortex_count': int(indices.size),
                'zero_cortex_indices': indices.tolist(),
            }
            usable_session_counts[run.subject][offset:offset + N_VERTICES] += (mask & ~zero)
        counts[run.subject] += 1
        input_records.append({'subject': run.subject, 'session': run.session,
                              'task': run.task, 'run': run.run,
                              'entities': dict(run.entities),
                              'model_session': counts[run.subject], 'timepoints': lh.shape[0],
                              **hemisphere_records})
    del lh, rh, data
    n_seed = int(assets.lh_cortex[:642].sum() + assets.rh_cortex[:642].sum())
    shape = (2 * N_VERTICES, n_seed, len(subjects), max(counts.values()))
    bytes_needed = int(np.prod(shape)) * 4
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir.parent).free < bytes_needed + 3 * shape[0] * n_seed * 8:
        raise ValueError(f'Insufficient free disk for profile tensor ({bytes_needed / 1024**3:.1f} GiB) and outputs')
    with tempfile.TemporaryDirectory(prefix='.pymshbm-', dir=output_dir.parent) as temp:
        staging = Path(temp)
        tensor_path = staging / 'profiles.npy'
        tensor = np.lib.format.open_memmap(tensor_path, mode='w+', dtype=np.float32, shape=shape)
        tensor[:] = np.nan  # Missing sessions are NaN, never artificial zero observations.
        avg_profile = np.zeros(shape[:2], dtype=np.float64)
        cortex = np.concatenate([assets.lh_cortex, assets.rh_cortex])
        for index, (run, record) in enumerate(zip(runs, input_records, strict=True), 1):
            logger.info('Computing connectivity profile %d/%d: sub-%s', index, len(runs), run.subject)
            profile = binary_profiles(read_surface_bold(run.lh), read_surface_bold(run.rh),
                                      assets.lh_cortex, assets.rh_cortex,
                                      allow_zero_cortex=allow_zero_cortex)
            for hemi, path in (('lh', run.lh), ('rh', run.rh)):
                if sha256_file(path) != record[hemi]['sha256']:
                    raise ValueError(f'Input changed during fitting: {path}')
            avg_profile += profile
            tensor[:, :, subjects.index(run.subject), record['model_session'] - 1] = normalize_profiles(profile, cortex)
            del profile
        tensor.flush()
        avg_profile /= len(runs)
        g_mu = centroids_from_labels(avg_profile, assets.labels)
        del avg_profile, tensor
        tensor = np.load(tensor_path, mmap_mode='r')
        logger.info('Estimating 15-network model for %d subjects, at most %d outer iterations', len(subjects), max_iter)
        params = params_training(tensor, g_mu, 15, max_iter=max_iter, save_all=True,
                                 output_dir=staging / 'model', subject_ids=subjects)
        validate_fit(params)
        coverage_subject_files = {}
        for subject_index, subject in enumerate(subjects):
            directory = staging / f'sub-{subject}' / 'func'
            directory.mkdir(parents=True)
            coverage_path = directory / f'sub-{subject}_desc-mshbm_coverage.npz'
            usable_counts = usable_session_counts[subject]
            np.savez_compressed(
                coverage_path, cortex_mask=cortex, usable_session_count=usable_counts,
                usable_in_any_session=usable_counts > 0,
                full_session_coverage=cortex & (usable_counts == counts[subject]))
            coverage_subject_files[subject] = coverage_path.relative_to(staging).as_posix()
            posterior = params.s_lambda[:, :, subject_index]
            labels = np.argmax(posterior, axis=1).astype(np.int32) + 1
            labels[posterior.sum(axis=1) == 0] = 0
            for hemi, indices in (('L', slice(0, N_VERTICES)), ('R', slice(N_VERTICES, None))):
                stem = f'sub-{subject}_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg'
                _write_labels(directory / f'{stem}.label.gii', labels[indices], hemi, assets.colors)
                metadata = {'SpatialReference': 'fsaverage6', 'Atlas': 'DU15NET',
                            'Sources': [r[h]['path'] for r in input_records if r['subject'] == subject for h in ('lh', 'rh')],
                            'Description': 'Argmax of the jointly fitted group-estimation posterior; zero is unassigned.'}
                (directory / f'{stem}.json').write_text(json.dumps(metadata, indent=2) + '\n')
            np.savez_compressed(directory / f'sub-{subject}_desc-mshbm_posterior.npz',
                                posterior=posterior, labels=labels)
        software = {name: version(name) for name in ('pymshbm', 'numpy', 'scipy', 'nibabel')}
        software['python'] = platform.python_version()
        relative_change = None
        if len(params.record) > 1 and abs(params.record[-2]) > 0:
            relative_change = abs((params.record[-1] - params.record[-2]) / params.record[-2])
        converged = relative_change is not None and relative_change <= 1e-5
        if python_source_hashes() != source_hashes:
            raise ValueError('Python source changed during fitting; no completed dataset was published')
        provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                      'workflow': 'Buckner group-estimation posterior extraction',
                      'python_source_sha256': source_hashes,
                      'software': software, 'assets': assets.provenance,
                      'subjects_in_model_order': subjects, 'runs': input_records,
                      'coverage': {
                          'index_base': 0,
                          'run_indices': 'Local to each hemisphere, cortical zero time courses only',
                          'vertex_order': 'left hemisphere then right hemisphere',
                          'usable_definition': 'Finite nonconstant cortical input time course; may include preparation imputations',
                          'zero_observation_policy': 'Literal zero profiles, not NaN missing sessions',
                          'subject_files': coverage_subject_files,
                      },
                      'settings': {'num_clusters': 15, 'max_iter': max_iter, 'conv_th': 1e-5,
                                   'correlation_threshold': .1, 'seed_mesh': 'fsaverage3',
                                   'allow_zero_cortex': bool(allow_zero_cortex),
                                   'additional_denoising': False, 'additional_resampling': False,
                                   'additional_individual_mrf': False, 'profile_dtype': 'float32'},
                      'fit': {'iterations': params.iter_inter, 'objective_history': params.record,
                              'reached_iteration_limit': params.iter_inter >= max_iter,
                              'relative_cost_change': relative_change, 'converged': converged,
                              'stop_reason': 'relative_cost' if converged else 'iteration_limit'},
                      'validation_scope': 'See docs/reference-validation.md; synthetic numerical equivalence does not establish denoising or real-data equivalence.'}
        (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
        (staging / 'dataset_description.json').write_text(json.dumps({
            'Name': 'pyMSHBM DU15NET individual network maps', 'BIDSVersion': '1.11.1',
            'DatasetType': 'derivative', 'GeneratedBy': [{'Name': 'pyMSHBM', 'Version': software['pymshbm'],
                                                       'CodeURL': 'https://github.com/lobennett/pyMSHBM'}],
        }, indent=2) + '\n')
        (staging / '.bidsignore').write_text('model/\nprovenance.json\n**/*.npz\n')
        del tensor
        tensor_path.unlink()
        staging.rename(output_dir)
    return output_dir
