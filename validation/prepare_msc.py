#!/usr/bin/env python3
"""Prepare the predeclared ds000224 example; no model fitting or denoising.

Requires the pinned git-annex dataset with the selected files retrieved,
CBIG's standard meshes, Workbench, and installed pyMSHBM. Large files remain
outside version control. See docs/openneuro-validation-plan.md.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from pymshbm.io.assets import load_assets, sha256_file

DATASET_COMMIT = '727a0a4e25ec3f7bea1a20c955ea860206b31e77'
CBIG_COMMIT = '35b5664bec8822e2f77da5e090e96f91d0095be6'
SUBJECTS = ('MSC01', 'MSC02')
SESSIONS = ('func01', 'func02')


class CoverageError(ValueError):
    def __init__(self, message, missing, distance, sources):
        super().__init__(message)
        self.missing, self.distance, self.sources = missing, distance, sources


def censor(data, mask):
    """Handle full-length and already shortened CIFTIs without double censoring."""
    mask = np.asarray(mask)
    if mask.ndim != 1 or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError('Temporal mask must be a finite one-dimensional binary vector')
    keep = mask.astype(bool)
    if keep.sum() < 3:
        raise ValueError('Fewer than three retained timepoints')
    if len(data) == len(mask):
        return data[keep], 'applied_supplied_mask'
    if len(data) == int(keep.sum()):
        return data, 'already_censored_by_source'
    raise ValueError('CIFTI time-axis length matches neither full nor retained temporal mask')


def git_head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def fill_boundary(series, valid, cortex, coords, faces, maximum_mm, refine_distance=None,
                  *, allow_zero_cortex=False):
    """Copy nearest measured target-cortex series along surface edges, bounded in mm.

    Edge-path distances upper-bound continuous surface geodesics. This explicit
    imputation does not alter any vertex with data. Zero values are not treated
    as missing: only the independent resampling coverage ROI defines missingness.
    """
    edges = np.unique(np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                              axis=1), axis=0)
    lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    graph = coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(len(coords), len(coords))).tocsr()
    graph = graph + graph.T
    distance, _, sources = dijkstra(graph, directed=False, indices=np.flatnonzero(valid & cortex),
                                    min_only=True, return_predecessors=True)
    missing = cortex & ~valid
    # Edge paths are conservative. Workbench can check paths across triangle
    # pairs for the few vertices not supported by this upper bound, at the
    # same fixed maximum distance. It returns -1 for points outside the limit.
    if refine_distance is not None:
        for vertex in np.flatnonzero(missing & (distance > maximum_mm)):
            refined = np.asarray(refine_distance(int(vertex)))
            supported = valid & cortex & np.isfinite(refined) & (refined >= 0) & (refined <= maximum_mm)
            if supported.any():
                candidates = np.flatnonzero(supported)
                source = candidates[np.argmin(refined[candidates])]
                sources[vertex], distance[vertex] = source, refined[source]
    unresolved = missing & (~np.isfinite(distance) | (distance > maximum_mm))
    if unresolved.any() and not allow_zero_cortex:
        raise CoverageError(f'Missing cortex extends beyond fixed {maximum_mm:g} mm imputation limit; '
                            f'maximum remaining distance bound is {distance[missing].max():g} mm',
                            missing, distance, sources)
    filled = series.copy()
    imputed = missing & ~unresolved
    filled[imputed] = series[sources[imputed]]
    filled[unresolved] = 0
    sources[unresolved] = -1
    return filled, distance, sources


def validate_coverage(series, cortex, unresolved, *, allow_zero_cortex=False, seed_vertices=642):
    """Only declared unresolved nonseed cortex may contain an all-zero series."""
    if unresolved.any() and not allow_zero_cortex:
        raise ValueError('Incomplete cortical coverage; explicit zero mode required')
    if unresolved[:seed_vertices].any():
        raise ValueError('Unresolved cortical seed vertex; no prepared dataset published')
    usable = cortex & ~unresolved
    if (not np.isfinite(series[cortex]).all() or np.any(np.ptp(series[usable], axis=1) == 0)
            or np.any(series[unresolved] != 0)):
        raise ValueError('Unusable cortical coverage; no prepared dataset published')


def metric(path, data, hemi):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    image = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(data[:, i], intent='NIFTI_INTENT_TIME_SERIES')
                 for i in range(data.shape[1])],
        meta=nib.gifti.GiftiMetaData({'AnatomicalStructurePrimary':
                                    'CortexLeft' if hemi == 'L' else 'CortexRight'}))
    nib.save(image, path)


def run(args):
    dataset, cbig, work, output = [Path(p).expanduser().resolve() for p in (
        args.dataset, args.cbig, args.workdir, args.output)]
    if git_head(dataset) != DATASET_COMMIT or git_head(cbig) != CBIG_COMMIT:
        raise ValueError('Dataset or CBIG checkout differs from the declared pinned revision')
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    # Never publish remnants of a failed preparation or a different cohort.
    # Failure evidence remains in its original directory; retries use a new one.
    work.mkdir(parents=True, exist_ok=False)
    allow_zero_cortex = getattr(args, 'allow_zero_cortex', False)
    script_hash = sha256_file(Path(__file__))
    assets = load_assets(args.assets)
    meshes = cbig / 'data/templates/surface/standard_mesh_atlases_20170508/resample_fsaverage'
    commands = []
    parent = Path(__file__).resolve().parents[2]
    podman = shutil.which('podman') or '/opt/podman/bin/podman'

    def wb(*command):
        tokens = [str(t) for t in command]
        if args.podman_container:
            tokens = [str(Path('/work') / Path(t).relative_to(parent))
                      if t.startswith(str(parent) + '/') else t for t in tokens]
            full = [podman, 'exec', '-e', 'OMP_NUM_THREADS=2', args.podman_container,
                    'wb_command', *tokens]
        else:
            full = [args.workbench, *tokens]
        print(' '.join(full), flush=True)
        commands.append(full)
        return subprocess.check_output(full, text=True, stderr=subprocess.STDOUT)

    version = wb('-version')
    inputs = {}

    def record(path, relative_root, kind):
        key = f'{kind}:{path.relative_to(relative_root)}'
        if key not in inputs:
            inputs[key] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
            if path.is_symlink():
                inputs[key]['annex_key'] = path.readlink().name

    records = []
    staged = work / 'prepared-bids'
    staged.mkdir(exist_ok=True)
    for subject in args.participant_label:
        areas = {}
        for hemi in ('L', 'R'):
            current = meshes / f'fs_LR-deformed_to-fsaverage.{hemi}.sphere.32k_fs_LR.surf.gii'
            target = meshes / f'fsaverage6_std_sphere.{hemi}.41k_fsavg_{hemi}.surf.gii'
            anatomy = dataset / (f'derivatives/surface_pipeline/sub-{subject}/fs_LR_Talairach/'
                                 f'fsaverage_LR32k/{subject}.{hemi}.midthickness.32k_fs_LR.surf.gii')
            for path in (current, target):
                record(path, cbig, 'cbig')
            record(anatomy, dataset, 'openneuro')
            target_anatomy = work / f'{subject}.{hemi}.midthickness.fsaverage6.surf.gii'
            wb('-surface-resample', anatomy, current, target, 'BARYCENTRIC', target_anatomy)
            areas[hemi] = current, target, anatomy, target_anatomy
        for session in SESSIONS:
            stem = f'sub-{subject}_ses-{session}_task-rest_bold_32k_fsLR'
            folder = dataset / (f'derivatives/surface_pipeline/sub-{subject}/'
                                f'processed_restingstate_timecourses/ses-{session}/cifti')
            source, tmask = folder / f'{stem}.dtseries.nii', folder / f'{stem}_tmask.txt'
            record(source, dataset, 'openneuro')
            record(tmask, dataset, 'openneuro')
            converted = work / f'{subject}_{session}.dtseries.nii'
            wb('-file-convert', '-cifti-version-convert', source, '2', converted)
            image = nib.load(converted)
            raw = np.asarray(image.dataobj)
            temporal_mask = np.loadtxt(tmask)
            data, action = censor(raw, temporal_mask)
            time_axis = image.header.get_axis(0)
            record_run = {'subject': subject, 'session': session, 'original_cifti_shape': list(raw.shape),
                          'original_mask_frames': len(temporal_mask), 'retained_frames': len(data),
                          'censor_action': action, 'nominal_tr_seconds': float(time_axis.step),
                          'retained_minutes': float(len(data) * time_axis.step / 60),
                          'source': str(source.relative_to(dataset)),
                          'mask': str(tmask.relative_to(dataset)), 'hemispheres': {}}
            np.savetxt(work / f'{subject}_{session}_retained_frame_indices.txt',
                       np.flatnonzero(temporal_mask), fmt='%d')
            brain_axis = image.header.get_axis(1)
            for hemi, cortex in (('L', assets.lh_cortex), ('R', assets.rh_cortex)):
                structure = 'CIFTI_STRUCTURE_CORTEX_LEFT' if hemi == 'L' else 'CIFTI_STRUCTURE_CORTEX_RIGHT'
                found = [(slc, model) for name, slc, model in brain_axis.iter_structures() if name == structure]
                if len(found) != 1:
                    raise ValueError(f'Expected one cortical brain model for {structure}')
                slc, model = found[0]
                if model.nvertices[structure] != 32492:
                    raise ValueError('Source must use the expected fsLR32k mesh')
                series = np.zeros((32492, len(data)), dtype=np.float32)
                series[model.vertex] = data[:, slc].T
                roi = np.zeros(32492, dtype=np.float32)
                roi[model.vertex] = 1
                if not np.isfinite(series[model.vertex]).all():
                    raise ValueError('Nonfinite source cortical data')
                name = f'{subject}_{session}.{hemi}'
                source_metric, source_roi = work / f'{name}.func.gii', work / f'{name}.roi.shape.gii'
                metric(source_metric, series, hemi)
                metric(source_roi, roi, hemi)
                current, target, anatomy, target_anatomy = areas[hemi]
                valid_path, resampled = work / f'{name}.valid.shape.gii', work / f'{name}.fsaverage6.func.gii'
                wb('-metric-resample', source_metric, current, target, 'ADAP_BARY_AREA', resampled,
                   '-area-surfs', anatomy, target_anatomy, '-current-roi', source_roi,
                   '-valid-roi-out', valid_path)
                valid = nib.load(valid_path).agg_data() > 0
                result = np.column_stack(nib.load(resampled).agg_data())
                missing = int(np.count_nonzero(cortex & ~valid))
                initial_missing = cortex & ~valid
                unresolved = initial_missing.copy()
                imputed = np.zeros_like(cortex)
                support = {'missing_mask': initial_missing, 'original_valid_mask': valid}
                qc = {'source_cortical_vertices': int(roi.sum()), 'target_cortex_vertices': int(cortex.sum()),
                      'initial_missing_target_cortex': missing,
                      'imputed_seed_vertices': 0, 'imputed_vertices': 0}
                if missing and args.boundary_fill_mm > 0:
                    coords, faces = nib.load(target_anatomy).agg_data()
                    refined_vertices = []

                    def refine_distance(vertex):
                        refined_vertices.append(vertex)
                        path = work / f'{name}.geodesic-{vertex}.shape.gii'
                        wb('-surface-geodesic-distance', target_anatomy, vertex, path,
                           '-limit', args.boundary_fill_mm)
                        return nib.load(path).agg_data()

                    try:
                        result, distance, sources = fill_boundary(
                            result, valid, cortex, coords, faces, args.boundary_fill_mm, refine_distance,
                            allow_zero_cortex=allow_zero_cortex)
                    except CoverageError as error:
                        failed_arrays = work / f'failed-{name}-coverage.npz'
                        unresolved = error.missing & (~np.isfinite(error.distance)
                                                       | (error.distance > args.boundary_fill_mm))
                        np.savez_compressed(failed_arrays, missing_mask=error.missing,
                                            original_valid_mask=valid, distance_mm=error.distance,
                                            nearest_valid_source_vertex=error.sources, unresolved_mask=unresolved)
                        qc.update(error=str(error), failed_hemisphere=hemi,
                                  triangle_distance_refined_vertices=refined_vertices,
                                  unresolved_vertices=np.flatnonzero(unresolved).tolist(),
                                  fixed_maximum_mm=args.boundary_fill_mm,
                                  coverage_arrays_sha256=sha256_file(failed_arrays))
                        record_run['hemispheres'][hemi] = qc
                        failure = {'run': record_run, 'inputs': inputs,
                                   'preparation_script_sha256': sha256_file(Path(__file__)),
                                   'commands': commands, 'workbench': version}
                        (work / f'failed-{name}-coverage.json').write_text(json.dumps(failure, indent=2) + '\n')
                        raise
                    support.update(distance_mm=distance, nearest_valid_source_vertex=sources)
                    unresolved = initial_missing & (~np.isfinite(distance) | (distance > args.boundary_fill_mm))
                    imputed = initial_missing & ~unresolved
                    qc.update(imputed_vertices=int(imputed.sum()),
                              imputed_seed_vertices=int(imputed[:642].sum()),
                              imputation_maximum_mm=args.boundary_fill_mm,
                              triangle_distance_refined_vertices=refined_vertices,
                              observed_maximum_distance_mm=float(distance[imputed].max()) if imputed.any() else None,
                              observed_median_distance_mm=float(np.median(distance[imputed])) if imputed.any() else None)
                if allow_zero_cortex:
                    result[unresolved] = 0
                missing = int(unresolved.sum())
                constant = int(np.count_nonzero(np.ptp(result[cortex], axis=1) == 0))
                qc.update(missing_target_cortex=missing, constant_target_cortex=constant,
                          unresolved_zero_vertices=np.flatnonzero(unresolved).tolist(),
                          unresolved_seed_vertices=int(unresolved[:642].sum()))
                support.update(observed_mask=cortex & valid, imputed_mask=imputed, unresolved_zero_mask=unresolved)
                record_run['hemispheres'][hemi] = qc
                print(f'{subject} {session} {hemi}: {qc}', flush=True)
                try:
                    validate_coverage(result, cortex, unresolved, allow_zero_cortex=allow_zero_cortex)
                except ValueError:
                    (work / 'failed-coverage.json').write_text(json.dumps(record_run, indent=2) + '\n')
                    raise
                # Explicitly discard noncortical values in the target mesh, shared by both engines.
                result[~cortex] = 0
                directory = staged / f'sub-{subject}' / f'ses-{session}' / 'func'
                directory.mkdir(parents=True, exist_ok=True)
                outstem = f'sub-{subject}_ses-{session}_task-rest_hemi-{hemi}_space-fsaverage6_desc-preproc_bold'
                final = directory / f'{outstem}.func.gii'
                metric(final, result, hemi)
                support_file = directory / f'{outstem}_coverage.npz'
                np.savez_compressed(support_file, **support)
                qc['coverage_sha256'] = sha256_file(support_file)
                qc['coverage_file'] = str(support_file.relative_to(staged))
                metadata = {'Sources': [f'bids:msc:{source.relative_to(dataset)}'],
                            'SpatialReference': 'fsaverage6', 'SkullStripped': True,
                            'Description': 'Author-denoised MSC cortical data; supplied temporal mask applied '
                                           'only when needed; Workbench ADAP_BARY_AREA fsLR32k to fsaverage6. '
                                           'Missing target-cortex boundary series explicitly imputed from nearest '
                                           'valid target cortex on the surface within recorded distance limit. '
                                           'Any explicitly permitted unresolved cortex remains zero; inspect coverage masks.',
                            'BoundaryImputation': qc,
                            'OriginalRepetitionTime': float(time_axis.step),
                            'RetainedOriginalFrameIndices': np.flatnonzero(temporal_mask).tolist(),
                            'CensorAction': action}
                final.with_suffix('').with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
                qc['prepared_bold_sha256'] = sha256_file(final)
            records.append(record_run)
            del image, raw, data
    if sha256_file(Path(__file__)) != script_hash:
        raise ValueError('Preparation source changed during execution; no dataset published')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': 'ds000224',
                'snapshot': '1.0.4', 'dataset_git_revision': DATASET_COMMIT, 'cbig_git_revision': CBIG_COMMIT,
                'selected_subjects': args.participant_label,
                'preparation_script_sha256': script_hash, 'workbench': version,
                'allow_zero_cortex': allow_zero_cortex,
                'inputs': inputs, 'runs': records, 'commands': commands,
                'resampling': 'ADAP_BARY_AREA, individual midthickness surfaces, current cortical ROI',
                'target_geometry': 'Individual source midthickness resampled BARYCENTRIC on registered spheres',
                'boundary_imputation': {'method': 'nearest valid target cortex by surface-edge shortest path; '
                                        'Workbench triangle-crawl refinement only if edge distance exceeds limit',
                                        'maximum_mm': args.boundary_fill_mm},
                'additional_denoising_filtering_smoothing': False, 'assets': assets.provenance}
    (staged / 'preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (staged / 'dataset_description.json').write_text(json.dumps({
        'Name': 'MSC ds000224 prepared for pyMSHBM numerical validation', 'BIDSVersion': '1.11.1',
        'DatasetType': 'derivative', 'License': 'CC0',
        'SourceDatasets': [{'DOI': '10.18112/openneuro.ds000224.v1.0.4'}],
        'DatasetLinks': {'msc': 'https://openneuro.org/datasets/ds000224/versions/1.0.4'},
        'GeneratedBy': [{'Name': 'pyMSHBM MSC validation preparation',
                         'Description': 'Shared externally prepared inputs; see preparation.json'}],
    }, indent=2) + '\n')
    (staged / '.bidsignore').write_text('preparation.json\n**/*_coverage.npz\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    staged.rename(output)
    print(f'Prepared {len(records)} real-data runs: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'cbig', 'assets', 'workdir', 'output'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('--podman-container')
    parser.add_argument('--workbench', default='wb_command')
    parser.add_argument('--boundary-fill-mm', type=float, choices=(0., 5.), default=0.,
                        help='Explicit optional boundary imputation; fixed at 5 mm for this benchmark')
    parser.add_argument('--allow-zero-cortex', action='store_true',
                        help='Separate masked-coverage protocol: leave unresolved nonseed cortex zero')
    parser.add_argument('--participant-label', nargs='+', choices=[f'MSC{i:02d}' for i in range(1, 11)],
                        default=list(SUBJECTS))
    run(parser.parse_args())
