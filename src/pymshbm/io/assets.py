"""User-managed, checksum-verified scientific reference assets."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import nibabel.freesurfer as fs
import numpy as np
from scipy.io import loadmat

BUCKNER_REVISION = '988ec48cf5453c6009a6f82dbddf5d5b99eda3cd'
CBIG_REVISION = '35b5664bec8822e2f77da5e090e96f91d0095be6'
REFERENCE_HASHES = {
    'MSHBM_prior_15.mat': '16326e1367349a04ed7d81e16c855ad4fe0187af815d973e64d8d5d8ca47c4c3',
    'ColorMap_15.txt': 'd6bdfdd7860a9f08e7d78f0b77996d80a31d59b1640814008164359fdef17d1e',
}
N_VERTICES = 40962


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceAssets:
    labels: np.ndarray
    lh_cortex: np.ndarray
    rh_cortex: np.ndarray
    colors: tuple
    provenance: dict


def _cortex(path):
    indices = fs.read_label(str(path))
    if (indices.size == 0 or np.any(indices < 0) or np.any(indices >= N_VERTICES)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError(f'Invalid fsaverage6 cortex indices: {path}')
    mask = np.zeros(N_VERTICES, dtype=bool)
    mask[indices] = True
    return mask


def load_assets(directory):
    """Read prepared assets; verify pinned prior, masks and vertex counts."""
    directory = Path(directory)
    names = (*REFERENCE_HASHES, 'lh.cortex.label', 'rh.cortex.label')
    if not (directory / 'assets.json').is_file() or any(
            not (directory / name).is_file() for name in names):
        raise FileNotFoundError('Reference assets missing. Run pymshbm-fetch-assets '
                                'DEST --freesurfer-dir SUBJECTS_DIR first.')
    manifest = json.loads((directory / 'assets.json').read_text())
    for name in names:
        digest = sha256_file(directory / name)
        recorded = manifest.get('files', {}).get(name, {}).get('sha256')
        if digest != recorded or (name in REFERENCE_HASHES and digest != REFERENCE_HASHES[name]):
            raise ValueError(f'Reference asset checksum mismatch: {name}')
    prior = loadmat(directory / 'MSHBM_prior_15.mat')
    labels = []
    for hemi in ('lh', 'rh'):
        lab = np.asarray(prior[f'{hemi}_labels_fs6']).ravel()
        if lab.shape != (N_VERTICES,) or not np.array_equal(np.unique(lab), np.arange(16)):
            raise ValueError('Expected fsaverage6 DU15NET labels 0 through 15')
        labels.append(lab.astype(np.int32))
    lines = (directory / 'ColorMap_15.txt').read_text().splitlines()
    colors = tuple((lines[i], *map(int, lines[i + 1].split())) for i in range(0, 30, 2))
    return ReferenceAssets(np.concatenate(labels), _cortex(directory / 'lh.cortex.label'),
                           _cortex(directory / 'rh.cortex.label'), colors, manifest)


def prepare_assets(output_dir, freesurfer_dir, *, reference_dir=None):
    """Download the pinned Buckner assets and copy local fsaverage6 masks.

    FreeSurfer subjects can come from fMRIPrep's sourcedata/freesurfer.
    Third-party assets remain external to the Python distribution.
    """
    output_dir = Path(output_dir).resolve()
    subjects = Path(freesurfer_dir).resolve()
    sources = {f'{h}.cortex.label': subjects / 'fsaverage6' / 'label' / f'{h}.cortex.label'
               for h in ('lh', 'rh')}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f'FreeSurfer fsaverage6 cortex.label missing: {path}')
        _cortex(path)
    if output_dir.exists():
        raise FileExistsError(f'Asset destination already exists: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pymshbm-assets-', dir=output_dir.parent) as temp:
        staging = Path(temp)
        manifest = {'buckner_revision': BUCKNER_REVISION, 'cbig_revision': CBIG_REVISION,
                    'mesh': 'fsaverage6', 'seed_mesh': 'fsaverage3', 'files': {}}
        for name, expected in REFERENCE_HASHES.items():
            url = f'https://raw.githubusercontent.com/bucknerlab/PrecisionNetworkMapping/{BUCKNER_REVISION}/MSHBM/{name}'
            dest = staging / name
            if reference_dir is None:
                with urlopen(url, timeout=60) as response, dest.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
            else:
                shutil.copyfile(Path(reference_dir) / 'MSHBM' / name, dest)
            if sha256_file(dest) != expected:
                raise ValueError(f'Pinned reference checksum mismatch: {name}')
            manifest['files'][name] = {'source': url, 'sha256': expected}
        for name, source in sources.items():
            shutil.copyfile(source, staging / name)
            manifest['files'][name] = {'source': str(source), 'sha256': sha256_file(source)}
        (staging / 'assets.json').write_text(json.dumps(manifest, indent=2) + '\n')
        load_assets(staging)
        staging.rename(output_dir)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare pinned DU15NET and fsaverage6 cortex assets')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--freesurfer-dir', type=Path, required=True, help='FreeSurfer SUBJECTS_DIR (contains fsaverage6)')
    parser.add_argument('--reference-dir', type=Path, help='Local PrecisionNetworkMapping checkout; otherwise download pinned files')
    args = parser.parse_args(argv)
    try:
        print(prepare_assets(args.output_dir, args.freesurfer_dir, reference_dir=args.reference_dir))
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
