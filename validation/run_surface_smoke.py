"""Generate fsaverage6-size synthetic BIDS BOLD and run the installed CLI.

This exercises file I/O and the complete pipeline, not real-data validity.
Run with: uv run python validation/run_surface_smoke.py --assets-dir ASSETS --work-dir NEW_DIR
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

from pymshbm.io.assets import load_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    assets = load_assets(args.assets_dir)
    inputs = work / 'input'
    directory = inputs / 'sub-smoke' / 'func'
    directory.mkdir(parents=True)
    (inputs / 'dataset_description.json').write_text(json.dumps({
        'Name': 'Synthetic fsaverage6 smoke input', 'BIDSVersion': '1.11.1',
        'DatasetType': 'derivative', 'GeneratedBy': [{'Name': 'pyMSHBM synthetic generator'}],
    }))
    rng = np.random.default_rng(20260910)
    for run in range(1, 3):
        shared = rng.normal(size=(40, 16))
        for h, offset, mask in (('L', 0, assets.lh_cortex), ('R', 40962, assets.rh_cortex)):
            labels = assets.labels[offset:offset + 40962]
            # Keep substantial session/vertex variation: nearly noiseless
            # templates can quantize resultant lengths above one in CBIG.
            values = (shared[:, labels] + rng.normal(size=(40, 40962))).astype(np.float32)
            values[:, ~mask] = 0
            image = nib.gifti.GiftiImage(darrays=[
                nib.gifti.GiftiDataArray(row, intent='NIFTI_INTENT_TIME_SERIES') for row in values
            ])
            nib.save(image, directory / f'sub-smoke_task-rest_run-{run}_space-fsaverage6_hemi-{h}_bold.func.gii')
    command = [sys.executable, '-m', 'pymshbm.cli.bids', 'bids', str(inputs), str(work / 'output'),
               '--assets-dir', str(args.assets_dir.resolve()), '--max-iter', '1']
    started = time.monotonic()
    subprocess.run(command, check=True)
    provenance = json.loads((work / 'output' / 'provenance.json').read_text())
    for hemi in ('L', 'R'):
        path = work / 'output' / 'sub-smoke' / 'func' / f'sub-smoke_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii'
        labels = nib.load(path).darrays[0].data
        assert labels.shape == (40962,)
        assert set(np.unique(labels)) <= set(range(16))
    report = {'scope': 'full-size synthetic BIDS-to-label smoke, not a real-data equivalence test',
              'command': command, 'elapsed_seconds': time.monotonic() - started,
              'shape': [81924, 'cortical fsaverage3 seeds', 1, 2],
              'iterations': provenance['fit']['iterations'], 'status': 'passed'}
    (work / 'smoke-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
