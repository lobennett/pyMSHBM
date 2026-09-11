#!/usr/bin/env python3
"""Retrieve only the declared MSC scans, temporal masks, and midthickness meshes.

Requires git and git-annex. Public OpenNeuro S3 access requires no credentials.
git-annex verifies the downloaded files against their content-addressed keys.
"""

import argparse
from pathlib import Path
import subprocess

REVISION = '727a0a4e25ec3f7bea1a20c955ea860206b31e77'


def run(directory, subjects):
    directory = Path(directory).expanduser().resolve()

    def git(*args):
        subprocess.run(['git', '-C', str(directory), *args], check=True)

    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', '1.0.4',
                        'https://github.com/OpenNeuroDatasets/ds000224.git', str(directory)], check=True)
    revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected ds000224 v1.0.4 revision {REVISION}; found {revision}')
    git('fetch', 'origin', 'git-annex:refs/remotes/origin/git-annex', '--depth', '1')
    git('annex', 'init', 'pymshbm-validation')
    git('annex', 'enableremote', 's3-PUBLIC')
    paths = []
    for subject in subjects:
        for session in ('func01', 'func02'):
            paths.append(f'derivatives/surface_pipeline/sub-{subject}/processed_restingstate_timecourses/'
                         f'ses-{session}/cifti')
        for hemisphere in ('L', 'R'):
            paths.append(f'derivatives/surface_pipeline/sub-{subject}/fs_LR_Talairach/fsaverage_LR32k/'
                         f'{subject}.{hemisphere}.midthickness.32k_fs_LR.surf.gii')
    git('annex', 'get', '--jobs=4', *paths)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--participant-label', nargs='+', default=['MSC01', 'MSC02'],
                        choices=[f'MSC{i:02d}' for i in range(1, 11)])
    args = parser.parse_args()
    run(args.directory, args.participant_label)
