"""Execute the unchanged CBIG normalization lines on 1,175-feature inputs.

Run --prepare-only, execute normalization_high_dim_oracle in Octave with
validation/runtime on its path, then run --collect-existing. Existing oracle
fixtures and adapters are not modified by this independent regression.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'upstream/CBIG_MSHBM_estimate_group_priors.m'
ADAPTER = ROOT / 'runtime/normalization_high_dim_oracle.m'
INPUT = ROOT / 'fixtures/normalization_high_dim_input.mat'
OUTPUT = ROOT / 'fixtures/normalization_high_dim_output.mat'
FIXTURE = ROOT / 'fixtures/cbig_normalization_high_dim.npz'
LOG = ROOT / 'fixtures/normalization_high_dim_octave.log'
SEED = 938411


def prepare():
    source = SOURCE.read_text()
    start = '                series = bsxfun(@minus,series,mean(series, 2));'
    body = start.strip() + source.rsplit(start, 1)[1].split('                data.series', 1)[0]
    ADAPTER.write_text('''function normalization_high_dim_oracle(root)
% Infrastructure wrapper; normalization body copied verbatim from pinned CBIG.
input = load(fullfile(root, 'fixtures', 'normalization_high_dim_input.mat'));
normalized = normalize_only(input.profiles, input.cortex);
single_normalized = zeros(size(normalized), 'single');
for row = 1:size(input.profiles, 1)
    single_normalized(row,:) = normalize_only(input.profiles(row,:), input.cortex(row));
end
runtime_version = version;
runtime_computer = computer;
runtime_blas = __octave_config_info__('BLAS_LIBS');
save('-mat7-binary', fullfile(root, 'fixtures', 'normalization_high_dim_output.mat'), ...
     'normalized', 'single_normalized', 'runtime_version', 'runtime_computer', 'runtime_blas');
fprintf('ORACLE_VERSION=%s\\nCOMPUTER=%s\\nBLAS=%s\\n', runtime_version, runtime_computer, runtime_blas);
end

function series = normalize_only(profiles, cortex)
% Cast and supplied cortex mask replace MRI and mesh I/O only.
series = single(profiles);
series(~logical(cortex),:) = 0;
''' + body + '\nend\n')
    rng = np.random.default_rng(SEED)
    profiles = np.zeros((16, 1175), dtype=np.float32)
    counts = (1, 2, 3, 10, 37, 100, 117, 256, 511, 700, 1000, 1174)
    for row, count in zip(range(3, 15), counts, strict=True):
        profiles[row, rng.choice(1175, count, replace=False)] = 1
    profiles[1] = 1  # Outside cortex: must remain all zero after normalization.
    profiles[2] = np.arange(1175)  # Centered zero: must remain unnormalized.
    profiles[15] = rng.uniform(-1, 1, 1175)  # Exercise mean reduction too.
    cortex = np.ones(16, dtype=bool)
    cortex[1] = False
    savemat(INPUT, {'profiles': profiles, 'cortex': cortex[:, None]})


def collect():
    inputs = loadmat(INPUT)
    output = loadmat(OUTPUT)
    np.savez_compressed(FIXTURE, profiles=inputs['profiles'], cortex=inputs['cortex'].ravel().astype(bool),
                        normalized=output['normalized'], single_normalized=output['single_normalized'])
    def text_field(name):
        return str(output[name].ravel()[0])
    files = (SOURCE, ADAPTER, FIXTURE, LOG, Path(__file__))
    report = {
        'scope': 'Executed CBIG float32 normalization, 16 profiles and one-row calls at D=1175',
        'cbig_revision': '35b5664bec8822e2f77da5e090e96f91d0095be6',
        'seed': SEED,
        'runtime': {key: text_field('runtime_' + key) for key in ('version', 'computer', 'blas')},
        'source_changes': ['replace MRI reader with supplied single profiles and mesh reader with supplied cortex mask; normalization body is unchanged'],
        'commands': ['python validation/generate_normalization_high_dim.py --prepare-only',
                     "octave --quiet --eval \"addpath('validation/runtime'); normalization_high_dim_oracle('validation');\"",
                     'python validation/generate_normalization_high_dim.py --collect-existing'],
        'sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'execution_artifact_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (INPUT, OUTPUT)},
    }
    (ROOT / 'fixtures/normalization_high_dim_provenance.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--collect-existing', action='store_true')
    args = parser.parse_args()
    if not args.collect_existing:
        prepare()
    if not args.prepare_only:
        if not args.collect_existing:
            with LOG.open('w') as log:
                subprocess.run(['octave', '--quiet', '--eval', f"addpath('{ROOT}/runtime'); normalization_high_dim_oracle('{ROOT}');"], stdout=log, stderr=subprocess.STDOUT, check=True)
        collect()
