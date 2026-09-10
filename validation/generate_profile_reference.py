"""Prepare/collect Octave oracles for correlation, binarization, normalization, centroids.

After --prepare-only, run Octave with validation/runtime and validation/upstream
on its path: profile_oracle('ABSOLUTE_PATH_TO_VALIDATION'). Then --collect-existing.
The adapters are assembled from the bundled, unchanged CBIG source files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent


def prepare():
    profile = (ROOT / "upstream/CBIG_ComputeCorrelationProfile.m").read_text()
    threshold = profile.rsplit("    tmp = sort(tmp(:), 'descend');", 1)[1].split("        write_fmri", 1)[0]
    threshold = "tmp = sort(tmp(:), 'descend');" + threshold
    # Remove only the conditional wrapper; threshold is fixed at 0.1.
    threshold = threshold.replace("    if(str2num(threshold) < 1)\n", "")
    right_threshold = "\n".join(line for line in profile.splitlines() if line.strip() in (
        "corr_mat2(corr_mat2 <  t) = 0;", "corr_mat2(corr_mat2 >= t) = 1;"))
    estimation = (ROOT / "upstream/CBIG_MSHBM_estimate_group_priors.m").read_text()
    normalization = estimation.rsplit("                series = bsxfun(@minus,series,mean(series, 2));", 1)[1].split("                data.series", 1)[0]
    normalization = "series = bsxfun(@minus,series,mean(series, 2));" + normalization
    centroid_source = (ROOT / "upstream/CBIG_IndCBM_generate_MSHBM_params.m").read_text()
    centroid = "% 0 label" + centroid_source.split("% 0 label", 1)[1].split("clustered.d", 1)[0]
    adapter = """function profile_oracle(root)
% Equations extracted from CBIG, see ../upstream/LICENSE-CBIG.md.
input = load(fullfile(root, 'fixtures', 'profile_input.mat'));
lh_seed_ind = logical(input.lh_cortex(1:input.seed_vertices));
rh_seed_ind = logical(input.rh_cortex(1:input.seed_vertices));
s_series = [input.lh(:,lh_seed_ind) input.rh(:,rh_seed_ind)];
corr_mat1 = CBIG_corr(s_series, input.lh);
corr_mat1(isnan(corr_mat1)) = 0;
corr_mat2 = CBIG_corr(s_series, input.rh);
corr_mat2(isnan(corr_mat2)) = 0;
correlations = [corr_mat1 corr_mat2]';
threshold = '0.1';
tmp = [corr_mat1 corr_mat2];
""" + threshold + right_threshold + """
binary = [corr_mat1 corr_mat2]';
series = single(input.profiles);
series(~logical(input.profile_cortex),:) = 0;
""" + normalization + """
normalized = series;
series = input.avg_profile;
labels = input.labels;
""" + centroid + """
centroids = mtc;
save('-mat7-binary',fullfile(root,'fixtures','profile_output.mat'), 'correlations', 'binary', 'normalized', 'centroids', 't');
fprintf('ORACLE_VERSION=%s\\n',version);
end
"""
    (ROOT / "runtime/profile_oracle.m").write_text(adapter)
    rng = np.random.default_rng(97620)
    lh, rh = rng.normal(size=(80, 30)), rng.normal(size=(80, 32))
    lh_cortex, rh_cortex = np.ones(30, bool), np.ones(32, bool)
    lh_cortex[[2, 16]] = False
    rh_cortex[[4, 19]] = False
    lh[:, 2] = np.nan
    rh[:, 4] = np.nan
    lh[0, 16] = np.nan  # One NaN invalidates the entire target correlation column.
    rh[17, 19] = np.nan
    profiles = rng.integers(0, 5, size=(18, 7)) / 4
    profiles[0] = np.arange(7)  # Centered zero: upstream leaves this row unscaled.
    profiles[1] = 0
    cortex = np.ones(18, bool)
    cortex[2] = False
    avg = rng.integers(0, 5, size=(18, 7)) / 4
    labels = np.arange(18) % 3 + 1
    labels[2] = 0
    inputs = dict(lh=lh, rh=rh, lh_cortex=lh_cortex, rh_cortex=rh_cortex,
                  seed_vertices=12, profiles=profiles, profile_cortex=cortex,
                  avg_profile=avg, labels=labels[:, None])
    savemat(ROOT / "fixtures/profile_input.mat", inputs)


def collect():
    inputs = loadmat(ROOT / "fixtures/profile_input.mat", simplify_cells=True)
    output = loadmat(ROOT / "fixtures/profile_output.mat", simplify_cells=True)
    arrays = {key: value for key, value in (inputs | output).items() if not key.startswith("__")}
    np.savez_compressed(ROOT / "fixtures/cbig_profiles.npz", **arrays)
    files = [ROOT / "runtime/profile_oracle.m", ROOT / "fixtures/cbig_profiles.npz"]
    files += sorted((ROOT / "upstream").glob("*"))
    provenance = {
        "scope": "synthetic array kernels: correlation, global 10% binarization, profile normalization, label-based centroids",
        "cbig_revision": "35b5664bec8822e2f77da5e090e96f91d0095be6",
        "runtime": "GNU Octave 6.4.0",
        "seed": 97620,
        "source_changes": ["replace mesh and file I/O with deterministic arrays and supplied cortex masks", "extract single-run no-splitting branch and pass fixed threshold 0.1", "replace profile MRI read with supplied single array for normalization and double array for centroids"],
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (ROOT / "fixtures/profile-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    args = parser.parse_args()
    if not args.collect_existing:
        prepare()
    if not args.prepare_only:
        if not args.collect_existing:
            subprocess.run(["octave", "--quiet", "--eval", f"addpath('{ROOT}/runtime'); addpath('{ROOT}/upstream'); profile_oracle('{ROOT}');"], check=True)
        collect()
