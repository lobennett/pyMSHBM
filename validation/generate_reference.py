"""Execute vendored, pinned CBIG MATLAB equations using Octave.

Run from the repository: .venv/bin/python validation/generate_reference.py
Requires octave on PATH; --podman-container NAME uses an existing Octave
container with the repository mounted at /work/pyMSHBM.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "upstream/CBIG_MSHBM_estimate_group_priors.m"
REVISION = "35b5664bec8822e2f77da5e090e96f91d0095be6"


def build_adapter():
    source = SOURCE.read_text()
    # These are contiguous upstream slices. Only the save block and path/I/O
    # preamble are excluded; every initialization/EM equation is unchanged.
    initialization = source.split("% mu: DxL.", 1)[1].split("%% EM", 1)[0]
    initialization = "% mu: DxL." + initialization
    em = source.split("%% EM", 1)[1].split("rmpath(", 1)[0]
    saving = em.index("        % set s_lambda")
    next_cost = em.index("    cost_inter = update_cost_inter;", saving)
    em = em[:saving] + "    end\n" + em[next_cost:]
    saving = em.index("    if(~exist(fullfile(project_dir, 'priors')))")
    em = em[:saving] + "end\n"
    functions = source.split("%% sub-functions", 1)[1].split("function data = fetch_data", 1)[0]
    preamble = """function Params = cbig_kernel(data, g_mu, max_iter)
% Adapted infrastructure only; equations from CBIG, see ../upstream/LICENSE-CBIG.md.
setting_params.num_sub = size(data.series, 3);
setting_params.num_session = size(data.series, 4);
setting_params.num_clusters = size(g_mu, 2);
setting_params.dim = size(data.series, 2) - 1;
setting_params.ini_concentration = 500;
setting_params.epsilon = 1e-4;
setting_params.conv_th = 1e-5;
setting_params.max_iter = max_iter;
setting_params.g_mu = g_mu;
"""
    adapter = preamble + initialization + "\n%% EM\n" + em + "\nend\n" + functions
    # Octave throws for a failed scalar bracket search where MATLAB can return
    # an unsuccessful exit flag. invAd's returned outu never depends on fzero.
    adapter = adapter.replace("= fzero(@", "= octave_fzero(@")
    (ROOT / "runtime/cbig_kernel.m").write_text(adapter)
    return adapter


def deterministic_inputs():
    rng = np.random.default_rng(73619)
    n, d, s, t, clusters = 300, 101, 3, 2, 3
    g_mu = rng.normal(size=(d, clusters))
    g_mu -= g_mu.mean(axis=0, keepdims=True)
    g_mu /= np.linalg.norm(g_mu, axis=0, keepdims=True)
    labels = np.arange(n) % clusters
    data = np.empty((n, d, s, t))
    for subject in range(s):
        subject_shift = rng.normal(scale=0.04, size=(d, clusters))
        for session in range(t):
            session_shift = rng.normal(scale=0.03, size=(d, clusters))
            rows = g_mu[:, labels].T + subject_shift[:, labels].T + session_shift[:, labels].T
            rows += rng.normal(scale=0.08, size=(n, d))
            # An ambiguous vertex exercises soft posteriors as well as maxima.
            rows[-3] = g_mu[:, 0] + g_mu[:, 1] + rng.normal(scale=0.05, size=d)
            rows -= rows.mean(axis=1, keepdims=True)
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            rows[-2:] = 0
            data[:, :, subject, session] = rows
    return data, g_mu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--podman-container")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--case", choices=("complete", "missing", "soft", "single"), default="complete")
    args = parser.parse_args()
    adapter = build_adapter()
    data, g_mu = deterministic_inputs()
    suffix = "" if args.case == "complete" else "_" + args.case
    if args.case == "missing":
        data[:, :, 2, 1] = np.nan
    if args.case == "soft":
        g_mu[:, 1] = g_mu[:, 0]  # Identical directions exercise nontrivial 0.5 posteriors.
    if args.case == "single":
        data = data.astype(np.float32)
    savemat(ROOT / f"fixtures/synthetic_input{suffix}.mat", {"series": data, "g_mu": g_mu})
    if args.prepare_only:
        return
    task_root = "/work/pyMSHBM/validation" if args.podman_container else str(ROOT)
    expression = (
        f"addpath('{task_root}/runtime'); addpath('{task_root}/upstream'); "
        f"input=load('{task_root}/fixtures/synthetic_input{suffix}.mat'); "
        "data.series=input.series; "
        "tic; Params=cbig_kernel(data,input.g_mu,5); elapsed_seconds=toc; "
        f"save('-mat7-binary','{task_root}/fixtures/synthetic_output{suffix}.mat','Params','elapsed_seconds'); "
        "fprintf('ORACLE_VERSION=%s\\n',version);"
    )
    command = ["octave", "--no-gui", "--quiet", "--eval", expression]
    if args.podman_container:
        command = [shutil.which("podman") or "/opt/podman/bin/podman", "exec", args.podman_container] + command
    if args.collect_existing:
        result = subprocess.CompletedProcess(command, 0, stdout=(ROOT / f"fixtures/octave{suffix}.log").read_text())
    else:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (ROOT / f"fixtures/octave{suffix}.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
    loaded = loadmat(ROOT / f"fixtures/synthetic_output{suffix}.mat", simplify_cells=True)
    output = loaded["Params"]
    arrays = {key: value for key, value in output.items() if key != "Record"}
    arrays["record"] = np.atleast_1d(output["Record"])
    arrays["data"] = data
    arrays["g_mu"] = g_mu
    labels = np.argmax(arrays["s_lambda"], axis=1) + 1
    labels[np.all(arrays["s_lambda"] == 0, axis=1)] = 0
    arrays["labels"] = labels
    np.savez_compressed(ROOT / f"fixtures/cbig_synthetic{suffix}.npz", **arrays)
    manifest = {
        "scope": "synthetic normalized group-prior EM kernel only",
        "cbig_revision": REVISION,
        "buckner_revision": "988ec48cf5453c6009a6f82dbddf5d5b99eda3cd",
        "runtime": [line for line in result.stdout.splitlines() if line.startswith("ORACLE_VERSION=")][0],
        "elapsed_seconds": float(loaded["elapsed_seconds"]),
        "case": args.case,
        "input_dtype": str(data.dtype),
        "shape_N_D_S_T": list(data.shape),
        "clusters": int(g_mu.shape[1]),
        "seed": 73619,
        "max_iter": 5,
        "adapter_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
        "runtime_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "runtime/cbig_kernel.m", ROOT / "runtime/mtimesx.m", ROOT / "runtime/octave_fzero.m")},
        "upstream_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "upstream").glob("*"))},
        "fixture_sha256": hashlib.sha256((ROOT / f"fixtures/cbig_synthetic{suffix}.npz").read_bytes()).hexdigest(),
        "iteration": int(output["iter_inter"]),
        "source_changes": ["replace file/mesh/argument setup with supplied arrays and identical settings", "omit output-save and addpath/rmpath blocks", "replace mtimesx infrastructure with explicit per-page matrix multiplication", "catch Octave fzero bracket failures and return failure exitflag; invAd returns outu independently of fzero"],
    }
    (ROOT / f"fixtures/oracle-provenance{suffix}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
