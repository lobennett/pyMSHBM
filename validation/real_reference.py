"""Resumable real-data comparison against executed, pinned CBIG equations.

Stages: prepare, run-profiles, check-profiles, run-kernel, collect. Reference
arrays are constructed only in Octave/MATLAB. Python's profile code runs only
in check-profiles; collect reads the separately executed production CLI fit.
Large MAT transports and reports belong in an external, ignored workdir.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent
MOUNT_ROOT = ROOT.parents[1]
TOLERANCES = {
    "mu": (2e-5, 1e-4), "s_psi": (2e-5, 1e-4), "s_t_nu": (2e-5, 1e-4),
    "theta": (1e-4, 1e-4), "s_lambda": (1e-4, 1e-4),
    "epsil": (.05, 2e-4), "sigma": (.05, 2e-4), "kappa": (.05, 2e-4),
    "cost_em": (.1, 5e-6), "cost_intra": (.1, 5e-6),
    "cost_inter": (.1, 5e-6), "record": (.1, 5e-6),
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_errors(actual, expected, atol, rtol, *, rows=2048):
    """Bound comparison scratch memory; nonfinite real-data outputs fail."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    report = {"atol": atol, "rtol": rtol, "actual_shape": list(actual.shape),
              "reference_shape": list(expected.shape), "actual_dtype": str(actual.dtype),
              "reference_dtype": str(expected.dtype), "passed": False}
    if actual.shape != expected.shape:
        return report
    actual, expected = np.atleast_1d(actual), np.atleast_1d(expected)
    absolute = relative = scaled = 0.
    unequal = violations = nonfinite = 0
    for start in range(0, actual.shape[0], rows):
        a = actual[start:start + rows].astype(np.float64)
        b = expected[start:start + rows].astype(np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        nonfinite += int(np.count_nonzero(~finite))
        unequal += int(np.count_nonzero(a != b))
        if np.any(finite):
            delta = np.abs(a[finite] - b[finite])
            denominator = np.abs(b[finite])
            absolute = max(absolute, float(delta.max(initial=0)))
            # Stabilized relative error matches the existing synthetic report.
            relative = max(relative, float((delta / np.maximum(denominator, atol or np.finfo(float).tiny)).max(initial=0)))
            limit = atol + rtol * denominator
            ratio = np.divide(delta, limit, out=np.zeros_like(delta), where=limit > 0)
            scaled = max(scaled, float(ratio.max(initial=0)))
            violations += int(np.count_nonzero(delta > limit))
    report.update(max_absolute_error=absolute, max_relative_error=relative,
                  max_tolerance_fraction=scaled, exact_mismatches=unequal,
                  tolerance_violations=violations, nonfinite_pairs=nonfinite,
                  passed=not violations and not nonfinite)
    return report


def label_errors(actual, expected, subjects, clusters):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError("Labels must have matching (vertices, subjects) shape")
    report = {"unpermuted_label_mismatches": int(np.count_nonzero(actual != expected)),
              "agreement": float(np.mean(actual == expected)), "subjects": {}}
    for index, subject in enumerate(subjects):
        a, b = actual[:, index], expected[:, index]
        union = (a != 0) | (b != 0)
        dice = {}
        for network in range(1, clusters + 1):
            aa, bb = a == network, b == network
            denominator = int(aa.sum() + bb.sum())
            dice[str(network)] = float(2 * np.count_nonzero(aa & bb) / denominator) if denominator else None
        report["subjects"][subject] = {
            "mismatches": int(np.count_nonzero(a != b)), "agreement": float(np.mean(a == b)),
            "assigned_union_agreement": float(np.mean(a[union] == b[union])) if union.any() else None,
            "python_unassigned": int(np.count_nonzero(a == 0)),
            "reference_unassigned": int(np.count_nonzero(b == 0)), "per_network_dice": dice}
    return report


def published_label_errors(output, actual, expected, subjects):
    """Read final GIFTIs, preserving subject/hemisphere and native network order."""
    import nibabel as nib

    n = actual.shape[0] // 2
    report = {"files": {}, "passed": True}
    for subject_index, subject in enumerate(subjects):
        for hemi, indices, structure in (("L", slice(0, n), "CortexLeft"),
                                          ("R", slice(n, None), "CortexRight")):
            path = (Path(output) / f"sub-{subject}/func" /
                    f"sub-{subject}_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii")
            entry = {"path": str(path), "passed": False}
            try:
                image = nib.load(path)
                if not isinstance(image, nib.gifti.GiftiImage) or len(image.darrays) != 1:
                    raise ValueError("Expected one label array in a GIFTI image")
                labels = image.darrays[0].data
                entry.update(sha256=digest(path),
                    hemisphere_matches=image.meta.get("AnatomicalStructurePrimary") == structure,
                    label_intent_matches=image.darrays[0].intent == nib.nifti1.intent_codes["NIFTI_INTENT_LABEL"],
                    parameter_labels=array_errors(labels, actual[indices, subject_index], 0, 0),
                    reference_labels=array_errors(labels, expected[indices, subject_index], 0, 0))
                entry["passed"] = (entry["hemisphere_matches"] and entry["label_intent_matches"] and
                                   entry["parameter_labels"]["passed"] and entry["reference_labels"]["passed"])
            except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
                entry["reason"] = str(exc)
            report["files"][f"sub-{subject}_hemi-{hemi}"] = entry
            report["passed"] = report["passed"] and entry["passed"]
    return report


def stopping(record, iteration, max_iter):
    record = np.asarray(record).ravel()
    change = (float(abs((record[-1] - record[-2]) / record[-2]))
              if record.size > 1 and record[-2] != 0 else None)
    if change is not None and not np.isfinite(change):
        change = None
    converged = change is not None and change <= 1e-5
    return {"iterations": int(iteration), "relative_cost_change": change,
            "converged": converged, "reached_iteration_limit": iteration >= max_iter,
            "stop_reason": "relative_cost" if converged else "iteration_limit" if iteration >= max_iter else "unexplained"}


def canonical_params(params, shape, clusters):
    n, d, s, t = shape
    shapes = {"mu": (d, clusters), "s_psi": (d, clusters, s),
              "s_t_nu": (d, clusters, t, s), "theta": (n, clusters),
              "s_lambda": (n, clusters, s), "epsil": (clusters,),
              "sigma": (clusters,), "kappa": (clusters,), "cost_em": (s,),
              "cost_inter": (), "cost_intra": (), "iter_inter": ()}
    result = {}
    for key, value in params.items():
        key = "record" if key == "Record" else key
        result[key] = np.asarray(value).reshape(shapes[key]) if key in shapes else np.asarray(value).ravel()
    return result


def profile_adapters():
    """Extract the same literal numerical slices as the synthetic oracle."""
    profile = (ROOT / "upstream/CBIG_ComputeCorrelationProfile.m").read_text()
    threshold = "tmp = sort(tmp(:), 'descend');" + profile.rsplit(
        "    tmp = sort(tmp(:), 'descend');", 1)[1].split("        write_fmri", 1)[0]
    threshold = threshold.replace("    if(str2num(threshold) < 1)\n", "")
    right = "\n".join(line for line in profile.splitlines() if line.strip() in (
        "corr_mat2(corr_mat2 <  t) = 0;", "corr_mat2(corr_mat2 >= t) = 1;"))
    estimation = (ROOT / "upstream/CBIG_MSHBM_estimate_group_priors.m").read_text()
    normalization = "series = bsxfun(@minus,series,mean(series, 2));" + estimation.rsplit(
        "                series = bsxfun(@minus,series,mean(series, 2));", 1)[1].split("                data.series", 1)[0]
    centroid_source = (ROOT / "upstream/CBIG_IndCBM_generate_MSHBM_params.m").read_text()
    centroid = "% 0 label" + centroid_source.split("% 0 label", 1)[1].split("clustered.d", 1)[0]
    profile_adapter = """function [binary, normalized, t] = real_profile_kernel(input, assets)
% Numerical slices from pinned CBIG; see validation/upstream/LICENSE-CBIG.md.
lh_seed_ind = logical(assets.lh_cortex(1:assets.seed_vertices));
rh_seed_ind = logical(assets.rh_cortex(1:assets.seed_vertices));
s_series = [input.lh(:,lh_seed_ind) input.rh(:,rh_seed_ind)];
corr_mat1 = CBIG_corr(s_series, input.lh);
corr_mat1(isnan(corr_mat1)) = 0;
corr_mat2 = CBIG_corr(s_series, input.rh);
corr_mat2(isnan(corr_mat2)) = 0;
clear input s_series;
threshold = '0.1';
tmp = [corr_mat1 corr_mat2];
""" + threshold + right + """
binary = uint8([corr_mat1 corr_mat2]');
clear tmp corr_mat1 corr_mat2;
series = single(binary);
series(~logical(assets.profile_cortex),:) = 0;
""" + normalization + """
normalized = series;
end
"""
    centroid_adapter = """function centroids = real_centroid_kernel(series, labels)
% Numerical slice from pinned CBIG; see validation/upstream/LICENSE-CBIG.md.
""" + centroid + "\ncentroids = mtc;\nend\n"
    return profile_adapter, centroid_adapter


def prepare(args):
    from pymshbm.core.buckner import validate_cortical_bold
    from pymshbm.io.assets import load_assets
    from pymshbm.io.bids import discover_surface_runs
    from pymshbm.pipeline.buckner import python_source_hashes, read_surface_bold

    work = args.workdir
    if work.exists() and any(work.iterdir()):
        raise FileExistsError("Preparation requires an empty workdir; use the resumable run stages for prepared data")
    # Existing fixture provenance establishes immutable upstream and kernel hashes.
    pinned = json.loads((ROOT / "fixtures/oracle-provenance.json").read_text())
    for name, value in pinned["upstream_files_sha256"].items():
        if digest(ROOT / "upstream" / name) != value:
            raise ValueError(f"Pinned source hash mismatch: {name}")
    for name, value in pinned["runtime_files_sha256"].items():
        if digest(ROOT / "runtime" / name) != value:
            raise ValueError(f"Pinned runtime hash mismatch: {name}")
    source_hashes = python_source_hashes()
    assets = load_assets(args.assets)
    runs = discover_surface_runs(args.bids_input, participant_labels=args.participant_label,
                                 session_labels=args.session_label, task=args.task)
    subjects = sorted({run.subject for run in runs})
    counts = Counter(run.subject for run in runs)
    shape = [len(assets.labels), int(assets.lh_cortex[:642].sum() + assets.rh_cortex[:642].sum()),
             len(subjects), max(counts.values())]
    work.mkdir(parents=True, exist_ok=True)
    (work / "runtime").mkdir(exist_ok=True)
    for name, content in zip(("real_profile_kernel.m", "real_centroid_kernel.m"), profile_adapters(), strict=True):
        (work / "runtime" / name).write_text(content)
    records, indices = [], Counter()
    for number, run in enumerate(runs, 1):
        indices[run.subject] += 1
        stem = f"run-{number:03d}"
        lh, rh = read_surface_bold(run.lh), read_surface_bold(run.rh)
        if lh.shape[0] != rh.shape[0]:
            raise ValueError("Hemisphere timepoints differ")
        zero_indices = {}
        for hemi, data, mask in (("lh", lh, assets.lh_cortex), ("rh", rh, assets.rh_cortex)):
            zero = validate_cortical_bold(data, mask, allow_zero_cortex=args.allow_zero_cortex)
            zero_indices[hemi] = np.flatnonzero(zero).tolist()
        savemat(work / f"{stem}-input.mat", {"lh": lh, "rh": rh}, do_compression=False)
        records.append({"stem": stem, "subject": run.subject, "session": run.session,
                        "run": run.run, "task": run.task, "timepoints": int(lh.shape[0]),
                        "subject_index": subjects.index(run.subject) + 1, "model_session": indices[run.subject],
                        "lh": {"path": str(run.lh), "sha256": digest(run.lh),
                               "zero_cortex_count": len(zero_indices["lh"]), "zero_cortex_indices": zero_indices["lh"]},
                        "rh": {"path": str(run.rh), "sha256": digest(run.rh),
                               "zero_cortex_count": len(zero_indices["rh"]), "zero_cortex_indices": zero_indices["rh"]},
                        "input_mat_sha256": digest(work / f"{stem}-input.mat")})
        del lh, rh, data
        print(f"Prepared {stem}: sub-{run.subject} model session {indices[run.subject]}", flush=True)
    savemat(work / "config.mat", {"shape": shape, "max_iter": args.max_iter,
                                  "run_stems": np.asarray([r["stem"] for r in records], dtype=object),
                                  "subject_indices": [r["subject_index"] for r in records],
                                  "session_indices": [r["model_session"] for r in records],
                                  "lh_cortex": assets.lh_cortex, "rh_cortex": assets.rh_cortex,
                                  "profile_cortex": np.r_[assets.lh_cortex, assets.rh_cortex][:, None],
                                  "labels": assets.labels[:, None], "seed_vertices": 642})
    runtime = [Path(__file__), *sorted((work / "runtime").glob("*.m")),
               *[ROOT / "runtime" / name for name in pinned["runtime_files_sha256"]],
               *sorted((ROOT / "runtime").glob("real*.m")), *sorted((ROOT / "upstream").glob("*"))]
    manifest = {"scope": "Real surface BOLD: independently executed upstream profiles, centroids and group-prior kernel",
                "cbig_revision": pinned["cbig_revision"], "buckner_revision": pinned["buckner_revision"],
                "bids_input": str(args.bids_input.resolve()), "assets_dir": str(args.assets.resolve()),
                "assets": assets.provenance, "subjects": subjects, "runs": records,
                "python_source_sha256": source_hashes,
                "shape_N_D_S_T": shape, "clusters": 15, "max_iter": args.max_iter,
                "allow_zero_cortex": args.allow_zero_cortex,
                "tolerances": TOLERANCES, "profile_tolerances": {"binary": [0, 0], "normalized": [2e-5, 1e-4], "centroids": [2e-5, 1e-4]},
                "config_sha256": digest(work / "config.mat"),
                "source_runtime_sha256": {str(path.resolve()): digest(path) for path in runtime},
                "adaptations": pinned["source_changes"] + [
                    "Read BIDS GIFTI arrays as double using shared file/orientation reader; transport raw arrays in per-run MAT v5 files",
                    "Use pinned correlation, joint top-10% threshold, profile normalization, and label-centroid source slices",
                    "Cast binary 0/1 arrays to uint8 only for storage; upstream normalization begins with single and centroid average with double",
                    "Average independently computed reference binary profiles in their selected run order",
                    "Load per-run upstream single normalized arrays into one NxDxSxT tensor; missing sessions are NaN",
                    "Record final stopping flags from original cost record and unchanged outer threshold; inner log counts are diagnostic"]}
    if python_source_hashes() != source_hashes:
        raise ValueError("Python source changed during preparation")
    write_json(work / "manifest.json", manifest)


def load_manifest(work):
    from pymshbm.pipeline.buckner import python_source_hashes

    manifest = json.loads((work / "manifest.json").read_text())
    if digest(work / "config.mat") != manifest["config_sha256"]:
        raise ValueError("Prepared config changed")
    for path, expected in manifest["source_runtime_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Prepared source/runtime changed: {path}")
    if manifest.get("python_source_sha256") != python_source_hashes():
        raise ValueError("Python source changed since preparation")
    return manifest


def profile_evidence(work, manifest):
    """Bind comparison claims to this manifest and the exact reference arrays."""
    path = work / "profile-comparison.json"
    if not path.exists():
        return {"passed": False, "binding_passed": False, "binding_errors": ["profiles not compared"]}
    report = json.loads(path.read_text())
    errors = []
    if report.get("manifest_sha256") != digest(work / "manifest.json"):
        errors.append("profile report belongs to a different manifest")
    if report.get("python_source_sha256") != manifest["python_source_sha256"]:
        errors.append("profile comparison used different Python source")
    expected_stems = [r["stem"] for r in manifest["runs"]]
    if [r.get("stem") for r in report.get("runs", [])] != expected_stems:
        errors.append("profile report has different selected runs")
    for run in report.get("runs", []):
        # Derive paths only from the prepared run list.
        if run.get("stem") not in expected_stems:
            continue
        profile = work / f"{run['stem']}-reference.mat"
        if not profile.exists() or digest(profile) != run.get("reference_mat_sha256"):
            errors.append(f"reference profile changed: {run['stem']}")
    centroids = work / "reference-centroids.mat"
    if not centroids.exists() or digest(centroids) != report.get("reference_centroids_mat_sha256"):
        errors.append("reference centroids changed")
    report["binding_passed"] = not errors
    report["binding_errors"] = errors
    report["passed"] = report.get("passed", False) and not errors
    return report


def octave_path(path, container):
    path = Path(path).resolve()
    return "/work/" + str(path.relative_to(MOUNT_ROOT)) if container else str(path)


def run_octave(args, function, log_name, index=None):
    def quote(path):
        return octave_path(path, args.podman_container).replace("'", "''")
    expression = (f"addpath('{quote(ROOT / 'runtime')}'); addpath('{quote(ROOT / 'upstream')}'); "
                  f"addpath('{quote(args.workdir / 'runtime')}'); "
                  f"{function}('{quote(args.workdir)}'" + (f",{index}" if index is not None else "") + ");")
    command = ["octave", "--no-gui", "--quiet", "--eval", expression]
    if args.podman_container:
        command = [shutil.which("podman") or "/opt/podman/bin/podman", "exec", args.podman_container] + command
    print("Executing " + function + (f" run {index}" if index is not None else ""), flush=True)
    with (args.workdir / log_name).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_profiles(args):
    manifest = load_manifest(args.workdir)
    for index, record in enumerate(manifest["runs"], 1):
        if (args.workdir / f"{record['stem']}-reference.mat").exists():
            print(f"Resume: keeping completed {record['stem']}", flush=True)
            continue
        if digest(args.workdir / f"{record['stem']}-input.mat") != record["input_mat_sha256"]:
            raise ValueError(f"Prepared input changed: {record['stem']}")
        run_octave(args, "real_run_profiles", f"{record['stem']}-octave.log", index)
    if not (args.workdir / "reference-centroids.mat").exists():
        run_octave(args, "real_run_centroids", "centroids-octave.log")


def check_profiles(args):
    from pymshbm.core.buckner import binary_profiles, centroids_from_labels, normalize_profiles
    from pymshbm.io.assets import load_assets
    from pymshbm.pipeline.buckner import read_surface_bold

    manifest = load_manifest(args.workdir)
    assets = load_assets(manifest["assets_dir"])
    average = np.zeros(manifest["shape_N_D_S_T"][:2], dtype=np.float64)
    cortex = np.r_[assets.lh_cortex, assets.rh_cortex]
    report = {"scope": "Independent real-data profile and initialization comparison", "runs": [],
              "manifest_sha256": digest(args.workdir / "manifest.json"),
              "python_source_sha256": manifest["python_source_sha256"]}
    for record in manifest["runs"]:
        for hemi in ("lh", "rh"):
            if digest(record[hemi]["path"]) != record[hemi]["sha256"]:
                raise ValueError(f"BOLD input changed: {record[hemi]['path']}")
        actual = binary_profiles(read_surface_bold(record["lh"]["path"]),
                                 read_surface_bold(record["rh"]["path"]), assets.lh_cortex, assets.rh_cortex,
                                 allow_zero_cortex=manifest.get("allow_zero_cortex", False))
        average += actual
        path = args.workdir / f"{record['stem']}-reference.mat"
        expected = loadmat(path, variable_names=["binary"])["binary"]
        run_report = {"stem": record["stem"], "binary": array_errors(actual, expected, *manifest["profile_tolerances"]["binary"])}
        del expected
        normalized = normalize_profiles(actual, cortex)
        del actual
        loaded = loadmat(path, variable_names=["normalized", "t", "elapsed_seconds"])
        run_report["normalized"] = array_errors(normalized, loaded["normalized"], *manifest["profile_tolerances"]["normalized"])
        run_report["reference_threshold"] = float(loaded["t"].item())
        run_report["reference_elapsed_seconds"] = float(loaded["elapsed_seconds"].item())
        run_report["reference_mat_sha256"] = digest(path)
        del normalized, loaded
        report["runs"].append(run_report)
        write_json(args.workdir / "profile-comparison.partial.json", report)
        print(f"Compared {record['stem']}: binary mismatches {run_report['binary']['exact_mismatches']}", flush=True)
    average /= len(manifest["runs"])
    actual = centroids_from_labels(average, assets.labels)
    del average
    expected = loadmat(args.workdir / "reference-centroids.mat")["g_mu"]
    report["centroids"] = array_errors(actual, expected, *manifest["profile_tolerances"]["centroids"])
    report["reference_centroids_mat_sha256"] = digest(args.workdir / "reference-centroids.mat")
    savemat(args.workdir / "python-centroids.mat", {"g_mu": actual})
    report["passed"] = report["centroids"]["passed"] and all(
        r["binary"]["passed"] and r["normalized"]["passed"] for r in report["runs"])
    load_manifest(args.workdir)  # Detect source edits during this comparison.
    write_json(args.workdir / "profile-comparison.json", report)
    return report


def collect(args):
    manifest = load_manifest(args.workdir)
    python_file = args.python_params or args.python_output / "model/priors/Params_Final.mat"
    reference_file = args.workdir / "reference-params.mat"
    shape, clusters = manifest["shape_N_D_S_T"], manifest["clusters"]
    actual = canonical_params(loadmat(python_file, simplify_cells=True)["Params"], shape, clusters)
    production_file = args.python_output / "model/priors/Params_Final.mat"
    production = canonical_params(loadmat(production_file, simplify_cells=True)["Params"], shape, clusters)
    capture_matches = all(key in actual and np.array_equal(value, actual[key], equal_nan=True)
                          for key, value in production.items())
    loaded = loadmat(reference_file, simplify_cells=True)
    expected = canonical_params(loaded["Params"], shape, clusters)
    report = {"scope": manifest["scope"], "manifest_sha256": digest(args.workdir / "manifest.json"),
              "python_params": {"path": str(python_file.resolve()), "sha256": digest(python_file)},
              "python_capture_matches_production_saved_fields": capture_matches,
              "reference_params_sha256": digest(reference_file),
              "runtime": str(loaded["runtime_version"]), "reference_elapsed_seconds": float(loaded["elapsed_seconds"]),
              "parameters": {}}
    for key, (atol, rtol) in manifest["tolerances"].items():
        report["parameters"][key] = (array_errors(actual[key], expected[key], atol, rtol)
            if key in actual and key in expected else {"passed": False, "reason": "parameter missing from saved fit"})
    def labels(params):
        posterior = params["s_lambda"]
        result = posterior.argmax(axis=1) + 1
        result[np.all(posterior == 0, axis=1)] = 0
        return result
    actual_labels, expected_labels = labels(actual), labels(expected)
    report["labels"] = label_errors(actual_labels, expected_labels, manifest["subjects"], clusters)
    report["published_labels"] = published_label_errors(args.python_output, actual_labels, expected_labels,
                                                        manifest["subjects"])
    np.savez_compressed(args.workdir / "label-comparison.npz", python=actual_labels, reference=expected_labels,
                        subjects=manifest["subjects"])
    report["stopping"] = {name: stopping(value["record"], int(value["iter_inter"]), manifest["max_iter"])
                          for name, value in (("python", actual), ("reference", expected))}
    a, b = report["stopping"].values()
    report["stopping"]["passed"] = all(a[key] == b[key] for key in (
        "iterations", "converged", "reached_iteration_limit", "stop_reason"))
    log = (args.workdir / "kernel-octave.log").read_text()
    report["reference_log_diagnostics"] = {"inter_region_blocks": log.count("Inter-region iteration "),
        "em_iterations": log.count("It is EM iteration.."), "intra_direction_updates": log.count("update s_psi and sigma"),
        "em_iteration_limit_warnings": log.count("vem can not converge"), "warnings": log.count("[WARNING]")}
    report["profiles"] = profile_evidence(args.workdir, manifest)
    provenance = json.loads((args.python_output / "provenance.json").read_text())
    python_runs = provenance["runs"]
    match_inputs = (provenance["subjects_in_model_order"] == manifest["subjects"] and
                    len(python_runs) == len(manifest["runs"]) and
                    all(p["subject"] == r["subject"] and p["model_session"] == r["model_session"] and
                        all(p[h]["sha256"] == r[h]["sha256"] for h in ("lh", "rh"))
                        for p, r in zip(python_runs, manifest["runs"])) and
                    provenance["settings"]["max_iter"] == manifest["max_iter"] and
                    provenance["settings"].get("allow_zero_cortex", False) == manifest.get("allow_zero_cortex", False) and
                    provenance["assets"] == manifest["assets"])
    report["same_production_inputs_assets_and_settings"] = match_inputs
    report["same_production_python_source"] = provenance.get("python_source_sha256") == manifest["python_source_sha256"]
    report["passed"] = (capture_matches and match_inputs and report["same_production_python_source"] and
                         report["published_labels"]["passed"] and report["profiles"]["passed"] and report["stopping"]["passed"] and
                         report["labels"]["unpermuted_label_mismatches"] == 0 and
                         all(value["passed"] for value in report["parameters"].values()))
    write_json(args.workdir / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run-profiles", "check-profiles", "run-kernel", "collect"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--bids-input", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--participant-label", nargs="+")
    parser.add_argument("--session-label", nargs="+")
    parser.add_argument("--task", default="rest")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--allow-zero-cortex", action="store_true",
                        help="Prepare an explicitly masked-coverage case; unusable cortical seeds remain errors")
    parser.add_argument("--podman-container")
    parser.add_argument("--python-output", type=Path)
    parser.add_argument("--python-params", type=Path)
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    if args.stage == "prepare":
        if args.bids_input is None or args.assets is None or args.max_iter < 1:
            parser.error("prepare requires --bids-input, --assets, and positive --max-iter")
        prepare(args)
    elif args.stage == "run-profiles":
        run_profiles(args)
    elif args.stage == "run-kernel":
        manifest = load_manifest(args.workdir)
        evidence = profile_evidence(args.workdir, manifest)
        if not evidence["binding_passed"]:
            raise ValueError("Cannot run kernel: " + "; ".join(evidence["binding_errors"]))
        if (args.workdir / "reference-params.mat").exists():
            print("Resume: keeping completed reference-params.mat")
        else:
            run_octave(args, "real_run_kernel", "kernel-octave.log")
    else:
        if args.stage == "collect" and args.python_output is None:
            parser.error("collect requires --python-output")
        report = check_profiles(args) if args.stage == "check-profiles" else collect(args)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
