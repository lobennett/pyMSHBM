"""Reporting must reveal mismatches, missing fields, and stopping differences."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
from scipy.io import savemat
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "real_reference", Path(__file__).resolve().parents[1] / "validation/real_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_error_report_rejects_mismatch_and_nonfinite_outputs():
    result = reference.array_errors(np.array([0., 1.1]), np.array([0., 1.]), .05, 2e-4)
    assert not result["passed"]
    assert np.isclose(result["max_absolute_error"], .1)
    assert np.isclose(result["max_relative_error"], .1)
    assert not reference.array_errors(np.array([np.nan]), np.array([np.nan]), 1e-4, 1e-4)["passed"]


def test_labels_are_compared_without_permutations_and_zero_is_separate():
    actual = np.array([[1, 2], [2, 2], [0, 0]])
    expected = np.array([[2, 2], [1, 2], [0, 0]])
    report = reference.label_errors(actual, expected, ["MSC01", "MSC02"], 2)
    assert report["unpermuted_label_mismatches"] == 2
    assert report["subjects"]["MSC01"]["per_network_dice"]["1"] == 0
    assert report["subjects"]["MSC02"]["per_network_dice"]["1"] is None
    assert report["subjects"]["MSC02"]["agreement"] == 1


def test_stopping_limit_is_not_evidence_of_convergence():
    report = reference.stopping([100., 110., 120.], 3, 3)
    assert report["reached_iteration_limit"]
    assert not report["converged"]
    report = reference.stopping([100., 100.0001], 2, 5)
    assert report["converged"]
    assert not report["reached_iteration_limit"]


def test_mat_shape_normalization_preserves_single_subject_and_session():
    shape = (8, 7, 1, 1)
    arrays = reference.canonical_params({"s_t_nu": np.ones((7, 3)),
                                        "s_lambda": np.ones((8, 3)),
                                        "Record": np.array([[1., 2.]])}, shape, 3)
    assert arrays["s_t_nu"].shape == (7, 3, 1, 1)
    assert arrays["s_lambda"].shape == (8, 3, 1)
    assert arrays["record"].shape == (2,)


def test_adapter_assembly_does_not_change_synthetic_runtime():
    root = Path(__file__).resolve().parents[1] / "validation"
    before = (root / "runtime/profile_oracle.m").read_bytes()
    profile, centroid = reference.profile_adapters()
    assert "CBIG_corr(s_series, input.lh)" in profile
    assert "tmp = sort(tmp(:), 'descend');" in profile
    assert "series(all(series,2)~=0,:)" in profile
    assert "mtc = x' * r;" in centroid
    assert (root / "runtime/profile_oracle.m").read_bytes() == before


@pytest.mark.parametrize("corruption", ["label", "hemisphere"])
def test_published_gifti_checks_reject_altered_or_wrong_hemisphere(tmp_path, corruption):
    labels = np.array([1, 1, 2, 0, 2, 2, 1, 0])[:, None]
    directory = tmp_path / "sub-MSC01/func"
    directory.mkdir(parents=True)
    for hemi, values in (("L", labels[:4, 0]), ("R", labels[4:, 0])):
        image = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(values.astype(np.int32), intent="NIFTI_INTENT_LABEL")],
            meta=nib.gifti.GiftiMetaData({"AnatomicalStructurePrimary": "CortexLeft" if hemi == "L" else "CortexRight"}))
        nib.save(image, directory / f"sub-MSC01_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii")
    assert reference.published_label_errors(tmp_path, labels, labels, ["MSC01"])["passed"]
    left = directory / "sub-MSC01_space-fsaverage6_atlas-DU15NET_hemi-L_dseg.label.gii"
    image = nib.load(left)
    if corruption == "label":
        image.darrays[0].data[0] = 3
    else:
        image.meta["AnatomicalStructurePrimary"] = "CortexRight"
    nib.save(image, left)
    assert not reference.published_label_errors(tmp_path, labels, labels, ["MSC01"])["passed"]


def test_profile_evidence_rejects_copied_report_and_changed_reference(tmp_path):
    reference.write_json(tmp_path / "manifest.json", {"case": "A"})
    manifest = {"runs": [{"stem": "run-001"}], "python_source_sha256": {"core/buckner.py": "original"}}
    (tmp_path / "run-001-reference.mat").write_bytes(b"profile")
    (tmp_path / "reference-centroids.mat").write_bytes(b"centroids")
    report = {"passed": True, "manifest_sha256": reference.digest(tmp_path / "manifest.json"),
              "python_source_sha256": manifest["python_source_sha256"],
              "runs": [{"stem": "run-001", "reference_mat_sha256": reference.digest(tmp_path / "run-001-reference.mat")}],
              "reference_centroids_mat_sha256": reference.digest(tmp_path / "reference-centroids.mat")}
    reference.write_json(tmp_path / "profile-comparison.json", report)
    assert reference.profile_evidence(tmp_path, manifest)["binding_passed"]
    report["manifest_sha256"] = "different experiment"
    reference.write_json(tmp_path / "profile-comparison.json", report)
    assert not reference.profile_evidence(tmp_path, manifest)["binding_passed"]
    report["manifest_sha256"] = reference.digest(tmp_path / "manifest.json")
    reference.write_json(tmp_path / "profile-comparison.json", report)
    (tmp_path / "run-001-reference.mat").write_bytes(b"changed profile")
    assert not reference.profile_evidence(tmp_path, manifest)["binding_passed"]


def test_manifest_rejects_different_installed_python_source(tmp_path):
    savemat(tmp_path / "config.mat", {"shape": [8, 7, 1, 1]})
    reference.write_json(tmp_path / "manifest.json", {
        "config_sha256": reference.digest(tmp_path / "config.mat"), "source_runtime_sha256": {},
        "python_source_sha256": {"core/buckner.py": "different code"}})
    with pytest.raises(ValueError, match="Python source changed"):
        reference.load_manifest(tmp_path)


def test_collect_flags_missing_costs_and_checks_captured_fit_identity(tmp_path):
    from pymshbm.pipeline.buckner import python_source_hashes

    output = tmp_path / "python"
    (output / "model/priors").mkdir(parents=True)
    params = {"mu": np.ones((7, 3)), "s_psi": np.ones((7, 3, 1)),
              "s_t_nu": np.ones((7, 3, 1, 1)), "theta": np.ones((8, 3)) / 3,
              "s_lambda": np.ones((8, 3, 1)) / 3,
              "epsil": np.ones(3) * 500, "sigma": np.ones(3) * 500,
              "kappa": np.ones(3) * 500, "iter_inter": 2,
              "Record": np.array([100., 100.0001])}
    savemat(output / "model/priors/Params_Final.mat", {"Params": params})
    full = params | {"cost_em": 50., "cost_intra": 100.0001, "cost_inter": 100.0001}
    savemat(tmp_path / "reference-params.mat", {"Params": full, "runtime_version": "test", "elapsed_seconds": 1.})
    savemat(tmp_path / "reference-centroids.mat", {"g_mu": params["mu"]})
    savemat(tmp_path / "config.mat", {"shape": [8, 7, 1, 1]})
    manifest = {"scope": "test", "config_sha256": reference.digest(tmp_path / "config.mat"),
                "source_runtime_sha256": {}, "shape_N_D_S_T": [8, 7, 1, 1],
                "clusters": 3, "max_iter": 5, "subjects": ["MSC01"], "runs": [], "assets": {},
                "python_source_sha256": python_source_hashes(), "tolerances": dict(reference.TOLERANCES)}
    reference.write_json(tmp_path / "manifest.json", manifest)
    reference.write_json(tmp_path / "profile-comparison.json", {"passed": True,
        "manifest_sha256": reference.digest(tmp_path / "manifest.json"),
        "python_source_sha256": manifest["python_source_sha256"], "runs": [],
        "reference_centroids_mat_sha256": reference.digest(tmp_path / "reference-centroids.mat")})
    provenance = {"runs": [], "subjects_in_model_order": ["MSC01"],
        "settings": {"max_iter": 5}, "assets": {}, "python_source_sha256": python_source_hashes()}
    reference.write_json(output / "provenance.json", provenance)
    directory = output / "sub-MSC01/func"
    directory.mkdir(parents=True)
    for hemi, structure in (("L", "CortexLeft"), ("R", "CortexRight")):
        image = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(np.ones(4, dtype=np.int32), intent="NIFTI_INTENT_LABEL")],
            meta=nib.gifti.GiftiMetaData({"AnatomicalStructurePrimary": structure}))
        nib.save(image, directory / f"sub-MSC01_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii")
    (tmp_path / "kernel-octave.log").write_text("")
    args = SimpleNamespace(workdir=tmp_path, python_output=output, python_params=None)
    report = reference.collect(args)
    assert not report["passed"]
    assert report["parameters"]["cost_em"]["reason"] == "parameter missing from saved fit"
    savemat(output / "model/priors/Params_Final.mat", {"Params": full})
    assert reference.collect(args)["passed"]
    # Historical manifests omit this setting and therefore mean strict False.
    provenance["settings"]["allow_zero_cortex"] = True
    reference.write_json(output / "provenance.json", provenance)
    report = reference.collect(args)
    assert not report["same_production_inputs_assets_and_settings"]
    assert not report["passed"]
    manifest["allow_zero_cortex"] = True
    reference.write_json(tmp_path / "manifest.json", manifest)
    profile_path = tmp_path / "profile-comparison.json"
    profile_report = json.loads(profile_path.read_text())
    profile_report["manifest_sha256"] = reference.digest(tmp_path / "manifest.json")
    reference.write_json(profile_path, profile_report)
    assert reference.collect(args)["passed"]
    provenance["python_source_sha256"] = {"core/buckner.py": "different source"}
    reference.write_json(output / "provenance.json", provenance)
    report = reference.collect(args)
    assert not report["same_production_python_source"]
    assert not report["passed"]
    provenance["python_source_sha256"] = python_source_hashes()
    reference.write_json(output / "provenance.json", provenance)
    full["mu"] = full["mu"] + 1e-7  # Within tolerance, but from a different fit.
    savemat(tmp_path / "capture.mat", {"Params": full})
    args.python_params = tmp_path / "capture.mat"
    report = reference.collect(args)
    assert report["parameters"]["mu"]["passed"]
    assert not report["python_capture_matches_production_saved_fields"]
    assert not report["passed"]
    # The evaluator uses the recorded tolerance, not a mutable module constant.
    manifest["tolerances"]["mu"] = [0, 0]
    reference.write_json(tmp_path / "manifest.json", manifest)
    assert not reference.collect(args)["parameters"]["mu"]["passed"]


def test_prepare_requires_explicit_zero_cortex_opt_in(tmp_path, monkeypatch):
    from pymshbm.io.bids import SurfaceRun

    # Small imaging/asset stand-ins exercise the actual cortical validator.
    # Vertex 642 is cortical but outside the first 642 seed positions.
    lh, rh = tmp_path / "left.func.gii", tmp_path / "right.func.gii"
    lh.write_bytes(b"left input")
    rh.write_bytes(b"right input")
    data = np.arange(3 * 643, dtype=float).reshape(3, 643)
    data[:, 642] = 0
    assets = SimpleNamespace(labels=np.ones(1286, dtype=int),
        lh_cortex=np.ones(643, dtype=bool), rh_cortex=np.ones(643, dtype=bool), provenance={})
    monkeypatch.setattr("pymshbm.io.assets.load_assets", lambda _: assets)
    monkeypatch.setattr("pymshbm.io.bids.discover_surface_runs", lambda *a, **k:
        [SurfaceRun("MSC01", "func01", "rest", None, lh, rh)])
    monkeypatch.setattr("pymshbm.pipeline.buckner.read_surface_bold", lambda _: data.copy())
    args = SimpleNamespace(workdir=tmp_path / "strict", bids_input=tmp_path, assets=tmp_path,
        participant_label=None, session_label=None, task="rest", max_iter=5, allow_zero_cortex=False)
    with pytest.raises(ValueError):
        reference.prepare(args)
    args.workdir = tmp_path / "opt-in"
    args.allow_zero_cortex = True
    reference.prepare(args)
    manifest = json.loads((args.workdir / "manifest.json").read_text())
    assert manifest["allow_zero_cortex"] is True
    assert manifest["runs"][0]["lh"]["zero_cortex_indices"] == [642]
