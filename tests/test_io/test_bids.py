"""BIDS naming fixtures exercise discovery without reading BOLD payloads."""

from dataclasses import FrozenInstanceError

import pytest

from pymshbm.io.bids import BIDSInputError, discover_surface_runs


def surface(root, *, subject="01", session=None, task="rest", run="1",
            hemi="L", space="fsaverage6", den=None, extra="", folder=None):
    entities = [f"sub-{subject}"]
    parent = root / f"sub-{subject}"
    if session:
        entities.append(f"ses-{session}")
        parent /= f"ses-{session}"
    entities.append(f"task-{task}")
    if run:
        entities.append(f"run-{run}")
    if extra:
        entities.extend(extra.split("_"))
    entities.extend([f"hemi-{hemi}", f"space-{space}"])
    if den:
        entities.append(f"den-{den}")
    parent = folder or parent / "func"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / ("_".join(entities) + "_bold.func.gii")
    path.touch()
    return path


def pair(root, **kwargs):
    return tuple(surface(root, hemi=h, **kwargs) for h in ("L", "R"))


@pytest.mark.parametrize("nested", [False, True])
def test_discovers_pairs_in_direct_and_bids_roots(tmp_path, nested):
    root = tmp_path / "derivatives" / "fmriprep" if nested else tmp_path
    left, right = pair(root, subject="02", session="baseline", run="02")
    pair(root, subject="01", run=None)
    pair(root, subject="03", task="motor")
    runs = discover_surface_runs(tmp_path)
    assert [(r.subject, r.session, r.task, r.run) for r in runs] == [
        ("01", None, "rest", None), ("02", "baseline", "rest", "02")]
    assert (runs[1].lh, runs[1].rh) == (left, right)
    with pytest.raises(FrozenInstanceError):
        runs[0].subject = "changed"


def test_filters_prefixes_and_multiple_participants(tmp_path):
    pair(tmp_path, subject="01", session="a", task="motor", run="02")
    pair(tmp_path, subject="02", session="a", task="motor", run="02")
    pair(tmp_path, subject="03", session="a", task="motor", run="02")
    pair(tmp_path, subject="01", session="b", task="motor", run="02")
    pair(tmp_path, subject="01", session="a", task="motor", run="03")
    runs = discover_surface_runs(tmp_path, participant_labels=["sub-01", "02"],
                                 session_labels=["ses-a"], task="motor",
                                 run_labels=["run-02"])
    assert [(r.subject, r.session, r.run) for r in runs] == [
        ("01", "a", "02"), ("02", "a", "02")]


def test_accepts_explicit_fsaverage_41k_density(tmp_path):
    pair(tmp_path, space="fsaverage", den="41k")
    assert len(discover_surface_runs(tmp_path)) == 1


@pytest.mark.parametrize("extra", ["acq-fast", "dir-AP", "echo-1", "rec-original",
                                    "desc-preproc", "part-mag"])
def test_never_cross_pairs_distinct_entities(tmp_path, extra):
    surface(tmp_path, hemi="L", extra=extra)
    surface(tmp_path, hemi="R")
    with pytest.raises(BIDSInputError, match="[Mm]issing.*hemisphere"):
        discover_surface_runs(tmp_path)


def test_never_cross_pairs_space_or_density(tmp_path):
    surface(tmp_path, hemi="L")
    surface(tmp_path, hemi="R", space="fsaverage", den="41k")
    with pytest.raises(BIDSInputError, match="[Mm]issing.*hemisphere"):
        discover_surface_runs(tmp_path)


@pytest.mark.parametrize("extra", ["desc-clean", "echo-1", "rec-alternative"])
def test_rejects_multiple_variants_of_one_physical_run(tmp_path, extra):
    pair(tmp_path)
    pair(tmp_path, extra=extra)
    with pytest.raises(BIDSInputError, match="[Aa]mbiguous"):
        discover_surface_runs(tmp_path)


def test_rejects_duplicate_equivalent_space_exports(tmp_path):
    pair(tmp_path)
    pair(tmp_path, space="fsaverage", den="41k")
    with pytest.raises(BIDSInputError, match="[Aa]mbiguous"):
        discover_surface_runs(tmp_path)


def test_acquisitions_and_directions_are_distinct_sessions(tmp_path):
    pair(tmp_path, extra="acq-fast_dir-AP")
    pair(tmp_path, extra="acq-slow_dir-PA")
    runs = discover_surface_runs(tmp_path)
    assert len(runs) == 2
    assert dict(runs[0].entities)["acq"] == "fast"
    assert dict(runs[1].entities)["dir"] == "PA"


def test_rejects_duplicate_hemisphere_entities(tmp_path):
    pair(tmp_path)
    original = tmp_path / "sub-01" / "func" / "sub-01_task-rest_run-1_hemi-L_space-fsaverage6_bold.func.gii"
    original.with_name("sub-01_task-rest_run-1_space-fsaverage6_hemi-L_bold.func.gii").touch()
    with pytest.raises(BIDSInputError, match="[Dd]uplicate"):
        discover_surface_runs(tmp_path)


@pytest.mark.parametrize("space,den", [("fsLR", "32k"), ("fsaverage5", None),
                                       ("fsaverage", None), ("fsaverage6", "10k")])
def test_unsupported_mesh_is_actionable(tmp_path, space, den):
    pair(tmp_path, space=space, den=den)
    with pytest.raises(BIDSInputError, match="fsaverage6"):
        discover_surface_runs(tmp_path)


def test_cifti_only_input_is_actionable(tmp_path):
    folder = tmp_path / "sub-01" / "func"
    folder.mkdir(parents=True)
    (folder / "sub-01_task-rest_space-fsLR_den-91k_bold.dtseries.nii").touch()
    with pytest.raises(BIDSInputError, match="CIFTI.*fsaverage6"):
        discover_surface_runs(tmp_path)


def test_raw_bids_is_actionable(tmp_path):
    folder = tmp_path / "sub-01" / "func"
    folder.mkdir(parents=True)
    (folder / "sub-01_task-rest_bold.nii.gz").touch()
    with pytest.raises(BIDSInputError, match="fMRIPrep"):
        discover_surface_runs(tmp_path)


def test_no_data_and_missing_requested_participant_are_errors(tmp_path):
    with pytest.raises(BIDSInputError, match="[Nn]o.*BOLD"):
        discover_surface_runs(tmp_path)
    pair(tmp_path)
    with pytest.raises(BIDSInputError, match="02"):
        discover_surface_runs(tmp_path, participant_labels=["01", "02"])


def test_unrelated_nested_derivatives_are_not_scanned(tmp_path):
    pair(tmp_path)
    pair(tmp_path / "derivatives" / "another-pipeline", subject="99")
    assert [r.subject for r in discover_surface_runs(tmp_path)] == ["01"]


def test_filename_subject_must_agree_with_directory(tmp_path):
    folder = tmp_path / "sub-02" / "func"
    pair(tmp_path, folder=folder)
    with pytest.raises(BIDSInputError, match="directory"):
        discover_surface_runs(tmp_path)


def test_legacy_fmriprep_container_is_accepted(tmp_path):
    pair(tmp_path / "fmriprep")
    assert len(discover_surface_runs(tmp_path)) == 1


def test_requested_task_excludes_incomplete_other_tasks(tmp_path):
    pair(tmp_path)
    surface(tmp_path, task="motor")
    assert len(discover_surface_runs(tmp_path)) == 1


def test_missing_input_directory_has_actionable_error(tmp_path):
    with pytest.raises(BIDSInputError, match="directory does not exist"):
        discover_surface_runs(tmp_path / "missing")


def test_missing_hemisphere_entity_is_not_silently_ignored(tmp_path):
    path = surface(tmp_path)
    path.rename(path.with_name(path.name.replace("_hemi-L", "")))
    with pytest.raises(BIDSInputError, match="hemisphere"):
        discover_surface_runs(tmp_path)


def test_duplicate_bids_entity_is_rejected(tmp_path):
    surface(tmp_path, extra="task-motor")
    with pytest.raises(BIDSInputError, match="duplicate BIDS entity"):
        discover_surface_runs(tmp_path)


def test_non_alphanumeric_filename_entity_is_rejected(tmp_path):
    pair(tmp_path, extra="desc-pre-proc")
    with pytest.raises(BIDSInputError, match="Malformed"):
        discover_surface_runs(tmp_path)
