"""Real command-line parsing and filesystem safety checks for BIDS."""

import json
import subprocess
import sys

import pytest

from pymshbm.cli.bids import main


def inputs(tmp_path):
    root = tmp_path / "input"
    folder = root / "sub-01" / "func"
    folder.mkdir(parents=True)
    for hemi in ("L", "R"):
        (folder / f"sub-01_task-rest_hemi-{hemi}_space-fsaverage6_bold.func.gii").touch()
    return root


@pytest.mark.parametrize("leading", [[], ["bids"]])
def test_dry_run_prints_manifest_without_output_or_assets(tmp_path, capsys, leading):
    root = inputs(tmp_path)
    output = tmp_path / "output"
    main([*leading, str(root), str(output), "--dry-run"])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["runs"][0]["subject"] == "01"
    assert manifest["runs"][0]["model_session"] == 1
    assert manifest["preprocessing"]["additional_denoising"] is False
    assert manifest["runs"][0]["lh"].endswith("hemi-L_space-fsaverage6_bold.func.gii")
    assert not output.exists()


def test_module_cli_supports_filters_and_dry_run(tmp_path):
    root = inputs(tmp_path)
    result = subprocess.run([sys.executable, "-m", "pymshbm.cli.bids", "bids",
                             str(root), str(tmp_path / "output"),
                             "--participant-label", "sub-01", "--task", "rest",
                             "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["runs"]) == 1


def test_assets_required_for_actual_execution(tmp_path, capsys):
    root = inputs(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as exc:
        main([str(root), str(output)])
    assert exc.value.code != 0
    assert "--assets-dir" in capsys.readouterr().err
    assert not output.exists()


def test_invalid_iteration_count_fails_without_writing(tmp_path, capsys):
    root = inputs(tmp_path)
    with pytest.raises(SystemExit):
        main([str(root), str(tmp_path / "output"), "--dry-run", "--max-iter", "0"])
    assert "positive" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_dry_run_preserves_existing_output(tmp_path, capsys):
    root = inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("untouched")
    main([str(root), str(output), "--dry-run"])
    capsys.readouterr()
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "untouched"


def test_cli_reports_input_problem_without_traceback(tmp_path):
    result = subprocess.run([sys.executable, "-m", "pymshbm.cli.bids", str(tmp_path),
                             str(tmp_path / "output"), "--dry-run"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "BOLD" in result.stderr
    assert "Traceback" not in result.stderr


def test_manifest_model_sessions_restart_for_each_subject(tmp_path, capsys):
    root = inputs(tmp_path)
    for subject in ("01", "02"):
        folder = root / f"sub-{subject}" / "ses-followup" / "func"
        folder.mkdir(parents=True)
        for hemi in ("L", "R"):
            (folder / f"sub-{subject}_ses-followup_task-rest_hemi-{hemi}_space-fsaverage6_bold.func.gii").touch()
    main([str(root), str(tmp_path / "output"), "--dry-run"])
    runs = json.loads(capsys.readouterr().out)["runs"]
    assert [(r["subject"], r["session"], r["model_session"]) for r in runs] == [
        ("01", None, 1), ("01", "followup", 2), ("02", "followup", 1)]


@pytest.mark.parametrize('enabled', [False, True])
def test_cli_forwards_explicit_zero_cortex_policy(tmp_path, monkeypatch, capsys, enabled):
    import pymshbm.pipeline.buckner as pipeline
    root = inputs(tmp_path)
    received = {}
    def run(runs, output, assets, **kwargs):
        received.update(kwargs)
        return output
    monkeypatch.setattr(pipeline, 'run_buckner_workflow', run)
    flags = ['--allow-zero-cortex'] if enabled else []
    main([str(root), str(tmp_path / 'output'), '--assets-dir', str(tmp_path / 'assets'), *flags])
    assert received['allow_zero_cortex'] is enabled
    capsys.readouterr()


def test_dry_run_records_zero_cortex_opt_in_without_claiming_coverage(tmp_path, capsys):
    root = inputs(tmp_path)
    main([str(root), str(tmp_path / 'output'), '--dry-run', '--allow-zero-cortex'])
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['allow_zero_cortex'] is True
    assert 'zero_cortex_count' not in manifest['runs'][0]
