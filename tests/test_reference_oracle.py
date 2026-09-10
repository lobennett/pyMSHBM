"""Numerical checks against an executed, independently sourced Octave oracle."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "validation"
spec = importlib.util.spec_from_file_location("compare_reference", ROOT / "compare_reference.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.fixture(scope="module", params=("complete", "missing", "soft", "single"))
def report(request):
    return comparison.compare(request.param)


@pytest.mark.parametrize("parameter", comparison.TOLERANCES)
def test_continuous_parameter_matches_executed_cbig(report, parameter):
    assert report["parameters"][parameter]["passed"], report["parameters"][parameter]


def test_unpermuted_labels_and_stopping_iteration_match(report):
    assert report["unpermuted_label_mismatches"] == 0
    assert report["iteration"]["python"] == report["iteration"]["octave"]


@pytest.mark.parametrize("suffix", ("", "_missing", "_soft", "_single"))
def test_reference_source_and_fixture_are_pinned(suffix):
    provenance = json.loads((ROOT / f"fixtures/oracle-provenance{suffix}.json").read_text())
    for name, expected in provenance["upstream_files_sha256"].items():
        assert hashlib.sha256((ROOT / "upstream" / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT / f"fixtures/cbig_synthetic{suffix}.npz").read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert hashlib.sha256((ROOT / "runtime/cbig_kernel.m").read_bytes()).hexdigest() == provenance["adapter_sha256"]
    for name, expected in provenance["runtime_files_sha256"].items():
        assert hashlib.sha256((ROOT / "runtime" / name).read_bytes()).hexdigest() == expected


def test_single_case_really_uses_single_input():
    import numpy as np
    with np.load(ROOT / "fixtures/cbig_synthetic_single.npz") as fixture:
        assert fixture["data"].dtype == np.float32


def test_soft_case_exercises_uncertain_posteriors():
    import numpy as np
    with np.load(ROOT / "fixtures/cbig_synthetic_soft.npz") as fixture:
        assert np.count_nonzero((fixture["s_lambda"] > 0.01) & (fixture["s_lambda"] < 0.99)) > 100
