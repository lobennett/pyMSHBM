"""Compare the near-unit resultant boundary to executed upstream intra updates."""
import hashlib
import json
from pathlib import Path

import numpy as np

from pymshbm.core.group_priors import intra_subject_var
from pymshbm.types import MSHBMParams

ROOT = Path(__file__).resolve().parents[1] / "validation"


def test_quantized_boundary_matches_upstream_intra_stage():
    report = json.loads((ROOT / "fixtures/intra-boundary-report.json").read_text())
    path = ROOT / "fixtures/cbig_intra_boundary.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["fixture_sha256"]
    assert report["diagnostic"]["converged"]
    assert report["diagnostic"]["iterations"] == 2
    with np.load(path) as fixture:
        fields = {k[6:]: fixture[k] for k in fixture.files if k.startswith("input_")}
        # MATLAB saves trailing singleton dimensions implicitly.
        fields["s_psi"] = fields["s_psi"][..., None]
        fields["s_t_nu"] = fields["s_t_nu"][..., None]
        fields["iter_inter"] = int(fields["iter_inter"])
        fields["theta"] = np.empty((0, 15))
        params = intra_subject_var(MSHBMParams(**fields), report["settings"])
        np.testing.assert_allclose(params.s_psi[:, :, 0], fixture["expected_s_psi"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(params.sigma, fixture["expected_sigma"], atol=1e-3, rtol=1e-6)
