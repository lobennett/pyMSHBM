"""Emit measured errors for the independently executed profile kernels."""
import json
from pathlib import Path

import numpy as np

from pymshbm.core.buckner import binary_profiles, centroids_from_labels, normalize_profiles

root = Path(__file__).resolve().parent
with np.load(root / "fixtures/cbig_profiles.npz") as f:
    binary = binary_profiles(f["lh"], f["rh"], f["lh_cortex"], f["rh_cortex"], seed_vertices=int(f["seed_vertices"]))
    normalized = normalize_profiles(f["profiles"], f["profile_cortex"])
    centroids = centroids_from_labels(f["avg_profile"], f["labels"], num_clusters=3)
    report = {
        "scope": "synthetic array profile kernels only",
        "binary_mismatches": int(np.count_nonzero(binary != f["binary"])),
        "normalization_max_absolute_error": float(np.max(np.abs(normalized - f["normalized"]))),
        "centroids_max_absolute_error": float(np.max(np.abs(centroids - f["centroids"]))),
        "normalization_tolerance": {"atol": 2e-7, "rtol": 2e-6},
        "centroid_tolerance": {"atol": 1e-12, "rtol": 1e-12},
        "passed": bool(np.array_equal(binary, f["binary"]) and np.allclose(normalized, f["normalized"], atol=2e-7, rtol=2e-6) and np.allclose(centroids, f["centroids"], atol=1e-12, rtol=1e-12)),
    }
(root / "fixtures/profile-convergence-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
