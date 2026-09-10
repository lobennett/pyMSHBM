"""Compare Python group-prior EM to the executed CBIG fixture; emit JSON."""

import json
import argparse
from pathlib import Path
import platform
import time

import numpy as np

from pymshbm.core.group_priors import estimate_group_priors

ROOT = Path(__file__).resolve().parent
TOLERANCES = {
    "mu": (2e-5, 1e-4), "s_psi": (2e-5, 1e-4),
    "s_t_nu": (2e-5, 1e-4), "theta": (1e-4, 1e-4),
    "s_lambda": (1e-4, 1e-4),
    "epsil": (0.05, 2e-4), "sigma": (0.05, 2e-4),
    "kappa": (0.05, 2e-4),
    "cost_em": (0.1, 5e-6), "cost_intra": (0.1, 5e-6),
    "cost_inter": (0.1, 5e-6), "record": (0.1, 5e-6),
}


def compare(case="complete"):
    suffix = "" if case == "complete" else "_" + case
    fixture = np.load(ROOT / f"fixtures/cbig_synthetic{suffix}.npz")
    data, g_mu = fixture["data"], fixture["g_mu"]
    settings = {
        "num_sub": data.shape[2], "num_session": data.shape[3],
        "num_clusters": g_mu.shape[1], "dim": data.shape[1] - 1,
        "ini_concentration": 500, "epsilon": 1e-4,
        "conv_th": 1e-5, "max_iter": 5,
    }
    start = time.monotonic()
    params = estimate_group_priors(data, g_mu, settings)
    report = {
        "scope": "synthetic normalized group-prior EM kernel only",
        "case": case,
        "python": platform.python_version(), "numpy": np.__version__,
        "elapsed_seconds": time.monotonic() - start,
        "parameters": {},
    }
    for name, (atol, rtol) in TOLERANCES.items():
        actual, expected = np.asarray(getattr(params, name)), fixture[name]
        same_shape = actual.shape == expected.shape
        delta = np.abs(actual.astype(float) - expected.astype(float)) if same_shape else np.array([np.inf])
        if same_shape:
            delta = np.where(np.isnan(actual) & np.isnan(expected), 0, delta)
        report["parameters"][name] = {
            "atol": atol, "rtol": rtol, "max_absolute_error": float(delta.max()),
            "max_relative_error": float(np.nanmax(delta / np.maximum(np.abs(expected), atol))) if same_shape else None,
            "passed": bool(same_shape and np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)),
        }
    labels = np.argmax(params.s_lambda, axis=1) + 1
    labels[np.all(params.s_lambda == 0, axis=1)] = 0
    report["unpermuted_label_mismatches"] = int(np.count_nonzero(labels != fixture["labels"]))
    report["iteration"] = {"python": int(params.iter_inter), "octave": int(fixture["iter_inter"])}
    report["passed"] = (
        all(value["passed"] for value in report["parameters"].values())
        and report["unpermuted_label_mismatches"] == 0
        and report["iteration"]["python"] == report["iteration"]["octave"]
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("complete", "missing", "soft", "single"), default="complete")
    args = parser.parse_args()
    report = compare(args.case)
    suffix = "" if args.case == "complete" else "_" + args.case
    (ROOT / f"fixtures/convergence-report{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
