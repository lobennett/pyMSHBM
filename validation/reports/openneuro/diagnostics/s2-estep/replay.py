"""Replay the saved E-step inputs without pyMSHBM or a whole fit.
Requires NumPy and SciPy. Run after s2_estep_witness_tail in Octave.
"""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
import python_estep_frozen as frozen
root = Path(__file__).resolve().parent
inputs = loadmat(root / "input.mat", simplify_cells=True)
expected = loadmat(root / "octave-tail.mat", simplify_cells=True)["output"]
report = {}
for name, target in expected.items():
    state = inputs[name]
    nu = np.array(state["s_t_nu"], order="C")
    theta = np.array(state["theta"], order="C")
    data = np.array(inputs["series"], order="C")
    kappa = np.atleast_1d(state["kappa"])
    log_c = frozen.cdln(kappa, data.shape[1] - 1)
    entry = {}
    for common_products in (False, True):
        actual = np.empty_like(target["posterior"])
        original = frozen._profile_dot
        try:
            for subject in range(data.shape[2]):
                if common_products:
                    products = iter([target["raw_dot"][:, :, subject, t].copy() for t in range(data.shape[3])])
                    frozen._profile_dot = lambda *_: next(products)
                actual[:, :, subject], _ = frozen._e_step_subject(data[:, :, subject, :],
                    nu[:, :, :, subject], kappa, log_c, theta)
        finally:
            frozen._profile_dot = original
        delta = np.abs(actual - target["posterior"])
        entry["upstream_products" if common_products else "native_products"] = {
            "maximum_absolute_posterior_error": float(delta.max()),
            "posterior_tolerance_violations": int(np.count_nonzero(delta > 1e-4 + 1e-4 * np.abs(target["posterior"])))
        }
    report[name] = entry
print(json.dumps(report, indent=2))
