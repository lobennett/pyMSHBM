"""Save a portable source-executed intra-subject boundary regression.

Inputs are the captured deterministic surface-smoke state and the result from
runtime/cbig_intra_diagnostic.m. This is an isolated stage comparison, not an
end-to-end surface oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

root = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = loadmat(args.input, simplify_cells=True)
    reference = loadmat(args.output, simplify_cells=True)
    arrays = {"input_" + k: np.asarray(v) for k, v in source["Params"].items()}
    arrays.update({"expected_" + k: np.asarray(v) for k, v in reference["Params"].items()})
    np.savez_compressed(root / "fixtures/cbig_intra_boundary.npz", **arrays)
    report = {
        "scope": "isolated intra_subject_var stage from deterministic surface smoke",
        "settings": source["setting_params"],
        "diagnostic": reference["Diagnostic"],
        "runtime": "GNU Octave 6.4.0",
        "input_state_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256((root / "fixtures/cbig_intra_boundary.npz").read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256((root / "runtime/cbig_intra_diagnostic.m").read_bytes()).hexdigest(),
        "source_changes": ["extracted intra_subject_var and helpers", "added diagnostic guard at10000 iterations (not reached)", "captured residual before assignment; converted loaded int settings to upstream double"],
    }
    (root / "fixtures/intra-boundary-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
