"""Extract the upstream intra-subject kernel with a diagnostic iteration guard.

This does not generate a convergence fixture: the added guard reports whether
the unchanged upstream iteration also fails to converge on a captured state.
"""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "upstream/CBIG_MSHBM_estimate_group_priors.m").read_text()
intra = source.split("function Params = intra_subject_var", 1)[1].split("function Params = vmf_clustering", 1)[0]
intra = "function [Params, Diagnostic] = cbig_intra_diagnostic" + intra
intra = intra.replace("iter_intra = iter_intra + 1;", """iter_intra = iter_intra + 1;
    if iter_intra > 10000
        Diagnostic.converged = false;
        Diagnostic.iterations = iter_intra-1;
        Diagnostic.relative_sigma_change = mean(abs(Params.sigma-sigma_update)./Params.sigma);
        return;
    end""")
# Record before assigning sigma so the last observed residual is meaningful.
intra = intra.replace("    Params.sigma = sigma_update;", "    Diagnostic.relative_sigma_change = mean(abs(Params.sigma-sigma_update)./Params.sigma);\n    Params.sigma = sigma_update;")
intra = intra.replace("Diagnostic.relative_sigma_change = mean(abs(Params.sigma-sigma_update)./Params.sigma);\n        return;", "return;")
last = intra.rfind("end")
intra = intra[:last] + "Diagnostic.converged = true;\nDiagnostic.iterations = iter_intra;\n" + intra[last:]
helpers = "function out = Ad" + source.split("function out = Ad", 1)[1].split("function data = fetch_data", 1)[0]
helpers = helpers.replace("= fzero(@", "= octave_fzero(@")
(root / "runtime/cbig_intra_diagnostic.m").write_text(intra + helpers)
