# Executed OpenNeuro comparison artifacts

Read the [scientific report](../../../docs/openneuro-validation.md) and
[reproduction instructions](../../../docs/openneuro-reproduction.md) first.
**All three complete numerical comparisons below fail the original overall
acceptance criteria.** Exact maps in one case or high agreement in another do
not replace the failed parameter/label criteria.

| Case | Labels | Overall result |
| --- | --- | --- |
| [MSC01 before correction](msc01-before-normalization-fix/comparison.json) | 85 differences | Fail |
| [MSC01 after correction](msc01-after-normalization-fix/comparison.json) | Exact | Fail: continuous parameters |
| [MSC01/MSC02, explicit unresolved-zero protocol](msc0102-masked/comparison.json) | 1 difference | Fail: label, spatial prior and posterior |

Each case contains the original comparison, input/source manifest, independently
executed profile comparison, reference log, Python provenance, preparation
manifest, parameter-derived labels, maps and an artifact checksum index.
`python-params.npz` and `reference-params.npz` are lossless compressed numeric
exports of the actual MAT `Params` structures, loaded using
`scipy.io.loadmat(..., simplify_cells=True)`. Their source MAT hashes, array
shapes, dtypes and export hashes are recorded in `parameter-archives.json`.
Use `numpy.load(path, allow_pickle=False)` to inspect them. Comparisons were
performed against the original MAT files, not reconstructed or Python-generated
reference expectations.

The masked case also includes the actual published label GIFTIs in `labels/`,
original coverage masks in `input-coverage/`, per-subject usable-session counts
in `model-coverage/`, and the separate [observed-support comparison](msc0102-masked/coverage-support.json).
Observed support means valid resampling support before boundary imputation;
it does not imply equivalence of the source acquisition or preprocessing.

[Strict eligibility failures](eligibility/summary.json) retain every attempted
participant, including the explicitly reconstructed MSC02 record. Historical
source snapshots preserve the original and corrected strict implementations;
their stored source hashes identify those fits independently of package version
metadata. The [runtime inventory](reference-runtime.json) records the executed
Octave/OpenBLAS container; native MATLAB was not executed.

Raw BOLD, intermediate full-size profiles and original MAT tensors remain
outside git. The pinned public dataset and scripts reproduce those inputs.
Absolute paths in original provenance are retained as recorded; use the
reproduction instructions to generate fresh manifests on another machine.
