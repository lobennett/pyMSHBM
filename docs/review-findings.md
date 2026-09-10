# Independent review, 2026-09-10

Reviewed the new BIDS/asset/profile/orchestration flow, numerical changes,
packaging/CI, and executable reference adapters against the supplied diff and
working tree. Did not rerun the full suite or duplicate the full-size smoke
investigation. Both findings below were resolved and independently rechecked
in a targeted follow-up. Their original descriptions remain as audit history.

## Findings

1. **Resolved P2 — Partially nonfinite medial-wall columns change cortical profiles.**
   `src/pymshbm/core/buckner.py:32` replaces individual nonfinite samples with
   zero. CBIG correlation instead propagates a nonfinite sample through that
   vertex's mean/norm and clears its complete correlation column afterward.
   Because the cutoff is global, the artificial correlations can also change
   cortical binary entries. Targeted reproduction: `default_rng(5)`, two
   `(10, 10)` normal arrays, cortex masks true except the last vertex,
   `lh[0, -1] = NaN`, `seed_vertices=5`. Current output differs at two cortical
   entries from the output with the entire affected medial column set to NaN.
   **Fix:** clear complete columns containing any nonfinite samples before
   normalization, retain rejection of nonfinite cortical data, and add a
   partial-NaN regression to the executable profile comparison.

2. **Resolved P2 — Executed numerical evidence does not exercise production input precision.**
   `validation/generate_reference.py:63` constructs double input profiles;
   the production tensor is single. In particular,
   `core/group_priors.py:122` lets NumPy multiply single data and double
   directions in double before casting, while mixed MATLAB arithmetic can
   have different accumulation and conversion semantics. The existing
   float32/float64 test compares Python runs with each other. Furthermore,
   `validation/runtime/mtimesx.m:7` unconditionally stores page products in
   double. That adapter needs an explicit audit of the pinned mtimesx dtype
   contract before using it to certify single-input execution. This is a
   validation gap, not a claim that a measured production mismatch has
   already been established. **Fix:** execute at least one single-input
   fixture, verify the adapter's result class against upstream mtimesx,
   compare every parameter and unpermuted label, and state the supported
   precision scope in the convergence report.

## Resolution verification

- The profile implementation now clears complete nonfinite medial columns.
  The reported seed-5 case is retained as a passing regression; nonfinite
  cortical data still raises an error. Existing executed profile comparisons
  continue to pass.
- The reference generator now casts the `single` case to float32 before
  writing the upstream input MAT file. The fixture's stored data dtype is
  explicitly tested. Continuous parameters, labels and stopping iteration
  agree with the executed oracle within the declared tolerances.
- Independently inspected `validation/upstream/mtimesx_upstream.c`: its dense
  non-scalar mixed single/double branches convert the double operand to
  single before calling `FloatTimesFloat`. The revised adapter performs
  those conversions and allocates a single result. The Python `_profile_dot`
  helper now follows the same conversion order, including the tested
  cancellation case that distinguishes conversion before multiplication.
- Provenance now hashes the generated estimator, `mtimesx.m`,
  `octave_fzero.m`, and the pinned upstream mtimesx C source; checksum tests
  pass for all four kernel cases.
- Targeted independent command:
  `.venv/bin/pytest tests/test_core/test_buckner_profiles.py tests/test_reference_profiles.py tests/test_reference_oracle.py tests/test_core/test_group_priors.py::test_single_profile_dot_casts_double_directions_before_multiplication -q`
  produced **72 passed in 0.43 s**. This is validation of the reviewed fixes,
  not a rerun of the complete suite or native MATLAB execution.

## Review boundaries and integration checks

- No additional concrete load-bearing BIDS pairing, asset checksum, label
  extraction, or package metadata defect was established in the examined
  flow. Python 3.11 is consistently declared in project metadata and lint
  configuration; CI covers 3.11, 3.13 and 3.14. Local wheel execution and CI
  outcomes are owned by integration and are not independently certified here.
- At initial review time `docs/reference-validation.md`, `docs/scientific-contract.md`
  and `validation/README.md` were absent despite references from source and
  the numerical audit. Integration was notified while it was writing docs;
  the scientific contract was present and reviewed at follow-up, while final
  reference documentation remains owned by integration.
- The recorded synthetic oracle agreement supports the extracted kernels,
  supplied arrays and listed tolerances. It does not establish equivalence
  to real Buckner data, native full MATLAB execution, MRI I/O, denoising,
  fsaverage asset identity across FreeSurfer installations, or biological
  validity. The audit already states most of these limitations correctly.
- The first full-size synthetic S1/T2 attempt encountered nonconvergence.
  The local `smoke-variable/smoke-report.json` subsequently records a
  successful 81,924-vertex, two-run synthetic workflow at `--max-iter 1`
  in 10.20 seconds using a fixture with better-estimable variation. This
  remains synthetic execution evidence, not real-data equivalence or an
  independently verified five-iteration run. The independent reviewer did
  not rerun that expensive workflow.
