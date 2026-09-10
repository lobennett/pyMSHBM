# BIDS input implementation evidence

Implemented `io/bids.py` and `cli/bids.py` with isolated filesystem fixtures.
The discovery tests were introduced before the modules existed; the first
test run failed with missing-module errors. An additional malformed-entity
test failed because discovery incorrectly accepted `desc-pre-proc`, then
passed after enforcing ASCII alphanumeric entity labels.

The supported naming is the current fMRIPrep functional derivative convention:
`sub-<label>_[specifiers]_hemi-[LR]_space-<space>_bold.func.gii`.
Its documentation identifies fsaverage6 as the approximately 41k-vertex
surface. Source checked 2026-09-10:
[fMRIPrep output documentation](https://fmriprep.org/en/stable/outputs.html#functional-derivatives).
Canonical `space-fsaverage_den-41k` is also accepted. Actual mesh dimensions
are checked by the workflow, rather than inferred as valid from names alone.

Discovery searches only subject `func` and subject/session `func` folders,
under a supplied derivative directory, BIDS `derivatives/fmriprep`, or the
legacy `fmriprep` container. It does not recursively mix unrelated derivative
pipelines. Filters accept participant, session and run labels with or without
their prefixes; the task defaults to `rest`.

All filename entities other than `hemi` must match exactly for a pair.
Duplicate files and alternative echo, reconstruction, description, space or
density exports for one physical run are errors. Acquisition and direction
entities distinguish physical runs. This avoids treating multiple echoes or
multiple processed versions of one run as independent model sessions.
The manifest preserves all entities and maps each pair to a within-subject
model-session index. Labels retain their original zero padding.

Missing hemispheres, inconsistent subject/session folders, malformed entities,
missing requested labels, empty inputs, raw/volume-only data, CIFTI-only data,
and incompatible surface meshes produce actionable input errors. Additional
incompatible exports are ignored when valid fsaverage6 pairs are present.

The CLI accepts an optional leading `bids`, allows dry runs without reference
assets, prints a JSON manifest, and does not create or change output files in
dry-run mode. Real execution lazily imports `run_buckner_workflow` and requires
`--assets-dir`. Iterations must be positive. No confound cleaning or resampling
is performed. fMRIPrep-derived inputs therefore do not establish equivalence
to the original Buckner denoising pipeline.

Validation: `.venv/bin/pytest tests/test_io/test_bids.py
tests/test_cli/test_bids_cli.py -q` — **40 passed**. Tests exercise discovery,
CLI subprocesses, input errors, model-session ordering, and output preservation.
Full fitting and built-wheel entry-point checks belong to integration validation.
