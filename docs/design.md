# pyMSHBM: BIDS inputs and upstream numerical convergence

## Scope and authorization

Create Logan Bennett's private `pyMSHBM` repository from Russell Poldrack's
implementation, preserving its history and MIT attribution. The user authorized
implementation, auditing, testing, and repository creation. Use GPT-6 Astra.

## Scientific contract

The reference is executable Buckner PrecisionNetworkMapping at
`988ec48cf5453c6009a6f82dbddf5d5b99eda3cd`, together with a pinned CBIG revision.
Its script samples fsaverage3-indexed cortex vertices from fsaverage6 BOLD,
computes Pearson correlations, thresholds globally at the top 10%, averages
profiles, initializes centroids from DU15NET labels, mean-centers and unit
normalizes profiles, estimates group parameters for five outer iterations,
and extracts labels from group-estimation posterior probabilities. The script
does not apply an additional individual MRF fit. README descriptions differ
from these executable steps; document this explicitly.

Numerical equivalence is a measured property, not a consequence of passing
Python tests. Compare deterministic inputs against executed upstream MATLAB
code using Octave where possible, with adapters limited to I/O and unavailable
MATLAB infrastructure. Record source checksums, tolerances, fixtures, runtime,
continuous parameter errors and unpermuted labels. Do not claim full biological
or preprocessing equivalence from synthetic comparisons.

## Input/output contract

`pymshbm bids INPUT OUTPUT --assets-dir ASSETS` discovers a direct derivative
root or BIDS `derivatives/fmriprep`, pairs hemisphere files by BIDS entities,
and supports fsaverage6 GIFTI (also canonical fsaverage density 41k naming).
Each paired run is one model session, as in Buckner's wrapper. A dry-run gives
a reviewable manifest. Raw-only BIDS and fsLR CIFTI must produce actionable
errors, without silently resampling or preprocessing. Rest is the default
task. Confound cleaning is an explicit option with complete provenance;
fMRIPrep preprocessing alone does not duplicate Buckner denoising.

The production pipeline uses `run_buckner_workflow(runs, output_dir,
assets_dir, *, max_iter=5)`; run records have subject, session, task, run,
lh and rh fields, and paths are pathlib Paths. It returns an output Path.
The root worker owns this orchestration and assets. BIDS/CLI worker owns
discovery, run dataclass, CLI, input tests, and can call this API lazily.

Output includes BIDS derivative dataset_description, label GIFTIs, posterior
and parameter files, run/subject mapping, hashes of inputs and reference
assets, versions and preprocessing choices. Fail before expensive fitting
for incompatible shapes, hemispheres, missing assets, nonfinite data or
invalid labels. Never silently overwrite results.

## Architecture and ownership

1. Numerical engine: audit and correct `core/group_priors.py` and `math/vmf.py`;
   regression tests and audit findings. Preserve API for legacy callers.
2. Inputs: `io/bids.py`, `cli/bids.py`, BIDS discovery and CLI tests.
3. Reference: `validation/`, upstream Octave adapter, reproducible fixture
   generator, convergence report and source pinning. Does not edit engine.
4. Integration: `pipeline/buckner.py`, canonical profiles/init, asset reader,
   pyproject/uv lock/CI, README, repository publication, full integration tests.

Use independent Astra workers for tasks 1–3 while integration proceeds locally.
All edits occur in the isolated clone on `feat/bids-upstream-convergence`.

## Validation and release

Run inherited tests before changes; add failing tests for corrected numerical
behavior and BIDS integration. Run actual upstream comparisons, full suite,
wheel build and uv installation/CLI from the built wheel. Require independent
review. Any unavailable human dataset or full MATLAB production comparison
remains explicitly unverified. Scientific assets from Buckner are fetched
from the pinned source or user supplied, not relicensed as our MIT code.
