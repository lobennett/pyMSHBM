# Real-data upstream comparison harness

`validation/real_reference.py` executes the pinned CBIG numerical source on
the same selected surface BOLD used by the production CLI. This harness does
not fetch, denoise, or resample data. Its inputs must already be paired
fsaverage6 BIDS GIFTIs and the checksum-verified DU15NET/fsaverage assets.
Dataset acquisition and conversion provenance must be recorded separately.

Complete cortical coverage is the default in both production and harness
preparation. The distinct masked-coverage protocol requires
`--allow-zero-cortex` in **both** `pymshbm bids` and the harness `prepare` stage:

```sh
pymshbm bids INPUT PYTHON_OUTPUT --assets-dir ASSETS \
  --participant-label MSC01 MSC02 --session-label func01 func02 \
  --max-iter 5 --allow-zero-cortex
python validation/real_reference.py prepare \
  --bids-input INPUT --assets ASSETS --workdir WORK \
  --participant-label MSC01 MSC02 --session-label func01 func02 \
  --max-iter 5 --allow-zero-cortex
```

The flag permits only entirely zero nonseed cortical time courses. Nonzero
constants and nonfinite cortical data remain errors, as do unusable cortical
seed vertices after preparation. The global correlation cutoff must remain
positive. It does not fill missing values. The manifest records the flag and
per-run, per-hemisphere zero cortical indices; `check-profiles` uses that saved
flag and `collect` requires the production provenance setting to agree.
Historical manifests that omit the flag mean `False`. Later stages do not
change the prepared policy through an added command-line flag.

Literal zero profiles differ from a NaN missing session: they remain in the
source's session averaging and cost/concentration calculations. The reference
profile and kernel equations are unchanged. A passing masked case would be
evidence conditional on that explicit zero-observation policy, not validation
of complete coverage or statistically neutral missingness.

The strict-case stages below are independently callable. For a masked case,
add the flag at preparation as shown above; the remaining stages are identical:

```sh
python validation/real_reference.py prepare \
  --bids-input INPUT --assets ASSETS --workdir WORK \
  --participant-label MSC01 MSC02 --session-label func01 func02 --max-iter 5
python validation/real_reference.py run-profiles \
  --workdir WORK --podman-container pymshbm-oracle
python validation/real_reference.py check-profiles --workdir WORK
python validation/real_reference.py run-kernel \
  --workdir WORK --podman-container pymshbm-oracle
python validation/real_reference.py collect \
  --workdir WORK --python-output PYTHON_OUTPUT
```

Use the actual session labels in the prepared BIDS directory. Omit selection
flags to use every matching resting run. Subject and model-session ordering
comes from the same BIDS discovery routine as production; the manifest records
every selected input hash. The comparison checks this ordering, input hashes,
asset provenance, and iteration setting against the completed Python output.
Production provenance also records SHA-256 hashes of every installed pymshbm
Python source file, using paths relative to the package so wheel installations
and checkouts can be compared. The workflow checks those hashes again before
publication and rejects source changes during fitting. Preparation snapshots
the same source hashes; later reference stages require them to remain unchanged.

Preparation requires an empty work directory. The container expects the
repository's parent mounted at `/work`, with this checkout at `/work/pyMSHBM`;
WORK must lie beneath that mounted parent. Omit `--podman-container` to use
local Octave. The run stages write logs directly to WORK, and completed MATs
are renamed atomically. Rerunning a stage keeps completed outputs and retries
unfinished runs. To deliberately rerun completed reference data, use a fresh
work directory. The kernel itself resumes only at completed-fit granularity.

Each raw BOLD run is transported separately in MAT v5, as double arrays.
Reference profiles are computed in Octave using the original `CBIG_corr`,
global top-10% cutoff, and profile-normalization source slices. The numerical
adapters are generated into WORK/runtime; existing synthetic adapters and
fixtures are never rewritten. Binary values are stored as uint8, normalized
profiles as float32, and the mean binary profile and initial centroids use
double precision. The reference mean and centroids are computed only from
reference binary profiles. The group wrapper loads reference normalized
profiles into one single-precision NxDxSxT array and calls the existing pinned
`cbig_kernel.m`. The shared GIFTI reader and mask/label loader are infrastructure
adaptations; MRI/FreeSurfer input readers themselves are outside this test.

The manifest records pinned revisions, hashes of upstream and runtime files,
the evaluator `real_reference.py`, raw input MATs, configuration, and every
adaptation. Existing synthetic
provenance is verified before preparation. All subsequent stages reject
changed runtime/configuration hashes. Comparison reports hash the observed
reference profile and parameter MATs. The profile report is bound to the
prepared manifest hash, Python source hashes, each reference profile MAT hash,
and the reference centroid MAT hash. Before running the kernel or collecting
results, the harness verifies these bindings. Changed arrays or a report from
another prepared case fail the binding checks. The kernel can still run for
diagnostic purposes when a correctly bound profile comparison has numerical
mismatches. Keep WORK and the production output
together for reproducibility; these large human-data transport files should
not be committed as portable synthetic fixtures.

`check-profiles` recomputes Python profiles because the production workflow
deletes its scratch tensor. It compares binary profiles exactly, then reports
raw normalized-tensor and centroid differences, including exact mismatch
counts, without rounding either array. Normalized profiles and centroids use
the existing direction tolerance (absolute 2e-5, relative 1e-4). All group
parameter tolerances match `compare_reference.py`: posterior/theta absolute
and relative 1e-4, concentration absolute .05 plus relative 2e-4, and cost
absolute .1 plus relative 5e-6. The stabilized maximum relative-error statistic
divides by `max(abs(reference), atol)`; tolerance decisions use the usual
`atol + rtol * abs(reference)` per element. Nonfinite real-data output values
are reported and fail the comparison.
Evaluation reads the tolerances recorded in the prepared manifest; changes to
the evaluator itself are rejected by its recorded source hash.

The production `Params_Final.mat` retains `cost_em`, `cost_intra`, and
`cost_inter` alongside the cost history, so collection reads the completed
production fit directly from `PYTHON_OUTPUT/model/priors/Params_Final.mat`.
For older runs that captured additional fitted fields separately, the optional
`--python-params FULL_PARAMS.mat` accepts a `Params` MAT structure, using
`Record` or `record` for its history. Its shared fields must match the production
`Params_Final.mat` exactly, binding this capture to that fit. Missing costs
explicitly fail the corresponding checks. No reference array is computed by
the Python fit.

`comparison.json` contains parameter errors, per-subject unpermuted label
agreement and per-network Dice, outer iteration counts, convergence and
iteration-limit flags, and reference log counts for inner iterations and
warnings. Collection reads every published subject/hemisphere label GIFTI
and compares its stored labels exactly with both its parameter-derived labels
and the reference labels. File hashes, hemisphere metadata, label intent, and
each comparison result are recorded; missing, corrupted, swapped, or unequal
maps fail collection. The production source hashes must match the prepared
and profile-comparison sources. Absent networks have null Dice. A five-iteration limit is not called
convergence unless the unchanged relative-cost criterion is also met. Labels
and stopping decisions must match exactly. A failed comparison exits 1 and
retains all evidence; no tolerance is selected from the observed result.

For N=81,924, D=1,175, S=2, T=2, the single reference tensor alone is about
1.54 GB. Each per-run profile MAT is about 481 MB; raw inputs, correlation
sorting, the mean profile, and EM intermediates require additional RAM and
disk. Run large reference/Python stages sequentially when sharing a constrained
machine. Small automated tests cover reporting, singleton shape restoration,
stopping decisions, and source extraction; an Octave wrapper smoke can use
the existing synthetic fixtures without downloading human data.

Preparation and model usability have different coverage meanings. OpenNeuro
per-run preparation NPZs retain `observed_mask`, `imputed_mask`, and
`unresolved_zero_mask`, alongside original support, distances, and imputation
sources. Production subject NPZs retain `usable_session_count`,
`usable_in_any_session`, and `full_session_coverage` within `cortex_mask`.
Usable input includes bounded-imputed series, so these subject arrays must not
be substituted for the original-observation masks. After collection, report
agreement over the complete cortex and common originally observed support:

```sh
python validation/compare_coverage_support.py \
  --input INPUT --workdir WORK --python-output PYTHON_OUTPUT \
  --assets ASSETS --output WORK/coverage-support.json
python validation/plot_real_comparison.py \
  --input INPUT --workdir WORK --python-output PYTHON_OUTPUT \
  --assets ASSETS --surfaces SURFACES --output FIGURES
```

See the [full OpenNeuro commands and eligibility record](openneuro-reproduction.md#separate-masked-coverage-case).

Passing this comparison establishes numerical agreement for the selected
real-data case under these shared preprocessing and mesh inputs. It does not
validate the scientific suitability of preprocessing or establish equivalence
of complete native MATLAB and Python workflows.
