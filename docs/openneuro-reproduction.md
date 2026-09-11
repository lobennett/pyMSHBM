# Reproduce the OpenNeuro comparison

This benchmark uses the Midnight Scan Club dataset from OpenNeuro,
accession **ds000224, version 1.0.4**. Cite Gordon et al., *Precision Functional
Mapping of Individual Human Brains*, Neuron 95, 791–807 (2017), and the dataset
[DOI](https://doi.org/10.18112/openneuro.ds000224.v1.0.4).
The snapshot is CC0. Its [README](https://github.com/OpenNeuroDatasets/ds000224/blob/1.0.4/README)
describes the authors' denoised, motion-censored surface derivatives.
The distributed CIFTIs used here still contain all 818 frames; their supplied
temporal masks must be applied. The preparation script checks this explicitly.

The [plan and amendments](openneuro-validation-plan.md) record selection,
eligibility failures, fixed tolerances and the boundary-imputation policy.
The [comparison harness](real-reference-harness.md) explains independence and
the limits of its Octave infrastructure adapters. This is a computational
comparison conditional on shared prepared inputs, not a replication of the
MSC authors' parcellation method or a native MATLAB workflow validation.

## Environment and assets

From a checkout of this repository:

```sh
uv sync --extra dev --extra validation
```

The current package version is **0.2.2**, including the explicit zero-cortex
opt-in. Its [installed-wheel verification](../validation/reports/openneuro/msc0102-masked/wheel-install-verification.json)
reproduces every development-fit parameter and Python source hash for the
completed MSC01/MSC02 case. The earlier
[0.2.1 wheel check](../validation/reports/openneuro/wheel-install-verification.json)
remains the historical result for corrected strict MSC01.

Install git-annex, Podman (or a local Octave and Workbench), and prepare the
checksum-verified DU15NET/fsaverage6 assets as described in the main README.
The commands below use `.local-data/assets`. Real data, scratch arrays and
large parameter files are intentionally ignored by git; allow tens of GB
when retaining multiple attempted/pre-correction runs.

The executed reference used Ubuntu 22.04 amd64, Octave 6.4.0-2, OpenBLAS
0.3.20+ds-1 and Workbench 1.5.0-2; the [runtime inventory](../validation/reports/openneuro/reference-runtime.json)
records the executed environment. Its base image digest is pinned in
`validation/Containerfile.real-reference`. Other apt dependencies are not
a hermetic snapshot; preserve actual installed versions when rebuilding.
On an Apple Silicon host this container is emulated; recorded run times
are observations, not a controlled speed comparison with native Python.

```sh
podman build --platform linux/amd64 \
  -f validation/Containerfile.real-reference -t pymshbm-real-reference .
podman run --name pymshbm-oracle -d \
  -v "$(pwd)/..:/work" pymshbm-real-reference
```

The harness expects this checkout's directory to be named `pyMSHBM` beneath
the mounted parent. An existing running reference container can be reused.
Avoid running simultaneous full-size reference fits in the 8 GB container.

Obtain the pinned CBIG mesh assets (a partial clone limits payload downloads):

```sh
git clone --filter=blob:none --no-checkout https://github.com/ThomasYeoLab/CBIG.git ../mshbm-cbig
git -C ../mshbm-cbig sparse-checkout init --cone
git -C ../mshbm-cbig sparse-checkout set data/templates/surface/standard_mesh_atlases_20170508
git -C ../mshbm-cbig checkout 35b5664bec8822e2f77da5e090e96f91d0095be6
```

## Data preparation

The single-participant diagnostic can be reproduced with:

```sh
uv run python validation/fetch_msc.py .local-data/openneuro/ds000224 --participant-label MSC01
uv run python validation/prepare_msc.py \
  --dataset .local-data/openneuro/ds000224 --cbig ../mshbm-cbig \
  --assets .local-data/assets \
  --workdir .local-data/openneuro/preparation-msc01-reproduction \
  --output .local-data/openneuro/input-msc01-reproduction \
  --podman-container pymshbm-oracle --boundary-fill-mm 5 --participant-label MSC01
```

For a cohort, pass every selected identifier to both commands. Each participant
contributes func01 and func02. Selection must follow the published eligibility
record; adding a participant changes the jointly fitted model and its outputs.
Do not choose participants by numerical agreement.

Every preparation needs a **new work directory and output directory**. Failed
attempts keep their logs and intermediate coverage data. A distance-support
failure writes current subject/hemisphere QC and unresolved masks before
stopping; it must not be treated as a successfully prepared cohort.

Preparation performs these shared steps:

1. Convert CIFTI-1 to CIFTI-2 using Workbench, preserving data values.
2. Apply the supplied temporal mask once, recording retained original indices.
3. Resample each participant's midthickness geometry and cortical time series
   from fsLR32k to fsaverage6 using registered HCP/CBIG spheres and
   Workbench ADAP_BARY_AREA with individual anatomical areas and source ROI.
4. Explicitly impute unsupported target-cortex boundary vertices from supported
   target cortex within the fixed 5 mm surface-distance rule. Preserve masks,
   source vertices, distances and counts of affected fsaverage3 seed vertices.
5. In the default strict mode, zero noncortex and verify finite, nonconstant cortical signals. Write paired
   BIDS derivative GIFTIs, sidecars and a preparation manifest with input hashes.

It performs no additional nuisance regression, temporal filtering, smoothing
of measured vertices, or raw-BIDS preprocessing. Imputed vertices must not be
described as directly observed cortex.

## Fit and compare

Substitute the prepared INPUT and new OUTPUT/WORK directories. Keep max_iter=5
for the primary comparison. The pipeline treats each physical run as a model
session, preserving the selected cohort and its sorted order.

```sh
uv run pymshbm bids INPUT OUTPUT --assets-dir .local-data/assets --max-iter 5 --verbose
uv run python validation/real_reference.py prepare \
  --bids-input INPUT --assets .local-data/assets --workdir WORK --max-iter 5
uv run python validation/real_reference.py run-profiles --workdir WORK --podman-container pymshbm-oracle
uv run python validation/real_reference.py check-profiles --workdir WORK
uv run python validation/real_reference.py run-kernel --workdir WORK --podman-container pymshbm-oracle
uv run python validation/real_reference.py collect --workdir WORK --python-output OUTPUT
```

`comparison.json` records pass/fail criteria and raw errors. A failing
comparison exits nonzero while retaining the evidence. Source hashes bind
the prepared comparison, actual production fit and evaluator. After changing
the Python implementation, create a new WORK and OUTPUT; preserve previous
results rather than overwriting or rebinding them to new code.

## Separate masked-coverage case

The [eligibility summary](../validation/reports/openneuro/eligibility/summary.json)
retains the strict fixed-5-mm failures for MSC02 through MSC10. The original
complete-cortex MSC01/MSC02 cohort was not fitted. The
[separately declared follow-up](openneuro-validation-plan.md#separately-declared-unresolved-zero-follow-up)
uses MSC01/MSC02 func01/func02 with the same radius and original tolerances,
but leaves unresolved nonseed cortex identically zero. It is a different
coverage policy; it does not replace the strict failures.

Preparation of all four follow-up runs is complete. MSC01 retains 523 and
717 frames; MSC02 retains 531 and 793 frames. Each MSC01 run has 373 left and
306 right bounded-imputed vertices. Each MSC02 run has 373 left and 305 right
bounded-imputed vertices, with right-hemisphere vertex **1659** unresolved and
zero in both sessions (zero-based hemisphere-local index). Each participant
has eight imputed cortical seeds per run, all supported within 5 mm, and no
unresolved seeds. The [completed comparison](../validation/reports/openneuro/msc0102-masked/comparison.json)
has exact profiles and matching convergence at iteration three, but one label
mismatch and spatial-prior/posterior tolerance failures give **overall failure**.
See the [results and coverage-specific analysis](openneuro-validation.md#completed-multi-participant-unresolved-zero-follow-up).

Use fresh directories for reproduction. These commands select the same four
runs and declare the opt-in at preparation, Python fitting, and reference
preparation:

```sh
MSC_INPUT=.local-data/openneuro/input-msc0102-masked-reproduction
MSC_WORK=.local-data/openneuro/reference-msc0102-masked-reproduction
MSC_OUTPUT=.local-data/openneuro/python-msc0102-masked-reproduction
MSC_FIGURES=.local-data/openneuro/figures-msc0102-masked-reproduction

uv run python validation/fetch_msc.py .local-data/openneuro/ds000224 \
  --participant-label MSC01 MSC02
uv run python validation/prepare_msc.py \
  --dataset .local-data/openneuro/ds000224 --cbig ../mshbm-cbig \
  --assets .local-data/assets \
  --workdir .local-data/openneuro/preparation-msc0102-masked-reproduction \
  --output "$MSC_INPUT" --podman-container pymshbm-oracle \
  --participant-label MSC01 MSC02 --boundary-fill-mm 5 --allow-zero-cortex

uv run pymshbm bids "$MSC_INPUT" "$MSC_OUTPUT" \
  --assets-dir .local-data/assets --participant-label MSC01 MSC02 \
  --session-label func01 func02 --max-iter 5 --allow-zero-cortex --verbose
uv run python validation/real_reference.py prepare \
  --bids-input "$MSC_INPUT" --assets .local-data/assets --workdir "$MSC_WORK" \
  --participant-label MSC01 MSC02 --session-label func01 func02 \
  --max-iter 5 --allow-zero-cortex
uv run python validation/real_reference.py run-profiles \
  --workdir "$MSC_WORK" --podman-container pymshbm-oracle
uv run python validation/real_reference.py check-profiles --workdir "$MSC_WORK"
uv run python validation/real_reference.py run-kernel \
  --workdir "$MSC_WORK" --podman-container pymshbm-oracle
uv run python validation/real_reference.py collect \
  --workdir "$MSC_WORK" --python-output "$MSC_OUTPUT"
```

All-zero nonseed time courses are the only extra permitted cortical inputs.
Nonzero constants, nonfinite cortical values, unresolved or constant cortical
seeds, and nonpositive global correlation cutoffs remain errors. The
preparation checks that only independently identified unsupported vertices
remain zero. The production CLI performs no resampling or imputation itself.
In the source equations, a zero profile is a literal zero observation that
still participates in session averaging and cost/concentration calculations;
it is not a NaN missing session. A vertex unusable in every supplied session
can remain label zero, and its exclusion must be explicit in coverage summaries.

Each preparation `*_coverage.npz` contains `observed_mask`, `imputed_mask`,
and `unresolved_zero_mask`, plus original validity, missingness, imputation
distances, and source vertices. Each production subject's
`sub-*_desc-mshbm_coverage.npz` instead contains `cortex_mask`,
`usable_session_count`, `usable_in_any_session`, and `full_session_coverage`,
in left-then-right vertex order. **Usable includes imputed input**; it does not
identify originally observed support. Preserve both sets of arrays.

After fitting and collection, compare the complete cortex and common originally
observed support separately, then render the maps using the display surfaces
described below:

```sh
uv run python validation/compare_coverage_support.py \
  --input "$MSC_INPUT" --workdir "$MSC_WORK" --python-output "$MSC_OUTPUT" \
  --assets .local-data/assets --output "$MSC_WORK/coverage-support.json"
uv run --extra validation python validation/plot_real_comparison.py \
  --input "$MSC_INPUT" --workdir "$MSC_WORK" --python-output "$MSC_OUTPUT" \
  --assets .local-data/assets --surfaces SURFACES --output "$MSC_FIGURES"
```

The strict reports and source snapshots remain the record for the earlier
analysis. A failed follow-up must likewise retain its coverage and numerical
reports without changing the original thresholds.

## Figures

For figures, provide FreeSurfer 8.1.0-1 fsaverage6 `lh.inflated` and `rh.inflated`
display meshes in SURFACES. They affect rendering only, not fitting:

```sh
uv run --extra validation python validation/plot_real_comparison.py \
  --input INPUT --assets .local-data/assets --surfaces SURFACES \
  --workdir WORK --python-output OUTPUT --output FIGURES
```

Figures use exact vertex-level counts. Binary overlays highlight all faces
touching an affected vertex, so their displayed area is larger than the set
of affected vertices. Quantitative agreement always uses unpermuted vertex
labels and the original numeric arrays, not rasterized figures.
