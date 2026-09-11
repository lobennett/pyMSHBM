# OpenNeuro real-data comparison plan

Declared before fitting or examining any MS-HBM comparison results on 2026-09-10.

Use Midnight Scan Club **ds000224 version 1.0.4**, git revision
`727a0a4e25ec3f7bea1a20c955ea860206b31e77`, participants MSC01 and MSC02,
sessions func01 and func02. Selection is the first two participants and first
two resting-state sessions by identifier, not a selection by output agreement.
This is a computational validation case, not a population study or a replication
of the dataset authors' network maps.

Use the authors' processed resting-state CIFTI derivatives and their supplied
temporal masks. Inspect whether censoring is already represented by dropped
frames; apply masks only to full-length data. Preserve retained timepoints and
record this decision per run. Do not add nuisance regression, temporal filtering,
or smoothing. Convert cortical fsLR32k data to fsaverage6 using pinned HCP/CBIG
registered spheres, Workbench ADAP_BARY_AREA, and individual midthickness area
surfaces (the same geometry resampled for target areas). Record valid cortical
coverage; do not silently fill missing cortical data. This conversion is shared
preparation, outside the numerical equivalence claim.

Fit the complete two-participant/two-session cohort jointly, with the pinned
DU15NET labels, fsaverage6 cortex masks, and default maximum five outer
iterations. Run Python and the actual pinned CBIG numerical source via Octave
on identical prepared BOLD, independently constructing profiles and centroids.
Retain input hashes, source/runtime revisions, commands, run durations, logs,
all model parameters, objective histories, and unpermuted network labels.

Acceptance criteria are the existing synthetic comparison tolerances:

* Binary profiles and final network labels: exact equality.
* Directions, normalized profiles, centroids: absolute 2e-5, relative 1e-4.
* Spatial priors and posteriors: absolute 1e-4, relative 1e-4.
* Concentrations: absolute .05, relative 2e-4.
* Costs: absolute .1, relative 5e-6.
* Outer iteration count and convergence/iteration-limit decisions: exact.

Report full-cortex and per-network agreement/Dice, parameter error summaries,
and any failed criteria. Do not adjust tolerances after observing the results.
If five iterations reach the cap without the source relative-cost convergence
criterion (1e-5), report that distinction. A secondary fit of **both** engines
with a maximum 50 iterations may assess convergence; it must remain separately
labeled, because it changes Buckner's default cap. Resource failures may require
sequential execution or a separately reported smaller case, never relabeling
the planned cohort as completed.

The available reference runtime is Octave 6.4 with documented infrastructure
adapters. Even a passing real-data result would not establish complete native
MATLAB workflow equivalence, independent preprocessing equivalence, or biological
validity. A licensed MATLAB run remains a separate validation milestone.

## Pre-fit coverage amendment

The first coverage check, before any model fit, found 373/37,476 target LH
cortical vertices unsupported by the source fsLR cortical ROI (including
7 of the first 642 seed-mesh vertices). Preparation stopped and preserved
`failed-coverage.json`. This reflects a boundary mismatch between cortical
atlas definitions; source values cannot simply be assumed there.

For this conditional numerical benchmark, permit **explicit nearest-neighbor
boundary imputation with a fixed 5 mm maximum**, using shortest paths along
the edges of the individual's resampled midthickness mesh. Edge paths provide
an upper bound on continuous surface geodesic distance. Copy the entire time
series from the nearest supported target cortical vertex only into initially
unsupported target cortex; preserve all measured vertices. Stop if any missing
vertex exceeds the bound. This is not additional temporal denoising or smoothing
of measured vertices, but it is spatial imputation beyond ordinary resampling.

Retain original coverage, missing/filled masks, source-vertex indices and
distances, count imputed seed vertices per hemisphere/run, and report their
extent. This shared preparation makes the resulting numerical comparison
conditional on the imputation; it does not validate native upstream
preprocessing or the biological interpretation of imputed boundary vertices.

The conservative edge-path check stopped again on the right hemisphere: four
vertices exceeded 5 mm (maximum edge-path bound 5.678 mm). Before fitting,
refine distances only for such vertices using Workbench's traversal across
triangle pairs, with the **same 5 mm limit**. Its nearest supported target
cortical vertex may be reachable across triangles within that bound even
when an edge-only path is longer. Record all refined vertex IDs. Do not
raise the distance limit; fail if this refinement still finds no support.

## Eligibility result before fitting

MSC01 passes the fixed 5 mm support rule for both hemispheres and sessions.
MSC02's right hemisphere still has unsupported cortex after triangle-based
refinement. The planned two-participant preparation therefore **fails**;
retain its log and do not increase the radius. Continue a separately identified
**MSC01-only, two-session computational benchmark**, selected by the declared
input-support rule before examining any model output. It cannot establish
multi-participant real-data equivalence or identifiable population variability.
The originally planned two-participant case must be reported as not completed.

## Multi-participant follow-up

The separately retained MSC01 fit exposes the S=1 concentration singularity:
between-subject dispersion has no finite identifiable estimate. Its strict
comparison fails (85 labels; different outer stopping iterations), despite
identical binary profiles. Do not discard or relabel this as a passing test.

To evaluate a nontrivial multi-participant case, assess additional participants
in identifier order beginning with MSC03, using the **unchanged input coverage
and 5 mm support checks**, and pair MSC01 with the first eligible participant.
Continue to use func01 and func02, max_iter=5 and the original numerical
tolerances. Report every attempted participant and eligibility outcome.
This follow-up is prompted by structural identifiability, not selection for
agreement; retain any failed numerical result from the resulting cohort too.

## Separately declared unresolved-zero follow-up

The identifier-order search found no second eligible participant among MSC02
through MSC10 under the unchanged 5 mm rule. Preserve every strict failure.
Before any multi-participant fit, declare a distinct masked-coverage comparison
of the originally selected MSC01 and MSC02, func01 and func02. Retain the same
resampling and bounded imputation, but leave any remaining unsupported target
cortex identically zero. Never impute beyond 5 mm; require all cortical seed
vertices to have usable, nonconstant series after bounded imputation. Stop if
a seed remains unresolved or any measured time course is constant/nonfinite.

Use an explicit `--allow-zero-cortex` opt-in; strict complete-cortex validation
remains the package default. Upstream correlation code clears undefined
correlations to zero, normalization retains zero profiles, and posterior
extraction assigns label zero where posterior mass is zero. An all-zero vertex
is NOT a NaN missing session: it still participates in source session averaging
and cost/concentration calculations. This comparison tests that literal behavior,
without claiming statistically neutral missing-data handling.

Preserve observed, bounded-imputed, and unresolved-zero masks per run, seed
coverage, and per-subject usable-session counts. Report complete-cortex labels
and common originally observed support separately. Use the unchanged numerical
tolerances and maximum-five-iteration protocol (secondary maximum50 only if
needed for convergence, in both engines). Preserve the strict MSC01 results and
source snapshots before adding this opt-in. This follow-up cannot establish
complete-coverage or independent-preprocessing equivalence.
