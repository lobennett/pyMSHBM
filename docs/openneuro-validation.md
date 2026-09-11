# Executed OpenNeuro validation

The completed **MSC01/MSC02, two-session comparison fails the declared overall
acceptance criteria**. All four binary and normalized float32 profile arrays
match exactly, and both engines converge at outer iteration three. Directions,
concentrations and costs pass their fixed tolerances. However, one of 163,848
subject–vertex labels differs, and spatial-prior and posterior errors exceed
tolerance. The remaining numerical discrepancy has not been explained.

This case uses the separately declared `--allow-zero-cortex` protocol: bounded
imputation stays within 5 mm, and one unsupported nonseed cortical vertex in
MSC02 remains zero. Agreement is conditional on those shared prepared inputs;
it does not establish complete-coverage, preprocessing or native MATLAB
workflow equivalence. The earlier strict preparation failures and MSC01 fits
remain part of the evidence.

The completed two-participant case has a
[full comparison report](../validation/reports/openneuro/msc0102-masked/comparison.json),
[coverage-specific comparison](../validation/reports/openneuro/msc0102-masked/coverage-support.json),
[MSC01 maps](../validation/reports/openneuro/msc0102-masked/MSC01-network-comparison.png),
[MSC02 maps](../validation/reports/openneuro/msc0102-masked/MSC02-network-comparison.png),
and [numerical summary](../validation/reports/openneuro/msc0102-masked/numerical-summary.png).
Its [detailed results](#completed-multi-participant-unresolved-zero-follow-up)
appear below.

The historical single-participant baseline and normalization-corrected fit are both retained:

- [Before-correction report](../validation/reports/openneuro/msc01-before-normalization-fix/comparison.json),
  [network maps](../validation/reports/openneuro/msc01-before-normalization-fix/MSC01-network-comparison.png),
  and [numerical summary](../validation/reports/openneuro/msc01-before-normalization-fix/numerical-summary.png).
- [After-correction report](../validation/reports/openneuro/msc01-after-normalization-fix/comparison.json),
  [network maps](../validation/reports/openneuro/msc01-after-normalization-fix/MSC01-network-comparison.png),
  and [numerical summary](../validation/reports/openneuro/msc01-after-normalization-fix/numerical-summary.png).

## Historical MSC01 case: dataset and shared preparation

The input is the authors' processed resting-state surface data from Midnight
Scan Club, OpenNeuro **ds000224 version 1.0.4**, dataset revision
`727a0a4e25ec3f7bea1a20c955ea860206b31e77`. The completed diagnostic uses
**MSC01, func01 and func02**. Both distributed CIFTIs contain 818 frames.
Preparation applies the supplied temporal masks once:

| Session | Retained frames | Nominal TR | Retained acquisition time |
| --- | ---: | ---: | ---: |
| func01 | 523 / 818 | 2.2 s | 19.18 min |
| func02 | 717 / 818 | 2.2 s | 26.29 min |

The retained time totals 45.47 minutes and includes gaps from censoring; it is
not a continuous recording of that duration. Data attribution, acquisition,
and executable commands are in [reproduction instructions](openneuro-reproduction.md).
The [preparation manifest](../validation/reports/openneuro/msc01-after-normalization-fix/preparation.json)
records original filenames, masks, hashes, retained frame counts, registered
spheres, individual surfaces, and Workbench commands.

The fsLR32k time series were resampled once to fsaverage6 using Workbench
1.5.0 `ADAP_BARY_AREA`, registered HCP/CBIG spheres, individual midthickness
areas, and the source cortical ROI. Some target cortex lacked direct support.
Under the declared fixed **5 mm** boundary rule, preparation copied the nearest
supported target cortical time series to 373 left and 306 right vertices per
session, including seven left and one right fsaverage3 seed vertices. The
maximum recorded distances were 4.863 mm left and 4.750 mm right. Four right
vertices required triangle-based distance refinement after the conservative
edge-path distance exceeded the bound. No radius was increased.

Those 679 vertices are imputed, not directly observed cortex. Their masks and
extent appear in the [coverage map](../validation/reports/openneuro/msc01-after-normalization-fix/MSC01-input-coverage.png).
Both implementations receive these same prepared GIFTIs. Preparation adds no
nuisance regression, temporal filtering, or smoothing of measured vertices.
This shared resampling and bounded imputation remain outside the numerical
equivalence claim; they do not reproduce native Buckner preprocessing.

The model has 81,924 vertices, 1,175 profile features, one participant, two
sessions, and 15 networks. It retains the Buckner default maximum of five outer
iterations and the original numerical tolerances. The reference executes
pinned CBIG source in Octave 6.4.0 through the documented I/O and numerical
infrastructure adapters. Native MATLAB was unavailable. See the
[source/runtime boundaries](reference-validation.md#adapter-boundaries) and
[real-data harness](real-reference-harness.md).

## Before and after the normalization correction

The initial comparison found exact binary profiles but small float32
normalization differences. NumPy's row selection produced a C-contiguous
array and changed feature-summation order. The correction keeps Fortran layout
and computes row norms before selection; a single-row path accumulates features
explicitly in float32. The normalization equations, label ordering, acceptance
thresholds, and model settings were retained.

An independent [1,175-feature oracle](../validation/fixtures/normalization_high_dim_provenance.json)
executes the unchanged CBIG normalization lines on binary and continuous
profiles, including separate single-row calls. It does not construct expected
outputs using the Python normalizer. The real-data
[normalization-only diagnostic](../validation/reports/openneuro/normalization-corrected-report.json)
and the separately prepared corrected full fit provide distinct evidence.

| Measurement | Before correction | After correction |
| --- | ---: | ---: |
| Binary profile mismatches, both sessions | 0 | 0 |
| Normalized float32 mismatches, both sessions | 174,288,266 | 0 |
| Maximum normalized-profile absolute error | 6.080e-6 | 0 |
| Initial centroid maximum absolute error | 4.718e-16 | 4.718e-16 |
| Unpermuted final label mismatches | 85 / 81,924 | 0 / 81,924 |
| Agreement over assigned cortical vertices | 99.8866% | 100% |
| Per-network Dice range, IDs 1–15 | 0.997541–0.999729 | 1.0 for every network |
| Python / reference outer iterations | 2 / 3 | 3 / 3 |
| Full acceptance result | **Fail** | **Fail** |

The corrected normalization is exact across all **192,521,400** stored values.
Initial centroids pass the fixed tolerance but are not bitwise identical.
Both engines leave 6,977 vertices unassigned. The published corrected left and
right label GIFTIs match their parameter-derived labels and the reference
labels exactly; hemisphere metadata and label intent also pass. The figures
show vertex-level agreement on display surfaces; quantitative counts come
from the original arrays, not rasterized overlays.

Both corrected fits stop through the relative-cost criterion at iteration
three, before the five-iteration cap. Their final relative cost changes differ:
9.772e-6 in Python and 2.841e-6 in the reference, both below 1e-5. Matching
stopping decisions does not mean that their objective trajectories are equal.

## Historical MSC01 parameter disagreement

The corrected fit retains the following errors at the original tolerances:

| Parameter | Maximum absolute error | Result |
| --- | ---: | --- |
| Group and subject directions, `mu`, `s_psi` | 0.006543 | Fail |
| Session directions, `s_t_nu` | 3.332e-5 | Fail: 8 elements |
| Spatial prior and posterior, `theta`, `s_lambda` | 0.033186 | Fail: 38 elements each |
| Between-subject concentration, `epsil` | 3.962e18 | Fail |
| Within-subject concentration, `sigma` | 111.259 | Fail |
| Shared concentration, `kappa` | 0 | Pass, exact |
| Final `cost_em`, `cost_intra`, `cost_inter` | 0 | Pass, exact |
| Entire outer cost record | 3,904 | Fail: 1 element |

Direction tolerances remain absolute 2e-5 plus relative 1e-4; posterior/theta
tolerances remain absolute and relative 1e-4; concentration tolerances remain
absolute .05 plus relative 2e-4; cost tolerances remain absolute .1 plus relative
5e-6. The [full report](../validation/reports/openneuro/msc01-after-normalization-fix/comparison.json)
contains element counts and stabilized relative errors. Some direction and
concentration errors increased after the correction even though profiles and
hard labels became exact. No failed field is excluded from the overall result.

With only one participant, between-subject variability is not identifiable.
The source inverse-concentration approximation has a pole as the mean
resultant approaches one; finite-precision differences can therefore produce
extreme concentration changes. This is a structural limitation of this case
and a plausible contributor to instability, **not a demonstrated sole cause
of the remaining discrepancies**. The present evidence does not explain away
failed parameters or establish reliable population variability estimates.

## Historical MSC01 installed-wheel verification

An isolated `uv tool install` of the built **pyMSHBM 0.2.1 wheel** was used to
rerun the corrected MSC01 case. Its installed Python source hashes and every
saved model parameter, including the objective history and three final costs,
match the development run bitwise. The
[wheel verification report](../validation/reports/openneuro/wheel-install-verification.json)
records these checks. The earlier development-run metadata still reports
package version 0.2.0; source hashes bind the actual implementation across
that metadata change. This checks packaging and reproducible Python fitting;
it does not turn the failed upstream parameter comparison into a pass.

## Completed multi-participant unresolved-zero follow-up

The original MSC01/MSC02 complete-cortex preparation failed before fitting.
The [strict eligibility archive](../validation/reports/openneuro/eligibility/summary.json)
retains the identifier-order search and all failures for MSC02 through MSC10,
with [export checksums](../validation/reports/openneuro/eligibility/files-sha256.json).
No second participant met the unchanged 5 mm support rule. The original
complete-cortex multi-participant fit therefore remains uncompleted.

A [separately declared protocol](openneuro-validation-plan.md#separately-declared-unresolved-zero-follow-up)
then fitted the originally selected MSC01 and MSC02 jointly, retaining func01
and func02, the five-iteration maximum and the original numerical tolerances.
The [preparation manifest](../validation/reports/openneuro/msc0102-masked/preparation.json)
records all four inputs:

| Participant | func01 retained frames | func02 retained frames | Imputed vertices per run | Unresolved cortex per run |
| --- | ---: | ---: | ---: | ---: |
| MSC01 | 523 / 818 | 717 / 818 | 679 | 0 |
| MSC02 | 531 / 818 | 793 / 818 | 678 | 1 |

Each run has 74,268 originally observed target cortical vertices. Eight cortical
seed vertices per participant/run were imputed within 5 mm; none remained
unresolved. MSC02 right-hemisphere vertex **1659** (zero-based, hemisphere-local)
remains zero in both sessions and receives label zero in both engines. No
imputation crossed the distance bound. The
[MSC01 coverage map](../validation/reports/openneuro/msc0102-masked/MSC01-input-coverage.png)
and [MSC02 coverage map](../validation/reports/openneuro/msc0102-masked/MSC02-input-coverage.png)
distinguish observed, bounded-imputed and unresolved input. An all-zero vertex
remains a literal observation in the source equations, not a NaN missing session.

The fitted tensor has 81,924 vertices, 1,175 features, two participants, two
sessions and 15 networks. Independent reference construction gives exact binary
and normalized float32 profiles across **385,042,800 values in each representation**.
Initial centroids pass tolerance, with maximum absolute error 6.661e-16, but
are not bitwise identical. The
[profile report](../validation/reports/openneuro/msc0102-masked/profile-comparison.json)
and [execution manifest](../validation/reports/openneuro/msc0102-masked/manifest.json)
bind the inputs, source hashes and fixed criteria.

| Measurement | Result at the original criteria |
| --- | --- |
| Group, subject and session directions | Pass; maximum absolute error 1.296e-6 |
| Between-/within-subject and shared concentrations | Pass |
| Final costs and entire outer objective history | Pass |
| Spatial prior `theta` | **Fail:** maximum absolute error 0.003601015; 63 elements exceed tolerance |
| Posterior `s_lambda` | **Fail:** maximum absolute error 0.007202029; 98 elements exceed tolerance |
| Final unpermuted labels | **Fail:** 1 mismatch, in MSC02, among 163,848 subject–vertex labels |
| Python / reference outer iterations | 3 / 3; both converge before the five-iteration cap |
| Overall acceptance | **Fail** |

The tolerance test combines absolute and relative allowances; passing fields
need not be exact. For example, the maximum `epsil` difference is 0.1400 and
the objective-history difference is 128, both within their fixed combined
allowances. Final cost fields and shared `kappa` are exact. Python and reference
final relative cost changes are 4.645e-6 and 4.532e-6, respectively, below 1e-5.
All published label GIFTIs match their own fitted posteriors, and their hemisphere
metadata and label intents pass. The reference comparison retains the single
MSC02 right-hemisphere label disagreement.

The [support-specific report](../validation/reports/openneuro/msc0102-masked/coverage-support.json)
verifies input-mask hashes and separates full cortex from originally observed
support. Of 74,947 target cortical vertices per participant, MSC01 matches
74,947 and MSC02 matches 74,946. On the common originally observed intersection
of 74,268 vertices, MSC01 matches all and MSC02 matches 74,267. Thus the label
disagreement occurs on originally observed cortex, not at unresolved vertex
1659. These restricted summaries do not replace the complete acceptance test.
The [bounded arithmetic diagnostic](../validation/reports/openneuro/diagnostics/s2-estep/diagnosis.json)
locates the disagreement at MSC02 right-hemisphere vertex **16145** (zero-based):
Python favors network 11 with probability 0.504394, while the reference favors
network 10 with probability 0.502808. Both runs originally observed this vertex.
Common observed support has Dice 1.0 for every MSC01 network and a minimum
0.9998714 for MSC02; the original exact-label criterion still fails.

A separate 51-row E-step replay supplies identical profiles and saved parameters
to both runtimes. Native matrix products differ by at most 1.550e-6 and the
resulting posterior by 3.161e-4. Supplying the exact upstream products to the
unchanged Python E-step tail reduces the latter error to 2.980e-8, with no
tolerance violations. This demonstrates an arithmetic contribution from the
different matrix-product backends: native Python uses Apple Accelerate and
the reference uses OpenBLAS. The inspected direction-norm reduction layouts
match explicit sequential feature summation; no further concrete port bug was
established in this diagnosis.

This replay uses final saved parameters and a smaller matrix; it **does not
reconstruct the original last E-step**, whose incoming spatial prior was not
captured. It does not attribute every accumulated difference or the label
crossing solely to BLAS. The [standalone diagnostic bundle](../validation/reports/openneuro/diagnostics/s2-estep/README.md)
includes inputs, outputs, extracted source, licenses, runtime hashes and replay
instructions. The complete comparison remains failed.

An isolated `uv tool install` of the built **pyMSHBM 0.2.2 wheel** reproduces
every saved development-fit parameter and installed Python source hash exactly,
including objective history and all three final costs. The installed run uses
Python 3.12.13; the development run uses Python 3.14.5. Both record NumPy 2.5.3,
SciPy 1.18.1 and nibabel 5.4.2. The
[wheel verification](../validation/reports/openneuro/msc0102-masked/wheel-install-verification.json),
[development provenance](../validation/reports/openneuro/msc0102-masked/python-provenance.json)
and [wheel provenance](../validation/reports/openneuro/msc0102-masked/wheel-provenance.json)
retain those checks. This is a successful packaging/reproduction result, while
the upstream numerical acceptance result remains failed.

The [real-reference runtime inventory](../validation/reports/openneuro/reference-runtime.json)
records Octave 6.4.0-2, OpenBLAS 0.3.20+ds-1 and Workbench 1.5.0-2 in an amd64
container emulated on an ARM host. The kernel report records 25.75 seconds for
reference fitting, excluding profile preparation; this is not a controlled
speed comparison with native Python. Native MATLAB execution, independent
preprocessing equivalence and biological validation remain outside the evidence.
