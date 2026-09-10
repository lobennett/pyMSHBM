# Group-estimation numerical audit

Reference: Buckner PrecisionNetworkMapping `988ec48cf5453c6009a6f82dbddf5d5b99eda3cd`
invokes CBIG's Kong2019 group-prior estimator. The executable CBIG source audited
here is pinned at `35b5664bec8822e2f77da5e090e96f91d0095be6`; its exact copy and
checksums are in `validation/upstream/` and `validation/fixtures/oracle-provenance.json`.
The oracle worker executes extracted upstream initialization and EM code in Octave;
it does not use a second Python implementation to manufacture expected values.
See `validation/README.md` and the convergence report for adapters and reproduction.

## Confirmed differences corrected

Line references below refer to the pinned
`validation/upstream/CBIG_MSHBM_estimate_group_priors.m`.

| Reference operation | Inherited Python behavior | Correction |
|---|---|---|
| Initialization theta, lines 234–236 | Sum across subjects divided by nonzero count | **Mean** across subjects, then divide by nonzero count; retain this unusual initialization literally |
| E-step theta, line 474 | Divide by nonzero count | Ordinary mean across subjects, including posterior underflow zeros |
| Middle EM, lines 250–280 | A single session fit and intra update per outer iteration | Alternate both to the reference relative-cost threshold, at most 50 times; reset kappa and session directions before each session fit |
| Kappa, line 412 | Sum sessions without dividing by their count | Sum vertices, NaN-mean sessions per subject/cluster, then sum subjects and clusters |
| M-step convergence, lines 401–452 | Direction-only stopping and arbitrary 50-step cap | Persistent subject/session convergence flags **and** relative kappa convergence |
| Intra update, lines 349–386 | Ordinary session sum, global sigma averaging, fresh flags, 50-step cap | NaN session sum for subject means, ordinary subject mean followed by NaN session mean for sigma, persistent subject flags |
| E-step zero masks, lines 460–471 | Add the vMF constant to zero profiles and assign uniform posterior to medial wall | Add constant only where every cluster dot product is nonzero; mask rows with any zero accumulated likelihood |
| Cost likelihood, lines 477–498 | Reuse masked E-step cache; omit NaNs everywhere | Separate cost likelihood with unconditional constant; preserve first-session NaNs and subsequent sequential NaN sums; ordinary final sums |
| Log zero cost, lines 489–494 | Clamp all probabilities to double's smallest normal | Replace infinite logs with `log(eps**20)` as upstream does |
| Inner EM stop, lines 500–509 | Return current cost and force a second iteration | Preserve MATLAB's comparison semantics and store **previous** cost on convergence; store current cost only at the 101-iteration limit |
| Hierarchical cost, lines 261–269 | Omit subject/session direction priors entirely | Add both vMF direction priors, preserving reduction order and NaN behavior |
| Outer record, lines 290–296 | Recompute posterior cost after inter update | Record the converged intra cost from **before** the inter update |
| `invAd`, lines 572–593 | Solve the Bessel inverse (scalar and batch methods differed) | Group fitting calls `cbig_inv_ad`, which returns upstream's initial approximation exactly |
| `Cdln` and mixed arithmetic, line 569 | Keep later calculations double | Match single-dominant arithmetic for posterior, weighted products, new session directions and costs; preserve double session-direction storage and double subject/group directions |

The `invAd` detail is an executable-source quirk: the MATLAB return variable is
`outu`, while the numerical root and overflow correction are stored in `out`.
The latter never becomes the return value. `cbig_inv_ad` intentionally reproduces
this behavior. The existing public `inv_ad` and `inv_ad_batch` mathematical
utilities remain unchanged, as does their use in the separately implemented
individual-parcellation algorithm. This audit does not certify that algorithm.

The effective dimension remains `D - 1` for mean-centered D-feature input. The
reference `Cdln` calculation, its 500/650 integration starting concentrations,
1000 midpoint integration grid, omitted common spherical constant and float32
return were already present; they were retained. This estimator does not replace
that approximation with a more accurate normalizer or clamp resultant lengths
away from one, since either would change the reference calculation.

## Test-first evidence

Before editing the respective production behavior, focused pytest runs showed:

- Five failures for initialization theta, zero-profile E-step, duplicated-session
  kappa invariance, missing prior costs and the literal CBIG inverse.
- Three failures for missing-session subject directions, recorded nested costs
  and previous-cost convergence semantics.
- Two failures for missing-session prior reduction order and single-session NaN
  cost propagation.
- A failure asserting upstream posterior/cost dtypes and quantized session
  directions.
- An empty-input failure showing that a fit with no assigned vertices reached a
  numerical loop rather than producing an actionable early error.

Each focused run passed after its corresponding correction. The inherited exact
Python-output snapshot encoded the incorrect algorithms (including theta mass
23 for 20 vertices). It was replaced with a posterior-mass regression; separate
executed-CBIG fixtures now supply the parameter goldens. The inherited float32
versus float64 and multiprocessing-versus-array tests remain and pass.

Initial combined verification:
` .venv/bin/pytest tests/test_core/test_group_priors.py tests/test_math/test_vmf.py tests/test_reference_oracle.py -q `
passed **55 tests** after the final empty-fit regression was added (1.00 s).
The reference worker subsequently extended the oracle fixtures; final aggregate
counts and fixture evidence are recorded in the convergence report.

## First measured comparison

For the executed upstream fixture with 300 vertices, 101 profile features,
3 subjects, 2 sessions, 3 networks, zero medial rows and a five-iteration outer
limit, both implementations stopped at outer iteration 4. There were **zero
unpermuted label mismatches**. Maximum absolute differences in the first matched
run were:

| Parameter | Maximum absolute difference |
|---|---:|
| Group directions mu | 7.92e-9 |
| Subject directions s_psi | 3.44e-8 |
| Session directions s_t_nu | 2.98e-8 |
| Posterior s_lambda | 7.60e-14 |
| Spatial prior theta | 0 |
| Inter concentration epsil | 0.00134 |
| Intra concentration sigma | 0.00567 |
| Kappa | 0 |
| Outer cost record | 0.0625 |

Concentration differences are approximately 1e-6 to 2.2e-6 relative, and cost
record difference is approximately 1.6e-7 relative. The checked-in comparison
report, not this rounded table, is authoritative for the final fixtures and
parameter tolerances. These are tolerance-based numerical comparisons, not
claims of bitwise identity across BLAS libraries or MATLAB and Octave.

A second independently executed fixture makes subject 3/session 2 entirely NaN.
It also matched every unpermuted label and stopped at iteration 5 in both
implementations. Maximum sigma difference was 0.01746 (4.5e-6 relative), direction
difference 1.25e-7, and posterior difference 7.5e-13. Reproduce it with
` .venv/bin/python validation/compare_reference.py --case missing `.

## Deliberate boundaries and remaining uncertainty

- The reference leaves its M-step and direct intra-variability updates uncapped.
  Python allows 10,000 iterations by default (`numerical_max_iter`) and raises
  `RuntimeError` if that safeguard is reached. It does not return an unconverged
  fit as a successful result. Middle and outer limits retain upstream semantics.
- A fit with zero finite total posterior mass raises `ValueError` instead of
  entering the upstream invalid division/nonconverging path.
- Concentration estimates can become extremely ill-conditioned as subject
  directions approach identity. An early tiny random fixture produced upstream
  epsil values around 1e11–1e14; small arithmetic differences can then yield large
  concentration differences. The stable golden fixture exercises estimable
  subject variation. It does not remove or conceal that model limitation.
- Missing-session averaging order is reproduced literally, including the
  surprising exclusion of an entire session from sigma/prior-cost averages if
  any subject's direction is NaN at that position. It has focused regression
  coverage and should not be described as a statistical redesign.
- Input shape validation, profile normalization and production preprocessing
  belong to the surrounding workflow. This kernel expects normalized profiles
  and valid centroids, with whole missing sessions represented as NaN.
- Synthetic kernel agreement does not establish biological validity, equivalence
  of fMRIPrep and Buckner denoising, production fsaverage6 runtime/memory behavior,
  or full MATLAB execution on a real dataset. Those claims require separate data
  and execution evidence.

## Full-surface smoke follow-up

A full fsaverage6 synthetic run (81,924 vertices, 1,175 features, one subject,
two sessions) exposed an additional inherited mismatch in intra-update stopping:
Python divided by `max(sigma, 1e-10)`, while upstream divides by signed `sigma`
(line 382). The test's almost noiseless network signals produced quantized
session directions with norms up to approximately 1.0000049 and transient
resultant lengths above one. The literal inverse approximation then temporarily
returns negative sigma. Upstream's signed mean relative change becomes negative,
and its persistent direction flags allow stopping; the Python floor prevented
that stopping and reached the 10,000-iteration safeguard.

The pre-update state was saved locally in
`.local-data/smoke/failed_intra_state.mat`. Independently executed upstream
`intra_subject_var` stopped after **two** iterations on that exact state, with
mean relative sigma change approximately **-0.2666667** and positive final sigma.
Restoring the signed denominator makes Python stop at the same point. A focused
regression first failed at the numerical guard, then passed after the correction.
No clipping, normalizer change, or tolerance relaxation was introduced.

The full-surface smoke generator's independent BOLD noise was increased from
0.15 to 1.0 relative to its unit-variance shared network signals, retaining its
seed and two 40-frame runs. This makes it a less degenerate I/O smoke fixture;
it remains synthetic and is not a realism or scientific-validity claim. The
production workflow separately rejects nonpositive final concentrations.

The refreshed smoke input completed the full CLI successfully at 81,924
vertices and one outer iteration. After the single-input arithmetic correction
below, the same input completed again; both hemispheres contained 40,962 labels,
all in 0–15. The final smoke objective was 629,391,744. Local final output is
`.local-data/smoke-variable/output-single`; it is not a checked-in scientific
golden dataset.

### Single-profile multiplication follow-up

The production disk-backed tensor is float32. The pinned `mtimesx` C source
explicitly converts a double matrix operand to single before multiplication
when the other non-scalar operand is single; ordinary MATLAB matrix products
have the same precision rule. NumPy instead promotes these mixed operands to
double. A small cancellation regression proved that casting only the result
cannot reproduce the source. `_profile_dot` now casts operands before the
product, and the single-input E-step preserves single scaling and session sums.

The independent oracle was corrected to preserve the native product dtype and
executed on float32 input. All original tolerances still pass: exact labels and
outer iteration 4, direction differences below 3.5e-8, sigma difference below
0.00567, posterior difference below 7.6e-14, and cost differences below 0.03125.
No tolerance was widened. The combined core/math/oracle test run passed 71 tests
before that fourth oracle case was added to the parameterized suite.
