# Scientific compatibility contract

The target is `MSHBM/MSHBM_Params_Training.m` and `MSHBM_wrapper.m` in Buckner's
PrecisionNetworkMapping revision `988ec48cf5453c6009a6f82dbddf5d5b99eda3cd`.
Because that repository does not pin its external CBIG dependency, this fork
pins CBIG to `35b5664bec8822e2f77da5e090e96f91d0095be6` and reports that choice.
It cannot establish which historical CBIG revision produced published Buckner
maps without additional information from that lab.

Buckner's README says pretrained priors are used without group estimation; the actual
training script loads DU15NET **labels**, derives directions from average
profiles, then estimates group parameters on the supplied participants with
`max_iter=5`. This fork follows the executable steps. The reference script omits
`save_all` even though its extractor requires `s_lambda`; Python retains those
posteriors explicitly. This is an output-enabling change, not an extra model fit.

| Stage | Compatibility path | Inherited path difference |
|---|---|---|
| Input | Ordered fsaverage6 surface vertices; each run is a session | Custom filename glob without BIDS pairing |
| Seeds | Cortex vertices among first 642 in each hemisphere | Nearest-seed ROI averaging |
| Connectivity | Pearson, one joint top-10% threshold, binary, ties retained | Fisher-Z continuous correlations |
| Mean profile | Equal mean over available subject/run profiles | Similar averaging, different source profiles |
| Initialization | DU15NET labels → centered unit profiles → summed network centroids | Optional k-means and centroid relabeling |
| Data normalization | CBIG single-precision cast, feature reduction order, and literal centered-row condition | No centering |
| Model | Pinned CBIG nested hierarchical EM, default five outer iterations | Missing nested iteration and other numerical deviations |
| Individual labels | Argmax of group-fit posterior; zero rows remain zero | Additional MRF fit can change maps |

The [numerical audit](numerical-audit.md) records each corrected equation,
mask, reduction order, dtype and stopping rule. Compatibility intentionally
preserves unexpected executable behavior (notably the returned inverse-vMF
approximation and the order used to average missing sessions). A statistically
revised algorithm would require a separately named mode and its own validation.

Python adds early input validation and a finite safeguard on loops the reference
leaves uncapped. By default it rejects constant/nonfinite cortical BOLD and nonpositive
global correlation cutoffs. The explicit `--allow-zero-cortex` opt-in permits
only entirely zero nonseed cortical inputs; nonzero constants, nonfinite
cortex and unusable cortical seeds remain errors. It checks all
fitted parameters before publishing results. These rejected inputs are outside
the supported equivalence domain. Single-subject or nearly identical sessions
can leave variability parameters ill-conditioned; do not interpret a successful
file write as evidence of model identifiability.

## Evidence required for a real-data equivalence claim

1. The same already-denoised surface time series, run ordering, cortex masks,
   DU15NET prior and CBIG revision in both workflows.
2. Compare sampled seed time series, correlations and binary profiles first;
   threshold ties can magnify tiny floating-point changes.
3. Compare initial centroids, normalized tensors, every saved parameter and
   posterior, objective history, iteration counts and unpermuted network IDs.
4. Run the complete upstream workflow with MATLAB and its actual dependencies.
   Octave infrastructure adapters and synthetic dimensions do not cover all
   MATLAB I/O, BLAS or population-specific behavior.
5. Retain a report with preregistered numerical tolerances and investigate
   disagreements rather than relaxing thresholds after seeing a failure.

The [executed OpenNeuro comparisons](openneuro-validation.md) supply real-data
evidence conditional on shared fsLR32k-to-fsaverage6 resampling and declared
boundary imputation within 5 mm. The completed MSC01/MSC02 follow-up explicitly
leaves one unresolved nonseed vertex zero. All four binary and normalized
profiles are exact and both engines converge at iteration three. Directions,
concentrations and costs pass tolerance, but spatial-prior/posterior failures
and one label mismatch mean the complete acceptance contract **is not met**.
The discrepancy remains on originally observed cortex as well. Its cause has
not been established.

The original complete-cortex multi-participant preparation failed its fixed
support rule; those failures and the historical MSC01 results remain preserved.
Installed-wheel reproduction matches the Python fit exactly but does not change
the failed upstream comparison. No native MATLAB run has been executed. These
results do not establish equivalent denoising, scientific suitability for a
particular dataset, or replication of published participant maps.
