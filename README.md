# pyMSHBM

A BIDS-friendly Python implementation of the Multi-Session Hierarchical Bayesian Model for individual cortical network mapping. This fork adds an audited compatibility path for the Buckner lab PrecisionNetworkMapping workflow.

**Validation status:** synthetic fixtures match pinned CBIG code executed in Octave. The MSC01/MSC02 OpenNeuro comparison matches profiles and convergence decisions, but remaining label and prior/posterior tolerance failures mean overall acceptance still fails. See [real-data validation](docs/openneuro-validation.md), [reference validation](docs/reference-validation.md), and the [scientific contract](docs/scientific-contract.md).

## Install

Python 3.11+:

```bash
uv tool install 'git+https://github.com/lobennett/pyMSHBM.git'
pymshbm --help
```

Pin a commit with `@<commit>` for reproducible analyses. Development setup:

```bash
git clone https://github.com/lobennett/pyMSHBM.git
cd pyMSHBM
uv sync --frozen --extra dev
uv run pymshbm --help
```

Python fitting does not require MATLAB, Octave, or a FreeSurfer executable. It does require reference masks and DU15NET labels.

## Inputs

Pass an fMRIPrep derivative directory or a BIDS root containing `derivatives/fmriprep`. The compatibility path requires paired left/right `fsaverage6` surface BOLD files:

```text
sub-01/ses-01/func/
├── sub-01_ses-01_task-rest_run-1_space-fsaverage6_hemi-L_bold.func.gii
└── sub-01_ses-01_task-rest_run-1_space-fsaverage6_hemi-R_bold.func.gii
```

`space-fsaverage_den-41k` names are also accepted. Each paired physical run becomes one model session. Multiple selected participants are fitted jointly, so cohort changes can alter individual maps.

The package does not denoise, censor, smooth, resample, or validate general BIDS datasets. Prepare appropriately cleaned fsaverage6 BOLD before fitting. Raw-only BIDS, volumetric BOLD, fsnative, fsaverage5, and fsLR CIFTI inputs are rejected.

## Reference assets

Use the FreeSurfer subjects directory from the same preprocessing environment. It must contain `fsaverage6/label/{lh,rh}.cortex.label`.

```bash
pymshbm-fetch-assets ./mshbm-assets \
  --freesurfer-dir /data/fmriprep/sourcedata/freesurfer
```

The command downloads pinned DU15NET labels, verifies hashes, and records cortex-mask hashes. For offline setup, add `--reference-dir /path/to/PrecisionNetworkMapping`.

## Inspect and run

```bash
# Show selected files and sessions without writing outputs.
pymshbm bids /data/bids /data/derivatives/pymshbm \
  --participant-label 01 02 --task rest --dry-run

# Fit the selected cohort.
pymshbm bids /data/bids /data/derivatives/pymshbm \
  --assets-dir ./mshbm-assets --participant-label 01 02
```

Use `--session-label` and `--run-label` to select subsets. Output directories are never overwritten. Numerical nonconvergence or invalid data prevents publication of a completed dataset.

Complete cortical coverage is required by default. `--allow-zero-cortex` permits only fully zero, nonseed cortical time courses for a separately declared masked-coverage analysis. It does not permit nonfinite values, nonzero constants, unusable seeds, or nonpositive global cutoffs; it also does not impute or resample data. Follow the [masked OpenNeuro protocol](docs/openneuro-reproduction.md#separate-masked-coverage-case).

The temporary float32 profile tensor is approximately:

```text
81,924 × cortical seeds × subjects × maximum runs × 4 bytes
```

For 1,175 seeds, this is about 0.36 GiB per subject/run slot before working memory and outputs.

## Outputs

```text
pymshbm/
├── dataset_description.json
├── provenance.json
├── model/priors/Params_Final.mat
├── model/ind_parcellation/
└── sub-01/func/
    ├── sub-01_space-fsaverage6_atlas-DU15NET_hemi-L_dseg.label.gii
    ├── sub-01_space-fsaverage6_atlas-DU15NET_hemi-R_dseg.label.gii
    └── sub-01_desc-mshbm_coverage.npz
```

Labels retain DU15NET IDs and colors. Provenance records inputs, hashes, subject/run mapping, software and asset versions, model settings, objective history, and convergence state. Coverage files distinguish usable vertices from complete per-session coverage.

## Scientific scope

The compatibility path samples seed vertices, computes Pearson correlations, applies one global top-10% binary threshold, initializes from DU15NET labels, and follows CBIG centering, precision, nested EM, and stopping rules. It does not add k-means initialization or a second MRF fit.

Inherited wrapper, extraction, individual-MRF, group-training, and cerebellar APIs remain available but are outside the audited whole-workflow compatibility contract. The older API is described in [the pipeline guide](docs/pipeline_guide.md).

## Test and reproduce

```bash
uv run --frozen pytest -q
uv build
uv run python validation/compare_reference.py
uv run python validation/compare_reference.py --case missing
```

See [validation instructions](validation/README.md) to regenerate independent references and [OpenNeuro reproduction](docs/openneuro-reproduction.md) for the human-data comparison.

## Attribution

The original Python implementation is by Russell Poldrack and retains its MIT license. The compatibility work derives from Jingnan Du, Noam Saadon-Grosman, the Buckner lab, and CBIG work by Thomas Yeo, Ru Kong, and Xue Aihuiping. Cite the original MS-HBM method, DU15NET sources, and both Python repositories. This fork is not endorsed by the upstream labs.
