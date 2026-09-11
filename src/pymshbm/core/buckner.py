"""Profile and atlas initialization steps in Buckner's executable workflow.

References: CBIG_ComputeCorrelationProfile, CBIG_IndCBM_generate_MSHBM_params,
and fetch_data in CBIG_MSHBM_estimate_group_priors. See docs/scientific-contract.md.
"""

import numpy as np


def validate_cortical_bold(values, cortex_mask, *, seed_vertices=642,
                           allow_zero_cortex=False):
    """Validate one hemisphere and return its all-zero cortical vertex mask.

    The opt-in permits only identically zero nonseed cortical time courses.
    These remain literal zero observations, not NaN missing sessions. Usable
    seed support and rejection of nonfinite/nonzero constant data never change.
    """
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(cortex_mask, dtype=bool)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError('Surface BOLD must have at least three timepoints')
    if mask.shape != (values.shape[1],) or not 0 < seed_vertices <= mask.size:
        raise ValueError('Cortex mask or seed resolution does not match BOLD')
    if not np.isfinite(values[:, mask]).all():
        raise ValueError('Nonfinite cortical BOLD values; inspect preprocessing')
    zero = mask & np.all(values == 0, axis=0)
    constant = np.zeros(mask.size, dtype=bool)
    constant[mask] = np.ptp(values[:, mask], axis=0) == 0
    if not allow_zero_cortex and np.any(constant):
        raise ValueError('Constant cortical time series; inspect cortical coverage')
    if np.any(constant[:seed_vertices]):
        raise ValueError('Unusable cortical seed time series; all cortical seeds must be nonconstant')
    if np.any(constant & ~zero):
        raise ValueError('Constant cortical time series is nonzero; zero-cortex opt-in does not permit it')
    return zero


def binary_profiles(lh, rh, lh_cortex, rh_cortex, *, seed_vertices=642,
                    allow_zero_cortex=False):
    """Return joint (vertices, seeds) binary profiles from (time, vertices).

    fsaverage meshes share vertex ordering: take cortex vertices among the
    first 642 fsaverage6 vertices, corresponding to fsaverage3. Correlations
    across BOTH hemispheres share one 10% cutoff; cutoff ties are retained.
    Constant cortical time series are rejected by default. The explicit
    allow_zero_cortex option permits all-zero nonseed targets, which retain zero
    profiles. A nonpositive threshold is always rejected.
    No temporal denoising is performed here.
    """
    arrays = []
    seeds = []
    for values, mask in ((lh, lh_cortex), (rh, rh_cortex)):
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        validate_cortical_bold(values, mask, seed_vertices=seed_vertices,
                               allow_zero_cortex=allow_zero_cortex)
        # Medial-wall NaNs are non-data. CBIG_corr turns their correlations to 0.
        values = values.copy()
        values[:, ~np.isfinite(values).all(axis=0)] = 0.
        centered = values - values.mean(axis=0)
        norms = np.linalg.norm(centered, axis=0)
        normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
        arrays.append(normalized)
        seeds.append(normalized[:, np.flatnonzero(mask[:seed_vertices])])
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ValueError('Left/right hemisphere timepoints do not match')
    seed_series = np.hstack(seeds)
    if seed_series.shape[1] < 2:
        raise ValueError('At least two cortical seed vertices are required')
    corr = np.vstack([arr.T @ seed_series for arr in arrays])
    # MATLAB round for positive numbers, followed by one-based order statistic.
    rank = int(np.floor(corr.size * .1 + .5))
    if rank < 1:
        raise ValueError('Too few correlations for the upstream 10% threshold')
    cutoff = np.partition(corr.ravel(), corr.size - rank)[corr.size - rank]
    if cutoff <= 0:
        raise ValueError('Global correlation threshold is nonpositive; inspect BOLD quality')
    return (corr >= cutoff).astype(np.uint8)


def _sum_profile_features(series):
    """Sum float32 features in CBIG's column-major accumulation order."""
    if series.shape[0] == 1:
        # A one-row Fortran array is also C-contiguous: NumPy would otherwise
        # use a pairwise fast-axis reduction instead of sequential addition.
        total = np.zeros((1, 1), dtype=np.float32)
        for feature in range(series.shape[1]):
            total[0, 0] += series[0, feature]
        return total
    return series.sum(axis=1, keepdims=True)


def normalize_profiles(profiles, cortex_mask):
    """Apply literal CBIG fetch_data mean centering and normalization.

    Its all(series,2) condition intentionally leaves rows containing a centered
    zero unscaled; this detail matters for averaged/continuous profiles.
    """
    # CBIG_MSHBM_read_fmri explicitly casts stored profiles to MATLAB single.
    # The source reduces columns in order. Boolean row selection makes a
    # C-contiguous copy, which changes NumPy's sum algorithm at full seed size.
    series = np.array(profiles, dtype=np.float32, copy=True, order="F")
    mask = np.asarray(cortex_mask, dtype=bool)
    if series.ndim != 2 or mask.shape != (series.shape[0],):
        raise ValueError('Profile/mask dimensions disagree')
    if not np.isfinite(series).all():
        raise ValueError('Profiles must be finite; missing sessions are represented separately')
    series[~mask] = 0
    series -= _sum_profile_features(series) / np.float32(series.shape[1])
    active = np.all(series != 0, axis=1)
    norms = np.sqrt(_sum_profile_features(series * series))
    series[active] /= norms[active]
    return series


def centroids_from_labels(avg_profile, labels, *, num_clusters=15):
    """Initialize network directions with DU15NET labels in their native order."""
    series = np.asarray(avg_profile, dtype=np.float64)
    labels = np.asarray(labels)
    if series.ndim != 2 or labels.shape != (series.shape[0],):
        raise ValueError('Template labels and profiles have different vertex dimensions')
    if not np.isfinite(series).all() or not np.isfinite(labels).all():
        raise ValueError('Template labels and profiles must be finite')
    if np.any(labels != np.floor(labels)) or np.any((labels < 0) | (labels > num_clusters)):
        raise ValueError('Template labels must be integers from zero to num_clusters')
    mask = (labels != 0) & (series.sum(axis=1) != 0)
    x = series[mask] - series[mask].mean(axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('A template cortical profile has zero variance')
    x /= norms
    mu = np.zeros((series.shape[1], num_clusters))
    for network in range(1, num_clusters + 1):
        direction = x[labels[mask] == network].sum(axis=0)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(f'No usable profile for network {network}')
        mu[:, network - 1] = direction / norm
    return mu
