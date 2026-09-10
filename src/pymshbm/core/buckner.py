"""Profile and atlas initialization steps in Buckner's executable workflow.

References: CBIG_ComputeCorrelationProfile, CBIG_IndCBM_generate_MSHBM_params,
and fetch_data in CBIG_MSHBM_estimate_group_priors. See docs/scientific-contract.md.
"""

import numpy as np


def binary_profiles(lh, rh, lh_cortex, rh_cortex, *, seed_vertices=642):
    """Return joint (vertices, seeds) binary profiles from (time, vertices).

    fsaverage meshes share vertex ordering: take cortex vertices among the
    first 642 fsaverage6 vertices, corresponding to fsaverage3. Correlations
    across BOTH hemispheres share one 10% cutoff; cutoff ties are retained.
    Constant cortical time series and a nonpositive threshold are rejected
    because they do not yield an interpretable sparse connectivity profile.
    No temporal denoising is performed here.
    """
    arrays = []
    seeds = []
    for values, mask in ((lh, lh_cortex), (rh, rh_cortex)):
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if values.ndim != 2 or values.shape[0] < 3:
            raise ValueError('Surface BOLD must have at least three timepoints')
        if mask.shape != (values.shape[1],) or not 0 < seed_vertices <= mask.size:
            raise ValueError('Cortex mask or seed resolution does not match BOLD')
        if not np.isfinite(values[:, mask]).all():
            raise ValueError('Nonfinite cortical BOLD values; inspect preprocessing')
        # Medial-wall NaNs are non-data. CBIG_corr turns their correlations to 0.
        values = values.copy()
        values[:, ~np.isfinite(values).all(axis=0)] = 0.
        centered = values - values.mean(axis=0)
        norms = np.linalg.norm(centered, axis=0)
        if np.any(norms[mask] == 0):
            raise ValueError('Constant cortical time series; inspect cortical coverage')
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


def normalize_profiles(profiles, cortex_mask):
    """Apply literal CBIG fetch_data mean centering and normalization.

    Its all(series,2) condition intentionally leaves rows containing a centered
    zero unscaled; this detail matters for averaged/continuous profiles.
    """
    # CBIG_MSHBM_read_fmri explicitly casts stored profiles to MATLAB single.
    series = np.array(profiles, dtype=np.float32, copy=True)
    mask = np.asarray(cortex_mask, dtype=bool)
    if series.ndim != 2 or mask.shape != (series.shape[0],):
        raise ValueError('Profile/mask dimensions disagree')
    if not np.isfinite(series).all():
        raise ValueError('Profiles must be finite; missing sessions are represented separately')
    series[~mask] = 0
    series -= series.mean(axis=1, keepdims=True)
    active = np.all(series != 0, axis=1)
    series[active] /= np.linalg.norm(series[active], axis=1, keepdims=True)
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
