"""Frozen Python E-step functions; see LICENSE-PYMSHBM.md.
Extracted verbatim from the executed 0.2.2 source for a bounded replay.
"""
import numpy as np
from scipy.special import iv as besseli

def _e_step_subject(data_s, nu_s, kappa, log_c, theta):
    N, _, T = data_s.shape
    L = len(kappa)
    dtype = np.float32 if data_s.dtype == np.float32 or nu_s.dtype == np.float32 else np.float64
    log_vmf_total = np.zeros((N, L), dtype=dtype)
    for t in range(T):
        lv = _profile_dot(data_s[:, :, t], nu_s[:, :, t]) * kappa.astype(dtype)
        add_constant = np.all(lv != 0, axis=1)
        lv[add_constant] = lv[add_constant].astype(np.float32) + log_c
        log_vmf_total += np.where(np.isnan(lv), 0, lv)
    zero_mask = np.any(log_vmf_total == 0, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        posterior = log_vmf_total.astype(np.float32) + np.log(theta)
        posterior -= posterior.max(axis=1, keepdims=True)
        sl = np.exp(posterior)
        sl /= sl.sum(axis=1, keepdims=True)
    sl[zero_mask] = 0.0
    return sl, log_vmf_total

def _profile_dot(X, nu):
    # Both MATLAB mtimes and pinned mtimesx cast the double operand to single
    # BEFORE multiplication whenever either non-scalar operand is single.
    if X.dtype == np.float32 or nu.dtype == np.float32:
        return X.astype(np.float32, copy=False) @ nu.astype(np.float32, copy=False)
    return X @ nu

def cdln(kappa: float | np.ndarray, d: int) -> float | np.ndarray:
    """Log normalizing constant of the vMF distribution.

    Uses numerical integration for large kappa to avoid Bessel overflow.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    scalar_input = kappa.ndim == 0
    kappa = np.atleast_1d(kappa)

    if d < 1200:
        k0 = 500
    elif d < 1800:
        k0 = 650
    else:
        raise ValueError(f"Dimension {d} too high, need to specify k0")

    out = (d / 2 - 1) * np.log(kappa) - np.log(besseli(d / 2 - 1, kappa))

    mask_overflow = kappa > k0
    if np.any(mask_overflow):
        fk0 = (d / 2 - 1) * np.log(k0) - np.log(besseli(d / 2 - 1, k0))
        kof = kappa[mask_overflow]
        n_grids = 1000
        ofintv = (kof - k0) / n_grids
        tempcnt = np.arange(1, n_grids + 1) - 0.5
        ks = k0 + np.outer(ofintv, tempcnt)
        half_d_minus_1 = 0.5 * (d - 1)
        ratio = half_d_minus_1 / ks
        adsum = np.sum(1.0 / (ratio + np.sqrt(1 + ratio**2)), axis=1)
        out[mask_overflow] = fk0 - ofintv * adsum

    out = out.astype(np.float32)
    if scalar_input:
        return float(out[0])
    return out
