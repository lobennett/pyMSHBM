"""Estimate group priors via hierarchical Bayesian EM.

Ports CBIG_MSHBM_estimate_group_priors.m with three nested EM loops:
    inter-subject -> intra-subject -> session-level vMF.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pymshbm.math.vmf import cbig_inv_ad, cdln
from pymshbm.types import MSHBMParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker process state for multiprocessing
# ---------------------------------------------------------------------------

_worker_data = None


def _init_worker(data_path):
    """Initialize worker process with memory-mapped data."""
    global _worker_data
    _worker_data = np.load(str(data_path), mmap_mode="r")


def _create_pool(data, S):
    """Create a ProcessPoolExecutor if data is memory-mapped and S > 1."""
    if S <= 1:
        return None
    if not isinstance(data, np.memmap):
        return None
    data_path = getattr(data, "filename", None)
    if data_path is None:
        return None
    n_workers = min(S, os.cpu_count() or 1)
    logger.info("  Creating worker pool: %d workers for %d subjects",
                n_workers, S)
    return ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(data_path,),
    )


# ---------------------------------------------------------------------------
# Per-subject worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _weighted_data_subject_worker(s, s_lambda_s, T):
    """Worker: compute weighted_data for one subject."""
    data = _worker_data
    D = data.shape[1]
    L = s_lambda_s.shape[1]
    result = np.empty((D, L, T), dtype=np.float32)
    for t in range(T):
        result[:, :, t] = data[:, :, s, t].astype(np.float32).T @ s_lambda_s
    return s, result


def _kappa_subject_worker(s, s_lambda_s, s_t_nu_s, T):
    """Worker: compute kappa numerator partial sum for one subject."""
    data = _worker_data
    return _kappa_subject(data[:, :, s, :], s_lambda_s, s_t_nu_s)


def _kappa_subject(data_s, sl, nu_s):
    values = np.stack([
        np.sum(sl * _profile_dot(data_s[:, :, t], nu_s[:, :, t]).astype(np.float32), axis=0)
        for t in range(data_s.shape[2])
    ], axis=1)
    counts = np.sum(~np.isnan(values), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum(values, axis=1) / counts.astype(np.float32)


def _e_step_subject_worker(s, s_t_nu_s, kappa, log_c, theta, N, L, T):
    """Worker: literal CBIG E-step, including zero dot-product masking."""
    return (s, *_e_step_subject(_worker_data[:, :, s, :], s_t_nu_s,
                               kappa, log_c, theta))


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


def _em_cost_subject_worker(s, s_lambda_s, s_t_nu_s, kappa, log_c,
                            log_theta, L, T):
    """Worker: compute EM cost for one subject."""
    cost = _em_cost_subject(_worker_data[:, :, s, :], s_lambda_s,
                            s_t_nu_s, kappa, log_c, log_theta)
    return s, cost


def _log_for_cost(values):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(values)
    return np.where(np.isinf(result), np.log(np.finfo(float).eps ** 20), result)


def _profile_dot(X, nu):
    # Both MATLAB mtimes and pinned mtimesx cast the double operand to single
    # BEFORE multiplication whenever either non-scalar operand is single.
    if X.dtype == np.float32 or nu.dtype == np.float32:
        return X.astype(np.float32, copy=False) @ nu.astype(np.float32, copy=False)
    return X @ nu


def _reference_log_probability(X, nu, kappa, log_c):
    # MATLAB single Cdln dominates the mixed-precision addition.
    dot = _profile_dot(X, nu)
    return (kappa.astype(dot.dtype) * dot).astype(np.float32) + log_c


def _em_cost_subject(data_s, sl, nu_s, kappa, log_c, log_theta):
    total = _reference_log_probability(data_s[:, :, 0], nu_s[:, :, 0], kappa, log_c)
    for t in range(1, data_s.shape[2]):
        lv = _reference_log_probability(data_s[:, :, t], nu_s[:, :, t], kappa, log_c)
        total = np.nansum(np.stack((total, lv), axis=2), axis=2)
    return (np.sum(sl * total) + np.sum(sl * log_theta)
            - np.sum(sl * _log_for_cost(sl)))


def _initial_s_lambda_subject_worker(s, s_t_nu_s, kappa, log_c, N, L, T):
    """Worker: compute initial s_lambda for one subject."""
    data = _worker_data
    log_vmf_total = np.zeros((N, L), dtype=np.float32)
    for t in range(T):
        X = data[:, :, s, t]
        nu = s_t_nu_s[:, :, t]
        lv = _reference_log_probability(X, nu, kappa, log_c)
        log_vmf_total += np.where(np.isnan(lv), 0.0, lv)

    log_vmf_total -= log_vmf_total.max(axis=1, keepdims=True)
    sl = np.exp(log_vmf_total)
    row_sums = sl.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    sl /= row_sums

    zero_mask = np.all(log_vmf_total == 0, axis=1)
    sl[zero_mask] = 0.0
    return s, sl


# ---------------------------------------------------------------------------
# Main estimation functions
# ---------------------------------------------------------------------------

def estimate_group_priors(
    data: np.ndarray,
    g_mu: np.ndarray,
    settings: dict,
) -> MSHBMParams:
    """Estimate group priors from training data.

    Args:
        data: (N, D, S, T) normalized FC profiles.
        g_mu: (D, L) group-level cluster centroids.
        settings: Dict with keys: num_sub, num_session, num_clusters,
                  dim (D-1), ini_concentration, epsilon, conv_th, max_iter.
                  Optional numerical_max_iter (default 10000) raises if the
                  reference's uncapped M-step or intra update fails to converge.

    Returns:
        MSHBMParams with estimated group priors.
    """
    N, D, S, T = data.shape
    logger.info("  EM estimation: N=%d vertices, D=%d features, "
                "S=%d subjects, T=%d sessions, L=%d clusters",
                N, D, S, T, settings["num_clusters"])

    pool = _create_pool(data, S)
    try:
        params = initialize_params(data, g_mu, settings, pool=pool)

        cost_inter = 0.0
        record = []

        for iteration in range(1, settings["max_iter"] + 1):
            params.iter_inter = iteration

            # Reset intra-subject params each outer iteration
            L = settings["num_clusters"]
            S = settings["num_sub"]
            params.sigma = np.full(L, settings["ini_concentration"],
                                   dtype=np.float64)
            params.s_psi = np.tile(g_mu[:, :, np.newaxis], (1, 1, S))

            # The reference alternates session clustering and intra-subject
            # fitting to convergence before each inter-subject update.
            intra_cost = 0.0
            for intra_em in range(1, 51):
                params.kappa = np.full(L, settings["ini_concentration"], dtype=float)
                params.s_t_nu = np.tile(g_mu[:, :, None, None], (1, 1, T, S))
                params = vmf_clustering_subject_session(params, settings, data, pool=pool)
                params = intra_subject_var(params, settings)
                update_cost = _compute_inter_cost(params, settings, data, pool=pool)
                with np.errstate(divide="ignore", invalid="ignore"):
                    change = np.abs(np.divide(update_cost - intra_cost, intra_cost))
                intra_cost = update_cost
                if change <= settings["epsilon"] or intra_em >= 50:
                    break
            params.cost_intra = float(intra_cost)
            params = inter_subject_var(params, settings)
            # Upstream records the cost BEFORE the inter-subject update.
            update_cost = params.cost_intra
            params.cost_inter = update_cost
            record.append(float(update_cost))

            if iteration > 1 and abs(cost_inter) > 0:
                rel_change = abs((update_cost - cost_inter) / cost_inter)
                logger.info("  Outer iter %d/%d: cost=%.4f  rel_change=%.2e",
                            iteration, settings["max_iter"],
                            update_cost, rel_change)
                if rel_change <= settings["conv_th"]:
                    logger.info("  Converged at outer iteration %d",
                                iteration)
                    break
            else:
                logger.info("  Outer iter %d/%d: cost=%.4f",
                            iteration, settings["max_iter"], update_cost)
            cost_inter = update_cost

        params.record = record
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return params


def initialize_params(
    data: np.ndarray,
    g_mu: np.ndarray,
    settings: dict,
    pool: ProcessPoolExecutor | None = None,
) -> MSHBMParams:
    """Initialize all parameters from group centroids and data.

    Args:
        data: (N, D, S, T) normalized FC profiles.
        g_mu: (D, L) group centroids.
        settings: Problem settings dict.
        pool: Optional ProcessPoolExecutor for parallel computation.

    Returns:
        MSHBMParams with initial values.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    c0 = settings["ini_concentration"]
    dim = settings["dim"]

    mu = g_mu.copy()
    epsil = np.full(L, c0, dtype=np.float64)
    sigma = np.full(L, c0, dtype=np.float64)
    kappa = np.full(L, c0, dtype=np.float64)
    s_psi = np.tile(g_mu[:, :, np.newaxis], (1, 1, S))
    s_t_nu = np.tile(g_mu[:, :, np.newaxis, np.newaxis], (1, 1, T, S))

    # Initialize s_lambda via vMF log likelihood
    s_lambda = _compute_initial_s_lambda(
        data, s_t_nu, kappa, dim, L, pool=pool)

    # Initialize theta
    theta = _compute_theta(s_lambda)

    return MSHBMParams(
        mu=mu,
        epsil=epsil,
        sigma=sigma,
        theta=theta,
        kappa=kappa,
        s_psi=s_psi,
        s_t_nu=s_t_nu,
        s_lambda=s_lambda,
        iter_inter=0,
        record=[],
    )


def vmf_clustering_subject_session(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    pool: ProcessPoolExecutor | None = None,
) -> MSHBMParams:
    """Inter-region level EM: update kappa, s_t_nu, s_lambda, theta.

    Args:
        params: Current parameters.
        settings: Problem settings.
        data: (N, D, S, T) FC profiles.
        pool: Optional ProcessPoolExecutor for parallel computation.

    Returns:
        Updated MSHBMParams.
    """
    N, D, S, T = data.shape
    epsilon = settings["epsilon"]
    c0 = settings["ini_concentration"]

    cost = np.zeros(S, dtype=np.float32)

    for iter_em in range(1, 102):
        # M-step: update kappa and s_t_nu
        # weighted_data depends only on s_lambda (constant within M-step),
        # so compute it once before the M-step inner loop.
        weighted_data = _compute_weighted_data(
            data, params.s_lambda, S, T, pool=pool)
        _m_step(params, settings, data, epsilon, c0,
                weighted_data=weighted_data, pool=pool)

        # E-step and reference stopping cost use distinct masking rules.
        _e_step(params, settings, data, pool=pool)

        # Check convergence using the separately normalized cost likelihood.
        update_cost = _compute_em_cost(params, settings, data, pool=pool)
        with np.errstate(divide="ignore", invalid="ignore"):
            # MATLAB uses (relative_change > epsilon) == 0; NaN compares false.
            converged = ~(np.abs((update_cost - cost) / cost) > epsilon)
        if np.all(converged):
            params.cost_em = cost.copy()
            break
        if iter_em > 100:
            params.cost_em = update_cost.copy()
            logger.warning("Inner EM reached reference iteration limit (101)")
            break
        cost = update_cost

    logger.debug("    Inner EM: %d iterations, kappa=%.1f",
                 iter_em, float(params.kappa[0]))
    return params


def _compute_weighted_data(
    data: np.ndarray,
    s_lambda: np.ndarray,
    S: int,
    T: int,
    pool: ProcessPoolExecutor | None = None,
) -> np.ndarray:
    """Compute weighted_data[d,l,s,t] = sum_n data[n,d,s,t] * s_lambda[n,l,s].

    Uses explicit BLAS matmuls per (s,t) slice for guaranteed GEMM dispatch.
    Result shape: (D, L, S, T).
    """
    D = data.shape[1]
    L = s_lambda.shape[1]
    result = np.empty((D, L, S, T), dtype=np.float32)

    if pool is not None:
        futures = [
            pool.submit(_weighted_data_subject_worker,
                        s, s_lambda[:, :, s], T)
            for s in range(S)
        ]
        for future in futures:
            s, wd_s = future.result()
            result[:, :, s, :] = wd_s
    else:
        for s in range(S):
            sl = s_lambda[:, :, s]  # (N, L)
            for t in range(T):
                # (D, N) @ (N, L) -> (D, L)
                result[:, :, s, t] = data[:, :, s, t].astype(np.float32).T @ sl

    return result


def _m_step(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    epsilon: float,
    c0: float,
    weighted_data: np.ndarray | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> None:
    """M-step: update kappa and s_t_nu.

    Args:
        weighted_data: Precomputed (D, L, S, T) from _compute_weighted_data.
            When provided, avoids redundant recomputation within the inner loop.
        pool: Optional ProcessPoolExecutor for parallel kappa computation.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]
    dim = settings["dim"]

    if weighted_data is None:
        weighted_data = _compute_weighted_data(
            data, params.s_lambda, S, T, pool=pool)

    flag_nu = np.zeros((T, S), dtype=bool)
    for _ in range(settings.get("numerical_max_iter", 10000)):
        # Update kappa — accumulate directly without (N,L,S,T) intermediate
        if pool is not None:
            futures = [
                pool.submit(_kappa_subject_worker,
                            s, params.s_lambda[:, :, s],
                            params.s_t_nu[:, :, :, s], T)
                for s in range(S)
            ]
            kappa_parts = [f.result() for f in futures]
        else:
            kappa_parts = [
                _kappa_subject(data[:, :, s, :], params.s_lambda[:, :, s],
                               params.s_t_nu[:, :, :, s])
                for s in range(S)]
        kappa_num = np.sum(np.sum(np.stack(kappa_parts, axis=1), axis=1))
        kappa_den = np.sum(np.sum(np.sum(params.s_lambda, axis=0), axis=1))

        if not np.isfinite(kappa_den) or kappa_den <= 0:
            raise ValueError("Group estimation has no finite assigned vertices")
        old_kappa = params.kappa.copy()
        if kappa_den > 0:
            rbar = kappa_num / kappa_den
            kappa_new = cbig_inv_ad(dim, rbar)
            kappa_new = max(kappa_new, c0)
            if np.isinf(kappa_new):
                kappa_new = params.kappa[0]
            params.kappa = np.full(L, kappa_new)

        # Update s_t_nu using precomputed weighted_data
        # lambda_X[d,l,t,s] = kappa * weighted_data[d,l,s,t]
        #                     + sigma * s_psi[d,l,s]
        lambda_X = (
            params.kappa.astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
            * weighted_data.transpose(0, 1, 3, 2)  # (D, L, T, S)
            + (params.sigma[np.newaxis, :, np.newaxis, np.newaxis]
            * params.s_psi[:, :, np.newaxis, :]).astype(np.float32)  # (D, L, 1, S) -> T
        )
        norms = np.linalg.norm(lambda_X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        nu_new = lambda_X / norms

        # Convergence check: cosine similarity between old and new
        cos_sim = np.sum(nu_new * params.s_t_nu.astype(np.float32), axis=0)  # (L, T, S)
        cos_sim = np.where(np.isnan(cos_sim), 1.0, cos_sim)
        flag_nu |= np.all(1 - cos_sim < epsilon, axis=0)
        all_converged = (np.all(flag_nu) and
                         np.mean(np.abs(old_kappa - params.kappa) / old_kappa) < epsilon)

        params.s_t_nu[...] = nu_new

        if all_converged:
            break
    else:
        raise RuntimeError("CBIG M-step did not converge within numerical_max_iter")


def _e_step(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    pool: ProcessPoolExecutor | None = None,
) -> list[np.ndarray]:
    """E-step: update s_lambda and theta.

    Returns:
        List of per-subject (N, L) accumulated E-step log-vmf arrays.
        These are not interchangeable with the reference stopping likelihood.
    """
    N, D, S, T = data.shape
    L = settings["num_clusters"]

    # Precompute log normalizing constant (kappa is uniform across L)
    dim = settings["dim"]
    log_c = cdln(params.kappa, dim)

    if pool is not None:
        futures = [
            pool.submit(_e_step_subject_worker,
                        s, params.s_t_nu[:, :, :, s],
                        params.kappa, log_c, params.theta, N, L, T)
            for s in range(S)
        ]
        log_vmf_cache = [None] * S
        for future in futures:
            s, s_lambda_s, log_vmf_total_s = future.result()
            params.s_lambda[:, :, s] = s_lambda_s
            log_vmf_cache[s] = log_vmf_total_s
    else:
        log_vmf_cache = []
        for s in range(S):
            sl, log_vmf_total = _e_step_subject(
                data[:, :, s, :], params.s_t_nu[:, :, :, s],
                params.kappa, log_c, params.theta)
            params.s_lambda[:, :, s] = sl
            log_vmf_cache.append(log_vmf_total)

    params.theta = params.s_lambda.mean(axis=2)
    return log_vmf_cache


def intra_subject_var(
    params: MSHBMParams,
    settings: dict,
) -> MSHBMParams:
    """Update s_psi and sigma (intra-subject variability level)."""
    S = settings["num_sub"]
    dim = settings["dim"]
    epsilon = settings["epsilon"]

    flag_psi = np.zeros(S, dtype=bool)
    for intra_iter in range(1, settings.get("numerical_max_iter", 10000) + 1):
        # Update s_psi — vectorized over S
        # s_t_nu: (D, L, T, S), sum over T -> (D, L, S)
        accum = (
            np.nansum(params.sigma[np.newaxis, :, None, None] * params.s_t_nu, axis=2)
            + params.epsil[np.newaxis, :, np.newaxis]
            * params.mu[:, :, np.newaxis]
        )
        norms = np.linalg.norm(accum, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        s_psi_new = accum / norms

        # Convergence check — vectorized
        cos_sim = np.sum(s_psi_new * params.s_psi, axis=0)  # (L, S)
        flag_psi |= np.all(1 - cos_sim < epsilon, axis=0)
        all_converged = np.all(flag_psi)
        params.s_psi = s_psi_new

        # Update sigma using the executed CBIG concentration approximation.
        # Preserve upstream order: ordinary subject mean, THEN NaN session
        # mean. A session missing for any subject is excluded at this stage.
        dots = np.sum(params.s_psi[:, :, None, :] * params.s_t_nu, axis=0)
        subject_mean = np.mean(dots, axis=2)
        counts = np.sum(~np.isnan(subject_mean), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rbar_all = np.nansum(subject_mean, axis=1) / counts
        sigma_new = cbig_inv_ad(dim, rbar_all)

        # Preserve the signed denominator: quantized session directions can
        # transiently yield negative sigma in the literal upstream algorithm.
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_converged = (
                np.mean(np.abs(params.sigma - sigma_new) / params.sigma) < epsilon
            )
        if all_converged and sigma_converged:
            params.sigma = sigma_new
            break
        params.sigma = sigma_new
    else:
        raise RuntimeError("CBIG intra-subject update did not converge within numerical_max_iter")

    logger.debug("    Intra-subject: %d iterations, "
                 "sigma=[%s]", intra_iter,
                 ", ".join(f"{s:.1f}" for s in params.sigma))
    return params


def inter_subject_var(
    params: MSHBMParams,
    settings: dict,
) -> MSHBMParams:
    """Update mu and epsil (inter-subject variability level)."""
    S = settings["num_sub"]
    dim = settings["dim"]
    c0 = settings["ini_concentration"]

    # Update mu
    mu_update = params.s_psi.sum(axis=2)  # (D, L)
    norms = np.linalg.norm(mu_update, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    params.mu = mu_update / norms

    # Update epsil using the executed CBIG concentration approximation
    rbar_all = np.einsum("dls,dl->l", params.s_psi, params.mu) / S
    epsil_new = cbig_inv_ad(dim, rbar_all)
    epsil_new = np.maximum(epsil_new, c0)
    # Keep old values where result is inf
    inf_mask = np.isinf(epsil_new)
    epsil_new[inf_mask] = params.epsil[inf_mask]
    params.epsil = epsil_new

    return params


def _compute_initial_s_lambda(
    data: np.ndarray,
    s_t_nu: np.ndarray,
    kappa: np.ndarray,
    dim: int,
    L: int,
    pool: ProcessPoolExecutor | None = None,
) -> np.ndarray:
    """Compute initial s_lambda from vMF log likelihoods."""
    N, D, S, T = data.shape
    s_lambda = np.zeros((N, L, S), dtype=np.float32)
    log_c = cdln(kappa, dim)

    if pool is not None:
        futures = [
            pool.submit(_initial_s_lambda_subject_worker,
                        s, s_t_nu[:, :, :, s], kappa, log_c, N, L, T)
            for s in range(S)
        ]
        for future in futures:
            s, sl = future.result()
            s_lambda[:, :, s] = sl
    else:
        for s in range(S):
            log_vmf_total = np.zeros((N, L), dtype=np.float32)
            for t in range(T):
                X = data[:, :, s, t]
                nu = s_t_nu[:, :, t, s]
                lv = _reference_log_probability(X, nu, kappa, log_c)
                log_vmf_total += np.where(np.isnan(lv), 0.0, lv)

            log_vmf_total -= log_vmf_total.max(axis=1, keepdims=True)
            sl = np.exp(log_vmf_total)
            row_sums = sl.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            sl /= row_sums

            zero_mask = np.all(log_vmf_total == 0, axis=1)
            sl[zero_mask] = 0.0
            s_lambda[:, :, s] = sl

    return s_lambda


def _compute_theta(s_lambda: np.ndarray) -> np.ndarray:
    """Reference initialization: subject mean divided by nonzero subject count.

    Subsequent E-steps use an ordinary subject mean instead.
    """
    nonzero_count = np.sum(s_lambda != 0, axis=2)
    theta_sum = s_lambda.mean(axis=2)
    theta = np.zeros_like(theta_sum)
    mask = nonzero_count > 0
    theta[mask] = theta_sum[mask] / nonzero_count[mask]
    return theta


def _compute_em_cost(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    log_vmf_cache: list[np.ndarray] | None = None,
    pool: ProcessPoolExecutor | None = None,
) -> np.ndarray:
    """Compute per-subject EM cost.

    Args:
        log_vmf_cache: Retained for API compatibility but ignored: the
            E-step has different masking rules from the stopping cost.
        pool: Optional ProcessPoolExecutor for parallel computation.
    """
    N, D, S, T = data.shape
    costs = np.zeros(S, dtype=np.float32)
    log_theta = _log_for_cost(params.theta)
    log_c = cdln(params.kappa, settings["dim"])
    # E-step cache uses different zero-row/NaN rules from the reference cost;
    # retain the argument for compatibility, but recompute the cost likelihood.
    if pool is not None:
        futures = [
            pool.submit(_em_cost_subject_worker, s, params.s_lambda[:, :, s],
                        params.s_t_nu[:, :, :, s], params.kappa, log_c,
                        log_theta, settings["num_clusters"], T)
            for s in range(S)
        ]
        for future in futures:
            s, costs[s] = future.result()
    else:
        for s in range(S):
            costs[s] = _em_cost_subject(
                data[:, :, s, :], params.s_lambda[:, :, s],
                params.s_t_nu[:, :, :, s], params.kappa, log_c, log_theta)
    return costs


def _compute_inter_cost(
    params: MSHBMParams,
    settings: dict,
    data: np.ndarray,
    pool: ProcessPoolExecutor | None = None,
) -> float:
    """Compute total inter-subject cost."""
    em_cost = params.cost_em
    if em_cost is None:
        em_cost = _compute_em_cost(params, settings, data, pool=pool)
    session_dots = np.sum(
        params.s_psi[:, :, np.newaxis, :] * params.s_t_nu, axis=0)
    session_terms = (
        (params.sigma[:, np.newaxis, np.newaxis] * session_dots).astype(np.float32)
        + cdln(params.sigma, settings["dim"])[:, np.newaxis, np.newaxis])
    session_prior = np.nansum(np.sum(np.sum(session_terms, axis=0), axis=1))
    subject_dots = np.sum(params.mu[:, :, np.newaxis] * params.s_psi, axis=0)
    subject_prior = np.sum(
        (params.epsil[:, np.newaxis] * subject_dots).astype(np.float32)
        + cdln(params.epsil, settings["dim"])[:, np.newaxis])
    return float(np.sum(em_cost) + session_prior + subject_prior)
