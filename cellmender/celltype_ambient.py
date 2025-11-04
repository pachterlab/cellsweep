"""Denoising count matrices using a Poisson + Negative Binomial model."""

import os
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from pydantic import validate_call, Field, ConfigDict
from typing import Annotated
from .utils import setup_logger, load_adata, determine_cutoff_umi_for_expected_cells, infer_empty_droplets, determine_cell_types

#* take the mean expression of each gene across all empty droplets, and normalize to sum to 1.
def infer_gene_ambient_fraction(adata, empty_droplet_method="threshold", umi_cutoff=None, expected_cells=None, verbose=0, quiet=False):
    """
    input: adata with adata.obs: is_empty (optional)
    output: adata with adata.obs: is_empty, and adata.var: ambient_fraction
      - is_empty: boolean indicating whether each cell is an empty droplet or not. If not present, it will be inferred using infer_empty_droplets().
      - ambient_fraction: fraction of ambient RNA comprised by each gene, computed as the mean expression of that gene across all empty droplets divided by the mean expression across all cells. This is added to adata.var as a new column named "ambient_fraction".
    """
    logger = setup_logger(verbose=verbose, quiet=quiet)

    # adata = adata.copy()
    # Ensure we have is_empty
    if "is_empty" not in adata.obs:
        logger.info("Inferring empty droplets since 'is_empty' not found in adata.obs.")
        adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)

    is_empty = adata.obs["is_empty"].values

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

    # Mean across all cells
    if sp.issparse(X):
        mean_all = np.asarray(X.mean(axis=0)).ravel()
        mean_empty = np.asarray(X[is_empty].mean(axis=0)).ravel()
    else:
        mean_all = X.mean(axis=0)
        mean_empty = X[is_empty].mean(axis=0)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ambient_fraction = np.divide(mean_empty, mean_all, where=mean_all > 0)
        ambient_fraction = np.nan_to_num(ambient_fraction, nan=0.0, posinf=0.0, neginf=0.0)

    adata.var["ambient_fraction"] = ambient_fraction

    logger.info("Added 'ambient_fraction' to adata.var.")
    return adata

#* Take the mean expression of each gene across all cells of a given cell type, and normalize to sum to 1.
def infer_celltype_profile(adata, celltype_key="celltype", empty_droplet_method="threshold", umi_cutoff=None, expected_cells=None, verbose=0, quiet=False):
    """
    input: adata with adata.obs: is_empty (optional), celltype
    output: adata with adata.obs: is_empty, celltype, and adata.uns: celltype_profile
      - is_empty: boolean indicating whether each cell is an empty droplet or not. If not present, it will be inferred using infer_empty_droplets().
      - celltype: string indicating the cell type of each cell.
      - celltype_profile: DataFrame (n_celltypes x n_genes) - mean expression of each gene across all cells of that type.
    """
    logger = setup_logger(verbose=verbose, quiet=quiet)

    if celltype_key not in adata.obs:
        raise KeyError(f"{celltype_key!r} not found in adata.obs")
    
    # adata = adata.copy()

    if "is_empty" not in adata.obs.columns:
        logger.info("Inferring empty droplets since 'is_empty' not found in adata.obs.")
        adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)

    # Extract matrix and group info
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()  # efficient row access

    celltypes = adata.obs[celltype_key].astype("category")
    unique_cts = celltypes.cat.categories

    # Preallocate array
    mean_expr = np.zeros((len(unique_cts), adata.n_vars), dtype=np.float32)

    # Compute mean expression per cell type
    for i, ct in enumerate(unique_cts):
        mask = (celltypes == ct).values
        n_cells = mask.sum()
        if n_cells == 0:
            continue

        subX = X[mask]
        if sp.issparse(subX):
            mean_expr[i, :] = np.asarray(subX.mean(axis=0)).ravel()
        else:
            mean_expr[i, :] = subX.mean(axis=0)

    # Store in adata.uns
    adata.uns["celltype_profile"] = pd.DataFrame(
        mean_expr,
        index=unique_cts,
        columns=adata.var_names
    )

    return adata


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def denoise_count_matrix(
    adata: str | ad.AnnData,
    adata_out: Annotated[str, Field(pattern=r"\.h5ad$")] = "adata_straightened.h5ad",
    max_iter: Annotated[int, Field(gt=0)] = 40,
    beta: Annotated[float, Field(ge=0, le=1)] = 0.03,
    eps: Annotated[float, Field(gt=0)] = 1e-9,
    integer_out: bool = False,
    fixed_celltype: bool = False,
    empty_droplet_method: str = "threshold",
    umi_cutoff: Annotated[int | None, Field(ge=0)] = None,
    expected_cells: Annotated[int | None, Field(ge=0)] = None,
    cell_ambient_fraction: Annotated[float, Field(ge=0, le=1)] = 0.01,
    empty_droplet_celltype_name: str = "Empty Droplet",
    verbose: Annotated[int, Field(ge=-2, le=2)] = 0,
    quiet: bool = False,
    log_file: str | None = None,
):
    """
    Denoise a count matrix using an Expectation-Maximization (EM) algorithm that
    models each observed count as a mixture of ambient RNA and true cell-type signal.

    This function operates on real cells only (excluding identified empty droplets),
    fixing the ambient expression profile and optionally fixing cell-type assignments.
    It iteratively estimates latent variables representing per-cell ambient fractions
    (alpha_i) and per-cell-type expression profiles (p_k), until convergence.

    Parameters
    ----------
    adata : str | AnnData
        Either an AnnData object or a path to an `.h5ad` file. Must contain:
        - `adata.X` : count matrix (cells x genes)
        - `adata.obs` :
            * `celltype` : categorical cell-type label for each cell
            * `is_empty` (optional) : boolean marking empty droplets. If absent,
              they are inferred using `empty_droplet_method`.
            * `cell_ambient_fraction` (optional) : fraction of ambient RNA per cell;
              defaults to `cell_ambient_fraction` argument if missing.
        - `adata.var` :
            * `ambient` (optional) : per-gene ambient RNA fraction.
        - `adata.uns` :
            * `celltype_profile` (optional) : DataFrame (n_celltypes x n_genes)
              giving mean expression for each cell type; inferred if absent.

    adata_out : str, default "adata_straightened.h5ad"
        Path to write the denoised AnnData object (must end with `.h5ad`).

    max_iter : int, default 40
        Maximum number of EM iterations.

    beta : float, default 0.03
        Smoothing parameter controlling update strength between iterations.

    eps : float, default 1e-9
        Numerical stability constant to prevent division by zero or log(0).

    integer_out : bool, default False
        If True, rounds denoised counts to nearest integer before saving.

    fixed_celltype : bool, default False
        If True, keeps cell-type assignments fixed during EM updates.

    empty_droplet_method : str, default "threshold"
        Strategy to infer empty droplets if `is_empty` is not present.
        Options may include "threshold", "quantile", or model-based approaches.

    umi_cutoff : int | None, default None
        Optional absolute UMI count threshold for classifying droplets as empty.

    expected_cells : int | None, default None
        Expected number of real cells, used when estimating thresholds.
        
    cell_ambient_fraction : float, default 0.01
        Default ambient fraction assigned to each cell when missing.

    empty_droplet_celltype_name : str, default "Empty Droplet"
        Name used in `celltype` to denote empty droplets.

    verbose : int, default 0
        Verbosity level (-2: silent, 0: normal, 2: debug).

    quiet : bool, default False
        Suppresses most log output when True.

    log_file : str | None, default None
        Optional path to save EM iteration logs.

    Returns
    -------
    AnnData
        Denoised AnnData object with updated `adata.X`, and
        added fields such as:
        - `adata.layers["denoised"]` : denoised count matrix
        - `adata.obs["cell_ambient_fraction"]` : estimated ambient fraction per cell
        - `adata.uns["em_convergence"]` : diagnostics and log-likelihood trace

    Notes
    -----
    The EM algorithm proceeds by:
      1. E-step: Estimate posterior probabilities of counts being ambient vs. true.
      2. M-step: Update cell-type expression profiles and per-cell ambient fractions.
      3. Iterate until convergence or reaching `max_iter`.

    Designed primarily for benchmarking correctness of parameter updates rather than
    for production-level denoising on empty droplets.
    """
    logger = setup_logger(log_file=log_file, verbose=verbose, quiet=quiet)

    adata = load_adata(adata, logger=logger)
    
    if "celltype" not in adata.obs.columns:
        raise KeyError("adata.obs must have column celltype.")
    
    adata = adata.copy()
    
    if "is_empty" not in adata.obs.columns:
        logger.info("adata.obs does not have 'is_empty' column. Inferring empty droplets using infer_empty_droplets().")
        adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)

    if "ambient_fraction" not in adata.var.columns:
        adata = infer_gene_ambient_fraction(adata, empty_droplet_method=empty_droplet_method, verbose=verbose, quiet=quiet)

    if "celltype_profile" not in adata.uns:
        logger.info("adata.uns does not have 'celltype_profile'. Inferring cell type profiles using infer_celltype_profile().")
        adata = infer_celltype_profile(adata, celltype_key="celltype", empty_droplet_method=empty_droplet_method, verbose=verbose, quiet=quiet)

    X = adata.X.astype(float)
    N, G = X.shape
    K = adata.uns["celltype_profile"].shape[0]

    a = adata.var["ambient_fraction"].copy()   # FIXED ambient
    is_empty = adata.obs["is_empty"].copy()   # FIXED empties
    z_true = adata.obs["celltype"].copy()
    z_true_str_to_int = {ct: i for i, ct in enumerate(adata.uns["celltype_profile"].index)}

    # work only on real cells
    real_mask = ~is_empty
    real_mask = np.asarray(real_mask)   # convert from Series → ndarray
    Xr = X[real_mask]           # (Nr, G)
    Nr = Xr.shape[0]

    # initial cell-type profiles: from truth
    p = adata.uns["celltype_profile"].copy()   # (K, G)

    # initial responsibilities: from truth if available
    gamma_type = np.zeros((Nr, K))
    for j, i in enumerate(np.where(real_mask)[0]):
        if z_true[i] != empty_droplet_celltype_name:
            gamma_type[j, z_true_str_to_int[z_true[i]]] = 1.0
        else:
            gamma_type[j] = 1.0 / K
    
    number_of_parameters = (K * G) + Nr + K - 1  # p_k (KxG), alpha_i (Nr), m_k (K-1)
    logger.info(f"Number of parameters in the cellmender model: {number_of_parameters:,} (p_k: {K*G:,}, alpha_i: {Nr:,}, m_k: {K-1:,})")

    # initial alpha: from truth
    if "cell_ambient_fraction" not in adata.obs.columns:
        logger.info("adata.obs does not have 'cell_ambient_fraction'. Setting to `cell_ambient_fraction` argument.")
        adata.obs.loc[real_mask, "cell_ambient_fraction"] = cell_ambient_fraction
        adata.obs.loc[~real_mask, "cell_ambient_fraction"] = 1.0
    alpha = adata.obs["cell_ambient_fraction"][real_mask].copy()

    alpha = np.asarray(alpha)
    a = np.asarray(a)
    p = np.asarray(p)
    loglike_prev = -np.inf
    m = X.mean(axis=0) # bulk mean profile

    for it in range(max_iter):
        # --- E step on real cells ---

        # The log probability of cell j assuming cell type k
        log_p_type = np.zeros((Nr, K))

        # Calculate the log probability given current alpha and beta
        for k in range(K):
            pi = alpha[:, None] * a + (1 - alpha)[:, None] * ((1 - beta) * p[k] + beta * m)
            pi = np.clip(pi, eps, 1.0) # avoid log(0)
            pi /= pi.sum(axis=1, keepdims=True)
            log_p_type[:, k] = np.sum(Xr * np.log(pi), axis=1)


        # --- M step on real cells ---

        if not fixed_celltype:
            # softmax to get percent of each cell type based off of log_p as "gamma_type"
            log_p_type -= log_p_type.max(axis=1, keepdims=True)
            r = np.exp(log_p_type)
            r /= r.sum(axis=1, keepdims=True)
            gamma_type = r
            
            # update p_k
            for k in range(K):
                p[k] = (gamma_type[:, k][:, None] * Xr).sum(axis=0) + 1.0  # pseudocount
                p[k] /= p[k].sum()

        # Maximize Alpha and Beta
        from scipy.optimize import minimize

        # --- objective ---
        def negS(params):
            alphas = params[:-1]
            beta = params[-1]
            s = (1.0 - beta) * (gamma_type @ p) + beta * m
            t = alphas[:, None] * a + (1.0 - alphas)[:, None] * s
            if np.any(t <= 0):
                return np.inf
            return -np.sum(Xr * np.log(t))

        # --- analytic gradient ---
        def grad_negS(params):
            alphas = params[:-1]
            beta = params[-1]
            s = (1.0 - beta) * (gamma_type @ p) + beta * m
            t = alphas[:, None] * a + (1.0 - alphas)[:, None] * s
            if np.any(t <= 0):
                return np.ones_like(params) * np.inf

            dS_dalpha = np.sum(Xr * (a - s) / t, axis=1)
            dS_dbeta = np.sum((1.0 - alphas)[:, None] * Xr * ((m - (gamma_type @ p))) / t)
            grad = -np.concatenate([dS_dalpha, [dS_dbeta]])  # negate for minimize()
            return grad
        
        start = np.concatenate([alpha, [beta]])
        bounds = [(0.0, 1.0)] * (Nr + 1)
        res = minimize(negS, start, jac=grad_negS, method="L-BFGS-B", bounds=bounds)
        alpha = res.x[:-1]
        beta = res.x[-1]

        # log-likelihood on real cells
        loglike = -res.fun
        if verbose:
            print(f"Iter {it+1:2d}: logL(real)={loglike:.3f}")
        if np.abs(loglike - loglike_prev) < 1e-4:
            break
        loglike_prev = loglike

    # build outputs back to full size
    alpha_hat = np.zeros(N)
    alpha_hat[real_mask] = alpha
    alpha_hat[~real_mask] = 1.0  # empties

    # predicted labels for real cells
    z_hat = np.full(N, -1)
    z_hat[real_mask] = np.argmax(gamma_type, axis=1)

    # --- Store previous matrix ---
    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()

    # --- Reconstruct expected clean matrix ---
    X_hat = np.zeros((N, G), dtype=float)
    real_mask = ~is_empty

    # For each real cell, compute expected signal from its inferred mixture
    m = p.mean(axis=0)
    for i, j in enumerate(np.where(real_mask)[0]):
        k = z_hat[j]
        mix_cell = (1 - beta) * p[k] + beta * m
        pi_j = alpha_hat[j] * a + (1 - alpha_hat[j]) * mix_cell
        pi_j /= pi_j.sum()  # normalize per gene
        X_hat[j] = X[j].sum() * pi_j  # rescale to same total counts

    # # Empties are pure ambient
    # X_hat[~real_mask] = X[~real_mask].toarray() if sp.issparse(X) else X[~real_mask]

    if integer_out:
        X_hat = np.round(X_hat).astype(int)
    adata.X = X_hat
    assert adata.X.shape == (adata.n_obs, adata.n_vars)
    assert len(adata.obs) == len(is_empty) == len(alpha_hat) == len(z_hat)
    assert adata.var.shape[0] == len(a) == p.shape[1]

    # --- Add inferred parameters back into obs/var/uns ---
    adata.obs["is_empty_hat"] = is_empty
    adata.obs["alpha_hat"] = alpha_hat
    adata.obs["z_hat"] = z_hat
    adata.uns["p_hat"] = p
    adata.var["ambient_hat"] = a
    adata.uns["loglike"] = loglike

    if adata_out:
        logger.info(f"Saving inferred adata to {adata_out!r}")
        adata.write_h5ad(adata_out)
    
    return adata