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
from scipy.optimize import minimize

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

    # Store in adata.uns as a numeric matrix and metadata separately
    adata.uns["celltype_profile"] = mean_expr  # (K × G) matrix
    adata.uns["celltype_names"] = np.array(unique_cts)  # K-length array
    adata.uns["celltype_profile_genes"] = np.array(adata.var_names)  # G-length array

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
    freeze_empty: bool = True,
    empty_droplet_method: str = "threshold",
    umi_cutoff: Annotated[int | None, Field(ge=0)] = None,
    expected_cells: Annotated[int | None, Field(ge=0)] = None,
    cell_ambient_fraction: Annotated[float, Field(ge=0, le=1)] = 0.01,
    empty_droplet_celltype_name: str = "Empty Droplet",
    tol: Annotated[float, Field(ge=0)] = 1e-6,
    random_state: Annotated[int | None, Field(ge=0)] = 42,
    verbose: Annotated[int, Field(ge=-2, le=2)] = 0,
    quiet: bool = False,
    log_file: str | None = None
):
    """
    Denoise a count matrix using an Expectation-Maximization (EM) algorithm that
    models each observed count as a mixture of ambient RNA, bulk RNA, and true cell-type 
    signal.

    This function optionally operates on real cells only (excluding identified empty droplets),
    fixing the ambient expression profile and optionally fixing cell-type assignments.
    It iteratively estimates latent variables representing per-cell ambient fractions
    (alpha_i), a bulk contamination factor (beta), per-cell-type expression profiles 
    (p_k), and an ambient contamination profile (a) until convergence.

    Parameters
    ----------
    adata : str | AnnData
        Either an AnnData object or a path to an `.h5ad` file. Must contain:
        - `adata.X` : cell count matrix (cells x genes)
        - `adata.obs` :
            * `celltype` : categorical cell-type label for each cell
            * `is_empty` (optional) : boolean marking empty droplets. If absent,
              they are inferred using `empty_droplet_method`.
            * `cell_ambient_fraction` (optional) : fraction of ambient RNA per cell;
              defaults to `cell_ambient_fraction` argument if missing.
        - `adata.var` :
            * `ambient` (optional) : per-gene ambient RNA fraction.
        - `adata.uns` :
            * `celltype_profile` (optional) : cell type matrix giving mean expression for each cell type (K x G); inferred if absent.
            * `celltype_names` (optional) : list of cell type names corresponding to rows of `celltype_profile`.
            * `celltype_profile_genes` (optional) : list of gene names corresponding to columns of `celltype_profile`.

    adata_out : str, default "adata_straightened.h5ad"
        Path to write the denoised AnnData object (must end with `.h5ad`).

    max_iter : int, default 40
        Maximum number of EM iterations.

    beta : float, default 0.03
        Initial fraction of counts attributed to bulk RNA contamination

    eps : float, default 1e-9
        Numerical stability constant to prevent division by zero or log(0).

    integer_out : bool, default False
        If True, rounds denoised counts to nearest integer before saving.

    fixed_celltype : bool, default False
        If True, keeps cell-type assignments fixed during EM updates.

    freeze_empty : bool, default True
        If True, does not attempt to reestimate empty droplets

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

    tol: float, default 1e-6
        The relative change in likelihood below which training is discontinued

    random_state: int | None, default 42
        Random seed

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
      1. E-step: Update expected value of true, ambient noise, and bulk noise counts for each cell and gene.
      2. M-step: Update parameters (alpha, beta, gamma, p_k, a).
      3. Iterate until convergence (relative change in ll < `tol`) or reaching `max_iter`.

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

    if "celltype_profile" not in adata.uns or "celltype_names" not in adata.uns:
        logger.info("adata.uns does not have 'celltype_profile'. Inferring cell type profiles using infer_celltype_profile().")
        adata = infer_celltype_profile(adata, celltype_key="celltype", empty_droplet_method=empty_droplet_method, verbose=verbose, quiet=quiet)

    C = adata.X.astype(float)
    N, G = C.shape
    K = adata.uns["celltype_profile"].shape[0]

    a = adata.var["ambient_fraction"].copy()   # FIXED ambient
    is_empty = adata.obs["is_empty"].copy()   # FIXED empties
    z_true = adata.obs["celltype"].copy()
    z_true_str_to_int = {ct: i for i, ct in enumerate(adata.uns["celltype_names"])}

    # work only on real cells
    real_mask = ~is_empty
    real_mask = np.asarray(real_mask)   # convert from Series → ndarray
    Nr = real_mask.sum()

    #!!! densify
    if sp.issparse(C):
        C = C.toarray()  # convert to dense numpy.ndarray
    else:
        C = np.asarray(C)  # ensure dense

    # initial cell-type profiles: from truth
    p = adata.uns["celltype_profile"].copy()   # (K, G)

    # initial responsibilities: from truth if available
    gamma = np.zeros((N, K))
    for i in np.where(real_mask)[0]:
        if z_true.iloc[i] != empty_droplet_celltype_name:
            gamma[i, z_true_str_to_int[z_true.iloc[i]]] = 1.0
        else:
            gamma[i] = 1.0 / K
    
    if freeze_empty:
        drop_param_num = Nr
    else:
        drop_param_num = N

    number_of_parameters = (drop_param_num + G) * (K + 1) + 1  # alpha_i (N), beta, gamma_type (N x K), p_k (K x G), a (G)
    logger.info(f"Number of parameters in the cellmender model: {number_of_parameters:,} (alpha_i: {drop_param_num:,}, beta: {1:,}, gamma_type: {drop_param_num*K:,}, p_k: {K*G:,}, a: {G:,})")


    # initial alpha
    if "cell_ambient_fraction" not in adata.obs.columns:
        logger.info("adata.obs does not have 'cell_ambient_fraction'. Setting to `cell_ambient_fraction` argument.")
        adata.obs.loc[real_mask, "cell_ambient_fraction"] = cell_ambient_fraction
        adata.obs.loc[~real_mask, "cell_ambient_fraction"] = 1.0
    alpha = adata.obs["cell_ambient_fraction"].copy()

    alpha = np.asarray(alpha)
    a = np.asarray(a)
    p = np.asarray(p)
    prev_ll = -np.inf
    m = np.asarray(C.mean(axis=0)).ravel().astype(float)
    m = (m + eps) / (m.sum() + G * eps) # bulk mean profile

    for it in range(1, max_iter + 1):
        # ----- E-step -----
        
        # compute per-source weights w for each (n,g)
        # Ambient component: 
        w_A = (1.0 - beta) * alpha[:, None] * a[None, :]
        # Bulk component: 
        w_M = beta * m[None, :]
        # Cell-type components: 
        b_n = (1.0 - beta) * (1.0 - alpha)  # shape (N,)
        w_P_sum = np.zeros((N, G), dtype=float)
        w_Pk = np.zeros((N, K, G), dtype=float)  # N x K x G
        for k in range(K):
            # p[k] shape (G,)
            # gamma[:,k] shape (N,)
            comp = (b_n[:, None] * gamma[:, k][:, None]) * p[k][None, :]
            w_Pk[:, k, :] = comp
            w_P_sum += comp
        
        # total weight p_total per (n,g)
        p_total = w_A + w_M + w_P_sum
        # numerical safeguard: ensure p_total >= eps
        p_total = np.maximum(p_total, eps)
        
        # Expected counts for each source
        C_A = C * (w_A / p_total)
        C_M = C * (w_M / p_total)
        # For P_k:
        C_Pk = np.zeros_like(w_Pk)
        for k in range(K):
            C_Pk[:, k, :] = C * (w_Pk[:, k, :] / p_total)

        def normalize_vector(x, eps=1e-12):
            x = np.asarray(x, dtype=float)
            s = x.sum()
            if s <= 0:
                return np.full_like(x, 1.0 / x.size)
            return x / max(s, eps)

        # ----- M-step -----
        if not fixed_celltype:
            # Update p^k (with Dirichlet pseudocount)
            p_new = np.zeros_like(p)
            for k in range(K):
                numer = C_Pk[:, k, :].sum(axis=0) + 1 # pseudocount
                denom = numer.sum()
                if denom <= 0:
                    p_new[k, :] = normalize_vector(np.full(G, 1.0 / G), eps)
                else:
                    p_new[k, :] = numer / denom
            p = p_new
            
            numer_gamma = C_Pk.sum(axis=2)  # (N x K)
            denom_gamma = numer_gamma.sum(axis=1, keepdims=True)  # (N x 1)

            # Update gamma_n: fraction of cell-derived counts contributed by each k
            gamma_new = np.zeros_like(gamma)
            # avoid division by zero; if denom==0, leave gamma as previous (or uniform)
            zero_rows = (denom_gamma.squeeze() == 0)
            gamma_new[~zero_rows, :] = numer_gamma[~zero_rows, :] / denom_gamma[~zero_rows]
            if np.any(zero_rows):
                # keep old gamma for zero rows or set uniform
                gamma_new[zero_rows, :] = gamma[zero_rows, :]
            gamma = gamma_new

        if fixed_celltype:
            numer_gamma = C_Pk.sum(axis=2)  # (N x K)
        
        # Update alpha_n 
        A_n = C_A.sum(axis=1)  # expected ambient counts per droplet
        Ccell_n = numer_gamma.sum(axis=1)  # expected cell-derived counts per droplet
        alpha = (A_n) / (A_n + Ccell_n)
        
        # if freezing empty droplets:
        if freeze_empty and np.any(~real_mask):
            alpha[~real_mask] = 1.0
        
        # Update beta (global bulk fraction) from expected bulk counts
        M_total = C_M.sum()
        total_counts = C.sum()
        beta = M_total / max(total_counts, 1.0)
        
        # Update ambient profile a (with Dirichlet pseudocount)
        a_numer = C_A.sum(axis=0) + 1 # pseudocount
        a = a_numer / max(a_numer.sum(), eps)

        # compute (approx) log-likelihood for monitoring (multinomial log-likelihood)
        # For numerical stability compute log of p_total but use small eps already set
        ll_matrix = C * np.log(p_total)
        ll = ll_matrix.sum()

        logger.info(f"Iter {it+1:2d}: log likelihood (approx)={ll:.3f}")
        if it > 1:
            if np.abs((ll-prev_ll)/ max(1.0, np.abs(prev_ll))) < tol:
                break
        prev_ll = ll
    
    # After EM, compute denoised expectations
    C_expected_cell = C_Pk.sum(axis=1)  # sum over k, shape (N,G)
    C_expected_ambient = C_A
    C_expected_bulk = C_M
    
    # Optionally integerize expected counts via floor + multinomial residuals per droplet
    def integerize_floor_multinomial(expected_cell, random_state=None):
        rng2 = np.random.default_rng(random_state)
        N, G = expected_cell.shape
        Cout = np.zeros_like(expected_cell, dtype=int)
        for n in range(N):
            base = np.floor(expected_cell[n]).astype(int)
            residual = expected_cell[n] - base
            Tadd = int(round(residual.sum()))
            if Tadd > 0:
                probs = residual / residual.sum()
                add = rng2.multinomial(Tadd, probs)
            else:
                add = np.zeros(G, dtype=int)
            Cout[n, :] = base + add
        return Cout
    
    if integer_out:
        C_integer_cell = integerize_floor_multinomial(C_expected_cell, random_state=random_state)
        adata.X = C_integer_cell
    else:
        adata.X = np.clip(C - C_expected_ambient - C_expected_bulk, 0, None)

    # predicted labels for real cells
    z_hat = np.full(N, -1)
    z_hat = np.argmax(gamma, axis=1)
    z_hat[~real_mask] = -1

    assert adata.X.shape == (adata.n_obs, adata.n_vars)
    assert len(adata.obs) == len(is_empty) == len(alpha) == len(z_hat)
    assert adata.var.shape[0] == len(a) == p.shape[1]

    # --- Add inferred parameters back into obs/var/uns ---
    adata.obs["alpha_hat"] = alpha
    adata.obs["z_hat"] = z_hat
    adata.uns["p_hat"] = p
    adata.var["ambient_hat"] = a
    adata.uns["loglike"] = prev_ll

    if adata_out:
        logger.info(f"Saving inferred adata to {adata_out!r}")
        adata.write_h5ad(adata_out)
    
    return adata