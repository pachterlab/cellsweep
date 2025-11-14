"""Denoising count matrices using a Poisson + Negative Binomial model."""

import os
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from pydantic import validate_call, Field, ConfigDict
from typing import Annotated, Optional
from .utils import setup_logger, load_adata, determine_cutoff_umi_for_expected_cells, infer_empty_droplets, determine_cell_types

#* take the mean expression of each gene across all empty droplets, and normalize to sum to 1.
def infer_gene_ambient_fraction(adata, empty_droplet_method="threshold", umi_cutoff=None, expected_cells=None, verbose=0, quiet=False, logger=None):
    """
    input: adata with adata.obs: is_empty (optional)
    output: adata with adata.obs: is_empty, and adata.var: ambient_fraction
      - is_empty: boolean indicating whether each cell is an empty droplet or not. If not present, it will be inferred using infer_empty_droplets().
      - ambient_fraction: fraction of ambient RNA comprised by each gene, computed as the mean expression of that gene across all empty droplets divided by the mean expression across all cells. This is added to adata.var as a new column named "ambient_fraction".
    """
    if not logger:
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
def infer_celltype_profile(adata, celltype_key="celltype", empty_droplet_method="threshold", umi_cutoff=None, expected_cells=None, verbose=0, quiet=False, logger=None):
    """
    input: adata with adata.obs: is_empty (optional), celltype
    output: adata with adata.obs: is_empty, celltype, and adata.uns: celltype_profile
      - is_empty: boolean indicating whether each cell is an empty droplet or not. If not present, it will be inferred using infer_empty_droplets().
      - celltype: string indicating the cell type of each cell.
      - celltype_profile: DataFrame (n_celltypes x n_genes) - mean expression of each gene across all cells of that type.
    """
    if not logger:
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

def dense_integerize(expected_cell, random_state=None): 
    """
    Converts dense float matrix to integer matrix through stochastic rounding
    expected_cell : matrix of expected true-cell counts (float)
    Returns: matrix (int) with floor + multinomial-distributed residual.
    """
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

def sparse_integerize(expected_cell: sp.csr_matrix, random_state=None):
    """
    Converts sparse float matrix to integer matrix through stochastic rounding
    expected_cell : CSR matrix of expected true-cell counts (float)
    Returns: CSR matrix (int) with floor + multinomial-distributed residual.
    """
    if not sp.isspmatrix_csr(expected_cell):
        expected_cell = expected_cell.tocsr()

    rng = np.random.default_rng(random_state)

    N, G = expected_cell.shape
    indptr = expected_cell.indptr
    indices = expected_cell.indices
    data = expected_cell.data

    # triplets for sparse output
    rows = []
    cols = []
    vals = []

    for n in range(N):
        rs = indptr[n]
        re = indptr[n+1]
        idx = indices[rs:re]       # nonzero gene indices
        vals_row = data[rs:re]     # expected values (floats)

        if len(idx) == 0:
            continue

        # --- floor ---
        base = np.floor(vals_row).astype(int)

        # --- residual ---
        residual = vals_row - base
        rsum = residual.sum()

        if rsum > 0:
            probs = residual / rsum
            Tadd = int(round(rsum))
            # multinomial only on nz genes
            add = rng.multinomial(Tadd, probs)
        else:
            add = np.zeros_like(base, dtype=int)

        out_row = base + add

        # keep only nonzero entries (optional)
        nz = out_row > 0
        if np.any(nz):
            rows.extend([n] * np.sum(nz))
            cols.extend(idx[nz])
            vals.extend(out_row[nz])

    # build CSR int matrix
    return sp.csr_matrix((np.array(vals, dtype=int),
                          (np.array(rows, dtype=int), np.array(cols, dtype=int))),
                         shape=(N, G))

def sparse_em(C, alpha, beta, a, m_global, gamma, p, K, N, G, 
              max_iter, tol, freeze_empty, fixed_celltype, 
              real_mask, eps, dirichlet_lambda, 
              verbose, logger):
    """
    Helper for denoise_count_matrix. Performs sparse compatible EM on model
    """
    # sparse structure
    indptr = C.indptr
    indices = C.indices
    data = C.data

    prev_ll = -np.inf

    # ============ EM LOOP ============
    for it in range(1, max_iter + 1):

        p_numer = np.zeros((K, G), dtype=float)
        a_numer = np.zeros(G, dtype=float)
        numer_gamma = np.zeros((N, K), dtype=float)
        A_n = np.zeros(N, dtype=float)
        M_total = 0.0

        # triplists for sparse ambient and bulk expected matrices
        rows_A = []
        cols_A = []
        vals_A = []

        rows_M = []
        cols_M = []
        vals_M = []

        # triplists for expected CELL (already existed)
        rows_list = []
        cols_list = []
        vals_list = []

        ll = 0.0

        b_n = (1.0 - beta) * (1.0 - alpha)

        # Iterate over sparse matrix row-by-row
        for n in range(N):
            rs = indptr[n]; re = indptr[n+1]
            if rs == re:
                continue

            idx = indices[rs:re]
            vals = data[rs:re].astype(float)
            nnz = len(idx)

            # ============================
            #     EMPTY DROPLET CASE
            # ============================
            if freeze_empty and (not real_mask[n]):

                w_a = (1.0 - beta) * alpha[n] * a[idx]
                w_m = beta * m_global[idx]
                p_tot = w_a + w_m + eps

                # expected ambient
                c_A = vals * w_a / p_tot
                A_n[n] = c_A.sum()
                a_numer[idx] += c_A

                rows_A.extend([n] * nnz)
                cols_A.extend(idx.tolist())
                vals_A.extend(c_A.tolist())

                # expected bulk
                c_M = vals * w_m / p_tot
                M_total += c_M.sum()

                rows_M.extend([n] * nnz)
                cols_M.extend(idx.tolist())
                vals_M.extend(c_M.tolist())

                ll += float(np.dot(vals, np.log(p_tot)))
                continue

            # ============================
            #     REAL DROPLET CASE
            # ============================

            w_a = (1.0 - beta) * alpha[n] * a[idx]
            w_m = beta * m_global[idx]

            b = b_n[n]
            w_p_sum = np.zeros(nnz, dtype=float)
            w_p_k = np.zeros((K, nnz), dtype=float)

            for k in range(K):
                wpk = b * gamma[n, k] * p[k, idx]
                w_p_k[k] = wpk
                w_p_sum += wpk

            p_tot = w_a + w_m + w_p_sum + eps
            c_scale = vals / p_tot

            # expected ambient
            c_A = c_scale * w_a
            a_numer[idx] += c_A
            A_n[n] = c_A.sum()
            rows_A.extend([n] * nnz)
            cols_A.extend(idx.tolist())
            vals_A.extend(c_A.tolist())

            # expected bulk
            c_M = c_scale * w_m
            M_total += c_M.sum()
            rows_M.extend([n] * nnz)
            cols_M.extend(idx.tolist())
            vals_M.extend(c_M.tolist())

            # expected cell counts
            row_expected = np.zeros(nnz, dtype=float)
            for k in range(K):
                c_P = c_scale * w_p_k[k]
                numer_gamma[n, k] = c_P.sum()
                p_numer[k, idx] += c_P
                row_expected += c_P

            rows_list.extend([n] * nnz)
            cols_list.extend(idx.tolist())
            vals_list.extend(row_expected.tolist())

            ll += float(np.dot(vals, np.log(p_tot)))

        # ============================
        #     M STEP
        # ============================

        # update p
        if not fixed_celltype:
            for k in range(K):
                numer = p_numer[k] + dirichlet_lambda
                denom = numer.sum()
                p[k] = numer / max(denom, eps)

            row_sums = numer_gamma.sum(axis=1)
            nz = row_sums > 0
            gamma[nz] = numer_gamma[nz] / row_sums[nz, None]
            gamma[~nz] = gamma[~nz]

        # update alpha
        Ccell_n = numer_gamma.sum(axis=1)
        alpha = A_n / np.maximum(A_n + Ccell_n, eps)
        if freeze_empty:
            alpha[~real_mask] = 1.0

        # update beta
        total_counts = C.sum()
        beta = float(M_total / max(total_counts, 1.0))

        # update ambient profile
        a_numer += dirichlet_lambda
        a = a_numer / max(a_numer.sum(), eps)

        if verbose:
            logger.info(f"EM Iter {it:3d}: ll={ll:.3f} beta={beta:.6f}")

        if it > 1 and abs((ll - prev_ll) / max(abs(prev_ll), 1.0)) < tol:
            if verbose:
                logger.info("Converged.")
            break

        prev_ll = ll

    # ============================
    # CONSTRUCT EXPECTED MATRICES
    # ============================

    # C_A sparse
    C_expected_ambient = sp.csr_matrix(
        (np.array(vals_A), (np.array(rows_A), np.array(cols_A))),
        shape=(N, G)
    )

    # C_M sparse
    C_expected_bulk = sp.csr_matrix(
        (np.array(vals_M), (np.array(rows_M), np.array(cols_M))),
        shape=(N, G)
    )

    # C_cell sparse (if needed)
    C_expected_cell = sp.csr_matrix(
        (np.array(vals_list), (np.array(rows_list), np.array(cols_list))),
        shape=(N, G)
    )

    return C_expected_cell, C_expected_ambient, C_expected_bulk, alpha, beta, gamma, p, a, prev_ll

def dense_em(C, alpha, beta, a, m_global, gamma, p, K, N, G,
             max_iter, tol, freeze_empty, fixed_celltype,
             real_mask, eps, dirichlet_lambda,
             verbose, logger):
    """
    Helper for denoise_count_matrix. Performs EM on model (dense arrays only)
    """
    
    if real_mask is None:
        real_mask = np.ones(N, dtype=bool)

    prev_ll = -np.inf


    # if freeze_empty, ensure empties have alpha=1 and gamma=0
    if freeze_empty:
        alpha[~real_mask] = 1.0
        gamma[~real_mask, :] = 0.0

    for it in range(1, max_iter + 1):
        # Precompute per-row scalars and per-column vectors
        b_n = (1.0 - beta) * (1.0 - alpha)      # (N,)
        w_a = (1.0 - beta) * (alpha[:, None]) * (a[None, :])      # (N,G)
        w_m = beta * m_global[None, :]                            # (1,G) broadcast to (N,G)

        # Compute per-k contributions
        w_Pk = (gamma[:, :, None] * p[None, :, :]) * b_n[:, None, None]  # (N,K,G)

        # sum over k to get w_p_sum
        w_p_sum = np.sum(w_Pk, axis=1)

        # total mixture probabilities 
        p_tot = w_a + w_m + w_p_sum + eps  # (N,G)

        # E-step: expected counts per component
        c_scale = C / p_tot                             # (N,G)

        C_A = c_scale * w_a                             # ambient expected counts (N,G)
        C_M = c_scale * w_m                             # bulk expected counts (N,G)

        # For cell-type components: expected counts per k: C_Pk (N,K,G)
        C_Pk = c_scale[:, None, :] * w_Pk               # broadcasting -> (N,K,G)

        # Compose expected cell counts (sum over k)
        C_cell = np.sum(C_Pk, axis=1)                   # (N,G)

        # For freeze_empty: override contributions from empty droplets so empties only contribute to ambient
        if freeze_empty:
            # For empty droplet rows, force cell to 0, ambient uses full expected ambient
            empties = ~real_mask
            if np.any(empties):
                C_cell[empties, :] = 0.0

        # Sufficient statistics for M-step:
        p_numer = np.sum(C_Pk, axis=0)  # (K, G)
        a_numer = np.sum(C_A, axis=0)   # (G,)
        numer_gamma = np.sum(C_Pk, axis=2)  # (N,K)

        # per-cell ambient counts
        A_n = np.sum(C_A, axis=1)        # (N,)
        # total bulk expected
        M_total = np.sum(C_M)            # scalar

        # compute log-likelihood
        ll = float(np.sum(C * np.log(p_tot)))

        # ---------------- M-step ----------------
        if not fixed_celltype:
            # update p
            for k in range(K):
                numer = p_numer[k, :] + dirichlet_lambda
                denom = numer.sum()
                p[k, :] = numer / max(denom, eps)

            # update gamma for real droplets only (keep empties unchanged if freeze_empty)
            row_sums = np.sum(numer_gamma, axis=1)  # (N,)
            nz = row_sums > 0
            gamma_new = gamma.copy()
            gamma_new[nz, :] = numer_gamma[nz, :] / row_sums[nz, None]
            gamma = gamma_new

        # update alpha
        Ccell_n = np.sum(numer_gamma, axis=1)  # (N,)
        alpha = A_n / np.maximum(A_n + Ccell_n, eps)
        if freeze_empty:
            alpha[~real_mask] = 1.0

        # update beta (with numerical guard)
        total_counts = np.sum(C)
        beta = float(M_total / max(total_counts, 1.0))

        # update ambient profile a
        a = (a_numer + dirichlet_lambda)
        a = a / max(a.sum(), eps)

        if verbose and logger is not None:
            logger.info(f"EM Iter {it:3d}: ll={ll:.3f} beta={beta:.6f}")

        # convergence check
        if it > 1 and abs((ll - prev_ll) / max(abs(prev_ll), 1.0)) < tol:
            if verbose and logger is not None:
                logger.info("Converged.")
            break
        prev_ll = ll

    # final expected matrices
    C_expected_cell = C_cell    # ndarray (N,G)
    C_expected_ambient = C_A
    C_expected_bulk = C_M

    return C_expected_cell, C_expected_ambient, C_expected_bulk, alpha, beta, gamma, p, a, prev_ll


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def denoise_count_matrix(
    adata: str | ad.AnnData,
    adata_out: Annotated[str, Field(pattern=r"\.h5ad$")] = "adata_straightened.h5ad",
    max_iter: Annotated[int, Field(gt=0)] = 40,
    beta: Annotated[float, Field(ge=0, le=1)] = 0.03,
    eps: Annotated[float, Field(gt=0)] = 1e-9,
    dirichlet_lambda: Optional[Annotated[float, Field(gt=0)]] = 0.1,
    integer_out: bool = False,
    fixed_celltype: bool = False,
    freeze_empty: bool = True,
    empty_droplet_method: str = "threshold",
    umi_cutoff: Optional[Annotated[int, Field(ge=0)]] = None,
    expected_cells: Optional[Annotated[int, Field(ge=0)]] = None,
    cell_ambient_fraction: Annotated[float, Field(ge=0, le=1)] = 0.01,
    empty_droplet_celltype_name: str = "Empty Droplet",
    tol: Optional[Annotated[float, Field(ge=0)]] = 1e-6,
    random_state: Optional[Annotated[int, Field(ge=0)]] = 42,
    verbose: Annotated[int, Field(ge=-2, le=2)] = 0,
    quiet: bool = False,
    log_file: Optional[str] = None
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

    dirichlet_lambda: float, default 0.01
        Pseudocount

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

    # ensure empty droplets are present
    if "is_empty" not in adata.obs.columns:
        logger.info("Inferring empty droplets.")
        adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff,
                                     expected_cells=expected_cells, verbose=verbose, quiet=quiet, logger=logger)

    if "ambient_fraction" not in adata.var.columns:
        logger.info("Inferring gene ambient fractions.")
        adata = infer_gene_ambient_fraction(adata, empty_droplet_method=empty_droplet_method,
                                            verbose=verbose, quiet=quiet, logger=logger)

    if "celltype_profile" not in adata.uns or "celltype_names" not in adata.uns:
        logger.info("Inferring celltype profiles.")
        adata = infer_celltype_profile(adata, celltype_key="celltype",
                                       empty_droplet_method=empty_droplet_method,
                                       verbose=verbose, quiet=quiet, logger=logger)

    C = adata.X
    N, G = C.shape
    K = adata.uns["celltype_profile"].shape[0]

    # ambient profile from var
    a = np.asarray(adata.var["ambient_fraction"], dtype=float).ravel()

    # empty mask
    is_empty = np.asarray(adata.obs["is_empty"].copy(), dtype=bool)
    real_mask = ~is_empty
    Nr = real_mask.sum()

    # count parameters
    number_of_parameters = Nr + 1 + (Nr * K) + (K * G)  # alpha_i (Nr), beta (1), gamma_type (Nr * K), p_k (K * G)
    logger.info(f"Number of parameters in the cellmender model: {number_of_parameters:,} (alpha_i: {Nr:,}, beta: {1:,}, gamma_type: {Nr*K:,}, p_k: {K*G:,})")

    # celltype mapping
    z_true = adata.obs["celltype"].copy()
    z_true_str_to_int = {ct: i for i, ct in enumerate(adata.uns["celltype_names"])}

    # initialize p from uns
    p = np.asarray(adata.uns["celltype_profile"], dtype=float)
    for k in range(K):
        p[k] = (p[k] + dirichlet_lambda) / (p[k].sum() + G * dirichlet_lambda)

    # initialize gamma
    gamma = np.zeros((N, K), dtype=float)
    for i in range(N):
        ct = z_true.iloc[i]
        if ct in z_true_str_to_int and ct != empty_droplet_celltype_name:
            gamma[i, z_true_str_to_int[ct]] = 1.0
        else:
            gamma[i, :] = 1.0 / K

    # initial alpha
    if "cell_ambient_fraction" not in adata.obs.columns:
        logger.info("adata.obs does not have 'cell_ambient_fraction'. Setting to `cell_ambient_fraction` argument.")
        adata.obs.loc[real_mask, "cell_ambient_fraction"] = cell_ambient_fraction
        adata.obs.loc[~real_mask, "cell_ambient_fraction"] = 1.0
    alpha = np.asarray(adata.obs["cell_ambient_fraction"].copy(), dtype=float).ravel()
    alpha = np.clip(alpha, eps, 1.0 - eps)

    if freeze_empty:
        alpha[~real_mask] = 1.0
        gamma[~real_mask, :] = 0.0

    # initial beta + bulk m
    beta = float(beta)
    m_raw = np.array(C.sum(axis=0)).ravel().astype(float)
    m_global = (m_raw + dirichlet_lambda) / (m_raw.sum() + G * dirichlet_lambda)

    if sp.issparse(C):
        if not sp.isspmatrix_csr(C):
            C = sp.csr_matrix(C)
        C_expected_cell, C_expected_ambient, C_expected_bulk, alpha, beta, gamma, p, a, prev_ll = sparse_em(C, alpha, beta, a, m_global, gamma, p, K, N, G, 
                                                                                                            max_iter, tol, freeze_empty, fixed_celltype, real_mask,                                                                                                         
                                                                                                            eps, dirichlet_lambda, verbose, logger)
    else:
        np.asarray(C)
        C_expected_cell, C_expected_ambient, C_expected_bulk, alpha, beta, gamma, p, a, prev_ll = dense_em(C, alpha, beta, a, m_global, gamma, p, K, N, G, 
                                                                                                           max_iter, tol, freeze_empty, fixed_celltype, real_mask, 
                                                                                                           eps, dirichlet_lambda, verbose, logger)

    # ============================
    #      DENOISED COUNTS
    # ============================
    C_minus_A = C - C_expected_ambient
    C_minus_A_minus_M = C_minus_A - C_expected_bulk
    C_denoised = C_minus_A_minus_M.maximum(0)

    # ===================================
    # STORE RESULTS AND RETURN
    # ===================================

    adata.layers["denoised"] = C_denoised
    adata.obs["alpha_hat"] = alpha
    z_hat = np.full(N, -1, dtype=int)
    z_hat[real_mask] = np.argmax(gamma[real_mask], axis=1)
    adata.obs["z_hat"] = z_hat
    adata.uns["p_hat"] = p
    adata.uns["beta_hat"] = beta
    adata.var["ambient_hat"] = a
    adata.uns["loglike"] = prev_ll

    # Replace adata.X
    if integer_out:
        # integerization: optional
        if sp.issparse(C_denoised):
            C_integer =  sparse_integerize(C_denoised, random_state=random_state)
        else:
            C_integer =  dense_integerize(C_denoised, random_state=random_state)
        adata.X = C_integer
    else:
        adata.X = C_denoised

    if adata_out:
        logger.info(f"Saving inferred adata to {adata_out!r}")
        if os.path.dirname(adata_out):
            os.makedirs(os.path.dirname(adata_out), exist_ok=True)
        adata.write_h5ad(adata_out)

    return adata