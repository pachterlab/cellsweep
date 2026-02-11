"""Denoising count matrices using a Multinomial Mixture Model."""

import gc
import os
from datetime import datetime
from typing import Annotated, Optional, Union

import anndata as ad
import numpy as np
import scipy.sparse as sp
from numba import get_num_threads, set_num_threads, get_thread_id, njit, prange
from pydantic import ConfigDict, Field, validate_call

from .utils import infer_empty_droplets, load_adata, setup_logger


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

    if "is_empty" not in adata.obs.columns:
        logger.info("Inferring empty droplets since 'is_empty' not found in adata.obs.")
        adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)

    is_empty = np.asarray(adata.obs["is_empty"].copy(), dtype=bool)

    # Extract matrix and group info
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()  # efficient row access

    celltypes = adata.obs.loc[~is_empty, celltype_key].astype("category")
    celltypes = celltypes.cat.remove_unused_categories()
    unique_cts = celltypes.cat.categories

    # Preallocate array
    mean_expr = np.zeros((len(unique_cts), adata.n_vars), dtype=np.float32)

    # Compute mean expression per cell type
    for i, ct in enumerate(unique_cts):
        mask = (adata.obs[celltype_key] == ct).values
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

def sparse_integerize(expected_cell: sp.csr_matrix, random_state=None):
    """
    Converts sparse float matrix to integer matrix through stochastic rounding
    expected_cell : CSR matrix of expected true-cell counts (float)
    Returns: CSR matrix (int) with floor + multinomial-distributed residual.
    """
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
            # vectorized Bernoulli draws
            add = rng.binomial(1, residual)
        else:
            add = np.zeros_like(base)

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

@njit(parallel=True, nogil=True)
def warm_up_e_step_numba(indptr, indices, data, alpha, beta, a, m_global,
                         gamma_idx, p, N, eps, freeze_empty_mask):
    """
    Parallel ambient calculation over rows to get initial alpha trajectories
    """
    numer_gamma = np.zeros(N, dtype=np.float64)
    A_n = np.zeros(N, dtype=np.float64)

    # We will fill entries sequentially but rows are parallelized
    # Build an array mapping entry index to its position: compute per-row offsets
    for n in prange(N):
        rs = indptr[n]
        re = indptr[n + 1]
        if re <= rs:
            continue

        # local accumulators for this row
        local_A = 0.0

        # index into global flattened arrays
        # we will write to positions [rs:re) which correspond to the CSR data indices
        # this is safe because each row writes to a unique slice

        if freeze_empty_mask[n]:
            #-----------------------------#
            # Non-Cellular Barcode Update #
            #-----------------------------#
            for jj in range(rs, re):
                g = indices[jj]        # gene index
                val = data[jj]

                # compute mixture weights for this nonzero
                wa = (1.0 - beta) * alpha[n] * a[g]
                wm = beta * m_global[g]

                # treat only ambient+bulk
                p_tot = wa + wm

                # fraction scale: vals / p_tot
                scale = val / np.maximum(p_tot, eps)

                # expected ambient and bulk at this entry
                cA = scale * wa

                # accumulate per-row totals
                local_A += cA
        else:
            #--------------------------------#
            # Cell-Containing Barcode Update #
            #--------------------------------#

            b = (1.0 - beta) * (1.0 - alpha[n])
            local_gamma = 0.0
            k = gamma_idx[n]   # int in [0, K) or -1

            for jj in range(rs, re):
                g = indices[jj]
                val = data[jj]

                # mixture weights
                wa = (1.0 - beta) * alpha[n] * a[g]
                wm = beta * m_global[g]
                w_cell = b * p[k, g]

                p_tot = wa + wm + w_cell
                scale = val / np.maximum(p_tot, eps)

                # expected contributions
                cA = scale * wa
                cC = scale * w_cell

                local_gamma += cC

                # accumulate per-row totals
                local_A += cA
            
            numer_gamma[n] = local_gamma

        # write back per-row numbers
        A_n[n] = local_A
    
    return numer_gamma, A_n

def warm_up(indptr, indices, data, alpha, beta, a, m_global, gamma_idx, p, N, 
            freeze_empties, freeze_empty_mask, real_mask, eps, alpha_cap):
    
    """
    Helper to initialize exclude_from_p_update mask based on which barcodes 
    are inclined to have alpha_n > alpha_cap.
    """

    numer_gamma, A_n = warm_up_e_step_numba(indptr=indptr, indices=indices, data=data, 
                                            alpha=alpha, beta=beta, a=a, m_global=m_global,
                                            gamma_idx=gamma_idx, p=p, N=N, eps=eps, 
                                            freeze_empty_mask=freeze_empty_mask)
    

    # see which alpha_n values will exceed alpha_cap
    Ccell_n = numer_gamma
    alpha_test = A_n / np.maximum(A_n + Ccell_n, eps)
    if freeze_empties:
        alpha_test[~real_mask] = 1.0

    exclude_from_p_update = (alpha_test > (alpha_cap + 1e-6)) & (~freeze_empty_mask)


    return exclude_from_p_update


# ---------- Numba-parallel E-step kernel ----------
@njit(parallel=True, nogil=True)
def e_step_numba(indptr, indices, data, alpha, beta, a, m_global,
                 gamma_idx, p, K, N, eps, log_eps, freeze_empty_mask,
                 freeze_ambient_profile, exclude_from_p_update,
                 p_numer_tls, a_numer_tls, numer_gamma, A_n,
                 ll_row, M_row, ambient_vals, bulk_vals, done):
    """
    Parallel E-step over rows. Returns per-entry arrays and per-row summaries.
    """
    p_numer_tls.fill(0.0)
    a_numer_tls.fill(0.0)
    numer_gamma.fill(0.0)
    A_n.fill(0.0)
    ll_row.fill(0.0)
    M_row.fill(0.0)

    # We will fill entries sequentially but rows are parallelized
    # Build an array mapping entry index to its position: compute per-row offsets
    for n in prange(N):
        tid = get_thread_id()
        rs = indptr[n]
        re = indptr[n + 1]
        if re <= rs:
            continue

        # local accumulators for this row
        local_ll = np.float64(0.0)
        local_M = np.float64(0.0)
        local_A = np.float64(0.0)

        # index into global flattened arrays
        # we will write to positions [rs:re) which correspond to the CSR data indices
        # this is safe because each row writes to a unique slice

        if freeze_empty_mask[n]:
            #-----------------------------#
            # Non-Cellular Barcode Update #
            #-----------------------------#
            for jj in range(rs, re):
                g = indices[jj]        # gene index
                val = data[jj]

                # compute mixture weights for this nonzero
                wa = (1.0 - beta) * alpha[n] * a[g]
                wm = beta * m_global[g]

                # treat only ambient+bulk
                p_tot = wa + wm

                # fraction scale: vals / p_tot
                scale = val / np.maximum(p_tot, eps)

                # expected ambient and bulk at this entry
                cA = scale * wa
                cM = scale * wm

                if done:
                    ambient_vals[jj] = cA
                    bulk_vals[jj] = cM

                # accumulate per-row totals
                local_A += cA
                if not freeze_ambient_profile:
                    a_numer_tls[tid, g] += cA
                local_M += cM
                local_ll += val * np.log(np.maximum(p_tot, log_eps))
        else:
            #--------------------------------#
            # Cell-Containing Barcode Update #
            #--------------------------------#
            b = (1.0 - beta) * (1.0 - alpha[n])
            local_gamma = np.float64(0.0)
            k = gamma_idx[n]   # int in [0, K) or -1
            allow_p_update = not exclude_from_p_update[n]

            for jj in range(rs, re):
                g = indices[jj]
                val = data[jj]

                # mixture weights
                wa = (1.0 - beta) * alpha[n] * a[g]
                wm = beta * m_global[g]
                w_cell = b * p[k, g]

                p_tot = wa + wm + w_cell
                scale = val / np.maximum(p_tot, eps)

                # expected contributions
                cA = scale * wa
                cM = scale * wm
                cC = scale * w_cell

                if allow_p_update:
                    p_numer_tls[tid, k, g] += cC
                local_gamma += cC

                if done:
                    ambient_vals[jj] = cA
                    bulk_vals[jj] = cM

                # accumulate per-row totals
                local_A += cA
                if not freeze_ambient_profile:
                    a_numer_tls[tid, g] += cA
                local_M += cM
                local_ll += val * np.log(np.maximum(p_tot, log_eps))
            
            numer_gamma[n] = local_gamma

            # ---------- HARD CELLTYPE REASSIGNMENT ----------#
            if not allow_p_update:
                best_k = k
                best_ll = -1e300

                for kk in range(K):
                    llk = np.float64(0.0)
                    for jj in range(rs, re):
                        g = indices[jj]
                        val = data[jj]

                        # Full Likelihood
                        p_mix = (1-beta) * ((1.0 - alpha[n]) * p[kk, g] + alpha[n] * a[g]) + beta * m_global[g]
                        llk += val * np.log(np.maximum(p_mix, log_eps))

                    if llk > best_ll:
                        best_ll = llk
                        best_k = kk
                
                # Reassign Cell-type
                gamma_idx[n] = best_k

        # write back per-row numbers
        A_n[n] = local_A
        M_row[n] = local_M
        ll_row[n] = local_ll

    return ambient_vals, bulk_vals, numer_gamma, A_n, ll_row, M_row

def sparse_em(C, alpha, beta, a, u, m_global, gamma_idx, p, K, N, G, alpha_cap, 
              max_iter, del0_ll_tol, min_ll_tol, tol_p, tol_f, freeze_empties, real_mask, 
              eps, celltype_lambda, repulsion_strength, max_frac_gene_repulsion,
              log_eps, verbose, logger, freeze_ambient_profile):
    
    """
    Helper for denoise_count_matrix. Performs sparse compatible EM on multinomial model
    """

    exclude_from_p_update = np.zeros(N, dtype=np.bool_)

    converged = False
    ll_converged = False
    
    indptr = C.indptr
    indices = C.indices
    data = C.data

    nnz = data.shape[0]

    # freeze mask as boolean array for Numba
    if freeze_empties:
        freeze_empty_mask = ~real_mask
    else:
        freeze_empty_mask = np.zeros(N, dtype=np.bool_)

    prev_ll = None
    prev_p = None
    tol_adaptive = None
    prev_f = None

    delta_f = np.inf
    delta_p = np.inf

    # Precompute row_of_entry (map each nnz index to its row) -> used for constructing CSR from per-entry arrays
    row_of_entry = np.repeat(np.arange(N, dtype=np.int64), np.diff(indptr))

    if freeze_ambient_profile:
        exclude_from_p_update = warm_up(indptr, indices, data, alpha, beta, a, m_global, gamma_idx, p, N, 
                                        freeze_empties, freeze_empty_mask, real_mask, eps, alpha_cap)
    else:
        exclude_from_p_update[:] = False

    nthreads = get_num_threads()
    p_numer_tls = np.zeros((nthreads, K, G), dtype=np.float32)
    a_numer_tls = np.zeros((nthreads, G), dtype=np.float32)

    numer_gamma = np.zeros(N, dtype=np.float32)
    A_n = np.zeros(N, dtype=np.float32)
    ll_row = np.zeros(N, dtype=np.float32)
    M_row = np.zeros(N, dtype=np.float32)

    ambient_vals = np.zeros(nnz, dtype=np.float32)
    bulk_vals = np.zeros(nnz, dtype=np.float32)

    # EM loop
    for it in range(1, max_iter + 1):
        done = (it == max_iter or converged)

        # ============================
        #     E STEP (numba parallel)
        # ============================

        ambient_vals, bulk_vals, numer_gamma, A_n, ll_row, M_row = e_step_numba(
            indptr=indptr, indices=indices, data=data, alpha=alpha, beta=beta, a=a, m_global=m_global,
            gamma_idx=gamma_idx, p=p, K=K, N=N, eps=eps, log_eps=log_eps, freeze_empty_mask=freeze_empty_mask,
            freeze_ambient_profile=freeze_ambient_profile, exclude_from_p_update = exclude_from_p_update,
            p_numer_tls=p_numer_tls, a_numer_tls=a_numer_tls, numer_gamma=numer_gamma, A_n=A_n, 
            ll_row=ll_row, M_row=M_row, ambient_vals = ambient_vals, bulk_vals=bulk_vals, done=done
        )

        # Reduce per-row scalars
        ll = float(np.sum(ll_row)) / N # average per-cell log-likelihood
        M_total = float(np.sum(M_row))

        # Build a_numer
        if not freeze_ambient_profile:
            a_numer = a_numer_tls.sum(axis=0)

        # Build p_numer: shape (K, G) 
        p_numer = p_numer_tls.sum(axis=0)

        # ============================
        #     M STEP
        # ============================

        # update alpha
        Ccell_n = numer_gamma
        alpha = A_n / np.maximum(A_n + Ccell_n, eps)
        if freeze_empties:
            alpha[~real_mask] = 1.0

        if not ll_converged and freeze_ambient_profile:
            # Stage 1: Don't allow suspect cells to update p
            exclude_from_p_update = (alpha > alpha_cap + 1e-6) & (~freeze_empty_mask)
            # Apply cap to all (non-empty) cells in stage 1
            alpha = np.minimum(alpha, alpha_cap)
        else:
            # Stage 2: allow full alpha
            exclude_from_p_update[:] = False

        # update beta
        total_counts = M_total + A_n.sum() + numer_gamma.sum()
        beta = M_total / np.maximum(total_counts, eps)


        # update ambient profile if indicated
        if not freeze_ambient_profile:
            for i in range(3):
                denom = np.maximum(a, eps)
                R = (u[:, None] * p) / denom[None, :]
                u = (R * a_numer[None, :]).sum(axis=1)
                total = np.maximum(u.sum(), eps)
                u = u / total
                a = u @ p    # shape (G,)
                a = a / a.sum()

        # update p (with repulsion)
        p = p_numer + celltype_lambda

        if not ll_converged and freeze_ambient_profile:
            # Stage 1: repulsion 
            cluster_mass = p_numer.sum(axis=1)  
            repel_lambda_k = repulsion_strength * cluster_mass  

            sub = repel_lambda_k[:, None] * a[None, :]          
            sub_cap = max_frac_gene_repulsion * p               
            sub = np.minimum(sub, sub_cap)

            p = p - sub
            p = np.maximum(p, eps)
            p = p / np.maximum(p.sum(axis=1)[:, None], eps)
        else:
            # Stage 2: repulsion disabled
            p = p / p.sum(axis=1)[:, None]

        # ============================
        #     Stopping Conditions
        # ============================

        # Calculate f to check convergence of alpha and beta
        if freeze_empties: 
            f = (1 - beta) * alpha[real_mask] + beta
        else:
            f = (1 - beta) * alpha + beta

        if verbose:
            if freeze_empties:
                alpha_eff = alpha[real_mask]
            else:
                alpha_eff = alpha

            alpha_median = np.median(alpha_eff)
            alpha_mean = np.mean(alpha_eff)
            alpha_max = np.max(alpha_eff)
            alpha_min = np.min(alpha_eff)

            logger.info(f"EM Iter {it:3d}: ll={ll:.4f} log_delta_p={np.log(delta_p):.4f} min_alpha={alpha_min:.4f} mean_alpha={alpha_mean:.4f} median_alpha={alpha_median:.4f} max_alpha={alpha_max:.4f} beta={beta:.4f}")
            
            if not ll_converged and freeze_ambient_profile:
                logger.debug(f"{exclude_from_p_update.sum()} cells want to exceed alpha_n > {alpha_cap}. They will be excluded from update of p_k and allowed cell-type reassignment")

            if converged:
                logger.info("Converged.")

        # Set adaptive threshold based on first likelihood step
        if it == 2:
            delta0 = abs((ll - prev_ll))
            tol_adaptive = delta0 * del0_ll_tol
        
        # Check convergence
        if it > 1 and not converged:
            delta_p = np.max(np.sum(np.abs(p - prev_p), axis=1))
            delta_f = np.quantile(np.abs(f - prev_f), 0.9)
            min_abs_tol = min_ll_tol * max(abs(prev_ll), 1.0)
            tol_adaptive = max(tol_adaptive, min_abs_tol)

            if abs((ll - prev_ll)) < tol_adaptive and not ll_converged:
                logger.debug(f"Absolute change in log-likelihood is < {tol_adaptive:.4f} (adaptive_tol). Checking for parameter convergence.")
                ll_converged = True
            
            if ll_converged:
                if delta_p < tol_p and delta_f < tol_f: 
                    logger.debug(f"delta_p < {tol_p:.6f} delta_f < {tol_f:.6f}. Parameters have converged.")
                    converged = True

        prev_ll = ll
        prev_f = f.copy()
        prev_p = p.copy()

        if done:
            break

    denoised_vals = data.copy()
    denoised_vals -= ambient_vals
    denoised_vals -= bulk_vals
    denoised_vals[denoised_vals < 0] = 0

    del ambient_vals, bulk_vals, p_numer_tls, a_numer_tls
    del numer_gamma, ll_row, M_row
    gc.collect()


    # Construct CSR expected matrix from per-entry arrays
    C_denoised = sp.csr_matrix((denoised_vals, (row_of_entry, indices)), shape=(N, G))

    return {"C_denoised": C_denoised,
            "alpha": alpha, 
            "beta": beta, 
            "gamma_idx": gamma_idx, 
            "p": p, 
            "a": a, 
            "ll": ll}    


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def denoise_count_matrix(
    adata: Union[str, ad.AnnData],
    adata_out: Optional[Annotated[str, Field(pattern=r"\.h5ad$")]] = None,
    round_X: bool = False,
    threads: Annotated[int, Field(gt=0)] = 1,
    freeze_empties: bool = True,
    freeze_ambient_profile: bool = True,
    empty_droplet_method: Optional[str] = "threshold",
    umi_cutoff: Optional[Annotated[int, Field(ge=0)]] = None,
    expected_cells: Optional[Annotated[int, Field(ge=0)]] = None,
    init_alpha: Annotated[float, Field(ge=0.1, le=0.9)] = 0.9,
    init_beta: Annotated[float, Field(ge=0.1, le=0.9)] = 0.1,
    alpha_cap: Annotated[float, Field(ge=0, le=1)] = 0.9,
    repulsion_strength: Annotated[float, Field(ge=0, le=1e-3)] = 1e-4,
    max_frac_gene_repulsion: Annotated[float, Field(gt=0, le=1)] = 0.2,
    celltype_lambda: Optional[Annotated[float, Field(ge=0)]] = 50,
    ambient_lambda: Optional[Annotated[float, Field(ge=0)]] = 50,
    bulk_lambda: Optional[Annotated[float, Field(ge=0)]] = 10,
    eps: Annotated[float, Field(gt=0)] = 1e-12,
    log_eps: Annotated[float, Field(gt=0)] = 1e-300,
    max_iter: Annotated[int, Field(gt=1)] = 2000,
    del0_ll_tol: Annotated[float, Field(gt=0)] = 1e-3,
    min_ll_tol: Annotated[float, Field(gt=0)] = 1e-6,
    tol_p: Annotated[float, Field(gt=0)] = 1e-4,
    tol_f: Annotated[float, Field(gt=0)] = 1e-4,
    random_state: Optional[Annotated[int, Field(ge=0)]] = 42,
    inplace: bool = False,
    verbose: Annotated[int, Field(ge=-2, le=2)] = 0,
    quiet: bool = False,
    log_file: Optional[str] = None
):
    """
    Denoise a count matrix using the Expectation-Maximization (EM) algorithm to fit a
    multinomial mixture model that assigns each observed count to either ambient 
    contamination, bulk contamination, or true cell-type expression.

    This function iteratively estimates latent variables representing the per-cell ambient 
    fractions (alpha_i), the bulk contamination factor (beta), the per-cell-type expression 
    profiles (p_k), and optionally the ambient contamination profile (a) until convergence.

    Parameters
    ----------
    adata : str | AnnData
        Either an AnnData object or a path to an `.h5ad` file. Must contain:
        - `adata.X` : cell count matrix (cells x genes)
        - `adata.obs` :
            * `celltype` : categorical cell-type label for each cell
            * `is_empty` (optional) : boolean marking non-cellular barcodes. If absent,
              they are inferred using `empty_droplet_method`.
            * `init_alpha` (optional) : initial estimate of fraction of ambient contamination per cell;
              defaults to `init_alpha` argument if missing.
        - `adata.var` :
            * `ambient_profile` (optional) : per-gene ambient RNA fraction.
        - `adata.uns` :
            * `celltype_profile` (optional) : cell type matrix giving mean expression for each cell type (K x G); inferred if absent.
            * `celltype_profile_genes` (optional) : list of gene names corresponding to columns of `celltype_profile`.

    adata_out : str, default "adata_straightened.h5ad"
        Path to write the denoised AnnData object (must end with `.h5ad`).

    round_X : bool, default False
        If True, rounds denoised counts to nearest integer before saving.

    threads : int, default 1
        number of numba threads

    freeze_empties : bool, default True
        If True, does not attempt to reestimate the percent contamination of empty droplets

    freeze_ambient_profile: bool, default True
        If True, does not update the ambient profile (a) 

    empty_droplet_method : str, default "threshold"
        Strategy to infer non-cellular barcodes if `is_empty` is not present.
        Options may include "threshold", "quantile", or model-based approaches.

    umi_cutoff : int | None, default None
        Optional absolute UMI count threshold for classifying droplets as empty.

    expected_cells : int | None, default None
        Expected number of real cells, used when estimating thresholds.

    init_alpha : float, default 0.9
       Initial value of alpha_n for each cell if `ambient_profile` column is not present. If `freeze_ambient_profile=True`, this value does not 
       significantly effect the final result, so we set equal to alpha_cap for convenience. If `freeze_ambient_pofile=False`, then init_alpha
       can be set lower. For the sake of stability, we recommend that this value be far above the expected contamination rate, within [0.1, 0.9].

    init_beta : float, default 0.1
        Initial beta (percent bulk contamination) value for each cell. We do not recommend initializing beta below 0.1 for the sake of stability. 
        Bulk and ambient contamination are not fully separable, so we set to a lower value than alpha_init to bias the assignment of contamination
        to ambient rather than bulk.

    alpha_cap : float default 0.9
        alpha_n is not allowed to surpass this value in the first stage of training (before ll convergence). Barcodes that attempt to pass this threshold
        will be excluded from updating p_k and will be allowed to change cell-types. Disabled for `freeze_ambient_profile=False`.

    repulsion_strength : float, default 1e-4
        Strength of repulsion between ambient and cell-type profiles during M-step.
        Higher values lead to greater separation between ambient and cell-type profiles.
        Note that repulsion is disabled for `freeze_ambient_profile=False`.

    max_frac_gene_repulsion : float, default 0.2
        Maximum fraction of each p_k entry that can be subtracted during repulsion.
        Note that repulsion is disabled for `freeze_ambient_profile=False`.

    celltype_lambda: float, default 50
        Pseudocount for cell-type profile updates. Will be divided by the number of genes G. Higher values lead to smoother cell-type profiles.

    ambient_lambda: float, default 50
        Pseudocount for ambient profile update. Will be divided by the number of genes G. Higher values lead to a smoother ambient profile.

    bulk_lambda: float, default 10
        Pseudocount for bulk profile update. Will be divided by the number of genes G. Higher values lead to a smoother bulk profile.

    eps : float, default 1e-12
        Numerical stability constant to prevent division by zero.

    log_eps : float, default 1e-300
        Numerical stability constant to log(0).

    max_iter : int, default 1000
        Maximum number of EM iterations.

    del0_ll_tol: float, default 1e-3
        The change in likelihood, relative to the first likelihood step, below which repulsion and cell-type reassignment are discontinued and convergence is checked.
    
    min_ll_tol: float, default 1e-6
        The change in likelihood, relative to the current likelihood step, below which repulsion and cell-type reassignment are discontinued and convergence is checked.
        This is intended to cap `del0_ll_tol` at the edge of floating-point precision.

    tol_p: float, default 1e-4
        The maximum change in p below which training is discontinued. This is in addition to the tol_f stopping criterion.

    tol_f: float, default 1e-4
        The maximum change in f = (1 - beta) * alpha + beta, below which training is discontinued. This is in addition to the tol_p stopping criterion.

    random_state: int | None, default 42
        Random seed for stochastic rounding. Only necessary if `round_X=True`.

    inplace : bool, default False
        If False, copy anndata rather than modify inplace

    verbose : int, default 0
        Verbosity level (2 debug, 1 info, 0 warning, -1 error, -2 critical).

    quiet : bool, default False
        Suppresses most log output when True.

    log_file : str | None, default None
        Optional path to save EM iteration logs.

    Returns
    -------
    AnnData
        Denoised AnnData object with updated `adata.X`, and added fields:
        - `adata.layers["raw"]` : raw count matrix
        - `adata.obs["alpha_hat"]` : final optimized alpha values
        - `adata.obs["celltype_hat"]` : final cell-type assignments (These should not change)
        - `adata.var["ambient_hat"]` : final optimized ambient distribution
        - `adata.uns["p_hat"]` : final optimized matrix of cell-type profiles (K x G)
        - `adata.uns["beta_hat"]` : final optimized beta
        - `adata.uns["em_convergence"]` : diagnostics and log-likelihood trace
        - `adata.uns["loglike"]` : final log-likelihood (note that this value is not the 
           complete log-likelihood, only the relative log-likelihood)

    Notes
    -----
    The EM algorithm proceeds by:
      1. E-step: Update expected value of true, ambient noise, and bulk noise counts for each cell and gene.
      2. M-step: Update parameters (alpha, beta, p_k, a).
      3. Iterate until convergence (relative change in ll < `del0_ll_tol`) or reaching `max_iter`.
    """
    from cellsweep import __version__

    # set thread number
    set_num_threads(threads)
    
    logger = setup_logger(log_file=log_file, verbose=verbose, quiet=quiet)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting cellsweep denoising at {timestamp}, cellsweep version {__version__}")

    adata = load_adata(adata, logger=logger, inplace=inplace)
    if "celltype" not in adata.obs.columns:
        raise KeyError("adata.obs must have column \"celltype\".")

    # ensure empty droplets are present
    if "is_empty" not in adata.obs.columns:
        logger.info("Inferring empty droplets.")
        adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff,
                                     expected_cells=expected_cells, verbose=verbose, quiet=quiet, logger=logger)

    if "celltype_profile" not in adata.uns or "celltype_names" not in adata.uns:
        logger.info("Inferring celltype profiles.")
        adata = infer_celltype_profile(adata, celltype_key="celltype",
                                       empty_droplet_method=empty_droplet_method,
                                       verbose=verbose, quiet=quiet, logger=logger)
            
    num_empty_droplets = adata.obs["is_empty"].sum()
    RECOMMENDED_MIN_EMPTY_DROPLETS = 10_000
    if freeze_ambient_profile:
        if num_empty_droplets < 30:
            logger.warning(f"{num_empty_droplets} empty barcodes found. Setting freeze_ambient_profile=False, as at least 30 empty barcodes are required to keep this setting True. Ambient profile estimation may be unreliable.")
            freeze_ambient_profile = False
        elif num_empty_droplets < RECOMMENDED_MIN_EMPTY_DROPLETS:
            logger.warning(f"Number of empty droplets ({num_empty_droplets}) is less than the recommended minimum ({RECOMMENDED_MIN_EMPTY_DROPLETS}). "
                        "Ambient profile estimation may be unreliable.")
    
    adata.layers["raw"] = adata.X
    C = adata.X
    N, G = C.shape
    K = adata.uns["celltype_profile"].shape[0]
    celltype_lambda = celltype_lambda / G

    # Convert matrix to csr format
    is_dense = False
    if not sp.issparse(C) or not sp.isspmatrix_csr(C):
        if not sp.issparse(C):
            logger.info("Input cell x gene matrix is not sparse. Converting to sparse.")
            is_dense=True
        C = sp.csr_matrix(C)

    # empty mask
    is_empty = np.asarray(adata.obs["is_empty"].copy(), dtype=bool)
    real_mask = ~is_empty
    Nr = real_mask.sum()

    logger.info(f"Number of celltypes: {adata.obs.loc[real_mask, 'celltype'].nunique()}")

    # count parameters
    if freeze_ambient_profile:
        number_of_parameters = 1 + Nr + (K * G)  # alpha (Nr), beta (1), p_k (K * G)
        logger.debug(f"Number of parameters in the cellsweep model: {number_of_parameters:,} (alpha: {Nr:,}, beta: {1:,}, p_k: {K*G:,})")
    else:
        number_of_parameters = K + 1 + Nr + (K * G)  # u, alpha (Nr), beta (1), p_k (K * G)
        logger.debug(f"Number of parameters in the cellsweep model: {number_of_parameters:,} (u: {K:,}, alpha: {Nr:,}, beta: {1:,}, p_k: {K*G:,})")

    # celltype mapping
    z_true = adata.obs["celltype"].copy()
    z_true_str_to_int = {ct: i for i, ct in enumerate(adata.uns["celltype_names"])}

    # initialize p from uns
    p = np.asarray(adata.uns["celltype_profile"], dtype=float)
    for k in range(K):
        p[k] = (p[k] + celltype_lambda) / (p[k].sum() + G * celltype_lambda)

    # Initialize vectors that will keep track of gamma
    gamma_idx = np.full(N, -1, dtype=np.int64)
    mapped = z_true.map(z_true_str_to_int)
    mask = mapped.notna().to_numpy()
    gamma_idx[mask] = mapped[mask].to_numpy()
    # empties
    gamma_idx[is_empty] = -1

    if verbose:
        gamma_idx_init = gamma_idx.copy()

    # initial beta + bulk m
    beta = float(init_beta)
    m_raw = np.array(C.sum(axis=0)).ravel().astype(float) + bulk_lambda/G
    m_global = m_raw / m_raw.sum()

    if "ambient_profile" not in adata.var: 
        if freeze_ambient_profile:
            logger.info("Inferring the ambient profile from empty droplets.")
            # Ensure we have is_empty
            if "is_empty" not in adata.obs:
                logger.info("Inferring empty droplets since 'is_empty' not found in adata.obs.")
                
                adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)
                is_empty = adata.obs["is_empty"].values

            C_empty = C[is_empty,:]

            a_raw = np.array(C_empty.sum(axis=0)).ravel().astype(float) + ambient_lambda/G
            a = a_raw / (a_raw.sum())
        
            u = np.zeros(K)
        else:
            logger.info("Inferring the initial gene ambient profile from cell-types. The ambient profile will be updated during training.")
            valid = gamma_idx >= 0
            counts = np.bincount(gamma_idx[valid], minlength=K)
            u = counts / counts.sum()
            a = u @ p
            a = a / a.sum()
        adata.var["ambient_profile"] = a
    else:
        a = np.asarray(adata.var["ambient_profile"])
        valid = gamma_idx >= 0
        counts = np.bincount(gamma_idx[valid], minlength=K)
        u = counts / counts.sum()
        
    # initial alpha
    if "init_alpha" not in adata.obs.columns:
        logger.info("adata.obs does not have 'init_alpha'. Setting to `init_alpha` argument.")
        adata.obs["init_alpha"] = init_alpha
        adata.obs.loc[is_empty, "init_alpha"] = 1.
    alpha = np.asarray(adata.obs["init_alpha"].copy(), dtype=float).ravel()
    alpha = np.clip(alpha, eps, 1.0 - eps)

    if freeze_empties:
        alpha[is_empty] = 1.0

    alpha = alpha.astype(np.float64)
    a = a.astype(np.float32)
    p = p.astype(np.float32)
    C = C.astype(np.float32)

    logger.info(f"Performing Sparse EM with {get_num_threads()} Numba thread(s)")
    em_dict = sparse_em(C=C, alpha=alpha, beta=beta, a=a, u=u, m_global=m_global, gamma_idx=gamma_idx, p=p, K=K, N=N, G=G, alpha_cap=alpha_cap,
                        max_iter=max_iter, del0_ll_tol=del0_ll_tol, min_ll_tol=min_ll_tol, tol_p=tol_p, tol_f=tol_f, freeze_empties=freeze_empties,
                        real_mask=real_mask, eps=eps, celltype_lambda=celltype_lambda, repulsion_strength= repulsion_strength, max_frac_gene_repulsion=max_frac_gene_repulsion,
                        log_eps=log_eps, verbose=verbose, logger=logger, freeze_ambient_profile=freeze_ambient_profile)

    C_denoised = em_dict['C_denoised']
    alpha = em_dict["alpha"]
    beta = em_dict["beta"]
    p = em_dict["p"]
    a = em_dict["a"]
    ll = em_dict["ll"]

    if verbose:
        celltype_mod_num = (gamma_idx_init != gamma_idx).sum()
        logger.debug(f"The model reassigned the celltype of {celltype_mod_num} cells")


    # ===================================
    # STORE RESULTS AND RETURN
    # ===================================

    assert C_denoised.shape == (N, G), "Denoised matrix has incorrect shape."
    adata.obs["alpha_hat"] = alpha
    z_hat = np.full(N, -1, dtype=int)
    z_hat[real_mask] = gamma_idx[real_mask] + 1
    adata.obs["z_hat"] = z_hat
    adata.uns["p_hat"] = p
    adata.uns["init_beta_hat"] = beta
    adata.var["ambient_hat"] = a
    adata.uns["loglike"] = ll

    # Replace adata.X
    if round_X:
        # integerization: optional
        C_integer =  sparse_integerize(C_denoised, random_state=random_state)
        adata.X = C_integer
    else:
        adata.X = C_denoised

    if is_dense:
        logger.info("Re-densifying output.")
        # adata.X = np.asarray(adata.X)
        adata.X = adata.X.toarray()

    if adata_out:
        logger.info(f"Saving inferred adata to {adata_out!r}")
        if os.path.dirname(adata_out):
            os.makedirs(os.path.dirname(adata_out), exist_ok=True)
        adata.write_h5ad(adata_out)
    else:
        logger.warning("adata_out not specified; not saving inferred adata to a file.")

    return adata