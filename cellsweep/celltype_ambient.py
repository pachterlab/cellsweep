"""Denoising count matrices using a Multinomial Mixture Model."""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import anndata as ad
import scipy.sparse as sp
from pydantic import validate_call, Field, ConfigDict
import numba
from numba import njit, prange, get_num_threads, get_thread_id
from typing import Annotated, Optional, Tuple
from .utils import setup_logger, load_adata, determine_cutoff_umi_for_expected_cells, infer_empty_droplets, determine_cell_types  # , plot_cellsweep_likelihood_over_epochs
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, VBox, Output
from IPython.display import display
from sklearn.isotonic import IsotonicRegression
import seaborn as sns
import gc

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

def cdf_vals(arr, n_pts=200):
    """Return (xs, ys) for empirical CDF plotting (sorted unique + linear interpolation)."""
    a = np.asarray(arr).ravel()
    if a.size == 0:
        return np.array([0.0]), np.array([0.0])
    xs = np.sort(a)
    ys = np.linspace(0, 1, len(xs), endpoint=True)
    # optionally thin the CDF for plotting speed
    if len(xs) > n_pts:
        idx = np.linspace(0, len(xs) - 1, n_pts).astype(int)
        return xs[idx], ys[idx]
    return xs, ys

def plot_epoch(a_history, p_history, m, epoch, max_G_to_plot=1000):
    """
    Plot distributions for a single epoch.
    - a_history: list of arrays (epochs x G)
    - p_history: list of arrays (epochs x K x G)
    - m: array (G,)
    - epoch: integer epoch index
    - q_grid_a, q_levels: precomputed quantile grid for heatmap
    - max_k_to_plot: maximum number of p_k curves to plot (thins if K large)
    """
    a = np.asarray(a_history[epoch])
    P = np.asarray(p_history[epoch])   # (K, G)
    K = P.shape[0]
    G = P.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Sort genes by current ambient distribution ----
    gene_order = np.argsort(a)
    
    a_sorted = a[gene_order]
    m_sorted = m[gene_order]
    P_sorted = P[:, gene_order]

    if G > max_G_to_plot:
        a_sorted = a_sorted[-1*max_G_to_plot:]
        m_sorted = m_sorted[-1*max_G_to_plot:]
        P_sorted = P_sorted[:, -1*max_G_to_plot:]

    # ---------------------------------------------------------
    # 1. Rank-aligned gene curves
    # ---------------------------------------------------------
    plt.subplot(2,2,1)
    plt.plot(a_sorted, label="ambient a", lw=2)

    for k in range(K):
        plt.plot(P_sorted[k], alpha=0.5, lw=1, label=f"p_{k}")

    plt.plot(m_sorted, label="bulk m", lw=2)
    plt.title(f"Rank-aligned gene profiles (epoch {epoch})")
    plt.xlabel("Genes (sorted by a)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")

    # ---------------------------------------------------------
    # 2. Ambient to Bulk Ratio
    # ---------------------------------------------------------
    plt.subplot(2,2,2)
    ratio = a / m
    ratio = ratio[gene_order]
    if G > max_G_to_plot:
        ratio = ratio[-1*max_G_to_plot:]
    plt.plot(ratio)
    plt.title("Ambient to bulk ratio (a/m)")
    plt.xlabel("Genes (sorted by a)")
    plt.ylabel("Ratio")

    # -------------------------
    # 3) Ridge plot of a across epochs
    # -------------------------
    plt.subplot(2,2,4)
    offset = 0
    for t in range(len(a_history)):
        sns.kdeplot(a_history[t], bw_adjust=0.7, fill=True)
        offset += 1.2
    plt.title("Ridge plot of a over epochs")
    plt.yticks([])

    # -------------------------
    # 4) Same as 1 but sorted by final ambient distribution
    # -------------------------
    # ---- Sort genes by current ambient distribution ----
    gene_order = np.argsort(a_history[-1])
    
    a_sorted = a[gene_order]
    m_sorted = m[gene_order]
    P_sorted = P[:, gene_order]

    if G > max_G_to_plot:
        a_sorted = a_sorted[-1*max_G_to_plot:]
        m_sorted = m_sorted[-1*max_G_to_plot:]
        P_sorted = P_sorted[:, -1*max_G_to_plot:]

    plt.subplot(2,2,3)
    plt.plot(a_sorted, label="ambient a", lw=2)

    for k in range(K):
        plt.plot(P_sorted[k], alpha=0.5, lw=1, label=f"p_{k}")

    plt.plot(m_sorted, label="bulk m", lw=2)
    plt.title(f"Rank-aligned gene profiles (epoch {epoch})")
    plt.xlabel("Genes (sorted by a)")
    plt.ylabel("Probability")
    plt.legend(loc="upper right")

    plt.tight_layout()
    # display and then close figure so Jupyter doesn't duplicate it
    display(fig)
    plt.close(fig)


def interactive_distribution_viewer(a_history, p_history, m, max_G_to_plot=1000):
    """
    Create an interactive viewer that shows epoch slider and redraws plots into a single Output widget.
    """

    slider = IntSlider(
        min=0,
        max=len(a_history) - 1,
        step=1,
        value=0,
        description="Epoch:",
        continuous_update=False
    )

    out = Output()

    # initial draw
    with out:
        plot_epoch(a_history, p_history, m, epoch=0, max_G_to_plot=max_G_to_plot)

    def _on_change(change):
        if change['name'] == 'value' and change['type'] == 'change':
            epoch = change['new']
            with out:
                out.clear_output(wait=True)   # clear previous content
                plot_epoch(a_history, p_history, m, epoch=epoch, max_G_to_plot=max_G_to_plot)

    slider.observe(_on_change)
    display(VBox([slider, out]))

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

# ---------- Numba-parallel E-step kernel ----------
@njit(parallel=True, nogil=True)
def e_step_numba(indptr, indices, data, alpha, beta, a, m_global,
                 gamma_idx, p, K, N, eps, log_eps, freeze_empty_mask,
                 freeze_ambient_profile, fixed_celltype,
                 p_numer_tls, a_numer_tls, done):
    """
    Parallel E-step over rows. Returns per-entry arrays and per-row summaries.
    """

    nnz_total = data.shape[0]
    ambient_vals = np.zeros(nnz_total, dtype=np.float64)
    bulk_vals = np.zeros(nnz_total, dtype=np.float64)

    p_numer_tls.fill(0.0)
    a_numer_tls.fill(0.0)

    numer_gamma = np.zeros(N, dtype=np.float64)
    A_n = np.zeros(N, dtype=np.float64)
    ll_row = np.zeros(N, dtype=np.float64)
    M_row = np.zeros(N, dtype=np.float64)

    # We will fill entries sequentially but rows are parallelized
    # Build an array mapping entry index to its position: compute per-row offsets
    for n in prange(N):
        tid = get_thread_id()
        rs = indptr[n]
        re = indptr[n + 1]
        if re <= rs:
            continue

        # local accumulators for this row
        local_ll = 0.0
        local_M = 0.0
        local_A = 0.0

        # index into global flattened arrays
        # we will write to positions [rs:re) which correspond to the CSR data indices
        # this is safe because each row writes to a unique slice

        if freeze_empty_mask[n]:
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
                cM = scale * wm
                cC = scale * w_cell

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

            # ---------- HARD REASSIGNMENT (CEM) ----------
            if not fixed_celltype:
                best_k = k
                best_ll = -1e300

                for kk in range(K):
                    llk = 0.0
                    for jj in range(rs, re):
                        g = indices[jj]
                        val = data[jj]

                        # Full Likelihood
                        p_mix = (1-beta) * ((1.0 - alpha[n]) * p[kk, g] + alpha[n] * a[g]) + beta * m_global[g]
                        llk += val * np.log(np.maximum(p_mix, log_eps))

                    if llk > best_ll:
                        best_ll = llk
                        best_k = kk
                
                gamma_idx[n] = best_k

        # write back per-row numbers
        A_n[n] = local_A
        M_row[n] = local_M
        ll_row[n] = local_ll
    
    if not done:
        ambient_vals = None
        bulk_vals = None

    return ambient_vals, bulk_vals, numer_gamma, A_n, ll_row, M_row

def sparse_em(C, alpha, beta, a, u, m_global, gamma_idx, p, K, N, G, 
              max_iter, tol, min_tol, freeze_empty, fixed_celltype, real_mask, 
              eps, dirichlet_lambda, log_eps, verbose, logger, 
              freeze_ambient_profile, debug):
    
    """
    Helper for denoise_count_matrix. Performs sparse compatible EM on model
    """
    
    converged = False
    
    if debug:
        a_tracker = []
        p_tracker = []
        
        a_tracker.append(a)
        p_tracker.append(p)
    else: 
        a_tracker = None
        p_tracker = None
    
    indptr = C.indptr.copy()
    indices = C.indices.copy()
    data = C.data.copy() 

    nnz = data.shape[0]

    # freeze mask as boolean array for Numba
    if freeze_empty:
        freeze_empty_mask = ~real_mask
    else:
        freeze_empty_mask = np.zeros(N, dtype=np.bool_)

    prev_ll = None
    tol_adaptive = None

    # Precompute row_of_entry (map each nnz index to its row) -> used for constructing CSR from per-entry arrays
    row_of_entry = np.empty(nnz, dtype=np.int64)
    for n in range(N):
        rs = indptr[n]; re = indptr[n+1]
        if re > rs:
            row_of_entry[rs:re] = n

    # EM loop
    for it in range(1, max_iter + 1):
        done = (it == max_iter or converged)

        # ============================
        #     E STEP (numba parallel)
        # ============================
        nthreads = get_num_threads()
        p_numer_tls = np.zeros((nthreads, K, G), dtype=np.float64)
        a_numer_tls = np.zeros((nthreads, G), dtype=np.float64)

        ambient_vals, bulk_vals, numer_gamma, A_n, ll_row, M_row = e_step_numba(
            indptr=indptr, indices=indices, data=data, alpha=alpha, beta=beta, a=a, m_global=m_global,
            gamma_idx=gamma_idx, p=p, K=K, N=N, eps=eps, log_eps=log_eps, freeze_empty_mask=freeze_empty_mask,
            freeze_ambient_profile=freeze_ambient_profile, fixed_celltype=fixed_celltype,
            p_numer_tls=p_numer_tls, a_numer_tls=a_numer_tls, done=done
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

        # update p 
        p = p_numer + dirichlet_lambda
        denoms = p.sum(axis=1)
        p = p / np.maximum(denoms[:, None], eps)

        # update alpha
        Ccell_n = numer_gamma
        alpha = A_n / np.maximum(A_n + Ccell_n, eps)
        if freeze_empty:
            alpha[~real_mask] = 1.0

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

        # ============================
        #     Stopping Conditions
        # ============================

        if verbose:
            if freeze_empty:
                alpha_eff = alpha[real_mask]
            else:
                alpha_eff = alpha

            alpha_median = np.median(alpha_eff)
            alpha_mean = np.mean(alpha_eff)
            alpha_max = np.max(alpha_eff)
            alpha_min = np.min(alpha_eff)

            logger.info(f"EM Iter {it:3d}: ll={ll:.3f} min_alpha={alpha_min:.4f} mean_alpha={alpha_mean:.4f} median_alpha={alpha_median:.4f} max_alpha={alpha_max:.4f} beta={beta:.6f}")

            if converged:
                logger.info("Converged.")

        if it == 2:
            delta0 = abs((ll - prev_ll))
            tol_adaptive = delta0 * tol
        
        if it > 1 and not converged:
            min_abs_tol = min_tol * max(abs(prev_ll), 1.0)
            tol_adaptive = max(tol_adaptive, min_abs_tol)

            # Relative change in likelihood stopping condition
            if abs((ll - prev_ll)) < tol_adaptive:
                logger.debug(f"Stopping early because change in log-likelihood is < {tol_adaptive} (tol_adaptive)")
                converged = True

        if debug:
            a_tracker.append(a.copy())
            p_tracker.append(p.copy())

        prev_ll = ll

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

    if debug:
        return {"C_denoised": C_denoised,
                "alpha": alpha, 
                "beta": beta, 
                "gamma_idx": gamma_idx, 
                "p": p, 
                "a": a, 
                "ll": ll, 
                "a_tracker": a_tracker, 
                "p_tracker": p_tracker}
    else:
        return {"C_denoised": C_denoised,
                "alpha": alpha, 
                "beta": beta, 
                "gamma_idx": gamma_idx, 
                "p": p, 
                "a": a, 
                "ll": ll}    


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def denoise_count_matrix(
    adata: str | ad.AnnData,
    adata_out: Optional[Annotated[str, Field(pattern=r"\.h5ad$")]] = "adata_denoised.h5ad",
    max_iter: Annotated[int, Field(gt=1)] = 500,
    init_alpha: Annotated[float, Field(ge=0, le=1)] = 0.9,
    beta: Annotated[float, Field(ge=0, le=1)] = 0.1,
    eps: Annotated[float, Field(gt=0)] = 1e-12,
    log_eps: Annotated[float, Field(gt=0)] = 1e-300,
    dirichlet_lambda: Optional[Annotated[float, Field(ge=0)]] = 10,
    integer_out: bool = False,
    threads: Optional[Annotated[int, Field(gt=0)]] = 1,
    fixed_celltype: bool = True,
    freeze_empty: bool = True,
    freeze_ambient_profile: bool = True,
    empty_droplet_method: str = "threshold",
    ambient_threshold: Optional[Annotated[float, Field(ge=0, le=1)]] = 0.0,
    umi_cutoff: Optional[Annotated[int, Field(ge=0)]] = None,
    expected_cells: Optional[Annotated[int, Field(ge=0)]] = None,
    tol: Annotated[float, Field(gt=0)] = 1e-3,
    min_tol: Annotated[float, Field(gt=0)] = 1e-6,
    random_state: Optional[Annotated[int, Field(ge=0)]] = 42,
    verbose: Annotated[int, Field(ge=-2, le=2)] = 0,
    quiet: bool = False,
    log_file: Optional[str] = None,
    debug: bool = False,
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
            * `cell_ambient_fraction` (optional) : initial estimate of fraction of ambient RNA per cell;
              defaults to `cell_ambient_fraction` argument if missing.
        - `adata.var` :
            * `ambient` (optional) : per-gene ambient RNA fraction.
        - `adata.uns` :
            * `celltype_profile` (optional) : cell type matrix giving mean expression for each cell type (K x G); inferred if absent.
            * `celltype_names` (optional) : list of cell type names corresponding to rows of `celltype_profile`.
            * `celltype_profile_genes` (optional) : list of gene names corresponding to columns of `celltype_profile`.

    adata_out : str, default "adata_straightened.h5ad"
        Path to write the denoised AnnData object (must end with `.h5ad`).

    max_iter : int, default 500
        Maximum number of EM iterations.

    init_alpha : float, default 0.9
       Initial value of alpha_n for each cell. Works better when set to a higher number than expected (expected is around 0.05 per cell).
    
    beta : float, default 0.1
        Initial beta (percent bulk contamination) value for each cell. Works better when set to a higher number than expected (expected is around 0.05).

    eps : float, default 1e-12
        Numerical stability constant to prevent division by zero.

    log_eps : float, default 1e-300
        Numerical stability constant to log(0).

    dirichlet_lambda: float, default 10
        Pseudocount. Will be divided by the number of genes G

    integer_out : bool, default False
        If True, rounds denoised counts to nearest integer before saving.

    threads : int, default 1
        number of numba threads

    fixed_celltype : bool, default False
        If True, keeps cell-type assignments fixed during EM updates.

    freeze_empty : bool, default True
        If True, does not attempt to reestimate the percent contamination of empty droplets from 100%

    freeze_ambient_profile: bool, default True
        If True, does not update the ambient profile (a) based upon alpha

    empty_droplet_method : str, default "threshold"
        Strategy to infer empty droplets if `is_empty` is not present.
        Options may include "threshold", "quantile", or model-based approaches.
    
    ambient_threshold : float | None, default 0.0
        Optional ambient RNA fraction threshold for classifying droplets as empty.

    umi_cutoff : int | None, default None
        Optional absolute UMI count threshold for classifying droplets as empty.

    expected_cells : int | None, default None
        Expected number of real cells, used when estimating thresholds.

    tol: float, default 1e-6
        The relative change in likelihood below which training is discontinued

    random_state: int | None, default 42
        Random seed

    verbose : int, default 0
        Verbosity level (2 debug, 1 info, 0 warning, -1 error, -2 critical).

    quiet : bool, default False
        Suppresses most log output when True.

    log_file : str | None, default None
        Optional path to save EM iteration logs.

    Returns
    -------
    AnnData
        Denoised AnnData object with updated `adata.X`, and
        added fields:
        - `adata.layers["raw"]` : raw count matrix
        - `adata.obs["cell_ambient_fraction"]` : estimated ambient fraction per cell
        - `adata.uns["em_convergence"]` : diagnostics and log-likelihood trace
        - `adata.obs["alpha_hat"]' : final optimized alpha values
        - `adata.obs["z_hat"]` : final cell-type assignments (These should not change)
        - `adata.uns["p_hat"]` : final optimized matrix of cell-type profiles (K x G)
        - `adata.uns["beta_hat"]` : final optimized beta
        - `adata.var["ambient_hat"]` : final optimized ambient distribution
        - `adata.uns["loglike"]` : final log-likelihood (note that this value is not the 
           complete log-likelihood, only the relative log-likelihood)

    Notes
    -----
    The EM algorithm proceeds by:
      1. E-step: Update expected value of true, ambient noise, and bulk noise counts for each cell and gene.
      2. M-step: Update parameters (alpha, beta, p_k, a).
      3. Iterate until convergence (relative change in ll < `tol`) or reaching `max_iter`.
    """
    from cellsweep import __version__

    # set thread number
    numba.set_num_threads(threads)
    
    logger = setup_logger(log_file=log_file, verbose=verbose, quiet=quiet)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting cellsweep denoising at {timestamp}, cellsweep version {__version__}")

    adata = load_adata(adata, logger=logger)
    if "celltype" not in adata.obs.columns:
        raise KeyError("adata.obs must have column celltype.")

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
            
    adata.layers["raw"] = adata.X
    C = adata.X
    N, G = C.shape
    K = adata.uns["celltype_profile"].shape[0]
    dirichlet_lambda = dirichlet_lambda / G

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
        p[k] = (p[k] + dirichlet_lambda) / (p[k].sum() + G * dirichlet_lambda)

    # Initialize vectors that will keep track of gamma
    # Full gamma not initialized to save memory

    gamma_idx = np.full(N, -1, dtype=np.int64)
    mapped = z_true.map(z_true_str_to_int)
    mask = mapped.notna().to_numpy()
    gamma_idx[mask] = mapped[mask].to_numpy()
    # empties
    gamma_idx[is_empty] = -1

    if not fixed_celltype:
        if verbose == 2:
            gamma_idx_init = gamma_idx.copy()

    # initial beta + bulk m
    beta = float(beta)
    m_raw = np.array(C.sum(axis=0)).ravel().astype(float)
    m_global = m_raw / m_raw.sum()

    if "ambient" not in adata.var: 
        if freeze_ambient_profile:
            logger.info("Inferring the gene ambient profile from empty droplets.")
            # Ensure we have is_empty
            if "is_empty" not in adata.obs:
                logger.info("Inferring empty droplets since 'is_empty' not found in adata.obs.")
                
                adata = infer_empty_droplets(adata, method=empty_droplet_method, umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)
                is_empty = adata.obs["is_empty"].values

            C_empty = C[is_empty,:]

            a_raw = np.array(C_empty.sum(axis=0)).ravel().astype(float)
            
            # zero out a values below a threshold (inclusive threshold)
            if ambient_threshold > 0:
                a = a_raw[a_raw < ambient_threshold] = 0
                a = a / a.sum()  # re-normalize
            else:
                a = a_raw / (a_raw.sum())
                
            u = np.zeros(K)
        else:
            logger.info("Inferring the initial gene ambient profile from cell-types. The ambient profile will be updated during training.")
            valid = gamma_idx >= 0
            counts = np.bincount(gamma_idx[valid], minlength=K)
            u = counts / counts.sum()
            a = u @ p
            a = a / a.sum()
        adata.var["ambient"] = a
    else:
        a = np.asarray(adata.var["ambient"])
        valid = gamma_idx >= 0
        counts = np.bincount(gamma_idx[valid], minlength=K)
        u = counts / counts.sum()
        
    # initial alpha
    if "cell_ambient_fraction" not in adata.obs.columns:
        logger.info("adata.obs does not have 'cell_ambient_fraction'. Setting to `cell_ambient_fraction` argument.")
        adata.obs["cell_ambient_fraction"] = init_alpha
        adata.obs.loc[is_empty, "cell_ambient_fraction"] = 1.
    alpha = np.asarray(adata.obs["cell_ambient_fraction"].copy(), dtype=float).ravel()
    alpha = np.clip(alpha, eps, 1.0 - eps)

    if freeze_empty:
        alpha[is_empty] = 1.0

    alpha = alpha.astype(np.float64)
    a = a.astype(np.float32)
    p = p.astype(np.float32)
    C = C.astype(np.float32)

    logger.info(f"Performing Sparse EM with {numba.get_num_threads()} Numba thread(s)")
    em_dict = sparse_em(C=C, alpha=alpha, beta=beta, a=a, u=u, m_global=m_global, gamma_idx=gamma_idx, p=p, K=K, N=N, G=G, 
                        max_iter=max_iter, tol=tol, min_tol=min_tol, freeze_empty=freeze_empty, fixed_celltype=fixed_celltype, 
                        real_mask=real_mask, eps=eps, dirichlet_lambda=dirichlet_lambda,log_eps=log_eps, verbose=verbose, 
                        logger=logger, freeze_ambient_profile=freeze_ambient_profile, debug=debug)

    C_denoised = em_dict['C_denoised']
    alpha = em_dict["alpha"]
    beta = em_dict["beta"]
    p = em_dict["p"]
    a = em_dict["a"]
    ll = em_dict["ll"]

    if debug:
        a_tracker = em_dict["a_tracker"]
        p_tracker = em_dict["p_tracker"]
        interactive_distribution_viewer(a_tracker, p_tracker, m_global)

    if not fixed_celltype:
        if verbose == 2:
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
    adata.uns["beta_hat"] = beta
    adata.var["ambient_hat"] = a
    adata.uns["loglike"] = ll

    # Replace adata.X
    if integer_out:
        # integerization: optional
        C_integer =  sparse_integerize(C_denoised, random_state=random_state)
        adata.X = C_integer
    else:
        adata.X = C_denoised

    if is_dense:
        logger.info("Re-densifying output.")
        adata.X = np.asarray(adata.X)

    if adata_out:
        logger.info(f"Saving inferred adata to {adata_out!r}")
        if os.path.dirname(adata_out):
            os.makedirs(os.path.dirname(adata_out), exist_ok=True)
        adata.write_h5ad(adata_out)
    else:
        logger.warning("adata_out not specified; not saving inferred adata to a file.")

    return adata