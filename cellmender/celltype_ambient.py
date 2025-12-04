"""Denoising count matrices using a Poisson + Negative Binomial model."""

import os
import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from pydantic import validate_call, Field, ConfigDict
import numba
from numba import njit, prange
from typing import Annotated, Optional, Tuple
from .utils import setup_logger, load_adata, determine_cutoff_umi_for_expected_cells, infer_empty_droplets, determine_cell_types  # , plot_cellmender_likelihood_over_epochs
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, VBox, Output
from IPython.display import display
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import UnivariateSpline
import seaborn as sns


#* take the mean expression of each gene across all empty droplets, and normalize to sum to 1.
def infer_gene_ambient_fraction(adata, empty_droplet_method="threshold", dirichlet_lambda=0.1, umi_cutoff=None, expected_cells=None, verbose=0, quiet=False, logger=None):
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

    X_empty = X[is_empty,:]
    G = X.shape[1]

    a_raw = np.array(X_empty.sum(axis=0)).ravel().astype(float)
    ambient_fraction = (a_raw + dirichlet_lambda) / (a_raw.sum() + G * dirichlet_lambda)


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

def init_alpha(adata, a, eps, dirichlet_lambda, alpha_clip):

    T = np.asarray(adata.X.sum(axis=1)).squeeze()
    o = adata.X / (T[:, None] + dirichlet_lambda)           # N x G normalized
    s = np.asarray((o * a[:, None]).sum(axis=1))      # ambient score per droplet

    # 3. predictor x = log(T+1)
    x = np.asarray(np.log(T + 1.0))

    # 4. Fit monotone decreasing smoother r(x):
    #    do isotonic regression on (x, s) with decreasing=True
    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    s_iso = iso.fit_transform(x, s)      # piecewise-constant monotone fit

    # 5. Optional: smooth the isotonic fit with a small spline to remove steps
    #    fit spline on sorted x
    order = np.argsort(x)
    xs = x[order]; ys = s_iso[order]
    spl = UnivariateSpline(xs, ys, s=1.0)   # s controls smoothing; tune as needed
    r_x = lambda t: spl(np.log(t+1.0))

    # 6. anchors for linear rescaling to [0,1]
    #    s_max = median s among small T (e.g., bottom 5%); s_min = median s among top 5%
    low_mask = T <= np.percentile(T, 5)
    high_mask = T >= np.percentile(T, 95)
    s_max = max(np.median(s[low_mask]), dirichlet_lambda)
    s_min = min(np.median(s[high_mask]), s_max - dirichlet_lambda)

    # 7. compute alpha_n (clamped)
    alpha = (r_x(T) - s_min) / (s_max - s_min + eps)
    alpha = np.clip(alpha, alpha_clip[0], alpha_clip[1])
    
    return alpha

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

# ---------- Numba-parallel E-step kernel ----------
@njit(parallel=True, nogil=True)
def e_step_numba(indptr, indices, data, alpha, beta, a, m_global,
                 gamma, p, K, N, G, eps, log_eps, freeze_empty_mask):
    """
    Parallel E-step over rows. Returns per-entry arrays and per-row summaries.
    """
    nnz_total = data.shape[0]
    ambient_vals = np.zeros(nnz_total, dtype=np.float64)
    bulk_vals = np.zeros(nnz_total, dtype=np.float64)
    cell_vals = np.zeros((nnz_total, K), dtype=np.float64)

    numer_gamma = np.zeros((N, K), dtype=np.float64)
    A_n = np.zeros(N, dtype=np.float64)
    ll_row = np.zeros(N, dtype=np.float64)
    M_row = np.zeros(N, dtype=np.float64)

    # We will fill entries sequentially but rows are parallelized
    # Build an array mapping entry index to its position: compute per-row offsets
    for n in prange(N):
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
        for jj in range(rs, re):
            g = indices[jj]        # gene index
            val = data[jj]

            # compute mixture weights for this nonzero
            wa = (1.0 - beta) * alpha[n] * a[g]
            wm = beta * m_global[g]

            # if freeze_empty and this row is empty, treat only ambient+bulk
            w_p_sum = 0.0
            if not freeze_empty_mask[n]:
                # compute sum over k of b * gamma[n,k] * p[k,g]
                b = (1.0 - beta) * (1.0 - alpha[n])
                for k in range(K):
                    wpk = b * gamma[n, k] * p[k, g]
                    # accumulate into cell_vals
                    cell_vals[jj, k] = wpk
                    w_p_sum += wpk

            p_tot = wa + wm + w_p_sum

            # fraction scale: vals / p_tot
            scale = val / np.maximum(p_tot, eps)

            # expected ambient and bulk at this entry
            cA = scale * wa
            cM = scale * wm

            ambient_vals[jj] = cA
            bulk_vals[jj] = cM

            # convert stored w_p_k to expected counts for each k
            if not freeze_empty_mask[n]:
                # multiply previously stored wpk by scale to get counts
                # (we stored wpk in cell_vals[jj,k])
                for k in range(K):
                    cell_vals[jj, k] = cell_vals[jj, k] * scale
                    numer_gamma[n, k] += cell_vals[jj, k]  # accumulate per-row k-sums
                # sum of cell counts for this entry implicitly added to row_expected if needed
            else:
                # ensure cell_vals row is zero
                for k in range(K):
                    cell_vals[jj, k] = 0.0

            # accumulate per-row totals
            local_A += cA
            local_M += cM
            local_ll += val * np.log(np.maximum(p_tot, log_eps))

        # write back per-row numbers
        A_n[n] = local_A
        M_row[n] = local_M
        ll_row[n] = local_ll

    return ambient_vals, bulk_vals, cell_vals, numer_gamma, A_n, ll_row, M_row

def sparse_em(C, alpha, beta, a, m_global, gamma, p, K, N, G, 
              max_iter, tol, freeze_empty, fixed_celltype, real_mask, 
              eps, dirichlet_lambda, 
              log_eps, verbose, logger, freeze_ambient_profile, debug):
    
    if debug:
        a_tracker = []
        p_tracker = []
        
        a_tracker.append(a)
        p_tracker.append(p)
    else: 
        a_tracker = None
        p_tracker = None
    
    """
    Helper for denoise_count_matrix. Performs sparse compatible EM on model
    """
    indptr = C.indptr.copy()
    indices = C.indices.copy()
    data = C.data.astype(np.float64).copy()  # ensure float64

    nnz = data.shape[0]

    N, G = C.shape

    # freeze mask as boolean array for Numba
    if freeze_empty:
        freeze_empty_mask = ~real_mask
    else:
        freeze_empty_mask = np.zeros(N, dtype=np.bool_)

    prev_ll = -np.inf

    # Precompute row_of_entry (map each nnz index to its row) -> used for constructing CSR from per-entry arrays
    row_of_entry = np.empty(nnz, dtype=np.int64)
    for n in range(N):
        rs = indptr[n]; re = indptr[n+1]
        if re > rs:
            row_of_entry[rs:re] = n

    # EM loop
    for it in range(1, max_iter + 1):
        # ============================
        #     E STEP (numba parallel)
        # ============================
        ambient_vals, bulk_vals, cell_vals, numer_gamma, A_n, ll_row, M_row = e_step_numba(
            indptr, indices, data, alpha, float(beta), a, m_global,
            gamma, p, K, N, G, float(eps), log_eps, freeze_empty_mask
        )

        # Reduce per-row scalars
        ll = float(np.sum(ll_row))
        M_total = float(np.sum(M_row))

        if not freeze_ambient_profile:
            # Build a_numer by summing ambient_vals into gene slots (fast, C-backed)
            a_numer = np.zeros(G, dtype=np.float64)
            # indices correspond to gene indices per entry
            np.add.at(a_numer, indices, ambient_vals)

        # Build p_numer: shape (K, G) by summing cell_vals rows into gene slots
        p_numer = np.zeros((K, G), dtype=np.float64)
        # for each k, add cell_vals[:,k] at positions indices
        for k in range(K):
            np.add.at(p_numer[k], indices, cell_vals[:, k])

        # ============================
        #     M STEP
        # ============================

        # update p and gamma
        if not fixed_celltype:
            for k in range(K):
                numer = p_numer[k] + dirichlet_lambda
                denom = numer.sum()
                p[k] = numer / max(denom, eps)

            row_sums = numer_gamma.sum(axis=1)  # shape (N,)

            if freeze_empty:
                # update only real droplets in-place to preserve same array object
                # protect division by zero with eps
                denom_real = np.maximum(row_sums[real_mask, None], eps)
                gamma[real_mask, :] = numer_gamma[real_mask, :] / denom_real
                # ensure empties explicitly zero
                gamma[~real_mask, :] = 0.0
            else:
                # update all rows in-place
                denom_all = np.maximum(row_sums[:, None], eps)
                gamma[:] = numer_gamma / denom_all


        # update alpha
        Ccell_n = numer_gamma.sum(axis=1)

        alpha = A_n / np.maximum(A_n + Ccell_n, eps)
        # Valid range enforcement
        alpha[real_mask] = np.clip(alpha[real_mask], eps, 1-eps)
        if freeze_empty:
            alpha[~real_mask] = 1.0

        # update beta
        total_counts = M_total + A_n.sum() + numer_gamma.sum()
        beta = M_total / np.maximum(total_counts, eps)

        if not freeze_ambient_profile:
            # update ambient profile
            a_numer += dirichlet_lambda
            a = a_numer / max(a_numer.sum(), eps)

        if freeze_empty:
            alpha_eff = alpha[real_mask]
        else:
            alpha_eff = alpha

        alpha_median = np.median(alpha_eff)
        alpha_mean = np.mean(alpha_eff)
        alpha_max = np.max(alpha_eff)
        alpha_min = np.min(alpha_eff)

        if verbose:
            logger.info(f"EM Iter {it:3d}: ll={ll:.3f} min_alpha={alpha_min:.4f} mean_alpha={alpha_mean:.4f} median_alpha={alpha_median:.4f} max_alpha={alpha_max:.4f} beta={beta:.6f}")

        if it > 1 and abs((ll - prev_ll) / max(abs(prev_ll), 1.0)) < tol:
            if verbose:
                logger.info("Converged.")
            break

        if debug:
            a_tracker.append(a)
            p_tracker.append(p)

        prev_ll = ll

    # Construct CSR expected matrices from per-entry arrays
    # C_expected_cell: per-entry cell sum across K
    cell_sum_per_entry = np.sum(cell_vals, axis=1)
    C_expected_cell = sp.csr_matrix((cell_sum_per_entry, (row_of_entry, indices)), shape=(N, G))
    C_expected_ambient = sp.csr_matrix((ambient_vals, (row_of_entry, indices)), shape=(N, G))
    C_expected_bulk = sp.csr_matrix((bulk_vals, (row_of_entry, indices)), shape=(N, G))

    return {"C_expected_cell" : C_expected_cell, 
            "C_expected_ambient": C_expected_ambient, 
            "C_expected_bulk": C_expected_bulk, 
            "alpha": alpha, 
            "beta": beta, 
            "gamma": gamma, 
            "p": p, 
            "a": a, 
            "prev_ll": prev_ll, 
            "a_tracker": a_tracker, 
            "p_tracker": p_tracker}

def dense_em(C, alpha, beta, a, m_global, gamma, p, K, N, G,
             max_iter, tol, freeze_empty, fixed_celltype, real_mask, 
             eps, dirichlet_lambda,
             log_eps, verbose, logger, freeze_ambient_profile, debug):
    """
    Helper for denoise_count_matrix. Performs EM on model (dense arrays only)
    """

    if debug:
        a_tracker = []
        p_tracker = []
        
        a_tracker.append(a)
        p_tracker.append(p)
    else: 
        a_tracker = None
        p_tracker = None
    
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
        p_tot = w_a + w_m + w_p_sum  # (N,G)
        p_tot_safe = np.where(p_tot > eps, p_tot, eps)

        # E-step: expected counts per component
        c_scale = C / p_tot_safe                        # (N,G)

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
        if not freeze_ambient_profile:
            a_numer = np.sum(C_A, axis=0)   # (G,)
        numer_gamma = np.sum(C_Pk, axis=2)  # (N,K)

        # per-cell ambient counts
        A_n = np.sum(C_A, axis=1)        # (N,)
        # total bulk expected
        M_total = np.sum(C_M)            # scalar

        # compute log-likelihood
        ll = float(np.sum(C * np.log(np.maximum(p_tot, log_eps))))

        # ---------------- M-step ----------------
        if not fixed_celltype:
            # update p
            for k in range(K):
                numer = p_numer[k, :] + dirichlet_lambda
                denom = numer.sum()
                p[k, :] = numer / max(denom, eps)

            # update gamma for real droplets only (keep empties unchanged if freeze_empty)
            row_sums = numer_gamma.sum(axis=1)
            if freeze_empty:
                gamma[real_mask] = numer_gamma[real_mask] / np.maximum(row_sums[real_mask, None], eps)
                gamma[~real_mask, :] = 0
            else:
                gamma = numer_gamma / np.maximum(row_sums[:,None], eps)

        # update alpha
        Ccell_n = np.sum(numer_gamma, axis=1)  # (N,)
        alpha = A_n / np.maximum(A_n + Ccell_n, eps)
        if freeze_empty:
            alpha[~real_mask] = 1.0

        # update beta (with numerical guard)
        total_counts = M_total + A_n.sum() + numer_gamma.sum()
        
        beta = float(M_total/np.maximum(total_counts,eps))

        if not freeze_ambient_profile:
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

        if debug:
            a_tracker.append(a)
            p_tracker.append(p)

        prev_ll = ll

    C_expected_cell = C_cell    # ndarray (N,G)
    C_expected_ambient = C_A
    C_expected_bulk = C_M

    return {"C_expected_cell" : C_expected_cell, 
            "C_expected_ambient": C_expected_ambient, 
            "C_expected_bulk": C_expected_bulk, 
            "alpha": alpha, 
            "beta": beta, 
            "gamma": gamma, 
            "p": p, 
            "a": a, 
            "prev_ll": prev_ll, 
            "a_tracker": a_tracker, 
            "p_tracker": p_tracker}


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def denoise_count_matrix(
    adata: str | ad.AnnData,
    adata_out: Optional[Annotated[str, Field(pattern=r"\.h5ad$")]] = "adata_straightened.h5ad",
    max_iter: Annotated[int, Field(gt=0)] = 150,
    eps_gamma: Annotated[int, Field(ge=0)] = 0,
    cell_ambient_clip: Tuple[Annotated[float, Field(ge=0, le=1)], Annotated[float, Field(ge=0, le=1)]] = (0.01, 0.9),
    beta: Annotated[float, Field(ge=0, le=1)] = 0.1,
    eps: Annotated[float, Field(gt=0)] = 1e-12,
    log_eps: Annotated[float, Field(gt=0)] = 1e-300,
    dirichlet_lambda: Optional[Annotated[float, Field(ge=0)]] = 1e-6,
    integer_out: bool = False,
    threads: Optional[Annotated[int, Field(gt=0)]] = 1,
    fixed_celltype: bool = False,
    freeze_empty: bool = True,
    freeze_ambient_profile: bool = True,
    empty_droplet_method: str = "threshold",
    umi_cutoff: Optional[Annotated[int, Field(ge=0)]] = None,
    expected_cells: Optional[Annotated[int, Field(ge=0)]] = None,
    empty_droplet_celltype_name: str = "Empty Droplet",
    tol: Optional[Annotated[float, Field(ge=0)]] = 1e-6,
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

    eps_gamma: float, default 0.02
        Initital cell-type softening to allow continuous cell-type estimation

    eps : float, default 1e-15
        Numerical stability constant to prevent division by zero or log(0).

    log_eps : float, default 1e-300
        Numerical stability constant to prevent division by zero or log(0).
        Lower than eps for log values

    dirichlet_lambda: float, default 0.01
        Pseudocount

    integer_out : bool, default False
        If True, rounds denoised counts to nearest integer before saving.

    threads : int, default 1
        number of numba threads

    fixed_celltype : bool, default False
        If True, keeps cell-type assignments fixed during EM updates.

    freeze_empty : bool, default True
        If True, does not attempt to reestimate empty droplets

    freeze_ambient_profile: bool, default True
        If True, does not update the ambient profile (a) based upon alpha

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
        - `adata.layers["raw"]` : raw count matrix
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

    # set thread number
    numba.set_num_threads(threads)
    
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
                                            dirichlet_lambda=dirichlet_lambda,
                                            verbose=verbose, quiet=quiet, logger=logger)

    if "celltype_profile" not in adata.uns or "celltype_names" not in adata.uns:
        logger.info("Inferring celltype profiles.")
        adata = infer_celltype_profile(adata, celltype_key="celltype",
                                       empty_droplet_method=empty_droplet_method,
                                       verbose=verbose, quiet=quiet, logger=logger)

    adata.layers["raw"] = adata.X.copy()
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
    
    gamma = gamma.astype(float)
    gamma = (1.0 - eps_gamma) * gamma + (eps_gamma / K)
    # renormalize rows (defensive)
    row_sums = gamma.sum(axis=1)
    gamma = gamma / np.maximum(row_sums[:, None], eps)

    # still keep empties zero if freeze_empty
    if freeze_empty:
        gamma[~real_mask, :] = 0.0

    # initial alpha
    if "cell_ambient_fraction" not in adata.obs.columns:
        logger.info("adata.obs does not have 'cell_ambient_fraction'. Setting to `cell_ambient_fraction` argument.")
        adata.obs["cell_ambient_fraction"] = init_alpha(adata, a, eps, dirichlet_lambda, cell_ambient_clip)
        adata.obs.loc[~real_mask, "cell_ambient_fraction"] = 1.
    alpha = np.asarray(adata.obs["cell_ambient_fraction"].copy(), dtype=float).ravel()
    alpha = np.clip(alpha, eps, 1.0 - eps)

    if freeze_empty:
        alpha[~real_mask] = 1.0
        gamma[~real_mask, :] = 0.0

    # initial beta + bulk m
    beta = float(beta)
    m_raw = np.array(C.sum(axis=0)).ravel().astype(float)
    m_global = (m_raw + dirichlet_lambda) / (m_raw.sum() + G * dirichlet_lambda)


    alpha = alpha.astype(np.float64)
    a = a.astype(np.float64)
    p = p.astype(np.float64)
    gamma = gamma.astype(np.float64)

    if sp.issparse(C):
        if not sp.isspmatrix_csr(C):
            C = sp.csr_matrix(C)
        logger.info(f"Performing Sparse EM with {numba.get_num_threads()} Numba thread(s)")
        em_dict = sparse_em(C, alpha, beta, a, m_global, gamma, p, K, N, G, 
                                max_iter, tol, freeze_empty, fixed_celltype, real_mask,                                                                                                         
                                eps, dirichlet_lambda, 
                                log_eps, verbose, logger, freeze_ambient_profile, debug)
    else:
        np.asarray(C)
        logger.info("Performing Dense EM")
        em_dict = dense_em(C, alpha, beta, a, m_global, gamma, p, K, N, G, 
                                max_iter, tol, freeze_empty, fixed_celltype, real_mask, 
                                eps, dirichlet_lambda, 
                                log_eps, verbose, logger, freeze_ambient_profile)

    C_expected_cell = em_dict['C_expected_cell']
    C_expected_ambient = em_dict['C_expected_ambient']
    C_expected_bulk = em_dict['C_expected_bulk']
    alpha = em_dict["alpha"]
    beta = em_dict["beta"]
    p = em_dict["p"]
    a = em_dict["a"]
    prev_ll = em_dict["prev_ll"]
    a_tracker = em_dict["a_tracker"]
    p_tracker = em_dict["p_tracker"]

    if debug:
        interactive_distribution_viewer(a_tracker, p_tracker, m_global)

    # ============================
    #      DENOISED COUNTS
    # ============================
    C_minus_A = C - C_expected_ambient
    C_minus_A_minus_M = C_minus_A - C_expected_bulk
    C_denoised = C_minus_A_minus_M.maximum(0)

    # ===================================
    # STORE RESULTS AND RETURN
    # ===================================

    assert C_denoised.shape == (N, G), "Denoised matrix has incorrect shape."
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
    else:
        logger.warning("adata_out not specified; not saving inferred adata to a file.")

    return adata