"""Denoising count matrices using a Poisson + Negative Binomial model."""

import numpy as np
import pandas as pd
import logging
import anndata as ad
import scipy.sparse as sp
from scipy.stats import poisson, nbinom
from .utils import setup_logger, determine_cutoff_umi_for_expected_cells, infer_empty_droplets, determine_cell_types

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


def denoise_count_matrix(adata, adata_out="adata_straightened.h5ad", max_iter=40, beta=0.03, eps=1e-9, empty_droplet_method="threshold", umi_cutoff=None, expected_cells=None, cell_ambient_fraction=0.01, empty_droplet_celltype_name="Empty Droplet", round_counts=True, verbose=0, quiet=False, log_file=None):
    """
    EM on *real* cells only, with:
      - ambient fixed to the true ambient
      - empty-vs-real fixed to the true empties
    This is to test whether the p_k and alpha_i parts are behaving.

    adata (an anndata object or path to h5ad file) must have:
    - adata.X
    - adata.obs:
      - celltype: cell type labels for each cell
      - is_empty (optional): boolean indicating whether each cell is an empty droplet or not. If not present, it will be inferred using infer_empty_droplets().
      - cell_ambient_fraction (optional): fraction of ambient RNA in each cell. If not present, it will be set to a default value (e.g., 0.01) for all cells. Can either pass in a constant, or a vector of length N (number of cells) to specify different ambient fractions for each cell.

    - adata.var:
      - ambient (optional): fraction of ambient RNA comprised by each gene
    
    - adata.uns
      - celltype_profile (optional): DataFrame (n_celltypes x n_genes) - mean expression of each gene across all cells of that type. If not present, it will be inferred using infer_celltype_profile().

    """
    logger = setup_logger(log_file=log_file, verbose=verbose, quiet=quiet)

    if isinstance(adata, str):
        if adata.endswith(".h5ad"):
            logger.info(f"Loading adata from {adata!r}")
            adata = ad.read_h5ad(adata)
        else:
            raise ValueError(f"Invalid adata input {adata!r}. Expected a path to an .h5ad file or an AnnData object.")
    elif isinstance(adata, ad.AnnData):
        pass
        # adata = adata.copy()
    else:
        raise ValueError(f"Invalid adata input {adata!r}. Expected a path to an .h5ad file or an AnnData object.")
    
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

    a = adata.var["ambient_fraction"].copy()           # FIXED ambient
    is_empty = adata.obs["is_empty"].copy()   # FIXED empties
    z_true = adata.obs["celltype"].copy()
    z_true_str_to_int = {ct: i for i, ct in enumerate(adata.uns["celltype_profile"].index)}

    # work only on real cells
    real_mask = ~is_empty
    real_mask = np.asarray(real_mask)   # convert from Series → ndarray
    Xr = X[real_mask]           # (Nr, G)
    Nr = Xr.shape[0]

    #!!! densify
    if sp.issparse(Xr):
        Xr = Xr.toarray()  # convert to dense numpy.ndarray
    else:
        Xr = np.asarray(Xr)  # ensure d

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

    for it in range(max_iter):
        m = p.mean(axis=0)

        # --- E step on real cells ---
        log_p_type = np.zeros((Nr, K))
        for k in range(K):
            pi_j = alpha[:, None] * a + (1 - alpha)[:, None] * ((1 - beta) * p[k] + beta * m)
            pi_j = np.clip(pi_j, eps, 1.0)
            pi_j /= pi_j.sum(axis=1, keepdims=True)
            log_p_type[:, k] = np.sum(Xr * np.log(pi_j), axis=1)

        # normalize over k
        log_p_type -= log_p_type.max(axis=1, keepdims=True)
        r = np.exp(log_p_type)
        r /= r.sum(axis=1, keepdims=True)
        gamma_type = r

        # --- M step on real cells ---
        # update p_k
        for k in range(K):
            p[k] = (gamma_type[:, k][:, None] * Xr).sum(axis=0) + 1.0  # smoothing
            p[k] /= p[k].sum()

        # update alpha_j by 1D search
        for j in range(Nr):
            mix_cell = (1 - beta) * (gamma_type[j] @ p) + beta * p.mean(axis=0)
            best_alpha, best_ll = alpha[j], -np.inf
            for a_try in np.linspace(0, 1, 31):
                pi_try = a_try * a + (1 - a_try) * mix_cell
                pi_try = np.clip(pi_try, eps, 1.0)
                ll = np.sum(Xr[j] * np.log(pi_try))
                if ll > best_ll:
                    best_ll, best_alpha = ll, a_try
            alpha[j] = best_alpha

        # log-likelihood on real cells
        loglike = np.sum(np.log(np.exp(log_p_type).sum(axis=1) + eps))
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

    if round_counts:
        X_hat = np.rint(X_hat).astype(int)
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