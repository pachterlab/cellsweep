"""
Simulate scRNA-seq count matrix with cell-type structure, noise, empty cells,
library-size variation, and dropout sparsity.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pydantic import validate_call, Field, ConfigDict
from typing import Annotated, Optional

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def simulate_cells(
    G : Annotated[int, Field(gt=0)] = 200,                         # number of genes
    N : Annotated[int, Field(gt=0)] = 1000,                        # number of cells (barcodes)
    k : Annotated[int, Field(gt=0)] = 5,                           # number of cell types
    markers_per_type : Annotated[int, Field(gt=0)] = 30,           # number of marker genes per cell type
    marker_boost : Annotated[float, Field(gt=0)] = 15.0,           # fold increase for marker genes in their type
    type_proportions: Optional[np.ndarray] = None,                 # vector of length k, normalized to 1
    empty_prob : Annotated[float, Field(ge=0, le=1)] = 0.8,        # fraction of empty barcodes
    alpha : Annotated[float, Field(ge=0, le=1)] = 0.01,            # fraction of size of ambient RNA relative to real cells
    expected_cell_size : Annotated[float, Field(gt=0)] = 10e3,     # expected library size for real cells
    libsize_logmean : Annotated[float, Field(gt=0)] = 0.0,         # mean of log library-size scaling
    libsize_logsd : Annotated[float, Field(gt=0)] = 0.5,           # sd of log library-size scaling
    dispersion : Annotated[float, Field(gt=0)] = 2.0,              # NB dispersion (lower => more overdispersion)
    dropout_midpoint : Annotated[float, Field(gt=0)] = 1.0,        # midpoint for dropout logistic curve
    dropout_slope : Annotated[float, Field(gt=0)] = 1.5,           # slope for dropout probability curve
    beta : Annotated[float, Field(ge=0, le=1)] = 0.03,             # fraction of counts to swap (bulk noise)
    singleton_prob : Annotated[float, Field(ge=0, le=1)] = 0.25,   # probability UMI has a single read
    rng_seed : Annotated[int, Field(ge=0)] = 42,                   # RNG seed
    gene_prefix : Annotated[str, Field(min_length=1)] = "Gene",    # gene name prefix
    cell_prefix : Annotated[str, Field(min_length=1)] = "Cell"     # cell name prefix
):
    """
    G: number of genes
    N: number of cells (barcodes)
    k: number of cell types
    markers_per_type: number of marker genes per cell type
    marker_boost: fold increase for marker genes in their type
    type_proportions: vector of length k, normalized to 1
    empty_prob: fraction of empty barcodes
    alpha: fraction of size of ambient RNA relative to real cells
    expected_cell_size: expected library size for real cells
    libsize_logmean: mean of log library-size scaling
    libsize_logsd: sd of log library-size scaling
    dispersion: NB dispersion (lower => more overdispersion)
    dropout_midpoint: midpoint for dropout logistic curve
    dropout_slope: slope for dropout probability curve
    beta: fraction of counts to swap (bulk noise)
    rng_seed: RNG seed
    gene_prefix: gene name prefix
    cell_prefix: cell name prefix

    Returns
    -------
    adata : AnnData
        anndata object with:
            - adata.X : (G x N) sparse count matrix
            - adata.layers['noise'] : background noise counts
            - adata.layers['real'] : true cell-derived counts
            - adata.obs : cell metadata (type, empty flag, lib_size, ambient fraction)
            - adata.var : gene names, marker annotation, ambient profile
            - adata.uns : simulation parameters, marker sets, and type profiles
    """
    rng = np.random.default_rng(rng_seed)

    # --- 1. Define celltype proportions --- 
    if type_proportions is None:
        type_proportions = rng.dirichlet(np.ones(k))
    else:
        type_proportions = np.asarray(type_proportions, dtype=float)
        type_proportions /= type_proportions.sum()

    # --- 2. Create per-type expected expression profiles ---
    type_expected = np.full((k, G), 1, dtype=float)
    all_genes = np.arange(G)
    rng.shuffle(all_genes)

    # Randomly pick marker genes for each type
    marker_sets = []
    pos = 0
    for t in range(k):
        markers = all_genes[pos:pos + markers_per_type]
        pos = (pos + markers_per_type) % G
        marker_sets.append(markers)
        type_expected[t, markers] *= marker_boost

    # random jitter for realism
    jitter = rng.normal(1.0, 0.5, size=type_expected.shape)
    type_expected *= np.clip(jitter, 0, 3)

    # normalize to probabilities
    type_expected /= type_expected.sum(axis=1, keepdims=True) 

    # --- 3. Assign cell types and empty barcodes ---
    cell_types = rng.choice(np.arange(k), size=N, p=type_proportions)
    is_empty = rng.random(N) < empty_prob

    # --- 4. Library-size variation ---
    lib_factors = np.exp(rng.normal(libsize_logmean, libsize_logsd, size=N)) * expected_cell_size
    lib_factors[is_empty] = 0.0

    # --- 5. Background noise distribution ---
    pop_expected = (type_proportions.reshape(-1, 1) * type_expected).sum(axis=0) # weighted sum of celltype profiles
    noise_lambda = alpha * expected_cell_size * pop_expected

    # --- 6. Generate counts ---
    counts = np.zeros((N, G), dtype=np.int32)
    noise = np.zeros((N, G), dtype=np.int32)
    real = np.zeros((N, G), dtype=np.int32)
    alphas = np.zeros(N, dtype=float)
    r = float(dispersion)
    r = max(r, 1e-6)

    for i in range(N):
        c_noise = rng.poisson(noise_lambda)
        if is_empty[i]:
            c_real = np.zeros(G, dtype=np.int32)
        else:
            mu = type_expected[cell_types[i]] * lib_factors[i]
            # Negative Binomial via Gamma-Poisson
            lam_real = rng.gamma(shape=r, scale=mu / r)
            c_real = rng.poisson(lam_real)
        counts[i, :] = c_real + c_noise
        noise[i, :] = c_noise
        real[i, :] = c_real

    # --- 7. Apply dropout / sparsity ---
    # Dropout prob = sigmoid(-slope*(log1p(mu_scaled) - midpoint))
    # so that low-expression genes are more likely to drop out
    log_means = np.log1p(type_expected[cell_types].T * lib_factors)
    dropout_probs = 1.0 / (1.0 + np.exp(-dropout_slope * (dropout_midpoint - log_means)))
    mask = rng.random(size=counts.shape) < dropout_probs.T
    counts[mask] = 0
    noise[mask] = 0
    real[mask] = 0

    # Compute ambient fraction after dropout
    pos_mask = counts.sum(axis=1) > 0
    alphas = np.zeros(N, dtype=float)
    alphas[pos_mask] = noise[pos_mask, :].sum(axis=1) / counts[pos_mask, :].sum(axis=1)
    alphas[is_empty] = 1.0

    # --- 8. Bulk noise / barcode swapping (count-weighted) ---
    if beta > 0:
        total_counts = counts.sum()
        n_swap = int(beta * total_counts)
        if n_swap > 0:
            flat_counts = counts.flatten()
            nonzero_idx = np.nonzero(flat_counts)[0]
            weights = flat_counts[nonzero_idx] / flat_counts[nonzero_idx].sum()

            # choose source indices proportional to count weight
            src_indices = rng.choice(nonzero_idx, size=n_swap, replace=True, p=weights)

            # map back to (cell x gene)
            g_src = src_indices // N
            c_src = src_indices % N

            # choose random destination cells
            c_dst = rng.integers(0, N, size=n_swap)

            for gs, cs, cd in zip(g_src, c_src, c_dst):
                if counts[cs, gs] > 0:
                    counts[cd, gs] += 1
                    noise[cd, gs] += 1 # treat swapped-in counts as noise
                    if rng.random() < singleton_prob:
                        counts[cs, gs] -= 1
                        real[cs, gs] -= 1

    # --- 9. Build AnnData ---
    gene_names = [f"{gene_prefix}_{g}" for g in range(G)]
    cell_names = [f"{cell_prefix}_{c}" for c in range(N)]

    # make sparse for memory efficiency
    X = sp.csr_matrix(counts) 
    X_noise = sp.csr_matrix(noise)
    X_real = sp.csr_matrix(real)

    obs = pd.DataFrame({
        "celltype": [f"Type_{t}" for t in cell_types],
        "is_empty": is_empty.astype(bool),
        "ambient_fraction": alphas,
        "lib_size": lib_factors
    }, index=cell_names)

    obs["celltype"] = obs["celltype"].mask(is_empty, "Empty Droplet")

    var = pd.DataFrame(index=gene_names)
    var["ambient_profile"] = pop_expected / pop_expected.sum()

    # annotate marker genes for convenience
    var["is_marker"] = False
    for t, genes in enumerate(marker_sets):
        var.loc[[f"{gene_prefix}_{g}" for g in genes], "is_marker"] = True

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Add layers for noise and real counts for reference
    adata.layers["noise"] = X_noise
    adata.layers["real"] = X_real

    adata.uns["simulation_params"] = dict(
        G=G, N=N, k=k, empty_prob=empty_prob, alpha=alpha,
        expected_real_size=expected_cell_size, libsize_logmean=libsize_logmean,
        libsize_logsd=libsize_logsd,
        dispersion=dispersion, dropout_midpoint=dropout_midpoint,
        dropout_slope=dropout_slope, beta=beta, 
        singleton_prob=singleton_prob, rng_seed=rng_seed,
        type_proportions=type_proportions.tolist()
    )
    adata.uns["marker_sets"] = {
        f"Type_{t}": [f"{gene_prefix}_{g}" for g in marker_sets[t]] for t in range(k)
    }

    cell_profiles = np.zeros((k, G), dtype=float)
    for t in range(k):
        cell_profiles[t, :] = type_expected[t, :] / type_expected[t, :].sum()

    adata.uns["type_profiles"] = cell_profiles

    return adata