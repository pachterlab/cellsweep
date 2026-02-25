"""
Simulate scRNA-seq count matrix with cell-type structure, noise, empty cells,
and library-size variation
"""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pydantic import validate_call, Field, ConfigDict
from typing import Annotated, Optional

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def simulate_cells(
    G : Annotated[int, Field(gt=0)] = 200,                         
    N : Annotated[int, Field(gt=0)] = 1000,                        
    k : Annotated[int, Field(gt=0)] = 5,                           
    markers_per_type : Annotated[int, Field(gt=0)] = 30,           
    marker_boost : Annotated[float, Field(gt=0)] = 15.0,           
    type_proportions: Optional[np.ndarray] = None,                 
    empty_prob : Annotated[float, Field(ge=0, le=1)] = 0.8,        
    alpha : Annotated[float, Field(ge=0, le=1)] = 0.01,            
    expected_cell_size : Annotated[float, Field(gt=0)] = 10e3,     
    libsize_logmean : Annotated[float, Field(gt=0)] = 0.0,         
    libsize_logsd : Annotated[float, Field(gt=0)] = 0.4,           
    dispersion : Annotated[float, Field(gt=0)] = 2.0,                      
    beta : Annotated[float, Field(ge=0, le=1)] = 0.03,             
    singleton_prob : Annotated[float, Field(ge=0, le=1)] = 0.25,   
    rng_seed : Annotated[int, Field(ge=0)] = 42,                   
    gene_prefix : Annotated[str, Field(min_length=1)] = "Gene",    
    cell_prefix : Annotated[str, Field(min_length=1)] = "Cell",
    housekeeping_frac: float = 0.08,          # fraction of genes that are housekeeping
    hk_logmean: float = -0.5,                 # mean log-expression of housekeeping genes
    hk_logsd: float   = 1.0,                  # variability in housekeeping gene expression
    marker_logsd: float = 0.5,               # variability in marker strength
    noise_logsd_empty = 0.8,   # empty droplets: very heterogeneous
    noise_logsd_cell = 0.5,    # cells: more constrained contamination
    bg_alpha: float = 0.1,   # background ambient load scaling
    bg_leakage: float = 0.01   # fraction of ambient profile that leaks into all genes

):
    rng = np.random.default_rng(rng_seed)

    # --- 1. Define cell type proportions ---
    if type_proportions is None:
        type_proportions = rng.dirichlet(np.ones(k))
    else:
        type_proportions = np.asarray(type_proportions, dtype=float)
        type_proportions /= type_proportions.sum()

    # --- 2. Create per-type expected expression profiles ---
    type_expected = np.zeros((k, G), dtype=float)  # start with zeros (true zeros for most genes)
    all_genes = np.arange(G)
    rng.shuffle(all_genes)

    # --- 2a. Assign marker genes ---
    marker_sets = []
    pos = 0
    
    for t in range(k):
        markers = all_genes[pos:pos + markers_per_type]
        pos = (pos + markers_per_type) % G
        marker_sets.append(markers)
        marker_strengths = marker_boost * rng.lognormal(
        mean=0.0,
        sigma=marker_logsd,
        size=len(marker_sets[t])
        )
        type_expected[t, markers] = marker_strengths

    # --- 2b. Assign housekeeping genes ---
    n_housekeeping = int(housekeeping_frac * G)
    marker_union = np.unique(np.concatenate(marker_sets))
    candidate_genes = np.setdiff1d(np.arange(G), marker_union)

    hk_genes = rng.choice(
        candidate_genes,
        size=n_housekeeping,
        replace=False
    )

    assert len(np.intersect1d(marker_union, hk_genes)) == 0

    hk_strengths = rng.lognormal(
        mean=hk_logmean,
        sigma=hk_logsd,
        size=len(hk_genes)
    )

    for t in range(k):
        type_expected[t, hk_genes] = hk_strengths

    # -- 2c. Add background ambient load ---
    bg_alpha = 0.01  
    p_background = rng.dirichlet(np.full(G, bg_alpha))

    # --- 2d. Normalize to probability profiles per cell type ---
    type_expected /= type_expected.sum(axis=1, keepdims=True)

    # --- 3. Assign cell types and empty barcodes ---
    cell_types = rng.choice(np.arange(k), size=N, p=type_proportions)
    is_empty = rng.random(N) < empty_prob

    # --- 4. Library-size variation ---
    lib_factors = np.exp(rng.normal(libsize_logmean, libsize_logsd, size=N)) * expected_cell_size
    lib_factors[is_empty] = 0.0

    # --- 5. Compute weighted ambient profile (overdispersed) ---
    pop_expected = (type_proportions.reshape(-1, 1) * type_expected).sum(axis=0)
    pop_expected /= pop_expected.sum()  # ensure proper probability distribution
    ambient_profile = (1 - bg_leakage) * pop_expected + bg_leakage * p_background
    ambient_profile /= ambient_profile.sum()

    # Parameters controlling ambient load variability
    noise_logmean = np.log(alpha * expected_cell_size)

    # --- 6. Generate counts ---
    counts = np.zeros((N, G), dtype=np.int32)
    noise = np.zeros((N, G), dtype=np.int32)
    real = np.zeros((N, G), dtype=np.int32)
    alphas = np.zeros(N, dtype=float)
    r = max(dispersion, 1e-6) # shape parameter for NB
    
    for i in range(N):
        # --- 6a. Draw latent ambient load for this droplet ---
        if is_empty[i]:
            # Empty droplets have wide variation in ambient capture
            lambda_i = rng.lognormal(mean=noise_logmean, sigma=noise_logsd_empty)
        else:
            # Cell-containing droplets have more constrained contamination
            lambda_i = rng.lognormal(mean=noise_logmean, sigma=noise_logsd_cell)

        # --- 6b. Sample ambient noise (Poisson conditional on lambda_i) ---
        c_noise = rng.poisson(lambda_i * ambient_profile)

        # --- 6c. Sample real counts (NB / Gamma–Poisson) ---
        if is_empty[i]:
            c_real = np.zeros(G, dtype=np.int32)
        else:
            mu = type_expected[cell_types[i]] * lib_factors[i]
            lam_real = rng.gamma(shape=r, scale=mu / r)
            c_real = rng.poisson(lam_real)

        # --- 6d. Combine ---
        counts[i, :] = c_real + c_noise
        noise[i, :] = c_noise
        real[i, :] = c_real

    # --- 7. Bulk noise / barcode swapping ---
    if beta > 0:
        total_counts = counts.sum()
        n_swap = int(beta * total_counts)
        if n_swap > 0:
            flat_counts = counts.flatten()
            nonzero_idx = np.nonzero(flat_counts)[0]
            weights = flat_counts[nonzero_idx] / flat_counts[nonzero_idx].sum()
            src_indices = rng.choice(nonzero_idx, size=n_swap, replace=True, p=weights)
            g_src = src_indices // N
            c_src = src_indices % N
            c_dst = rng.integers(0, N, size=n_swap)
            for gs, cs, cd in zip(g_src, c_src, c_dst):
                if counts[cs, gs] > 0:
                    counts[cd, gs] += 1
                    noise[cd, gs] += 1
                    if rng.random() < singleton_prob:
                        counts[cs, gs] -= 1
                        real[cs, gs] -= 1

    # --- 8. Compute ambient fraction ---
    pos_mask = counts.sum(axis=1) > 0
    alphas = np.zeros(N, dtype=float)
    alphas[pos_mask] = noise[pos_mask, :].sum(axis=1) / counts[pos_mask, :].sum(axis=1)
    alphas[is_empty] = 1.0

    # --- 9. Build AnnData ---
    gene_names = [f"{gene_prefix}_{g}" for g in range(G)]
    cell_names = [f"{cell_prefix}_{c}" for c in range(N)]

    X = sp.csr_matrix(counts)
    X_noise = sp.csr_matrix(noise)
    X_real = sp.csr_matrix(real)

    obs = pd.DataFrame({
        "cellid": [t+1 for t in cell_types],
        "celltype": [f"Type_{t}" for t in cell_types],
        "is_empty": is_empty.astype(bool),
        "true_ambient_fraction": alphas,
        "lib_size": lib_factors
    }, index=cell_names)
    obs["celltype"] = obs["celltype"].mask(is_empty, "Empty Droplet")
    obs["cellid"] = obs["cellid"].mask(is_empty, -1)

    var = pd.DataFrame(index=gene_names)
    var["true_ambient_profile"] = pop_expected / pop_expected.sum()
    var["is_marker"] = False
    for t, genes in enumerate(marker_sets):
        var.loc[[f"{gene_prefix}_{g}" for g in genes], "is_marker"] = True

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["noise"] = X_noise
    adata.layers["real"] = X_real

    adata.uns["simulation_params"] = dict(
        G=G, N=N, k=k, empty_prob=empty_prob, alpha=alpha,
        expected_real_size=expected_cell_size, libsize_logmean=libsize_logmean,
        libsize_logsd=libsize_logsd,
        dispersion=dispersion, beta=beta, 
        singleton_prob=singleton_prob, rng_seed=rng_seed,
        type_proportions=type_proportions.tolist()
    )
    adata.uns["marker_sets"] = np.array(marker_sets)

    cell_profiles = np.zeros((k, G), dtype=float)
    for t in range(k):
        cell_profiles[t, :] = type_expected[t, :] / type_expected[t, :].sum()
    adata.uns["type_profiles"] = cell_profiles

    return adata