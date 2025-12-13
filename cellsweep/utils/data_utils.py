"""Data Utils"""

import os
import numpy as np
import urllib.request
from scipy import io, sparse
import anndata as ad
import pandas as pd
from .logger_utils import setup_logger

def take_adata_cell_gene_intersection(adata1, adata2):
    """Return copies of adata1 and adata2 with only the intersecting cells and genes."""
    adata1, adata2 = adata1.copy(), adata2.copy()
    adata1.obs_names_make_unique()
    adata2.obs_names_make_unique()
    adata1.var_names_make_unique()
    adata2.var_names_make_unique()
    
    common_cells = adata1.obs_names.intersection(adata2.obs_names)
    common_genes = adata1.var_names.intersection(adata2.var_names)
    adata1_sub = adata1[common_cells, common_genes].copy()
    adata2_sub = adata2[common_cells, common_genes].copy()

    # ensure same cell/gene order
    adata1_sub = adata1_sub[common_cells, common_genes].copy()
    adata2_sub = adata2_sub[common_cells, common_genes].copy()

    return adata1_sub, adata2_sub

def determine_cutoff_umi_for_expected_cells(adata, expected_cells):
    knee = np.sort(np.ravel(adata.X.sum(axis=1)))[::-1]
    cutoff_umi = knee[expected_cells - 1]
    return cutoff_umi

def run_scanpy_preprocessing_and_clustering(adata, min_genes=100, min_cells=3, umi_top_percentile_to_remove=None, unique_genes_top_percentile_to_remove=None, mt_gene_percentile_to_remove=None, max_mt_percentage=25, n_top_genes=2000, hvg_flavor="seurat_v3", n_pcs=50, n_neighbors=15, leiden_resolution=1.0, seed=42, verbose=0, quiet=False):
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for this function. Please install scanpy, or reinstall cellsweep with pip install cellsweep[analysis].")
    
    logger = setup_logger(verbose=verbose, quiet=quiet)
    logger.info(f"Adata initial shape: {adata.shape}")
    
    #* cell filtering
    if min_genes:
        logger.info(f"Filtering cells with < {min_genes} genes")
        sc.pp.filter_cells(adata, min_genes=min_genes)
        logger.info(f"After filtering cells with < {min_genes} genes, adata shape: {adata.shape}")
    
    #* gene filtering
    if min_cells:
        logger.info(f"Filtering genes expressed in < {min_cells} cells")
        sc.pp.filter_genes(adata, min_cells=min_cells)
        logger.info(f"After filtering genes expressed in < {min_cells} cells, adata shape: {adata.shape}")
    
    #* Doublet removal (thresholding)
    to_remove = np.zeros(adata.n_obs, dtype=bool)
    if umi_top_percentile_to_remove is not None:
        logger.info(f"Filtering cells with total UMI counts in the top {umi_top_percentile_to_remove} percentile. This is done by calculating the total UMI counts for each cell and removing those that exceed the specified percentile threshold, which can help to eliminate potential doublets or multiplets that may have artificially high UMI counts.")
        total_umis = np.ravel(adata.X.sum(axis=1))
        umi_cutoff = np.percentile(total_umis, 100 - umi_top_percentile_to_remove)
        # adata = adata[adata.X.sum(axis=1) < umi_cutoff, :].copy()
        # logger.info(f"After UMI filtering, adata shape: {adata.shape}")
        to_remove |= (total_umis > umi_cutoff)
    if unique_genes_top_percentile_to_remove is not None:
        logger.info(f"Filtering cells with the number of unique genes expressed in the top {unique_genes_top_percentile_to_remove} percentile. This is done by calculating the number of unique genes expressed for each cell and removing those that exceed the specified percentile threshold, which can help to eliminate potential doublets or multiplets that may have artificially high gene diversity.")
        unique_genes_per_cell = np.ravel((adata.X > 0).sum(axis=1))
        unique_genes_cutoff = np.percentile(unique_genes_per_cell, 100 - unique_genes_top_percentile_to_remove)
        # adata = adata[unique_genes_per_cell < unique_genes_cutoff, :].copy()
        # logger.info(f"After unique gene filtering, adata shape: {adata.shape}")
        to_remove |= (unique_genes_per_cell > unique_genes_cutoff)
    
    #* MT filtering
    if max_mt_percentage is not None or mt_gene_percentile_to_remove is not None:
        adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
        if mt_gene_percentile_to_remove is not None:
            logger.info(f"Filtering cells with mitochondrial gene expression in the top {mt_gene_percentile_to_remove} percentile. This is done by calculating the percentage of counts that come from mitochondrial genes for each cell and removing those that exceed the specified percentile threshold, which can help to eliminate cells that may be stressed or dying, as they often have higher mitochondrial gene expression.")
            mt_cutoff = np.percentile(adata.obs["pct_counts_mt"], 100-mt_gene_percentile_to_remove)
            # adata = adata[adata.obs["pct_counts_mt"] <= mt_cutoff].copy()
            # logger.info(f"After mitochondrial gene filtering, adata shape: {adata.shape}")
            to_remove |= (adata.obs["pct_counts_mt"] > mt_cutoff)
        if max_mt_percentage is not None:
            logger.info(f"Filtering cells with > {max_mt_percentage}% mitochondrial gene expression. This is done by identifying mitochondrial genes (those starting with 'MT-') and calculating the percentage of counts that come from these genes for each cell. Cells that exceed the specified threshold are filtered out.")
            adata = adata[adata.obs.pct_counts_mt < max_mt_percentage, :]
            logger.info(f"After filtering cells with > {max_mt_percentage}% mitochondrial gene expression, adata shape: {adata.shape}")
    
    #* Do the actual percentile filtering based on the combined criteria
    n_removed = to_remove.sum()
    if n_removed > 0:
        adata = adata[~to_remove].copy()
        logger.info(f"After applying combined filtering criteria (UMI counts, unique genes, mitochondrial percentage), {n_removed} cells were removed. Remaining adata shape: {adata.shape}")

    #* normalization and log transformation
    if "counts" not in adata.layers:
        logger.info(f"'counts' layer not found in adata. Creating 'counts' layer from adata.X and normalizing total counts to 1e4. This is done by copying the raw count matrix into a new layer called 'counts' and then applying total-count normalization to ensure that each cell has the same total count (e.g., 10,000). This step is important for downstream analyses that assume normalized data.")
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
    if "log1p" not in adata.uns:
        logger.info(f"'log1p' not found in adata.uns. Applying log1p transformation to adata.X and storing in 'log1p' layer. This transformation is commonly used to stabilize variance and make the data more normally distributed, which can improve the performance of downstream analyses such as PCA and clustering.")
        sc.pp.log1p(adata)

    #* HGVs
    if "highly_variable" not in adata.var.columns:
        logger.info(f"Identifying highly variable genes using 'highly_variable_genes' function. This step identifies the top {n_top_genes} genes that show the most variability across cells, which are often the most informative for downstream analyses like clustering and dimensionality reduction.")
        layer = "counts" if hvg_flavor == "seurat_v3" else None
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=hvg_flavor, layer=layer)

    #* PCA
    if adata.varm is None or "PCs" not in adata.varm:
        logger.info(f"Running PCA on the log-transformed data using 'pca' function. This step reduces the dimensionality of the data while retaining as much variance as possible. The number of principal components to compute is set to {n_pcs}, and the SVD solver used is 'arpack'. Setting a random state ensures reproducibility of the results.")
        sc.tl.pca(adata, svd_solver="arpack", random_state=seed)

    #* KNN
    if adata.obsp is None or "distances" not in adata.obsp:
        logger.info(f"Computing the neighborhood graph of cells using 'neighbors' function. This step constructs a graph where cells are connected to their nearest neighbors based on the PCA representation. The number of neighbors to consider is set to {n_neighbors}, and the number of principal components used for this step is set to {n_pcs}. This graph is essential for downstream clustering and visualization.")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    #* Leiden clustering
    if "leiden" not in adata.obs.columns:
        logger.info(f"Running Leiden clustering on the neighborhood graph using 'leiden' function. This step identifies clusters of cells based on the connectivity of the graph. The 'flavor' parameter is set to 'igraph', which specifies the underlying algorithm used for clustering. The number of iterations is set to 2, and the resolution parameter controls the granularity of the clusters. Setting a random state ensures reproducibility of the results.")
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, resolution=leiden_resolution, random_state=seed)

    logger.info(f"Done!")
    return adata

valid_empty_droplet_methods = {"threshold"}
def infer_empty_droplets(adata, method="threshold", umi_cutoff=None, expected_cells=None, verbose=0, quiet=False, logger=None):
    """
    input: adata
    output: adata with adata.obs: is_empty
      - is_empty: boolean indicating whether each cell is an empty droplet or not. This is inferred using a simple heuristic: if the total counts for a cell are below a certain threshold (e.g., 100), it is considered an empty droplet. This threshold can be adjusted based on the dataset and expected cell types.
    """
    if not logger:
        logger = setup_logger(verbose=verbose, quiet=quiet)
    # adata = adata.copy()
    if method == "threshold":
        if umi_cutoff is None:
            if expected_cells is None:
                raise ValueError("For method 'threshold', either umi_cutoff or expected_cells must be provided.")
            umi_cutoff = determine_cutoff_umi_for_expected_cells(adata, expected_cells)
        adata.obs["is_empty"] = np.ravel(adata.X.sum(axis=1)) < umi_cutoff
    #!!! add more methods here
    else:
        raise ValueError(f"Invalid method {method!r} for inferring empty droplets. Valid methods are: {valid_empty_droplet_methods}")

    return adata

def determine_cell_types(adata, method="celltypist", filter_empty=True, empty_column="is_empty", celltype_column="celltype", umi_cutoff=None, expected_cells=None, model_pkl=None, celltypist_convert=False, celltypist_map_file=None, verbose=0, quiet=False, logger=None):
    """
    Adds a 'celltype' column to adata.obs based on the specified method.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for this function. Please install scanpy, or reinstall cellsweep with pip install cellsweep[analysis].")

    if not logger:
        logger = setup_logger(verbose=verbose, quiet=quiet)

    adata = adata.copy()
    
    # Identify real cells if present
    empty_droplet_category_name = "Empty Droplet"
    if filter_empty:
        logger.info(f"Filtering empty droplets using column '{empty_column}' in adata.obs. If this column is not present, it will be inferred using method '{method}' with umi_cutoff={umi_cutoff} and expected_cells={expected_cells}.")
        if empty_column not in adata.obs.columns:
            logger.info(f"'{empty_column}' column not found in adata.obs. Inferring empty droplets using method '{method}' with umi_cutoff={umi_cutoff} and expected_cells={expected_cells}.")
            adata = infer_empty_droplets(adata, method="threshold", umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet)
        real_mask = ~adata.obs[empty_column].astype(bool)
        adata_real = adata[real_mask].copy()
    else:
        real_mask = np.ones(adata.n_obs, dtype=bool)
        adata_real = adata

    if method == "celltypist":
        logger.info(f"Running cell type annotation using CellTypist with model_pkl={model_pkl}. This may take some time depending on the size of the dataset and the model used.")
        import celltypist
        if model_pkl is None:
            raise ValueError("model_pkl must be provided when method is 'celltypist'.")
        if not model_pkl in set(celltypist.models.models_description()["model"]):
            model_pkl_url = model_pkl
            model_pkl_dir = celltypist.models.data_path  # default directory for celltypist models
            model_pkl_name = model_pkl_url.split("/")[-1]
            model_pkl = os.path.join(model_pkl_dir, model_pkl_name)
            if not os.path.exists(model_pkl):
                os.makedirs(model_pkl_dir, exist_ok=True)
                urllib.request.urlretrieve(model_pkl_url, model_pkl)
            # model_pkl = celltypist.models.Model.load(model_pkl)
        
        if celltypist_convert:
            model_pkl = celltypist.models.Model.load(model_pkl)
            model_pkl.convert(map_file=celltypist_map_file)  # celltypist_map_file=None corresponds to human-to-mouse mapping
        
        if "counts" not in adata_real.layers:  # normalization
            logger.info(f"'counts' layer not found in adata_real. Creating 'counts' layer from adata_real.X and normalizing total counts to 1e4.")
            adata_real.layers["counts"] = adata_real.X.copy()
            sc.pp.normalize_total(adata_real, target_sum=1e4)
        if "log1p" not in adata_real.uns:
            logger.info(f"'log1p' not found in adata_real.uns. Applying log1p transformation to adata_real.X and storing in 'log1p' layer.")
            sc.pp.log1p(adata_real)
        predictions = celltypist.annotate(adata_real, model=model_pkl, majority_voting=True)
        pred_labels = predictions.predicted_labels[['majority_voting']]
        pred_labels = pred_labels.reindex(adata_real.obs_names)  # reorder to match adata_real.obs index
        adata_real.obs[celltype_column] = pred_labels['majority_voting'].values
        adata.obs[celltype_column] = adata_real.obs[celltype_column].reindex(adata.obs_names)
        if filter_empty:
            adata.obs[celltype_column] = adata.obs[celltype_column].cat.add_categories([empty_droplet_category_name]).fillna(empty_droplet_category_name)
    #!!! add more methods here
    else:
        raise ValueError(f"Unknown method {method} for determining cell types.")

    return adata

def check_counts_less_equal(adata_raw, adata_denoised):
    if adata_raw is None or adata_denoised is None:
        print("One of the adatas is None, cannot check counts.")
        return False
    
    adata_raw, adata_denoised = take_adata_cell_gene_intersection(adata_raw, adata_denoised)
    
    X_raw = adata_raw.X
    X_cb = adata_denoised.X

    if sparse.issparse(X_raw):
        diff_ok = not (X_cb > X_raw).nnz
    else:
        diff_ok = np.all(X_cb <= X_raw)

    return diff_ok

def zero_out_low_counts(adata, threshold=0.5):
    X = adata.X

    if sparse.issparse(X):
        X = X.copy()
        X.data[X.data < threshold] = 0
        X.eliminate_zeros()        # remove explicit zeros
        adata.X = X
    else:
        adata.X = np.where(X < threshold, 0, X)
    
    return adata

def create_base_adata(n_cells=5000, n_genes=1000, seed=42):
    np.random.seed(seed)

    # Create a random sparse count matrix
    X = sparse.random(
        n_cells, n_genes,
        density=0.05,
        format="csr",
        data_rvs=lambda n: np.random.poisson(lam=2, size=n)
    )

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])

    return ad.AnnData(X=X, obs=obs, var=var)

def find_single_branch_leaf_dir(base_dir):
    """
    Recursively descend into base_dir. At each level:
      - If there is more than one subdirectory → raise ValueError.
      - If there is exactly one → descend into it.
      - If there are no subdirectories → this is the leaf directory containing files.

    Returns:
        str: the path to the directory that contains files.
    """
    current = base_dir

    while True:
        entries = os.listdir(current)
        subdirs  = [d for d in entries if os.path.isdir(os.path.join(current, d))]
        files    = [f for f in entries if os.path.isfile(os.path.join(current, f))]

        # If files exist here, this is the leaf-level directory
        if files:
            return current

        # No files, but more than one subdirectory → ambiguous structure
        if len(subdirs) > 1:
            raise ValueError(f"Multiple directory branches found in {current}: {subdirs}")

        # No subdirectories → it’s empty or malformed
        if len(subdirs) == 0:
            raise ValueError(f"No files and no subdirectories in {current}; cannot descend further.")

        # Exactly one directory → descend
        current = os.path.join(current, subdirs[0])


def matrices_equal(A, B):
    # Case 1: both sparse
    if sparse.issparse(A) and sparse.issparse(B):
        return (A != B).nnz == 0

    # Case 2: one sparse, one dense → convert sparse to dense
    if sparse.issparse(A):
        A = A.toarray()
    if sparse.issparse(B):
        B = B.toarray()

    # Case 3: dense arrays
    return np.array_equal(A, B)