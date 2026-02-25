"""Data Utils"""

import os
import numpy as np
import urllib.request
import tarfile
from scipy import io, sparse
from scipy.ndimage import gaussian_filter1d
from scipy.stats import entropy
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

    # assert np.array_equal(adata1_sub.obs_names, adata2_sub.obs_names)
    # assert np.array_equal(adata1_sub.var_names, adata2_sub.var_names)

    return adata1_sub, adata2_sub

def determine_cutoff_umi_for_expected_cells(adata, expected_cells):
    knee = np.sort(np.ravel(adata.X.sum(axis=1)))[::-1]
    cutoff_umi = knee[expected_cells - 1]
    return cutoff_umi

def run_scanpy_preprocessing_and_clustering(adata, filter_empty_droplets=False, umi_cutoff=None, expected_cells=None, min_genes=100, min_counts=None, min_cells=3, umi_top_percentile_to_remove=None, unique_genes_top_percentile_to_remove=None, mt_gene_percentile_to_remove=None, max_mt_percentage=25, n_top_genes=2000, hvg_flavor="seurat_v3", n_pcs=50, n_neighbors=15, leiden_resolution=1.0, seed=42, verbose=0, quiet=False):
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for this function. Please install scanpy, or reinstall cellsweep with pip install cellsweep[analysis].")
    
    logger = setup_logger(verbose=verbose, quiet=quiet)
    logger.info(f"Adata initial shape: {adata.shape}")

    adata = adata.copy()

    #* empty droplet filtering
    if filter_empty_droplets:
        logger.info(f"Filtering empty droplets using 'infer_empty_droplets' function with method 'threshold'. This is done by calculating the total UMI counts for each cell and removing those that fall below a certain threshold, which helps to eliminate empty droplets that do not contain any cells.")
        if "is_empty" not in adata.obs.columns:
            logger.info(f"'is_empty' column not found in adata.obs. Using 'infer_empty_droplets' function to identify empty droplets.")
            adata = infer_empty_droplets(adata, method="threshold", umi_cutoff=umi_cutoff, expected_cells=expected_cells, verbose=verbose, quiet=quiet, logger=logger)
        adata = adata[~adata.obs["is_empty"]].copy()
        logger.info(f"After filtering empty droplets, adata shape: {adata.shape}")
    
    #* cell filtering
    if min_counts:
        logger.info(f"Filtering cells with < {min_counts} counts")
        sc.pp.filter_cells(adata, min_counts=min_counts)
        logger.info(f"After filtering cells with < {min_counts} counts, adata shape: {adata.shape}")

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

valid_empty_droplet_methods = {"threshold", "mx_filter"}
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
                # raise ValueError("For method 'threshold', either umi_cutoff or expected_cells must be provided.")
                logger.warning("Expected cells and UMI cutoff being determined automatically. This is still experimental. To determine manually, please provide either expected_cells or umi_cutoff as a parameter.")
                expected_cells, umi_cutoff = automatic_umi_cutoff_detection(adata)
            else:
                umi_cutoff = determine_cutoff_umi_for_expected_cells(adata, expected_cells)
        adata.obs["is_empty"] = np.ravel(adata.X.sum(axis=1)) < umi_cutoff
    elif method == "mx_filter":
        if umi_cutoff is None:
            logger.warning("UMI cutoff being determined automatically using mx_filter method. To determine manually, please provide umi_cutoff as a parameter.")
            umi_cutoff = get_umi_cutoff_from_adata(adata, sum_axis=1, comps=[2], select_axis=None)
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
        files    = [f for f in entries if os.path.isfile(os.path.join(current, f)) and not f.endswith('.tar.gz')]

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

def get_tar_top_level_dir(tar_path):
    with tarfile.open(tar_path, "r:*") as tar:
        names = tar.getnames()

        # Find the first top-level directory
        top_level = names[0].split("/")[0]
        return top_level

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

def automatic_umi_cutoff_detection(adata, min_counts=10):
    counts = np.sort(np.ravel(adata.X.sum(axis=1)))[::-1]
    counts = counts[counts > min_counts]
    ranks = np.arange(1, len(counts) + 1)

    x = ranks  # np.log10(ranks)
    y = np.log10(counts)

    # smooth to reduce noise in tail
    y_smooth = gaussian_filter1d(y, sigma=2)

    # second derivative
    d2 = np.gradient(np.gradient(y_smooth, x), x)

    knee_idx = np.argmin(d2)   # most negative curvature
    knee_rank_barcode = ranks[knee_idx]
    knee_count_umi = counts[knee_idx]

    return knee_rank_barcode, knee_count_umi


def normalize_by_median_gene_expression(adata, layer=None, min_genes=None, min_cells=None, normalize=True, nonzero=True):
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy is required for this function. Please install scanpy, or reinstall cellsweep with pip install cellsweep[analysis].")

    adata = adata.copy()
    
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    
    revert_to_raw = False
    if normalize:
        if "counts" in adata.layers:
            print("Data already normalized")
        else:
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata, target_sum=1e4, layer="counts", inplace=True)
            revert_to_raw = True
    else:
        if "counts" in adata.layers and layer != "counts":
            print("normalize=True, but 'counts' layer already exists. Setting layer to 'counts'.")
            layer = "counts"

    if layer and layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")
    
    X = adata.layers[layer] if layer is not None else adata.X

    if sparse.issparse(X):
        X = X.tocsc()

    gene_medians = np.zeros(adata.n_vars)

    for j in range(adata.n_vars):
        col = X[:, j]
        if sparse.issparse(col):
            vals = col.data
        else:
            vals = col

        if nonzero:
            vals = vals[vals > 0]
        gene_medians[j] = np.median(vals) if len(vals) > 0 else np.nan

    adata.var["nonzero_median_norm_total"] = gene_medians

    med = adata.var["nonzero_median_norm_total"].values

    if normalize:
        X = adata.layers["counts"] if layer is None else adata.layers[layer]  # use original counts for normalization to avoid dividing already normalized values by the median

    if sparse.issparse(X):
        X = X.tocsr()
        X_gene = X.multiply(1.0 / med)
    else:
        X_gene = X / med

    # clean up infinities / NaNs
    if sparse.issparse(X_gene):
        X_gene.data[~np.isfinite(X_gene.data)] = 0
    else:
        X_gene[~np.isfinite(X_gene)] = 0

    
    if sparse.issparse(X_gene):
        X_gene = X_gene.tocsr()

    adata.layers["norm_total_gene_median"] = X_gene

    if revert_to_raw:
        adata.X = adata.layers["counts"]
        del adata.layers["counts"]
    
    # ensure adata.X is sparse
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    return adata


#* from mx_filter: https://github.com/cellatlas/mx/blob/master/mx/utils.py
# def nd(arr):
#     return np.asarray(arr).reshape(-1)

# def write_list(fname, lst=list):
#     with open(fname, "w") as f:
#         for idx, ele in enumerate(lst):
#             f.write(f"{ele}\n")

# def knee(mtx, sum_axis):
#     u = nd(mtx.sum(sum_axis))  # counts per barcode
#     x = np.sort(u)[::-1]  # sorted
#     v = np.log1p(x).reshape(-1, 1)  # log1p and reshaped for gmm
#     return (u, x, v)


# def knee_select(mtx, select_axis):
#     u = nd(mtx[:, select_axis])  # counts per barcode
#     x = np.sort(u)[::-1]  # sorted
#     v = np.log1p(x).reshape(-1, 1)  # log1p and reshaped for gmm
#     return (u, x, v)


def gmm(x, v, comps):
    from sklearn.mixture import GaussianMixture
    n_comps = comps.pop(0)

    gm = GaussianMixture(n_components=n_comps, random_state=42)
    labels = gm.fit_predict(v)
    prob = gm.predict_proba(v)
    ent = entropy(prob, axis=1)

    # index of v where low count cell is
    cutoff = 0
    if n_comps == 2:
        ind = np.argmax(ent)
        # log1p_cutoff = v[ind][0]
        cutoff = x[ind]
    elif n_comps > 2:
        # sort means, and pick the range of the top two
        means = np.sort((np.exp(gm.means_) - 1).flatten())
        r = np.logical_and(x > means[-2], x < means[-1])  # make ranage
        df = pd.DataFrame({"ent": ent, "idx": np.arange(ent.shape[0]).astype(int)})[r]
        # get the index (of x) where the entropy is the max (in range r)
        amax = df["ent"].argmax()
        idx = df.iloc[amax]["idx"].astype(int)
        cutoff = x[idx]

    # n_iter -= 1
    n_iter = len(comps)
    if n_iter <= 0:
        return (cutoff, (x > cutoff).sum())
    return gmm(x[x > cutoff], v[x > cutoff], comps)  # , n_comps, n_iter)


# def run_mx_filter(
#     matrix_fn,
#     axis_data_fn,
#     matrix_fn_out,
#     axis_data_out_fn,
#     sum_axis=1,
#     comps=[2],
#     select_axis=None,  # if you want to do the knee only on certain columns
# ):
#     # read matrix
#     mtx = io.mmread(matrix_fn).toarray()

#     # read barcodes
#     axis_data = []
#     # read_str_list(axis_data_fn, axis_data)
#     if axis_data_fn.split(".")[-1] == "gz":
#         axis_data = pd.read_csv(axis_data_fn, header=None, compression="gzip").values.flatten()

#     else:
#         axis_data = pd.read_csv(axis_data_fn, header=None).values.flatten()

#     (mtx_f, axis_data_f) = mx_filter(mtx, axis_data, sum_axis, comps, select_axis)

#     # save filtered matrix
#     io.mmwrite(matrix_fn_out, sparse.csr_matrix(mtx_f))

#     # save filtered metadata
#     write_list(axis_data_out_fn, axis_data_f)

# def mx_filter(mtx, axis_data, sum_axis, comps, select_axis):
#     # find knee
#     # check this, do it twice?
#     u, x, v = knee(mtx, sum_axis)
#     if select_axis:
#         u, x, v = knee_select(mtx, select_axis)

#     (cutoff, ncells) = gmm(x, v, comps=comps)

#     print(f"Filtered to {ncells:,.0f} cells with at least {cutoff:,.0f} UMIs.")

#     # mask matrix and netadata
#     mask = u > cutoff
#     mtx_f = mtx[mask]
#     axis_data_f = np.array(axis_data)[mask]
#     return (mtx_f, axis_data_f)

def get_umi_cutoff_from_adata(
    adata,
    sum_axis=1,
    comps=[2],
    select_axis=None,
):
    """
    Compute UMI cutoff from AnnData using knee + GMM logic.
    Returns only the cutoff.
    """

    # ---- Step 1: compute UMI totals safely ----
    X = adata.X

    if select_axis is not None:
        # selecting specific columns (genes)
        if sparse.issparse(X):
            u = np.asarray(X[:, select_axis].sum(axis=1)).ravel()
        else:
            u = X[:, select_axis].sum(axis=1)
    else:
        if sparse.issparse(X):
            u = np.asarray(X.sum(axis=sum_axis)).ravel()
        else:
            u = X.sum(axis=sum_axis)

    # ---- Step 2: sort ----
    x = np.sort(u)[::-1]
    v = np.log1p(x).reshape(-1, 1)

    # ---- Step 3: run GMM (avoid mutating comps) ----
    cutoff, ncells = gmm(x, v, comps=comps.copy())

    return cutoff