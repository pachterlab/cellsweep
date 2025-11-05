"""Utils"""

import os
import shutil
import subprocess
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import logging
from scipy import io, sparse
from datetime import datetime
import anndata as ad
from scipy.stats import pearsonr
import torch
import tarfile
from upsetplot import from_contents, UpSet

def setup_logger(log_file = None, log_level = None, verbose = 0, quiet = False):    
    if log_level is None:
        if quiet or verbose < -1:  # -q
            log_level = logging.CRITICAL
        elif verbose == -1:
            log_level = logging.ERROR
        elif verbose == 0:  # no -q/-v
            log_level = logging.WARNING
        elif verbose == 1:  # -v (and not -q)
            log_level = logging.INFO
        elif verbose >= 2:  # -vv (and not -q)
            log_level = logging.DEBUG
        else:
            raise ValueError(f"Invalid verbose level {verbose}. Use -q for quiet, -v for verbose, and -vv for very verbose.")
    
    if log_file is True:
        # default log file name with timestamp
        start_time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_file = f"cellmender_log_{start_time_string}.log"
    
    if log_file:
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        open(log_file, "w").close()  # create or overwrite the log file to ensure it is empty before logging starts. This prevents appending to an existing log file from previous runs, which could lead to confusion when analyzing logs.
        # if os.path.exists(log_file):
        #     raise FileExistsError(f"Log file {log_file} already exists. Please choose a different log file name.")
        print(f"Logging to {log_file}")

    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        return logger
    
    logger.propagate = False
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def make_upset_plot(upset_data_dict: dict[str: list[str]], out_path: str = None, title: str = None, show: bool = True):
    """
    eg upset_data_dict: {
        "Method A": ["cell1", "cell2", "cell3"],
        "Method B": ["cell2", "cell3", "cell4"],
        "Method C": ["cell1", "cell4", "cell5"],
    }
    """
    upset_data_dict = {k: v for k, v in upset_data_dict.items() if v is not None}  # iterate through dict, and if a value is None, skip that entry
    upset_data = from_contents(upset_data_dict)
    ax_dict = UpSet(upset_data, show_counts=True).plot()
    if title:
        ax_dict["intersections"].set_title(title, fontsize=14, pad=15)
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if not show:
        plt.close()
    return ax_dict

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

def knee_plot(adata, expected_cells=None, out_path=None, show=True):
    # Compute total counts per barcode
    knee = np.sort(np.ravel(adata.X.sum(axis=1)))[::-1]
    barcodes = np.arange(1, len(knee) + 1)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    # Plot (x=barcodes, y=knee)
    ax.plot(barcodes, knee, linewidth=2, color="gray")

    if expected_cells is not None:
        cutoff_umi = knee[expected_cells - 1]

        # Correct axes
        ax.axvline(x=expected_cells, color="k", ls="--", linewidth=1.5)
        ax.axhline(y=cutoff_umi, color="k", ls="--", linewidth=1.5)

        # Keep only barcodes up to expected_cells
        keep_mask = barcodes <= expected_cells
        ax.plot(barcodes[keep_mask], knee[keep_mask], linewidth=2, color="blue")

        print(f"UMI cutoff for expected cells ({expected_cells}): {cutoff_umi:.2f}")

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Auto axis limits based on data range
    ax.set_xlim(barcodes.min(), barcodes.max())
    # ax.set_ylim(knee.min(), knee.max())
    ax.set_ylim(1, knee.max())

    # Labels and styling
    ax.set_title("Knee Plot", fontsize=18)
    ax.set_xlabel("Barcodes", fontsize=18)
    ax.set_ylabel("UMI counts per barcode", fontsize=18)
    ax.grid(True, which="both", color="lightgray")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=14)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if not show:
        plt.close()



def plot_difference_heatmap(adata1, adata2, cell_subset=200, gene_subset=200, show_cell_names=True, show_gene_names=True, seed=42, title="Expression Difference Heatmap", out_path=None, show=True):
    np.random.seed(seed)
    # adata1, adata2 = adata1.copy(), adata2.copy()
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)
    
    # convert to dense
    X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

    if X1.shape != X2.shape:
        raise ValueError(f"Shape mismatch: {X1.shape} vs {X2.shape}")

    # difference matrix
    diff = X1 - X2

    # random subset of cells and genes
    if cell_subset is None and gene_subset is None:
        print("No subsets specified; plotting full heatmap may be slow.")
    if cell_subset is None:
        cell_subset = diff.shape[0]
    if gene_subset is None:
        gene_subset = diff.shape[1]

    rng = np.random.default_rng(seed)
    cell_idx = rng.choice(diff.shape[0], min(cell_subset, diff.shape[0]), replace=False)
    gene_idx = rng.choice(diff.shape[1], min(gene_subset, diff.shape[1]), replace=False)
    diff_sub = diff[np.ix_(cell_idx, gene_idx)]

    # Get names
    cell_labels = adata1.obs_names[cell_idx]
    gene_labels = adata1.var_names[gene_idx]

    if not show_cell_names:
        cell_labels = range(len(cell_idx))
    if not show_gene_names:
        gene_labels = range(len(gene_idx))

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        diff_sub,
        cmap="coolwarm",
        center=0,
        xticklabels=gene_labels,
        yticklabels=cell_labels,
        cbar_kws={"label": "Expression difference (adata2 - adata1)"}
    )
    plt.title(title)
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def plot_matrix_scatterplot(adata1, adata2, figsize=(8, 8), s=20, scale="log", alpha=0.6, cmap='viridis', x_axis='adata1', y_axis='adata2', sample_frac=1.0, seed=42, out_path=None, show=True):
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)

    X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

    # Flatten matrices to 1D arrays
    x = np.array(X1).flatten()
    y = np.array(X2).flatten()

    if sample_frac < 1.0:
        np.random.seed(seed)
        n = int(len(x) * sample_frac)
        idx = np.random.choice(len(x), n, replace=False)
        x, y = x[idx], y[idx]
    else:
        print(f"Using all {len(x)} points for scatterplot. This may be slow if the dataset is large.")
    
    if scale == "log":
        # Replace 0 values with 0.5 (so they appear at log2(0.5) = -1)
        x = np.where(x < 0.5, 0.5, x)
        y = np.where(y < 0.5, 0.5, y)
    
    # Calculate density using KDE
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    # Sort by density so densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatterplot
    scatter = ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap=cmap, 
                        edgecolors='none')
    
    # Set equal ranges for x and y axes
    all_vals = np.concatenate([x, y])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin_factor = 1.1
    ax.set_xlim(vmin / margin_factor, vmax * margin_factor)
    ax.set_ylim(vmin / margin_factor, vmax * margin_factor)

    if scale == "log":
        # Set log scale for both axes
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)

        # Define tick locations at powers of 2
        xticks = np.logspace(np.floor(np.log2(vmin)), np.ceil(np.log2(vmax)), num=int(np.ceil(np.log2(vmax)) - np.floor(np.log2(vmin))) + 1, base=2)
        yticks = np.logspace(np.floor(np.log2(vmin)), np.ceil(np.log2(vmax)), num=int(np.ceil(np.log2(vmax)) - np.floor(np.log2(vmin))) + 1, base=2)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        # Optionally format tick labels as 2^n
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda val, _: f"$2^{{{int(np.log2(val))}}}$"))
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda val, _: f"$2^{{{int(np.log2(val))}}}$"))
    
    # Add diagonal line for reference
    ax.plot([vmin / margin_factor, vmax * margin_factor], 
            [vmin / margin_factor, vmax * margin_factor], 
            'k--', alpha=0.3, linewidth=1, zorder=0)
    
    # Labels and formatting
    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel(y_axis, fontsize=12)
    ax.set_title(f"{y_axis} vs {x_axis} CellxGene Scatterplot", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density', fontsize=11)
    
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()



def plot_per_cell_correlation(adata1, adata2, title="Per-cell Expression Correlation", out_path=None, show=True):
    # adata1, adata2 = adata1.copy(), adata2.copy()
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)

    X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

    correlations = np.array([
        pearsonr(X1[i, :], X2[i, :])[0]
        for i in range(X1.shape[0])
    ])

    y_max = 10 ** np.ceil(np.log10(len(correlations)))

    sns.histplot(correlations, bins=10, color='steelblue')
    plt.xlim(0, 1)
    plt.ylim(1, y_max)
    plt.ylabel("Number of cells")
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if not show:
        plt.close()

def run_scanpy_preprocessing_and_clustering(adata, min_genes=100, min_cells=3, umi_top_percentile_to_remove=None, unique_genes_top_percentile_to_remove=None, mt_gene_percentile_to_remove=None, max_mt_percentage=25, n_top_genes=2000, hvg_flavor="seurat_v3", n_pcs=50, n_neighbors=15, leiden_resolution=1.0, seed=42, verbose=0, quiet=False):
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
def infer_empty_droplets(adata, method="threshold", umi_cutoff=None, expected_cells=None, verbose=0, quiet=False):
    """
    input: adata
    output: adata with adata.obs: is_empty
      - is_empty: boolean indicating whether each cell is an empty droplet or not. This is inferred using a simple heuristic: if the total counts for a cell are below a certain threshold (e.g., 100), it is considered an empty droplet. This threshold can be adjusted based on the dataset and expected cell types.
    """
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

def determine_cell_types(adata, method="celltypist", filter_empty=True, empty_column="is_empty", umi_cutoff=None, expected_cells=None, model_pkl=None, verbose=0, quiet=False):
    """
    Adds a 'celltype' column to adata.obs based on the specified method.
    """
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
        celltypist.models.download_models(force_update = False)
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
        adata_real.obs['celltype'] = pred_labels['majority_voting'].values
        adata.obs["celltype"] = adata_real.obs["celltype"].reindex(adata.obs_names)
        if filter_empty:
            adata.obs["celltype"] = adata.obs["celltype"].cat.add_categories([empty_droplet_category_name]).fillna(empty_droplet_category_name)
    #!!! add more methods here
    else:
        raise ValueError(f"Unknown method {method} for determining cell types.")

    return adata

def plot_alluvial(*adatas, merged_df_csv, out_path, names=None, displayed_column="celltype", wompwomp_env="wompwomp_env", wompwomp_path="wompwomp", verbose=0):
    logger = setup_logger(verbose=verbose)
    
    if not os.path.exists(wompwomp_path):
        raise ValueError(f"wompwomp_path {wompwomp_path} does not exist.")
    
    # check if wompwomp_env exists as a path or environment
    if not (os.path.exists(wompwomp_env) or wompwomp_env in subprocess.run("conda env list", shell=True, capture_output=True, text=True).stdout):
        raise ValueError(f"wompwomp_env {wompwomp_env} does not exist as a path or conda environment.")

    for adata in adatas.copy():
        if adata is None:
            if len(names) == len(adatas):
                names.remove(names[adatas.index(adata)])
            adatas.remove(adata)

    for i, adata in enumerate(adatas):
        if displayed_column not in adata.obs.columns:
            raise ValueError(f"adata at position {i} does not have '{displayed_column}' in .obs columns.")

    if names is None:
        names = [f"adata_{i}" for i in range(len(adatas))]
    
    if len(names) != len(adatas):
        raise ValueError(f"Length of names ({len(names)}) does not match number of adatas ({len(adatas)}).")
    
    # if not os.path.exists(merged_df_csv):
    # Step 1: Get intersection of barcodes (shared cell IDs)
    common_barcodes = set(adatas[0].obs_names)
    for ad in adatas[1:]:
        common_barcodes &= set(ad.obs_names)

    # Step 2: Make sure all adatas only include the common cells
    common_barcodes = list(common_barcodes)
    adatas_filtered = [ad[common_barcodes, :].copy() for ad in adatas]

    # Step 3: Extract and merge 'displayed_column' columns
    merged_df = pd.concat(
        [ad.obs[displayed_column].rename(name) for ad, name in zip(adatas_filtered, names)],
        axis=1
    )

    merged_df.to_csv(merged_df_csv)

    names_str = " ".join(names)
    conda_run_flag = "-p" if "/" in wompwomp_env else "-n"
    wompwomp_cmd = f"conda run {conda_run_flag} {wompwomp_env} {wompwomp_path}/exec/wompwomp plot_alluvial --df {merged_df_csv} --graphing_columns {names_str} --coloring_algorithm left -o {out_path}"
    logger.info(f"Running wompwomp for {displayed_column}")
    logger.debug(wompwomp_cmd)
    subprocess.run(wompwomp_cmd, shell=True, check=True)


def make_raw_and_processed_dotplots(adata_raw, adata_processed, marker_genes, celltype_column="celltype", cluster_column="leiden", title_raw=None, title_processed=None, out_path_raw="raw_dotplot.png", out_path_processed="processed_dotplot.png"):
    if adata_raw is None or adata_processed is None:
        print("One of the adatas is None, skipping dotplot generation.")
        return

    common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)
    adata_raw_only_cellbender_cells = adata_raw[common_cells].copy()
    adata_raw_only_cellbender_cells.obs = adata_raw_only_cellbender_cells.obs.join(adata_processed.obs[[celltype_column, cluster_column]], how='left')

    print(title_raw)
    sc.pl.dotplot(adata_raw_only_cellbender_cells, marker_genes, groupby=cluster_column, standard_scale="var", save="raw_tmp.png")  # title=title_raw
    print("------------------------------")
    print(title_processed)
    sc.pl.dotplot(adata_processed, marker_genes, groupby=cluster_column, standard_scale="var", save="processed_tmp.png")  # title=title_processed
    print("------------------------------")

    shutil.move("figures/dotplot_raw_tmp.png", out_path_raw)
    shutil.move("figures/dotplot_processed_tmp.png", out_path_processed)
    os.rmdir("figures")

def count_cellmender_parameters(log_path):
    with open(log_path, "r") as f:
        for line in f:
            if "Number of parameters in the cellmender model" in line:
                return line.strip()
    return None

def count_cellbender_parameters(ckpt_tar_path):
    # Extract the tar.gz file
    with tarfile.open(ckpt_tar_path, "r:gz") as tar:
        tar.extractall("./extracted_checkpoint")
        
    # List the extracted files to find the actual checkpoint
    extracted_files = os.listdir("./extracted_checkpoint")

    # Load the actual checkpoint file
    checkpoint_file = "./extracted_checkpoint/" + extracted_files[0]
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    state_dict = checkpoint["params"]
    total_params = sum(p.numel() for p in state_dict.values())
    
    # Print architecture information
    print("\n=== Model Architecture (inferred from state_dict) ===")
    print(f"Total parameters: {total_params:,}\n")
    
    for name, param in state_dict.items():
        param_count = param.numel()
        print(f"{name:50s} | Shape: {str(param.shape):20s} | Params: {param_count:,}")
    
    # Group by layer (optional)
    print("\n=== Parameter Summary by Layer ===")
    layer_counts = {}
    for name, param in state_dict.items():
        layer_name = name.split('.')[0]  # Get first part before '.'
        if layer_name not in layer_counts:
            layer_counts[layer_name] = 0
        layer_counts[layer_name] += param.numel()
    
    for layer, count in layer_counts.items():
        print(f"{layer:30s}: {count:,} parameters")
    
    # remove the extracted checkpoint directory after loading the checkpoint
    shutil.rmtree("./extracted_checkpoint")

    return total_params


def read_r_matrix_into_anndata(file_prefix):
    """
    Read in the matrix output from R SoupX processing into an AnnData object.
    Assumed files: 
    1. {file_prefix}.mtx
    2. {file_prefix}_genes.csv
    3. {file_prefix}_barcodes.csv
    """
    if file_prefix is None:
        print("No file prefix provided for reading R matrix into AnnData. Returning None.")
        return None

    for expected_file in [f"{file_prefix}.mtx", f"{file_prefix}_genes.csv", f"{file_prefix}_barcodes.csv"]:
        if not os.path.exists(expected_file):
            raise FileNotFoundError(f"Expected file not found: {expected_file}")

    X = io.mmread(f"{file_prefix}.mtx").T.tocsr()
    genes = pd.read_csv(f"{file_prefix}_genes.csv", header=None)[0].to_list()
    barcodes = pd.read_csv(f"{file_prefix}_barcodes.csv", header=None)[0].to_list()

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = barcodes
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

# anndata object, h5 path, h5ad path, or matrix directory (containing matrix.mtx, genes.tsv, barcodes.tsv)
def load_adata(adata, logger=None, verbose=0, quiet=False):
    if logger is None:
        logger = setup_logger(verbose=verbose, quiet=quiet)
    if isinstance(adata, str):
        if adata.endswith(".h5ad"):
            logger.info(f"Loading adata from {adata!r}")
            adata = ad.read_h5ad(adata)
        elif adata.endswith(".h5"):
            logger.info(f"Loading adata from {adata!r}")
            import scanpy as sc
            adata = sc.read_10x_h5(adata)
        elif os.path.isdir(adata):
            import scanpy as sc
            logger.info(f"Searching recursively for 10x-style dataset under {adata!r}")

            found_dirs = []
            for root, dirs, files in os.walk(adata):
                files_lower = [f.lower() for f in files]
                if "matrix.mtx" in files_lower and "barcodes.tsv" in files_lower and ("genes.tsv" in files_lower or "features.tsv" in files_lower):
                    found_dirs.append(root)

            if len(found_dirs) == 0:
                raise FileNotFoundError(f"No valid 10x dataset found under {adata!r}. Expected matrix.mtx, barcodes.tsv, and genes.tsv or features.tsv.")
            elif len(found_dirs) > 1:
                raise RuntimeError(
                    f"Multiple 10x-style datasets found under {adata!r}:\n" +
                    "\n".join(found_dirs) +
                    "\nPlease specify one directory explicitly."
                )

            tenx_dir = found_dirs[0]
            logger.info(f"Found 10x dataset in {tenx_dir!r}")
            
            use_gene_symbols = os.path.exists(os.path.join(tenx_dir, "genes.tsv"))
            adata = sc.read_10x_mtx(
                tenx_dir,
                var_names="gene_symbols" if use_gene_symbols else "gene_ids",
                make_unique=True
            )
        else:
            raise ValueError(f"Invalid adata input {adata!r}. Expected a path to an .h5ad file, an .h5 file, a matrix-containing directory, or an AnnData object.")
    elif isinstance(adata, ad.AnnData):
        pass
        # adata = adata.copy()
    else:
        raise ValueError(f"Invalid adata input {adata!r}. Expected a path to an .h5ad file, an .h5 file, a matrix-containing directory, or an AnnData object.")
    return adata