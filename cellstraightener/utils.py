"""Utils"""

import os
import subprocess
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import logging
from datetime import datetime
import anndata as ad
from scipy.stats import pearsonr
from upsetplot import from_contents, UpSet

def setup_logger(log_dir = None, log_level = None, verbose = 0, quiet = False):
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
    
    start_time_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"{start_time_string}.log")

        if os.path.exists(log_file_path):
            raise FileExistsError(f"Log file {log_file_path} already exists. Please choose a different run name.")
        print(f"Logging to {log_file_path}")

    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir:
            file_handler = logging.FileHandler(log_file_path)
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

def plot_matrix_scatterplot(adata1, adata2, title="Expression Scatterplot", sample_frac=1.0, seed=42, out_path=None, show=True):
    # adata1, adata2 = adata1.copy(), adata2.copy()
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)

    X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

    if X1.shape != X2.shape:
        raise ValueError(f"Shape mismatch: {X1.shape} vs {X2.shape}")
    
    # --- flatten both ---
    x = X1.ravel()
    y = X2.ravel()

    # --- optionally subsample to avoid millions of points ---
    if sample_frac < 1.0:
        np.random.seed(seed)
        n = int(len(x) * sample_frac)
        idx = np.random.choice(len(x), n, replace=False)
        x, y = x[idx], y[idx]
    else:
        print(f"Using all {len(x)} points for scatterplot. This may be slow if the dataset is large. Consider setting sample_frac < 1.0 to speed up.")

    # --- compute density (kernel density estimate) ---
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # --- sort by density for nice overlay ---
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # --- plot ---
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(x, y, c=z, s=3, cmap="viridis", edgecolor="none")
    plt.xlabel("adata1.X entries")
    plt.ylabel("adata2.X entries")
    plt.title(title)
    plt.colorbar(sc, label="Density")
    plt.grid(False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
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

def run_scanpy_preprocessing_and_clustering(adata, min_genes=100, min_cells=3, max_mt_percentage=25, n_top_genes=2000):
    # adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < max_mt_percentage, :]

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

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

def determine_cell_types(adata, method="celltypist", filter_empty=True, empty_column="is_empty", model_pkl=None, verbose=0, quiet=False):
    """
    Adds a 'celltype' column to adata.obs based on the specified method.
    """
    logger = setup_logger(verbose=verbose, quiet=quiet)
    # adata = adata.copy()
    
    # Identify real cells if present
    if filter_empty and empty_column in adata.obs.columns:
        real_mask = ~adata.obs[empty_column].astype(bool)
        adata_real = adata[real_mask].copy()
    else:
        real_mask = np.ones(adata.n_obs, dtype=bool)
        adata_real = adata

    if method == "celltypist":
        import celltypist
        if model_pkl is None:
            raise ValueError("model_pkl must be provided when method is 'celltypist'.")
        celltypist.models.download_models(force_update = False)
        predictions = celltypist.annotate(adata_real, model=model_pkl, majority_voting=True)
        pred_labels = predictions.predicted_labels[['majority_voting']]
        pred_labels = pred_labels.reindex(adata_real.obs_names)  # reorder to match adata_real.obs index
        adata_real.obs['celltype'] = pred_labels['majority_voting'].values
    else:
        raise ValueError(f"Unknown method {method} for determining cell types.")
    
    if filter_empty and empty_column in adata.obs.columns:
        adata.obs["celltype"] = "Empty Droplet"
        adata.obs.loc[real_mask, "celltype"] = pred_labels["majority_voting"].values
    else:
        adata = adata_real

    return adata

def plot_alluvial(*adatas, merged_df_csv, out_path, names=None, celltype_column_name="celltype", wompwomp_env="wompwomp_env", wompwomp_path="wompwomp"):
    if not os.path.exists(wompwomp_path):
        raise ValueError(f"wompwomp_path {wompwomp_path} does not exist.")

    for i, ad in enumerate(adatas):
        if celltype_column_name not in ad.obs.columns:
            raise ValueError(f"adata at position {i} does not have '{celltype_column_name}' in .obs columns.")

    if names is None:
        names = [f"adata_{i}" for i in range(len(adatas))]
    
    # if not os.path.exists(merged_df_csv):
    # Step 1: Get intersection of barcodes (shared cell IDs)
    common_barcodes = set(adatas[0].obs_names)
    for ad in adatas[1:]:
        common_barcodes &= set(ad.obs_names)

    # Step 2: Make sure all adatas only include the common cells
    common_barcodes = list(common_barcodes)
    adatas_filtered = [ad[common_barcodes, :].copy() for ad in adatas]

    # Step 3: Extract and merge celltype columns
    merged_df = pd.concat(
        [ad.obs['celltype'].rename(name) for ad, name in zip(adatas_filtered, names)],
        axis=1
    )

    merged_df.to_csv(merged_df_csv)

    names_str = " ".join(names)
    wompwomp_cmd = f"conda run -n {wompwomp_env} {wompwomp_path}/exec/wompwomp plot_alluvial --df {merged_df_csv} --graphing_columns {names_str} --coloring_algorithm left -o {out_path}"
    print(wompwomp_cmd)
    # subprocess.run(wompwomp_cmd, shell=True, check=True)

