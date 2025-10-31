"""Utils"""

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr
from upsetplot import from_contents, UpSet


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
    adata1 = adata1[adata2.obs_names, adata2.var_names]

    return adata1_sub, adata2_sub

def plot_difference_heatmap(adata1, adata2, cell_subset=200, gene_subset=200, show_cell_names=True, show_gene_names=True, seed=42, title="Expression Difference Heatmap", out_path=None, show=True):
    np.random.seed(seed)
    adata1, adata2 = adata1.copy(), adata2.copy()
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)
    
    # convert to dense
    X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
    X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

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

def plot_per_cell_correlation(adata1, adata2, title="Per-cell Expression Correlation", out_path=None, show=True):
    adata1, adata2 = adata1.copy(), adata2.copy()
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

def run_scanpy_preprocessing_and_clustering(adata):
    adata = adata.copy()

    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    
    adata.var["mt"] = adata.var_names.str.upper().startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="sample")
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

def determine_cell_types(adata, method="celltypist", model_pkl=None):
    """
    Adds a 'celltype' column to adata.obs based on the specified method.
    """
    adata = adata.copy()
    if method == "celltypist":
        import celltypist
        if model_pkl is None:
            raise ValueError("model_pkl must be provided when method is 'celltypist'.")
        celltypist.models.download_models(force_update = False)
        predictions = celltypist.annotate(adata, model=model_pkl, majority_voting=True)
        pred_labels = predictions.predicted_labels[['majority_voting']]
        pred_labels = pred_labels.reindex(adata.obs_names)  # reorder to match adata.obs index
        adata.obs['celltype'] = pred_labels['majority_voting'].values
    else:
        raise ValueError(f"Unknown method {method} for determining cell types.")
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

