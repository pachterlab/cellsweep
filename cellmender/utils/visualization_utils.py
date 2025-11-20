"""Visualization Utils"""

import os
import shutil
import subprocess
import re
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
# from scipy.stats import pearsonr
import torch
import tarfile
from upsetplot import from_contents, UpSet
from .data_utils import take_adata_cell_gene_intersection
from .logger_utils import setup_logger

default_colors = [
    "#D55E00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#E69F00", "#CC79A7", "#666666", "#AD7700", "#1C91D4", "#007756", "#D5C711", "#005685",
    "#A04700", "#B14380", "#4D4D4D", "#FFBE2D", "#80C7EF", "#00F6B3", "#F4EB71", "#06A5FF", "#FF8320", "#D99BBD", "#8C8C8C", "#FFCB57", "#9AD2F2",
    "#2CFFC6", "#F6EF8E", "#38B7FF", "#FF9B4D", "#E0AFCA", "#A3A3A3", "#8A5F00", "#1674A9", "#005F45", "#AA9F0D", "#00446B", "#803800", "#8D3666",
    "#3D3D3D"
]

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

def knee_plot(adata, expected_cells=None, color_column=None, out_path=None, show=True):
    # Compute total counts per barcode
    knee = np.sort(np.ravel(adata.X.sum(axis=1)))[::-1]
    barcodes = np.arange(1, len(knee) + 1)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    # Plot (x=barcodes, y=knee)
    if color_column is not None:
        if color_column not in adata.obs.columns:
            raise ValueError(f"color_column '{color_column}' not found in adata.obs columns.")
        color = adata.obs[color_column].values
        sc = ax.scatter(barcodes, knee, c=color, s=8, cmap="viridis", alpha=0.9, edgecolors="none")
        plt.colorbar(sc, ax=ax, label=color_column)
    else:
        ax.plot(barcodes, knee, linewidth=2, color="gray")

    cutoff_umi = None
    if expected_cells is not None:
        cutoff_umi = knee[expected_cells - 1]

        # Correct axes
        ax.axvline(x=expected_cells, color="k", ls="--", linewidth=1.5)
        ax.axhline(y=cutoff_umi, color="k", ls="--", linewidth=1.5)

        # Keep only barcodes up to expected_cells
        if color_column is None:
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

    return cutoff_umi



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

# def plot_matrix_scatterplot(adata1, adata2, figsize=(8, 8), s=20, scale="log", alpha=0.6, cmap='viridis', x_axis='adata1', y_axis='adata2', sample_frac=1.0, seed=42, out_path=None, show=True):
#     adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)

#     X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
#     X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

#     # Flatten matrices to 1D arrays
#     x = np.array(X1).flatten()
#     y = np.array(X2).flatten()

#     if sample_frac < 1.0:
#         np.random.seed(seed)
#         n = int(len(x) * sample_frac)
#         idx = np.random.choice(len(x), n, replace=False)
#         x, y = x[idx], y[idx]
#     else:
#         print(f"Using all {len(x)} points for scatterplot. This may be slow if the dataset is large.")
    
#     if scale == "log":
#         # Replace 0 values with 0.5 (so they appear at log2(0.5) = -1)
#         x = np.where(x < 0.5, 0.5, x)
#         y = np.where(y < 0.5, 0.5, y)
    
#     # Calculate density using KDE
#     xy = np.vstack([x, y])
#     z = gaussian_kde(xy)(xy)
    
#     # Sort by density so densest points are plotted last
#     idx = z.argsort()
#     x, y, z = x[idx], y[idx], z[idx]
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Create scatterplot
#     scatter = ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap=cmap, 
#                         edgecolors='none')
    
#     # Set equal ranges for x and y axes
#     all_vals = np.concatenate([x, y])
#     vmin, vmax = all_vals.min(), all_vals.max()
#     margin_factor = 1.1
#     ax.set_xlim(vmin / margin_factor, vmax * margin_factor)
#     ax.set_ylim(vmin / margin_factor, vmax * margin_factor)

#     if scale == "log":
#         # Set log scale for both axes
#         ax.set_xscale('log', base=2)
#         ax.set_yscale('log', base=2)

#         # Define tick locations at powers of 2
#         xticks = np.logspace(np.floor(np.log2(vmin)), np.ceil(np.log2(vmax)), num=int(np.ceil(np.log2(vmax)) - np.floor(np.log2(vmin))) + 1, base=2)
#         yticks = np.logspace(np.floor(np.log2(vmin)), np.ceil(np.log2(vmax)), num=int(np.ceil(np.log2(vmax)) - np.floor(np.log2(vmin))) + 1, base=2)
#         ax.set_xticks(xticks)
#         ax.set_yticks(yticks)

#         # Optionally format tick labels as 2^n
#         ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda val, _: f"$2^{{{int(np.log2(val))}}}$"))
#         ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda val, _: f"$2^{{{int(np.log2(val))}}}$"))
    
#     # Add diagonal line for reference
#     ax.plot([vmin / margin_factor, vmax * margin_factor], 
#             [vmin / margin_factor, vmax * margin_factor], 
#             'k--', alpha=0.3, linewidth=1, zorder=0)
    
#     # Labels and formatting
#     ax.set_xlabel(x_axis, fontsize=12)
#     ax.set_ylabel(y_axis, fontsize=12)
#     ax.set_title(f"{y_axis} vs {x_axis} CellxGene Scatterplot", fontsize=14, fontweight='bold')
#     ax.set_aspect('equal')
#     ax.grid(True, alpha=0.3)
    
#     # Add colorbar
#     cbar = plt.colorbar(scatter, ax=ax)
#     cbar.set_label('Density', fontsize=11)
    
#     plt.tight_layout()
#     if out_path is not None:
#         plt.savefig(out_path, dpi=300)
#     if show:
#         plt.show()
#     else:
#         plt.close()


# def plot_per_cell_correlation(adata1, adata2, title="Per-cell Expression Correlation", out_path=None, show=True):
#     # adata1, adata2 = adata1.copy(), adata2.copy()
#     adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)

#     X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else np.array(adata1.X)
#     X2 = adata2.X.toarray() if hasattr(adata2.X, "toarray") else np.array(adata2.X)

#     correlations = np.array([
#         pearsonr(X1[i, :], X2[i, :])[0]
#         for i in range(X1.shape[0])
#     ])

#     y_max = 10 ** np.ceil(np.log10(len(correlations)))

#     sns.histplot(correlations, bins=10, color='steelblue')
#     plt.xlim(0, 1)
#     plt.ylim(1, y_max)
#     plt.ylabel("Number of cells")
#     plt.yscale("log")
#     plt.title(title)
#     plt.tight_layout()
#     if out_path:
#         plt.savefig(out_path, bbox_inches="tight")
#     if not show:
#         plt.close()

def plot_matrix_scatterplot(
    adata1, adata2,
    figsize=(8, 8), s=20, scale="log", alpha=0.6,
    cmap='viridis', x_axis='adata1', y_axis='adata2',
    max_points=1000, seed=42, out_path=None, show=True
):
    # -------------------------
    # 1. Match cells + genes
    # -------------------------
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)
    X1 = adata1.X
    X2 = adata2.X

    if not sparse.issparse(X1) or not sparse.issparse(X2):
        raise ValueError("This function requires sparse AnnData.X matrices.")

    n_cells, n_genes = X1.shape
    N = n_cells * n_genes  # flattened total size

    # -------------------------
    # 2. Sample flattened positions
    # -------------------------
    if max_points is not None and max_points < N:
        rng = np.random.default_rng(seed)
        flat_idx = rng.choice(N, size=max_points, replace=False)
    else:
        print(f"Using all {N:,} values – may be slow.")
        flat_idx = np.arange(N)

    # Convert flattened index → (row, col)
    rows = flat_idx // n_genes
    cols = flat_idx % n_genes

    # -------------------------
    # 3. Safe sparse value extraction
    # -------------------------
    def get_values_from_sparse(X, rows, cols):
        """
        Efficiently extracts X[row, col] from a sparse matrix.
        Handles:
          - scalar returns
          - 1×1 sparse
          - 1×k sparse
        Never densifies the full matrix.
        """
        out = np.empty(len(rows), dtype=float)
        unique_rows = np.unique(rows)

        for r in unique_rows:
            mask = (rows == r)
            c_sub = cols[mask]
            sub = X[r, c_sub]

            if sparse.issparse(sub):
                sub_dense = sub.toarray().ravel()
            else:
                sub_dense = np.array(sub).ravel()  # scalar, or small ndarray

            out[mask] = sub_dense

        return out

    x = get_values_from_sparse(X1, rows, cols)
    y = get_values_from_sparse(X2, rows, cols)

    # -------------------------
    # 4. Log handling
    # -------------------------
    if scale == "log":
        x = np.where(x < 0.5, 0.5, x)
        y = np.where(y < 0.5, 0.5, y)

    # -------------------------
    # 5. KDE density
    # -------------------------
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort by density (lowest first → densest points plotted last)
    order = z.argsort()
    x, y, z = x[order], y[order], z[order]

    # -------------------------
    # 6. Plotting
    # -------------------------
    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap=cmap, edgecolors='none')

    all_vals = np.concatenate([x, y])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = 1.1
    ax.set_xlim(vmin / margin, vmax * margin)
    ax.set_ylim(vmin / margin, vmax * margin)

    if scale == "log":
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

        xticks = np.logspace(
            np.floor(np.log2(vmin)),
            np.ceil(np.log2(vmax)),
            num=int(np.ceil(np.log2(vmax)) - np.floor(np.log2(vmin))) + 1,
            base=2
        )
        ax.set_xticks(xticks)
        ax.set_yticks(xticks)

        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"$2^{{{int(np.log2(v))}}}$")
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"$2^{{{int(np.log2(v))}}}$")
        )

    # diagonal reference
    ax.plot(
        [vmin / margin, vmax * margin],
        [vmin / margin, vmax * margin],
        'k--', alpha=0.3, linewidth=1, zorder=0
    )

    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel(y_axis, fontsize=12)
    ax.set_title(f"{y_axis} vs {x_axis} Cell×Gene Scatterplot", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Density", fontsize=11)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300)
    if not show:
        plt.close()
    else:
        plt.show()

# --------------------------------------------
# Robust sparse Pearson correlation
# --------------------------------------------
def sparse_row_pearson(x, y):
    """
    Compute Pearson correlation between two 1×G row vectors.
    Works for:
        - csr_matrix rows
        - csc_matrix rows
        - numpy arrays
    Avoids densifying the full matrix.
    """

    # Convert to CSR row
    if sparse.issparse(x):
        x = x.tocsr()
    if sparse.issparse(y):
        y = y.tocsr()

    # Extract sparse structure
    x_data = x.data
    y_data = y.data
    x_idx = x.indices
    y_idx = y.indices

    G = x.shape[1]   # number of genes

    # Means
    mean_x = x_data.sum() / G
    mean_y = y_data.sum() / G

    # Intersect non-zero indices for fast centered dot
    intersect = np.intersect1d(x_idx, y_idx, assume_unique=False)

    if len(intersect) > 0:
        x_i = x[0, intersect].toarray().ravel()
        y_i = y[0, intersect].toarray().ravel()
        num = np.sum((x_i - mean_x) * (y_i - mean_y))
    else:
        num = 0.0

    # Norms for x
    x_centered_sq = np.sum((x_data - mean_x)**2)
    zeros_x = G - len(x_data)
    x_centered_sq += zeros_x * (mean_x**2)

    # Norms for y
    y_centered_sq = np.sum((y_data - mean_y)**2)
    zeros_y = G - len(y_data)
    y_centered_sq += zeros_y * (mean_y**2)

    # Denominator
    denom = np.sqrt(x_centered_sq * y_centered_sq)

    if denom == 0:
        return np.nan

    return num / denom


# --------------------------------------------
# Main per-cell correlation plotting function
# --------------------------------------------
def plot_per_cell_correlation(
    adata1, adata2,
    bins=20,
    title="Per-cell Expression Correlation Histogram",
    out_path=None,
    show=True
):

    # Match intersection of cells + genes
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)
    X1 = adata1.X
    X2 = adata2.X

    correlations = []

    # Compute correlation row-by-row (sparse)
    for i in range(X1.shape[0]):
        corr = sparse_row_pearson(X1[i, :], X2[i, :])
        correlations.append(corr)

    correlations = np.array(correlations)

    # Plot
    y_max = 10 ** np.ceil(np.log10(len(correlations)))
    sns.histplot(correlations, bins=bins, color='steelblue')

    plt.xlim(0, 1)
    plt.ylim(1, y_max)
    plt.yscale("log")
    plt.xlabel("Cell Pearson Correlation")
    plt.ylabel("Number of cells")
    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if not show:
        plt.close()
    else:
        plt.show()

def plot_per_cell_difference(adata_raw, adata_denoised, bins=10, tool="denoised", out_path=None, show=True):
    """
    Compute D = X_raw − X_denoised (sparse), take per-row sums, and plot histogram.
    """

    # 1. Match cells + genes
    adata_raw, adata_denoised = take_adata_cell_gene_intersection(adata_raw, adata_denoised)

    X_raw = adata_raw.X
    X_denoised = adata_denoised.X

    # Ensure sparse compatibility
    if not sparse.issparse(X_raw) or not sparse.issparse(X_denoised):
        raise ValueError("Both adata.X matrices must be sparse matrices.")

    # 2. Compute sparse difference
    D = X_raw - X_denoised  # remains sparse, no densification
    print(f"Total differences for {tool}: {D.sum():,}")

    # 3. Per-cell (per-row) sums
    #    sum(axis=1) returns a (n,1) sparse matrix, convert safely:
    row_sums = np.array(D.sum(axis=1)).ravel()

    # 4. Plot
    y_max = 10 ** np.ceil(np.log10(len(row_sums)))

    sns.histplot(row_sums, bins=bins, color='steelblue')
    plt.yscale("log")
    plt.ylim(1, y_max)
    plt.ylabel("Number of cells", fontsize=12)
    plt.title(f"Per-cell Difference Histogram: raw − {tool}", fontsize=14)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    if not show:
        plt.close()
    else:
        plt.show()

    return row_sums  # return values if user wants to inspect/plot further

def plot_alluvial(*adatas, merged_df_csv, out_path, names=None, displayed_column="celltype", wompwomp_env="wompwomp_env", wompwomp_path="wompwomp", verbose=0):
    logger = setup_logger(verbose=verbose)
    
    if not os.path.exists(wompwomp_path):
        raise ValueError(f"wompwomp_path {wompwomp_path} does not exist.")
    
    # check if wompwomp_env exists as a path or environment
    if not (os.path.exists(wompwomp_env) or wompwomp_env in subprocess.run("conda env list", shell=True, capture_output=True, text=True).stdout):
        raise ValueError(f"wompwomp_env {wompwomp_env} does not exist as a path or conda environment.")

    new_adatas = []
    new_names = []

    for a, n in zip(adatas, names):
        if a is not None:
            new_adatas.append(a)
            new_names.append(n)

    adatas = new_adatas
    names = new_names

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
    wompwomp_exec_path = f"{wompwomp_path}/exec/biowompwomp"
    if not os.path.exists(wompwomp_exec_path):
        wompwomp_exec_path = f"{wompwomp_path}/exec/wompwomp"
    wompwomp_cmd = f"conda run {conda_run_flag} {wompwomp_env} {wompwomp_exec_path} plot_alluvial --df {merged_df_csv} --graphing_columns {names_str} --coloring_algorithm left --disable_optimize_column_order -o {out_path}"
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
    sc.pl.dotplot(adata_raw_only_cellbender_cells, marker_genes, groupby=cluster_column, standard_scale=None, save="raw_tmp.png")  # title=title_raw
    print("------------------------------")
    print(title_processed)
    sc.pl.dotplot(adata_processed, marker_genes, groupby=cluster_column, standard_scale=None, save="processed_tmp.png")  # title=title_processed
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

def find_ckpt_file(extracted_files):
    # candidate extensions for torch checkpoints
    torch_exts = (".pt", ".pth", ".ckpt", ".pt.tar")

    # 1. Prefer files with the known extensions
    for f in extracted_files:
        if f.endswith(torch_exts):
            return f

    # 2. If nothing matches, try to inspect other files
    #    Some CellBender checkpoints have no extension.
    for f in extracted_files:
        path = "./extracted_checkpoint/" + f
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "params", "model")):
                return f
        except Exception:
            pass

    raise RuntimeError("No valid PyTorch checkpoint found inside the tar.")

def count_cellbender_parameters(ckpt_tar_path):
    # Extract the tar.gz file
    with tarfile.open(ckpt_tar_path, "r:gz") as tar:
        tar.extractall("./extracted_checkpoint")
        
    # List the extracted files to find the actual checkpoint
    extracted_files = os.listdir("./extracted_checkpoint")

    # Load the actual checkpoint file
    ckpt_name = find_ckpt_file(extracted_files)
    checkpoint_file = "./extracted_checkpoint/" + ckpt_name
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    if "params" not in checkpoint:
        print("No 'params' key found in the checkpoint. Cannot count parameters.")
        return None
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

def parse_em_log(log_path):
    pattern = re.compile(r"EM Iter\s+(\d+): ll=([-\d\.]+)")
    iters = []
    lls = []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                iters.append(int(m.group(1)))
                lls.append(float(m.group(2)))

    return iters, lls

def plot_cellmender_likelihood_over_epochs(iters=None, lls=None, log_path=None, out_path=None, show=True):
    if log_path is None and (iters is None or lls is None):
        raise ValueError("Either log_path or both iters and lls must be provided.")
    
    if log_path is not None:
        iters, lls = parse_em_log(log_path)

    # Convert negative log-likelihoods to positive values
    lls_pos = [-x for x in lls]  # now all > 0

    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)

    # Plot only the positive values
    ax.plot(iters, lls_pos, marker="o", linewidth=2, color="steelblue")

    # Set y-axis to log scale ONCE
    ax.set_yscale("log")

    ax.set_title("EM Log-Likelihood Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("-Log-Likelihood")

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def identify_human_and_mouse_cells(adata, human_prefix="hg19_", mouse_prefix="mm10_"):
    is_human = adata.var_names.str.startswith(human_prefix)
    is_mouse = adata.var_names.str.startswith(mouse_prefix)
    adata.obs["human_counts_total"] = np.array(adata.X[:, is_human].sum(axis=1)).ravel()
    adata.obs["mouse_counts_total"] = np.array(adata.X[:, is_mouse].sum(axis=1)).ravel()
    return adata

def plot_cross_species_histogram(adata, out_path_human=None, out_path_mouse=None, show=True):
    if adata is None:
        return

    if "human_counts_total" not in adata.obs.columns or "mouse_counts_total" not in adata.obs.columns:
        adata = identify_human_and_mouse_cells(adata)

    sns.histplot(
        data=adata.obs[adata.obs["genome"] == "mm10"],
        x="human_counts_total",
        bins=50,
        alpha=0.6
    )
    if out_path_mouse:
        plt.savefig(out_path_mouse, dpi=300, bbox_inches="tight")

    sns.histplot(
        data=adata.obs[adata.obs["genome"] == "hg19"],
        x="mouse_counts_total",
        bins=50,
        alpha=0.6
    )
    if out_path_human:
        plt.savefig(out_path_human, dpi=300, bbox_inches="tight")
    if not show:
        plt.close()

def plot_joint_scatterplot(adata_raw, adata_processed, processed_name="processed", marginal_type="histogram", show_marginal_ticks=False, show_point_movement=False, max_points=15_000, seed=42, out_path=None, show=True):    
    if adata_processed is None:
        return  # nothing to plot
    
    if marginal_type not in ["histogram", "kde"]:
        raise ValueError("marginal_type must be either 'histogram' or 'kde'")

    adata_raw, adata_processed = take_adata_cell_gene_intersection(adata_raw, adata_processed)
    
    if adata_raw.n_obs > max_points:
        np.random.seed(seed)
        sampled_indices = np.random.choice(adata_raw.obs_names, size=max_points, replace=False)
        adata_raw = adata_raw[sampled_indices].copy()
        adata_processed = adata_processed[sampled_indices].copy()
    
    if "human_counts_total" not in adata_raw.obs.columns or "mouse_counts_total" not in adata_raw.obs.columns:
        adata_raw = identify_human_and_mouse_cells(adata_raw)
    if "human_counts_total" not in adata_processed.obs.columns or "mouse_counts_total" not in adata_processed.obs.columns:
        adata_processed = identify_human_and_mouse_cells(adata_processed)
    
    human_raw = adata_raw.obs["human_counts_total"].values + 1
    mouse_raw = adata_raw.obs["mouse_counts_total"].values + 1

    human_processed = adata_processed.obs["human_counts_total"].values + 1
    mouse_processed = adata_processed.obs["mouse_counts_total"].values + 1

    df = pd.DataFrame({
        "x": np.concatenate([human_raw, human_processed]),
        "y": np.concatenate([mouse_raw, mouse_processed]),
        "group": (["raw"] * len(human_raw)) + ([processed_name] * len(human_processed))
    })

    g = sns.JointGrid(data=df, x="x", y="y", hue="group", palette={"raw": "gray", processed_name: "blue"}, marginal_ticks=show_marginal_ticks)

    if show_point_movement:
        for xr, yr, xp, yp in zip(human_raw, mouse_raw, human_processed, mouse_processed):
            g.ax_joint.plot(
                [xr, xp],
                [yr, yp],
                color="lightgray",
                alpha=0.4,
                linewidth=0.5,
                zorder=1
            )

    # Main scatter axes log scale
    g.ax_joint.set_xscale("log")
    g.ax_joint.set_yscale("log")

    g.ax_joint.set_xlabel("Human counts + 1")
    g.ax_joint.set_ylabel("Mouse counts + 1")

    
    if marginal_type == "kde":
        g.plot(sns.scatterplot, sns.kdeplot, alpha=.7, linewidth=.5)
    elif marginal_type == "histogram":
        g.plot(sns.scatterplot, sns.histplot, alpha=.7, linewidth=.5)
    
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close()

def print_top_empty_genes(adata, top_n=10, out_path=None):
    if 'empty_counts' not in adata.var.columns:
        if 'is_empty' not in adata.obs.columns:
            raise ValueError("adata.obs must contain 'is_empty' column indicating empty droplets.")
        adata.var['empty_counts'] = np.array(adata.X[adata.obs['is_empty'].values, :].sum(axis=0)).flatten()
    
    # Get sorted indices
    idx = np.argsort(adata.var['empty_counts'])[::-1]
    
    top_genes = adata.var_names[idx[:top_n]]
    top_vals  = adata.var['empty_counts'].iloc[idx[:top_n]]
    
    for gene, val in zip(top_genes, top_vals):
        print(f"{gene}: {val}")
    
    if out_path:
        df = pd.DataFrame({
            "gene": adata.var_names[idx[:]],
            "empty_counts": adata.var['empty_counts'].iloc[idx[:]].values
        })
        df.to_csv(out_path, index=False)

def plot_empty_gene_counts(adata, out_path=None, show=True):
    if 'empty_counts' not in adata.var.columns:
        if 'is_empty' not in adata.obs.columns:
            raise ValueError("adata.obs must contain 'is_empty' column indicating empty droplets.")
        adata.var['empty_counts'] = np.array(adata.X[adata.obs['is_empty'].values, :].sum(axis=0)).flatten()

    sorted_vals = np.sort(adata.var['empty_counts'])[::-1]
    sorted_vals = sorted_vals[sorted_vals > 0]
    
    plt.figure(figsize=(10, 4))
    plt.plot(sorted_vals)
    plt.xlabel("Gene rank")
    plt.ylabel("Empty-droplet counts")
    plt.title(f"Counts per gene in empty droplets (total genes: {adata.n_vars})")
    plt.yscale("log")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close() 

def plot_ambient_hat_vs_empty_fraction(adata_raw, adata_cellmender, log=False, remove_zeroes=False, lower_quantile_removed=None, upper_quantile_removed=None, out_path=None, show=True):
    """
    Plots ambient_hat from CellMender vs empty fraction from raw data.
    """
    if upper_quantile_removed is not None and (not isinstance(upper_quantile_removed, (int, float)) or not (0 < upper_quantile_removed < 1)):
        raise ValueError("upper_quantile_removed must be a float between 0 and 1 (or None for no outlier removal).")
    
    if 'empty_fraction' not in adata_raw.var.columns:
        total_empty_counts = adata_raw.var['empty_counts'].sum()
        adata_raw.var['empty_fraction'] = adata_raw.var['empty_counts'] / total_empty_counts if total_empty_counts > 0 else 0

    # intersection of genes
    genes = adata_raw.var_names.intersection(adata_cellmender.var_names)

    x = adata_raw.var.loc[genes, 'empty_fraction']
    y = adata_cellmender.var.loc[genes, 'ambient_hat']

    # # --- remove NaNs ---
    # mask = ~(np.isnan(x) | np.isnan(y))
    # x, y = x[mask], y[mask]

    # --- optional outlier removal ---
    if upper_quantile_removed is not None:
        qx = np.quantile(x, upper_quantile_removed)
        qy = np.quantile(y, upper_quantile_removed)
        mask2 = (x <= qx) & (y <= qy)
        x, y = x[mask2], y[mask2]
    
    if lower_quantile_removed is not None:
        qx = np.quantile(x, lower_quantile_removed)
        qy = np.quantile(y, lower_quantile_removed)
        mask3 = (x >= qx) & (y >= qy)
        x, y = x[mask3], y[mask3]
    
    if remove_zeroes:
        mask4 = (x > 0) & (y > 0)
        x, y = x[mask4], y[mask4]

    # --- y = x line ---
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.3, lw=1)

    # --- density via KDE ---
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    sc = plt.scatter(x, y, s=8, c=z, alpha=0.5, cmap="viridis")
    cb = plt.colorbar(sc)
    cb.set_label("Empty fraction")

    # --- log scale if requested ---
    if log:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel("Empty fraction (raw)")
    plt.ylabel("Ambient_hat (cellmender)")
    plt.title("Empty fraction vs ambient_hat")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close() 








def plot_per_cell_correlation_multi(
    adata1_list,
    adata2_list,
    labels=None,
    title="Per-cell Expression Correlation (Multiple Comparisons)",
    colors=None,
    out_path=None,
    show=True,
    fill=False
):
    """
    Compute and plot per-cell Pearson correlation curves (KDE) 
    for multiple pairs of AnnData objects.
    """

    # --- Validation ---
    if len(adata1_list) != len(adata2_list):
        raise ValueError("adata1_list and adata2_list must have same length.")

    n = len(adata1_list)
    if labels is None:
        labels = [f"set_{i+1}" for i in range(n)]

    if (colors is not None) and (len(colors) != n):
        raise ValueError("colors list must match number of datasets.")

    if colors is None:
        # default seaborn color cycle
        colors = sns.color_palette("tab10", n)

    # --- Collect correlations from each pair ---
    corr_sets = []

    for ad1, ad2 in zip(adata1_list, adata2_list):

        # match intersection
        ad1i, ad2i = take_adata_cell_gene_intersection(ad1, ad2)
        X1 = ad1i.X
        X2 = ad2i.X

        correlations = []
        for i in range(X1.shape[0]):
            corr = sparse_row_pearson(X1[i, :], X2[i, :])
            correlations.append(corr)

        corr_sets.append(np.array(correlations))

    # --- Plotting ---
    plt.figure(figsize=(8, 6))

    # Estimate maximum count for y-limits
    max_count = 0
    for values in corr_sets:
        hist_counts, _ = np.histogram(values, bins=50, range=(0, 1))
        max_count = max(max_count, hist_counts.max())

    # Final y-limit with log-scale padding
    y_max = 10 ** np.ceil(np.log10(max_count))

    # Plot each KDE
    for values, label, color in zip(corr_sets, labels, colors):
        sns.kdeplot(
            values,
            bw_adjust=1,
            fill=fill,
            color=color,
            label=label
        )

    plt.xlim(0, 1)
    plt.ylim(1, y_max)
    plt.yscale("log")
    plt.xlabel("Cell Pearson Correlation")
    plt.ylabel("Density (log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if not show:
        plt.close()
    else:
        plt.show()


def plot_per_cell_difference_multi(
    adata_raw_list,
    adata_denoised_list,
    labels=None,
    colors=None,
    bins=50,
    title="Per-cell Difference Distribution: raw − denoised",
    out_path=None,
    show=True
):
    if len(adata_raw_list) != len(adata_denoised_list):
        raise ValueError("adata_raw_list and adata_denoised_list must have same length.")

    n = len(adata_raw_list)
    if labels is None:
        labels = [f"set_{i+1}" for i in range(n)]
    if colors is None:
        colors = sns.color_palette("tab10", n)

    diff_sets = []

    for ad_raw, ad_den in zip(adata_raw_list, adata_denoised_list):
        ad_raw_i, ad_den_i = take_adata_cell_gene_intersection(ad_raw, ad_den)
        X_raw = ad_raw_i.X
        X_den = ad_den_i.X

        if not sparse.issparse(X_raw) or not sparse.issparse(X_den):
            raise ValueError("adata.X must be sparse.")

        D = X_raw - X_den
        row_sums = np.array(D.sum(axis=1)).ravel()
        diff_sets.append(row_sums)

    plt.figure(figsize=(8, 6))

    for values, label, color in zip(diff_sets, labels, colors):
        sns.histplot(
            values,
            bins=bins,
            element="step",
            fill=False,
            stat="count",
            color=color,
            label=label
        )

    plt.yscale("log")
    plt.xlabel("Per-cell difference sum: raw − denoised")
    plt.ylabel("Number of cells (log)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    if not show:
        plt.close()
    else:
        plt.show()



def plot_knee_multi(
    adata_list,
    labels=None,
    colors=None,
    title="Knee Plot (UMI Counts per Barcode)",
    linewidth=2,
    out_path=None,
    filter_empty=False,
    show=True
):
    """
    Compute and plot knee curves for multiple AnnData objects.
    A knee curve is defined as sorted descending per-cell UMI totals.

    Parameters
    ----------
    adata_list : list of AnnData
        List of AnnData objects.

    labels : list of str (optional)
        Names for each curve.

    colors : list of colors (optional)
        Colors for each curve.

    title : str
        Plot title.

    linewidth : float
        Line width for curves.

    out_path : str (optional)
        If provided, save figure.

    show : bool
        Whether to display the plot.

    """
    n = len(adata_list)

    if labels is None:
        labels = [f"adata_{i+1}" for i in range(n)]

    if (colors is not None) and (len(colors) != n):
        raise ValueError("colors list must match number of adatas.")

    if colors is None:
        colors = sns.color_palette("tab10", n)

    plt.figure(figsize=(8, 6))

    # ---- Generate curves ----
    for adata, label, color in zip(adata_list, labels, colors):

        if filter_empty:
            if "is_empty" in adata.obs.columns:
                adata = adata[~adata.obs["is_empty"]].copy()
            else:
                print("Warning: filter_empty=True but 'is_empty' column not found in adata.obs. Proceeding without filtering.")

        X = adata.X

        # Sparse-safe
        if sparse.issparse(X):
            row_sums = np.array(X.sum(axis=1)).ravel()
        else:
            row_sums = X.sum(axis=1)

        knee = np.sort(row_sums)[::-1]              # descending
        barcodes = np.arange(1, len(knee) + 1)      # rank axis

        plt.plot(
            barcodes,
            knee,
            color=color,
            linewidth=linewidth,
            label=label
        )

    # ---- Formatting ----
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Barcode Rank (log)", fontsize=12)
    plt.ylabel("Total UMI Counts (log)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)

    if not show:
        plt.close()
    else:
        plt.show()
