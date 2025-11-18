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

def my_hello_world2():
    print("Hello, world!2")

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