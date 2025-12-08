"""Visualization Utils"""

import os
import shutil
import subprocess
import re
import matplotlib
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.scale import SymmetricalLogScale
from matplotlib.colors import LogNorm
import seaborn as sns
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
import logging
from scipy import io, sparse
from datetime import datetime
import anndata as ad
# from scipy.stats import pearsonr
import torch
import tarfile
from upsetplot import from_contents, UpSet
from .data_utils import take_adata_cell_gene_intersection, infer_empty_droplets
from .logger_utils import setup_logger

default_colors = [
    "#D55E00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#E69F00", "#CC79A7", "#666666", "#AD7700", "#1C91D4", "#007756", "#D5C711", "#005685",
    "#A04700", "#B14380", "#4D4D4D", "#FFBE2D", "#80C7EF", "#00F6B3", "#F4EB71", "#06A5FF", "#FF8320", "#D99BBD", "#8C8C8C", "#FFCB57", "#9AD2F2",
    "#2CFFC6", "#F6EF8E", "#38B7FF", "#FF9B4D", "#E0AFCA", "#A3A3A3", "#8A5F00", "#1674A9", "#005F45", "#AA9F0D", "#00446B", "#803800", "#8D3666",
    "#3D3D3D"
]

def auto_bins(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        return 20    # fallback
    bin_width = 2 * iqr * (len(x) ** (-1/3))
    bins = int((x.max() - x.min()) / bin_width)
    return max(10, min(bins, 200))

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

def plot_matrix_scatterplot(
    adata1, adata2,
    figsize=(8, 8), scale="log", point_type="matrix", density_type="scatter_with_density", alpha=0.6,
    cmap='viridis', x_axis='adata1', y_axis='adata2', out_path=None, show=True
):
    # -------------------------
    # 1. Match cells + genes
    # -------------------------
    if adata1 is None or adata2 is None:
        print("One of the adatas is None, skipping matrix scatterplot.")
        return 
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)
    
    if point_type == "matrix":
        X1 = adata1.X
        X2 = adata2.X

        if not sparse.issparse(X1) or not sparse.issparse(X2):
            raise ValueError("This function requires sparse AnnData.X matrices.")

        n_cells, n_genes = X1.shape
        N = n_cells * n_genes  # flattened total size

        def get_all_nonzero_pairs(X1, X2):
            X1 = X1.tocsr()
            X2 = X2.tocsr()

            # Extract all nonzero (row, col, value) in X1
            rows = np.repeat(np.arange(X1.shape[0]), np.diff(X1.indptr))
            cols = X1.indices
            x_vals = X1.data

            # Extract matching values from X2 (vectorized)
            y_vals = X2[rows, cols].A1

            return x_vals, y_vals

        print("Extracting all nonzero pairs from sparse matrices...")
        x, y = get_all_nonzero_pairs(X1, X2)
    elif point_type == "cell":
        x = np.array(adata1.X.sum(axis=1)).ravel()
        y = np.array(adata2.X.sum(axis=1)).ravel()
    elif point_type == "gene":
        x = np.array(adata1.X.sum(axis=0)).ravel()
        y = np.array(adata2.X.sum(axis=0)).ravel()
    else:
        raise ValueError(f"Unknown point_type '{point_type}'. Use 'matrix', 'cell', or 'gene'.")

    # -------------------------
    # 4. Log handling
    # -------------------------
    if scale == "log":
        x = np.where(x < 0.5, 0.5, x)
        y = np.where(y < 0.5, 0.5, y)

    # -------------------------
    # 6. Plotting
    # -------------------------
    print("Creating scatterplot...")
    fig, ax = plt.subplots(figsize=figsize)

    if density_type == "2d_hist":
        print("Calculating 2D histogram...")
        h = ax.hist2d(x, y, bins=2000, cmap=cmap, norm=LogNorm())
        cbar = plt.colorbar(h[3], ax=ax)
        cbar.set_label("count density")
        # hb = ax.hexbin(x, y, gridsize=4000, cmap=cmap, norm=LogNorm(), mincnt=1)
        # plt.colorbar(hb, ax=ax).set_label("density")
    elif density_type == "scatter":
        print("Calculating scatterplot...")
        ax.scatter(x, y, c='steelblue', s=2, alpha=0.05, edgecolors='none')
    elif density_type == "scatter_with_density":
        print("Calculating scatterplot...")
        import mpl_scatter_density
        from astropy.visualization import LogStretch, AsinhStretch
        from astropy.visualization.mpl_normalize import ImageNormalize
        norm = ImageNormalize(vmin=0., vmax=100_000, stretch=LogStretch(a=100_000))  # big a for more log-like
        ax.remove()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(x, y, norm=norm, cmap=cmap)
        # fig.colorbar(density, label='Number of points per pixel')
        cbar = plt.colorbar(density, ax=ax)
        cbar.set_label("Density", fontsize=11)
        cbar.set_ticks([1, 10, 100, 1000, 10000, 100000])
        cbar.set_ticklabels([r"$10^{{{}}}$".format(int(np.log10(tick))) for tick in cbar.get_ticks()])

    elif density_type == "scatter_with_kde":
        print("Calculating scatterplot...")
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Sort by density (lowest first → densest points plotted last)
        order = z.argsort()
        x, y, z = x[order], y[order], z[order]
        sc = ax.scatter(x, y, c=z, s=20, alpha=alpha, cmap=cmap, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Density", fontsize=11)
    else:
        raise ValueError(f"Unknown density_type '{density_type}'. Use '2d_hist', 'scatter', 'scatter_with_density', or 'scatter_with_kde'.")

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
    ax.set_title(f"{y_axis} vs {x_axis} {point_type} Scatterplot", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

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

def compute_sparse_pearson(
    X1, X2,
    mode="cell"
):
    """
    Compute Pearson correlations for either:
        mode="cell": row-wise (per cell), compares X1[i,:] vs X2[i,:]
        mode="gene": column-wise (per gene), compares X1[:,j] vs X2[:,j]

    Returns
    -------
    numpy array of correlations (length = n_cells or n_genes)
    """

    if mode not in ("cell", "gene"):
        raise ValueError("mode must be 'cell' or 'gene'")

    if mode == "cell":
        n = X1.shape[0]  # n_cells
        corrs = np.zeros(n)

        for i in range(n):
            corrs[i] = sparse_row_pearson(X1[i, :], X2[i, :])

        return corrs

    else:  # mode == "gene"
        # convert to CSC for fast column slicing
        X1_c = X1.tocsc()
        X2_c = X2.tocsc()

        n = X1.shape[1]  # n_genes
        corrs = np.zeros(n)

        for j in range(n):
            corrs[j] = sparse_row_pearson(X1_c[:, j].T, X2_c[:, j].T)

        return corrs

def sparse_cosine(x, y):
    # x and y are 1 × G sparse rows (CSR)
    num = x.multiply(y).sum()
    den = np.sqrt(x.multiply(x).sum()) * np.sqrt(y.multiply(y).sum())
    return float(num / den) if den != 0 else 0.0

# --------------------------------------------
# Main per-cell correlation plotting function
# --------------------------------------------
def plot_per_cell_correlation(
    adata1, adata2,
    bins=None,
    plot_type="cell",
    metric="cosine",
    title="Per-cell Expression Correlation Histogram",
    out_path=None,
    show=True
):
    if adata1 is None or adata2 is None:
        print("One of the adatas is None, skipping per-cell correlation plot.")
        return

    # Match intersection of cells + genes
    adata1, adata2 = take_adata_cell_gene_intersection(adata1, adata2)
    X1 = adata1.X
    X2 = adata2.X

    correlations = []

    # Compute correlation row-by-row (sparse)
    if metric == "pearson":
        x_label = f"{plot_type.capitalize()} Pearson Correlation"
        correlations = compute_sparse_pearson(X1, X2, mode=plot_type)
    elif metric == "cosine":
        x_label = f"{plot_type.capitalize()} Cosine Similarity"
        if plot_type == "gene":
            X1c = X1.tocsc()
            X2c = X2.tocsc()

            for j in range(X1.shape[1]):
                x_col = X1c[:, j].T  # CSC → 1×G
                y_col = X2c[:, j].T
                correlations.append(sparse_cosine(x_col, y_col))
        elif plot_type == "cell":
            for i in range(X1.shape[0]):
                sim = cosine_similarity(X1[i, :], X2[i, :])[0, 0]
                correlations.append(sim)
    else:
        raise ValueError("Unknown metric")

    correlations = np.array(correlations)

    # Plot
    if not bins:
        bins = int(2 * (len(correlations) ** (1/3)))
    y_max = 10 ** np.ceil(np.log10(len(correlations)))
    sns.histplot(correlations, bins=bins, color='steelblue')

    plt.xlim(0, 1)
    # plt.ylim(0.75, y_max)
    plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel("Number of cells")
    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    if not show:
        plt.close()
    else:
        plt.show()

def plot_per_cell_difference(adata_raw, adata_denoised, bins=None, plot_type="cell", tool_raw="raw", tool_processed="denoised", out_path=None, show=True):
    """
    Compute D = X_raw − X_denoised (sparse), take per-row sums, and plot histogram.
    """
    if adata_raw is None or adata_denoised is None:
        print("One of the adatas is None, skipping per-cell difference plot.")
        return

    # 1. Match cells + genes
    adata_raw, adata_denoised = take_adata_cell_gene_intersection(adata_raw, adata_denoised)

    X_raw = adata_raw.X
    X_denoised = adata_denoised.X

    # Ensure sparse compatibility
    if not sparse.issparse(X_raw) or not sparse.issparse(X_denoised):
        raise ValueError("Both adata.X matrices must be sparse matrices.")

    # 2. Compute sparse difference
    D = X_raw - X_denoised  # remains sparse, no densification
    total_matrix_difference = D.sum()
    print(f"Total differences between {tool_raw} and {tool_processed}: {total_matrix_difference:,}")

    # 3. Per-cell (per-row) sums
    #    sum(axis=1) returns a (n,1) sparse matrix, convert safely:
    if plot_type == "cell":
        sums = np.array(D.sum(axis=1)).ravel()
    elif plot_type == "gene":
        sums = np.array(D.sum(axis=0)).ravel()
    elif plot_type == "matrix":
        sums = np.array(total_matrix_difference).ravel()

    # 4. Plot
    y_max = 10 ** np.ceil(np.log10(len(sums)))

    if not bins:
        bins = int(2 * (len(sums) ** (1/3)))

    sns.histplot(sums, bins=bins, color='steelblue')
    plt.yscale("log")
    # plt.ylim(0.75, y_max)
    plt.ylabel("Number of cells", fontsize=12)
    plt.title(f"Per-{plot_type.capitalize()} Difference Histogram: {tool_raw} − {tool_processed}", fontsize=14)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    if not show:
        plt.close()
    else:
        plt.show()

    return sums  # return values if user wants to inspect/plot further

def plot_alluvial(*adatas, merged_df_csv=None, out_path=None, names=None, displayed_column="celltype", verbose=0):
    verbose = True if verbose >= 1 else False

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

    if merged_df_csv is not None:
        merged_df.to_csv(merged_df_csv)
    
    from wompywompy import plot_alluvial
    plot_alluvial(df=merged_df, graphing_columns=names, sorting_algorithm="neighbornet", coloring_algorithm="left", optimize_column_order=False, savefig=out_path, verbose=verbose)

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
    adata.obs['genome'] = np.where(adata.obs['human_counts_total'] >= adata.obs['mouse_counts_total'], 'hg19', 'mm10')  # predict genome
    return adata

def plot_cross_species_histogram(adata, processed_name="processed", doublet_cell_set=None, out_path_human=None, out_path_mouse=None, show=True):
    if adata is None:
        return

    if "human_counts_total" not in adata.obs.columns or "mouse_counts_total" not in adata.obs.columns:
        adata = identify_human_and_mouse_cells(adata)
    
    if isinstance(doublet_cell_set, set):  # remove doublets if provided
        adata = adata[~adata.obs_names.isin(doublet_cell_set)].copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    # --- Mouse cells: plotting human_counts_total in blue ---
    sns.histplot(
        data=adata.obs[adata.obs["genome"] == "mm10"],
        x="human_counts_total",
        bins=100,
        alpha=0.6,
        linewidth=1.5,
        color="blue",
        element="step",
        fill=False,
        ax=ax,
        label="Mouse cell human gene contamination"
    )

    # --- Human cells: plotting mouse_counts_total in gray ---
    sns.histplot(
        data=adata.obs[adata.obs["genome"] == "hg19"],
        x="mouse_counts_total",
        bins=100,
        alpha=0.6,
        linewidth=1.5,
        color="gray",
        element="step",
        fill=False,
        ax=ax,
        label="Human cell mouse gene contamination"
    )

    ax.set_xlabel("Cross-species counts")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    ax.set_title(f"Cross-species Gene Counts in {processed_name} Data")
    ax.legend(title="Genome", loc="upper right")

    if out_path_mouse or out_path_human:
        # Save once, with both distributions together
        save_path = out_path_mouse or out_path_human
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not show:
        plt.close()

def plot_joint_scatterplot(adata_raw, adata_processed, processed_name="processed", marginal_type="histogram", fill_histogram=True, marginal_color_number=4, bin_number=20, show_marginal_ticks=False, show_point_movement=False, max_points=None, seed=42, out_path=None, show=True):    
    if adata_processed is None:
        return  # nothing to plot
    
    if marginal_type not in ["histogram", "kde"]:
        raise ValueError("marginal_type must be either 'histogram' or 'kde'")

    adata_raw, adata_processed = take_adata_cell_gene_intersection(adata_raw, adata_processed)
    
    if max_points and adata_raw.n_obs > max_points:
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

    if marginal_color_number == 2:
        df = pd.DataFrame({
            "x": np.concatenate([human_raw, human_processed]),
            "y": np.concatenate([mouse_raw, mouse_processed]),
            "group": (["raw"] * len(human_raw)) + ([processed_name] * len(human_processed))
        })
        palette = {
            "raw": "#a6c8ff",         # light blue
            processed_name: "#0047b3",   # dark blue
        }
    elif marginal_color_number == 4:
        if "genome" not in adata_raw.obs.columns or "genome" not in adata_processed.obs.columns:
            raise ValueError("Both adata_raw and adata_processed must have 'genome' column in .obs for 4-color plotting.")
    
        genomes_raw = adata_raw.obs["genome"].values
        genomes_processed = adata_processed.obs["genome"].values
        genomes = np.concatenate([genomes_raw, genomes_processed])
        raw_processed = (["raw"] * len(human_raw) + ["processed"] * len(human_processed))
        
        df = pd.DataFrame({
            "x": np.concatenate([human_raw, human_processed]),
            "y": np.concatenate([mouse_raw, mouse_processed]),
            "group": [f"{'human' if g == 'hg19' else 'mouse'}_{rp}" for g, rp in zip(genomes, raw_processed)]
        })

        palette = {
            "human_raw": "#a6c8ff",         # light blue
            "human_processed": "#0047b3",   # dark blue
            "mouse_raw": "#ffd1a6",         # light orange
            "mouse_processed": "#cc5500",   # dark orange
        }

    else:
        raise ValueError("marginal_color_number must be either 2 or 4")

    # --- Create JointGrid ---
    g = sns.JointGrid(
        data=df,
        x="x", y="y",
        hue="group",
        palette=palette,
        marginal_ticks=show_marginal_ticks
    )
    
    #? apologies for the massive blocks
    if marginal_type == "histogram" and fill_histogram == False:
        # =============================
        #   Movement lines BELOW points
        # =============================
        if show_point_movement:
            for xr, yr, xp, yp in zip(human_raw, mouse_raw, human_processed, mouse_processed):
                g.ax_joint.plot([xr, xp], [yr, yp],
                                color="lightgray", alpha=0.25,
                                linewidth=0.4, zorder=0)

        # =============================
        #   Scatter (on top)
        # =============================
        sns.scatterplot(
            data=df,
            x="x", y="y",
            hue="group",
            palette=palette,
            ax=g.ax_joint,
            s=20,
            edgecolor="white",
            zorder=5
        )

        # =============================
        #   Marginals
        # =============================
        if marginal_type == "histogram":
            sns.histplot(
                data=df,
                x="x",
                hue="group",
                palette=palette,
                ax=g.ax_marg_x,
                bins=np.logspace(np.log10(df["x"].values.min()), np.log10(df["x"].values.max()), bin_number),
                fill=False,
                element="step",
                linewidth=1.2
            )
            sns.histplot(
                data=df,
                y="y",
                hue="group",
                palette=palette,
                ax=g.ax_marg_y,
                bins=np.logspace(np.log10(df["y"].values.min()), np.log10(df["y"].values.max()), bin_number),
                fill=False,
                element="step",
                linewidth=1.2
            )

        elif marginal_type == "kde":
            sns.kdeplot(
                data=df,
                x="x",
                hue="group",
                palette=palette,
                ax=g.ax_marg_x,
                fill=False,
                linewidth=1.5
            )
            sns.kdeplot(
                data=df,
                y="y",
                hue="group",
                palette=palette,
                ax=g.ax_marg_y,
                fill=False,
                linewidth=1.5
            )

        # =============================
        #  Remove marginal legends
        # =============================
        if g.ax_marg_x.get_legend():
            g.ax_marg_x.get_legend().remove()
        if g.ax_marg_y.get_legend():
            g.ax_marg_y.get_legend().remove()

        # =============================
        #  Final clean legend
        # =============================
        handles, labels = g.ax_joint.get_legend_handles_labels()
        g.ax_joint.legend(handles, labels, loc="lower left", title=processed_name)
        
        # =============================
        # Axes scaling + equal limits
        # =============================
        min_val = 0.8
        max_val = max(df["x"].max(), df["y"].max())
        g.ax_joint.set_xlim(min_val, max_val)
        g.ax_joint.set_ylim(min_val, max_val)
        g.ax_joint.set_aspect("equal", adjustable="box")

        # y = x line
        g.ax_joint.plot([min_val, max_val], [min_val, max_val],
                        color="gray", linestyle="--", linewidth=1)

        # log scale
        g.ax_joint.set_xscale("log")
        g.ax_joint.set_yscale("log")

        g.ax_joint.set_xlabel("Human counts + 1")
        g.ax_joint.set_ylabel("Mouse counts + 1")

        # ensure marginals have same height
        g.ax_marg_x.relim()
        g.ax_marg_x.autoscale_view()
        g.ax_marg_y.relim()
        g.ax_marg_y.autoscale_view()

        # Extract current heights
        x_max = g.ax_marg_x.get_ylim()[1]
        y_max = g.ax_marg_y.get_xlim()[1]

        # Use the global max so both marginals match visually
        max_height = max(x_max, y_max)

        # Apply uniform height
        g.ax_marg_x.set_ylim(0, max_height)
        g.ax_marg_y.set_xlim(0, max_height)
    else:  #? apologies for the massive blocks
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
        
        # enforce square plot
        min_val = 0.8
        max_val = max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])
        g.ax_joint.set_xlim(min_val, max_val)
        g.ax_joint.set_ylim(min_val, max_val)
        g.ax_joint.set_aspect('equal', adjustable='box')

        # plot y=x
        g.ax_joint.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1, zorder=0)

        # Main scatter axes log scale
        g.ax_joint.set_xscale("log")
        g.ax_joint.set_yscale("log")

        g.ax_joint.set_xlabel("Human counts + 1")
        g.ax_joint.set_ylabel("Mouse counts + 1")

        if marginal_type == "kde":
            g.plot(sns.scatterplot, sns.kdeplot, alpha=.7, linewidth=.5)
        elif marginal_type == "histogram":
            g.plot(sns.scatterplot, sns.histplot, alpha=.7, linewidth=.5)
        
        leg = g.ax_joint.legend(loc="lower left", title=processed_name)

        # ensure side histograms have same max height    
        x_hist_patches = g.ax_marg_x.patches
        y_hist_patches = g.ax_marg_y.patches

        max_height = 0
        if x_hist_patches:
            max_height = max(max_height, max(p.get_height() for p in x_hist_patches))
        if y_hist_patches:
            max_height = max(max_height, max(p.get_width() for p in y_hist_patches))  # note: width for y-hist

        max_height *= 1.05
        g.ax_marg_x.set_ylim(0, max_height)
        g.ax_marg_y.set_xlim(0, max_height)


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
    plot_type="cell",
    labels=None,
    title="Expression Correlation Distribution",
    colors=None,
    out_path=None,
    show=True,
    fill=False,
):
    """
    Compute and plot Pearson correlation curves (KDE)
    for multiple pairs of AnnData objects.

    plot_type:
        "cell" → row-wise correlation (per cell)
        "gene" → column-wise correlation (per gene)
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
        colors = sns.color_palette("tab10", n)

    if plot_type not in ("cell", "gene"):
        raise ValueError("plot_type must be 'cell' or 'gene'")

    # --- Collect correlations for each pair ---
    corr_sets = []

    for ad1, ad2 in zip(adata1_list, adata2_list):
        # match intersection
        ad1i, ad2i = take_adata_cell_gene_intersection(ad1, ad2)
        X1, X2 = ad1i.X, ad2i.X

        correlations = []

        if plot_type == "cell":
            # row-wise
            for i in range(X1.shape[0]):
                corr = sparse_row_pearson(X1[i, :], X2[i, :])
                correlations.append(corr)

        elif plot_type == "gene":
            # column-wise: convert to CSC for fast slicing
            X1c = X1.tocsc()
            X2c = X2.tocsc()

            for j in range(X1c.shape[1]):
                # slice column j → 1×G row vector
                corr = sparse_row_pearson(X1c[:, j].T, X2c[:, j].T)
                correlations.append(corr)

        corr_sets.append(np.array(correlations))

    # --- Plotting ---
    plt.figure(figsize=(8, 6))

    # Estimate maximum count for y-limits (for log scale)
    max_count = 0
    for values in corr_sets:
        hist_counts, _ = np.histogram(values, bins=50, range=(0, 1))
        max_count = max(max_count, hist_counts.max())

    y_max = 10 ** np.ceil(np.log10(max_count))

    # KDE curves
    for values, label, color in zip(corr_sets, labels, colors):
        sns.kdeplot(
            values,
            bw_adjust=1,
            fill=fill,
            color=color,
            label=label,
        )

    plt.xlim(0, 1)
    plt.ylim(1, y_max)
    plt.yscale("log")

    # axis label changes dynamically
    xlabel = "Cell Pearson Correlation" if plot_type == "cell" else "Gene Pearson Correlation"
    plt.xlabel(xlabel)

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
    plot_type="cell",
    colors=None,
    bins=100,
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
        if plot_type == "cell":
            sums = np.array(D.sum(axis=1)).ravel()
        elif plot_type == "gene":
            sums = np.array(D.sum(axis=0)).ravel()
        elif plot_type == "matrix":
            sums = np.array(D.sum()).ravel()
        diff_sets.append(sums)

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


def plot_iterative_difference_counts(
    adatas_dict,
    threshold=0.0,
    metric="cells",   # "cells" or "counts"
    expected_cells=None,
    colors=None,
    title="Difference per Iteration",
    out_path=None,
    show=True
):
    """
    Parameters
    ----------
    adatas_dict : dict
        key -> list of AnnData objects (iterations)

    threshold : float
        Used only when metric="cells"

    metric : {"cells", "counts"}
        "cells"  -> count rows where |row_sum| > threshold
        "counts" -> sum of absolute row differences

    Returns
    -------
    diff_results : dict
        key -> list of metric results for each adjacent iteration pair.
    """

    if metric not in ("cells", "counts", "number_of_cells"):
        raise ValueError('metric must be "cells", "counts", or "number_of_cells"')

    keys = list(adatas_dict.keys())

    # --- Colors ---
    if colors is None:
        palette = sns.color_palette("tab10", len(keys))
        colors = {k: c for k, c in zip(keys, palette)}

    diff_results = {}
    max_iter_count = 0

    plt.figure(figsize=(8, 6))

    # --- Compute metric for each method/key ---
    for key in keys:
        adata_list = adatas_dict[key]
        results = []

        for i in range(len(adata_list) - 1):
            A = adata_list[i]
            B = adata_list[i + 1]

            Ai, Bi = take_adata_cell_gene_intersection(A, B)
            X_A = Ai.X
            X_B = Bi.X

            if not sparse.issparse(X_A) or not sparse.issparse(X_B):
                raise ValueError("AnnData.X must be sparse.")

            D = X_A - X_B
            row_sums = np.array(D.sum(axis=1)).ravel()

            if metric == "cells":
                # Count cells whose |difference| exceeds threshold
                result = int(np.sum(np.abs(row_sums) > threshold))

            elif metric == "counts":
                # Total absolute difference
                result = float(np.sum(np.abs(row_sums)))
            
            elif metric == "number_of_cells":
                result = X_A.shape[0]

            results.append(result)

        diff_results[key] = results

        # --- Plotting ---
        x = np.arange(len(results))
        max_iter_count = max(max_iter_count, len(results))

        plt.plot(
            x, results,
            marker='o',
            color=colors[key],
            label=key
        )

    # --- Integer ticks ---
    plt.xticks(
        ticks=np.arange(max_iter_count),
        labels=[str(i) for i in range(max_iter_count)],
        fontsize=11
    )

    # --- Labels ---
    if metric == "cells":
        ylabel = f"# Cells With |Difference| > {threshold}"
    elif metric == "counts":
        ylabel = "Total Absolute Row-Sum Difference"
    elif metric == "number_of_cells":
        ylabel = "Total Number of Cells"
    
    if expected_cells is not None and metric in ("cells", "number_of_cells"):
        # plot horizontal line at expected_cells
        plt.axhline(y=expected_cells, color='gray', linestyle='--')
        plt.text(x=(max_iter_count-1), y=expected_cells - 0.03*(plt.ylim()[1] - plt.ylim()[0]), s=f'Expected cells: {expected_cells}', fontsize=10, color='gray', ha='right')

    plt.xlabel("Iteration Comparison (i → i+1)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(bottom=0)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)

    if not show:
        plt.close()
    else:
        plt.show()

    return diff_results


def detect_doublets_human_mouse(adata_raw, fraction_doublet=0.15, plot_empty=False, umi_cutoff=None, expected_cells=None, out_path=None, show=True):
    if "genome" not in adata_raw.var.columns:
        raise ValueError("The adata_raw.var must contain a 'genome' column indicating the genome for each gene.")
    if not set(adata_raw.var['genome']).issuperset({'hg19', 'mm10'}):
        raise ValueError("The adata_raw.var['genome'] column must contain both 'hg19' and 'mm10' values.")
    
    adata_raw_original = adata_raw.copy()
    if not plot_empty and "is_empty" not in adata_raw.obs.columns:
        adata_raw = infer_empty_droplets(adata_raw, method="threshold", umi_cutoff=umi_cutoff, expected_cells=expected_cells)  # adds adata.obs["is_empty"]
        adata_raw = adata_raw[~adata_raw.obs["is_empty"]].copy()
    
    if "hg19_total_counts" not in adata_raw.obs.columns:
        adata_raw.obs['hg19_total_counts'] = np.asarray(adata_raw[:, adata_raw.var['genome'] == 'hg19'].X.sum(axis=1)).ravel()
    if "mm10_total_counts" not in adata_raw.obs.columns:
        adata_raw.obs['mm10_total_counts'] = np.asarray(adata_raw[:, adata_raw.var['genome'] == 'mm10'].X.sum(axis=1)).ravel()

    if "total_counts" not in adata_raw.obs.columns:
        adata_raw.obs["total_counts"] = adata_raw.obs["hg19_total_counts"] + adata_raw.obs["mm10_total_counts"]
    if "frac_human" not in adata_raw.obs.columns:
        adata_raw.obs["frac_human"] = adata_raw.obs["hg19_total_counts"] / adata_raw.obs["total_counts"]
    if "frac_mouse" not in adata_raw.obs.columns:
        adata_raw.obs["frac_mouse"] = 1 - adata_raw.obs["frac_human"]
    
    adata_raw.obs["is_doublet"] = (adata_raw.obs["frac_human"] > fraction_doublet) & (adata_raw.obs["frac_human"] < (1 - fraction_doublet))
    # adata_raw = adata_raw[~adata_raw.obs["is_doublet"]].copy()
    sns.scatterplot(
        x=adata_raw.obs["hg19_total_counts"] + 1,
        y=adata_raw.obs["mm10_total_counts"] + 1,
        hue=adata_raw.obs["is_doublet"]
    )
    plt.xscale("log")
    plt.yscale("log")
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=300)

    if not show:
        plt.close()
    else:
        plt.show()

    if not plot_empty:
        adata_raw_original.obs["is_doublet"] = (
            adata_raw.obs["is_doublet"]
            .reindex(adata_raw_original.obs_names)   # align indices
            .fillna(False)                           # missing → False
        )
    else:
        adata_raw_original = adata_raw.copy()

    return adata_raw_original


def evaluate_simulation_denoising(adata_processed, adata_raw):
    pass