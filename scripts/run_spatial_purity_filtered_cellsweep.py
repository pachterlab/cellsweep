#!/usr/bin/env python
"""
Reviewer point 7 (spatial xenograft): run cellsweep with bins that may hold two species excluded up front.

An 8 um Visium HD bin at the tumor-host interface can genuinely contain cells of both species, which the model
does not represent (the spatial analog of a cross-species doublet in the single-cell experiment). Rather than
correcting such bins and filtering them afterwards, this run excludes them up front:

    --rule purity     (default) exclude non-empty bins with < PURITY of counts from one species
    --rule interface            exclude non-empty bins within INTERFACE_UM of the tumor-host interface

Empty bins are untouched under either rule and still define the ambient profile.

The two rules are not interchangeable. Because the ambient pool of this sample is 88% human, a bin's
cross-species fraction is essentially its contamination estimate (Spearman(purity, alpha_hat) = -0.85), so
`purity` removes the most contaminated bins whether or not they could contain two species: 54% of the bins it
excludes lie >48 um from the interface, and their mouse counts match the mouse ambient profile (r = 0.93) rather
than carrying mouse stromal markers. `interface` instead excludes only bins within about one cell diameter of the
compartment boundary, where genuine two-species content is physically possible, and is independent of the
per-bin species fraction. Use `interface` for spatial claims about where contamination sits, and `purity` as a
sensitivity analysis.

Cell-type labels are Leiden run separately within the human-pure and the mouse-pure bins, so that each species
gets its own profile instead of being merged into mixed clusters (see scripts/run_spatial_species_aware_cellsweep.py).
Purity uses the raw-count species fraction; the content-normalized fraction is stored alongside for reference.

Usage: python scripts/run_spatial_purity_filtered_cellsweep.py [--overwrite]
Figures: scripts/make_spatial_species_run_figures.py
"""
import os
import argparse

import numpy as np
import pandas as pd
import anndata as ad

from cellsweep import denoise_count_matrix
import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "visium_human_mouse")
run_dir = os.path.join(data_dir, "purity_filtered")
os.makedirs(run_dir, exist_ok=True)

PURITY = 0.9
INTERFACE_UM = 16   # ~one cell diameter; distances from scripts/analyze_spatial_xenograft_mixing.py
MIN_BINS_FOR_LEIDEN = 1000
leiden_resolution = 1.0
threads = 32
verbose = 1
OUT_PATHS = {"purity": os.path.join(run_dir, f"adata_cellsweep_pure{int(PURITY * 100)}.h5ad"),
             "interface": os.path.join(run_dir, f"adata_cellsweep_interface{INTERFACE_UM}.h5ad")}


def leiden_labels(adata, prefix):
    # the mouse-pure set is small (the host capsule is thin and every mouse bin carries tumor-derived counts);
    # below MIN_BINS_FOR_LEIDEN, sub-clustering is not stable (seurat_v3 HVG fails), so the species gets one profile
    if adata.n_obs < MIN_BINS_FOR_LEIDEN:
        print(f"{prefix}: {adata.n_obs} bins < {MIN_BINS_FOR_LEIDEN}, using a single cell-type label")
        return pd.Series(f"{prefix}_0", index=adata.obs_names)
    tmp = cs_utils.run_scanpy_preprocessing_and_clustering(adata, min_genes=None, min_cells=None, max_mt_percentage=None, n_top_genes=2000, n_pcs=50, n_neighbors=15, leiden_resolution=leiden_resolution, seed=42, verbose=0)
    return (prefix + "_" + tmp.obs["leiden"].astype(str)).reindex(adata.obs_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rule", choices=["purity", "interface"], default="purity")
    args = parser.parse_args()
    out_path = OUT_PATHS[args.rule]
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists, skipping")
        return

    original = ad.read_h5ad(os.path.join(data_dir, "adata_cellsweep.h5ad"))
    adata = ad.AnnData(X=original.layers["raw"].tocsr(), obs=original.obs[["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres", "is_empty"]].copy(), var=original.var[["gene_ids", "feature_types", "genome"]].copy())
    adata.obsm["spatial"] = original.obsm["spatial"]
    del original

    is_human = (adata.var["genome"] == "GRCh38").values
    X = adata.X.tocsc()
    h = np.asarray(X[:, is_human].sum(axis=1)).ravel()
    m = np.asarray(X[:, ~is_human].sum(axis=1)).ravel()
    is_empty = adata.obs["is_empty"].astype(bool).values
    fh = h / np.maximum(h + m, 1)
    s_h, s_m = np.median((h + m)[~is_empty & (fh >= PURITY)]), np.median((h + m)[~is_empty & (fh <= 1 - PURITY)])
    adata.obs["human_counts_total_"], adata.obs["mouse_counts_total_"] = h, m
    adata.obs["frac_human"] = fh
    adata.obs["frac_human_content_normalized"] = (h / s_h) / np.maximum(h / s_h + m / s_m, 1e-12)
    adata.obs["purity"] = np.maximum(fh, 1 - fh)
    adata.obs["genome"] = np.where(fh >= 0.5, "human", "mouse")

    if args.rule == "purity":
        keep = (~is_empty) & (adata.obs["purity"].values >= PURITY)
    else:
        d_iface = pd.read_parquet(os.path.join(data_dir, "reviewer7_cache", "bins.parquet"))["d_iface"].reindex(adata.obs_names)
        adata.obs["d_iface"] = d_iface.values
        keep = (~is_empty) & (d_iface.values > INTERFACE_UM)
    excluded = (~is_empty) & ~keep
    print(f"rule={args.rule}; non-empty bins: {(~is_empty).sum():,}; corrected: {keep.sum():,} "
          f"({(adata.obs['genome'].values[keep] == 'mouse').sum():,} mouse-majority); excluded: {excluded.sum():,}")

    adata = adata[is_empty | keep].copy()   # empties kept: they define the ambient profile
    corrected = adata[~adata.obs["is_empty"].astype(bool)]
    is_human_bin = (corrected.obs["genome"] == "human").values
    labels = pd.concat([leiden_labels(corrected[is_human_bin], "human"), leiden_labels(corrected[~is_human_bin], "mouse")])
    adata.obs["celltype"] = labels.reindex(adata.obs_names).fillna("empty").astype("category")
    print("cell-type labels:", labels.str.split("_").str[0].value_counts().to_dict(), "K =", labels.nunique())

    denoise_count_matrix(adata, adata_out=out_path, freeze_ambient_profile=True, empty_droplet_method="threshold", threads=threads, verbose=verbose, log_file=os.path.join(run_dir, os.path.basename(out_path).replace("adata_cellsweep_", "cellsweep_").replace(".h5ad", ".log")))


if __name__ == "__main__":
    main()
