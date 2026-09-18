#!/usr/bin/env python
"""
Reviewer point 7: cellsweep on a standard (non-xenograft) Visium HD tissue.

Dataset: 10x Genomics Visium HD 3' Mouse Brain (fresh frozen), Space Ranger 4.0.1, 8 um bins
https://www.10xgenomics.com/datasets/visium-hd-three-prime-mouse-brain-fresh-frozen

Mirrors the Visium HD pipeline in notebooks/spatial.ipynb: bins below the knee-plot UMI cutoff or outside
the tissue are empty, Leiden (resolution 1.0) on non-empty bins gives cell-type labels, and cellsweep is run with
default settings.

Usage: python scripts/run_spatial_mouse_brain.py [--overwrite]
Analysis/figures: scripts/analyze_spatial_mouse_brain_markers.py
"""
import os
import argparse
import subprocess

import numpy as np
import pandas as pd

from cellsweep import denoise_count_matrix
import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "visium_mouse_brain")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "visium_mouse_brain")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

base_url = "https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_3prime_Mouse_Brain/Visium_HD_3prime_Mouse_Brain"
resolution = "008um"
expected_cells = 350_000  # ~93% of the 376,419 in-tissue bins; the brain section is contiguous tissue
leiden_resolution = 1.0
threads = 32
verbose = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in ["binned_outputs", "segmented_outputs"]:
        if not os.path.exists(os.path.join(data_dir, name)):
            subprocess.run(["wget", "-O", f"{data_dir}/{name}.tar.gz", f"{base_url}_{name}.tar.gz"], check=True)
            subprocess.run(["tar", "-xzf", f"{data_dir}/{name}.tar.gz", "-C", data_dir], check=True)

    adata_path_cellsweep = os.path.join(data_dir, "adata_cellsweep.h5ad")
    if os.path.exists(adata_path_cellsweep) and not args.overwrite:
        print(f"{adata_path_cellsweep} exists, skipping")
        return

    matrix_dir_raw = os.path.join(data_dir, "binned_outputs", f"square_{resolution}")
    adata_raw = cs_utils.load_adata(os.path.join(matrix_dir_raw, "raw_feature_bc_matrix.h5"))
    adata_raw.var_names_make_unique()
    spatial_df = pd.read_parquet(os.path.join(matrix_dir_raw, "spatial", "tissue_positions.parquet"))
    adata_raw.obs = adata_raw.obs.merge(spatial_df.set_index("barcode"), left_index=True, right_index=True, how="left")
    adata_raw.obsm["spatial"] = adata_raw.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()

    umi_cutoff = cs_utils.knee_plot(adata_raw, transpose=True, expected_cells=expected_cells, out_path=os.path.join(out_dir, "knee_plot.png"), show=False)
    adata_raw = cs_utils.infer_empty_droplets(adata_raw, method="threshold", umi_cutoff=umi_cutoff, verbose=verbose)
    adata_raw.obs["is_empty"] = adata_raw.obs["is_empty"] | (~adata_raw.obs["in_tissue"].fillna(False).astype(bool))
    print(f"non-empty bins: {(~adata_raw.obs['is_empty']).sum()}")

    adata_processed_tmp = adata_raw[~adata_raw.obs["is_empty"]].copy()
    adata_processed_tmp = cs_utils.run_scanpy_preprocessing_and_clustering(adata_processed_tmp, min_genes=None, min_cells=None, max_mt_percentage=None, n_top_genes=2000, n_pcs=50, n_neighbors=15, leiden_resolution=leiden_resolution, seed=42, verbose=verbose)
    adata_raw.obs["celltype"] = adata_processed_tmp.obs["leiden"].reindex(adata_raw.obs.index).astype(str).replace("nan", "empty").astype("category")

    denoise_count_matrix(adata_raw, adata_out=adata_path_cellsweep, freeze_ambient_profile=True, empty_droplet_method="threshold", threads=threads, verbose=verbose, log_file=os.path.join(data_dir, "cellsweep.log"))


if __name__ == "__main__":
    main()
