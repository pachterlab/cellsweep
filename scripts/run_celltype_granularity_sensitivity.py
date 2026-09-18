#!/usr/bin/env python
"""
cellsweep sensitivity to the granularity of the cell-type labels (pbmc8k).

Extends the Leiden-resolution sweep in notebooks/benchmarking.ipynb
("## How sensitive are we to celltype method?") to span the full label hierarchy, from a
single label for every cell up to very fine over-clustering:

    all_one      K=1   every cell shares one label (the degenerate limit)
    lineage      K=2   lymphoid vs myeloid (CellTypist Immune_All_High collapsed)
    ct_high      K=7   CellTypist Immune_All_High (the default used in the manuscript)
    ct_low       K~20  CellTypist Immune_All_Low
    leiden_<r>         Leiden at r in {0.001, 0.005, 0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20}, i.e. K=1..190
    shuffled     K=7   ct_high labels randomly permuted across cells (negative control)

Step 1 writes every labeling to <data_dir>/celltype_granularity/labels.csv.
Step 2 runs cellsweep once per labeling, with identical settings to the default pbmc8k run.
All runs share the same raw matrix and empty-droplet calls (taken from the default output).

Usage: python scripts/run_celltype_granularity_sensitivity.py [--overwrite] [--jobs N]
Metrics/figures: scripts/analyze_celltype_granularity_sensitivity.py
"""
import os
import sys
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import anndata as ad

import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
dataset_name = "pbmc8k"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", dataset_name)
run_dir = os.path.join(data_dir, "celltype_granularity")
os.makedirs(run_dir, exist_ok=True)

default_cellsweep_path = os.path.join(data_dir, "pbmc8k_output_cellsweep.h5ad")
labels_csv = os.path.join(run_dir, "labels.csv")

# params from notebooks/config/pbmc8k.yaml
min_genes = 0
min_cells = 0
max_mt_percentage = None
n_top_genes = 2000
n_pcs = 25
n_neighbors = 20
expected_cells = 8381
cellsweep_max_iter = 2000
cellsweep_init_alpha = 0.9
cellsweep_init_beta = 0.1

existing_leiden_resolutions = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
new_leiden_resolutions = [0.001, 0.005, 0.01, 10.0, 20.0]  # 0.001 -> K=1, 0.005 -> K=2, 0.01 -> K=3

LINEAGE = {
    "T cells": "Lymphoid", "ILC": "Lymphoid", "B cells": "Lymphoid",
    "Monocytes": "Myeloid", "DC": "Myeloid", "pDC": "Myeloid", "HSC/MPP": "Myeloid",
}


def res_tag(r):
    return str(float(r)).replace(".", "_")


def load_raw():
    """Raw counts + empty-droplet calls + CellTypist-High labels, from the default run."""
    adata = ad.read_h5ad(default_cellsweep_path)
    adata.X = adata.layers["raw"].copy()
    del adata.layers["raw"]
    adata.obs = adata.obs[["is_empty", "celltype"]].copy()
    for key in list(adata.uns.keys()):
        del adata.uns[key]
    adata.var = adata.var[["gene_ids", "empty_counts"]].copy()
    adata.var_names_make_unique()
    return adata


def make_labels(overwrite=False):
    if os.path.exists(labels_csv) and not overwrite:
        return pd.read_csv(labels_csv, index_col=0, dtype=str)

    adata = load_raw()
    cells = adata[~adata.obs["is_empty"]].copy()
    labels = pd.DataFrame(index=cells.obs_names)

    labels["ct_high"] = cells.obs["celltype"].astype(str)
    labels["all_one"] = "cell"
    labels["lineage"] = labels["ct_high"].map(LINEAGE)
    assert labels["lineage"].notna().all(), labels.loc[labels["lineage"].isna(), "ct_high"].unique()

    rng = np.random.default_rng(0)
    labels["shuffled"] = rng.permutation(labels["ct_high"].values)

    # CellTypist Immune_All_Low (same call the notebook uses for High)
    ct = cs_utils.determine_cell_types(adata, model_pkl="Immune_All_Low.pkl", filter_empty=True, expected_cells=expected_cells, verbose=1)
    labels["ct_low"] = ct.obs["celltype"].astype(str).reindex(labels.index)

    # Leiden labels already used for the notebook sweep
    for r in existing_leiden_resolutions:
        path = os.path.join(data_dir, f"adata_cellsweep_leiden_{res_tag(r)}.h5ad")
        obs = ad.read_h5ad(path, backed="r").obs
        labels[f"leiden_{r:g}"] = obs["celltype"].astype(str).reindex(labels.index)

    # finer Leiden resolutions, same pipeline as the notebook
    for r in new_leiden_resolutions:
        tmp = cs_utils.run_scanpy_preprocessing_and_clustering(cells, min_genes=min_genes, min_cells=min_cells, max_mt_percentage=max_mt_percentage, n_top_genes=n_top_genes, n_pcs=n_pcs, n_neighbors=n_neighbors, leiden_resolution=r, seed=42, verbose=1)
        labels[f"leiden_{r:g}"] = tmp.obs["leiden"].astype(str).reindex(labels.index)

    labels = labels[[c for c in labels.columns if not c.startswith("leiden")] + sorted([c for c in labels.columns if c.startswith("leiden")], key=lambda c: float(c.split("_")[1]))]
    assert not labels.isna().any().any(), labels.isna().sum()
    labels.to_csv(labels_csv)
    return labels


def run_one(condition, threads, overwrite=False):
    out = os.path.join(run_dir, f"adata_cellsweep_{condition}.h5ad")
    if os.path.exists(out) and not overwrite:
        return condition, "exists"
    labels = pd.read_csv(labels_csv, index_col=0, dtype=str)
    adata = load_raw()
    adata.obs["celltype"] = pd.Categorical(labels[condition].reindex(adata.obs_names).fillna("Empty Droplet"))
    cs_utils_log = os.path.join(run_dir, f"cellsweep_{condition}.log")
    from cellsweep import denoise_count_matrix
    denoise_count_matrix(adata, adata_out=out, init_alpha=cellsweep_init_alpha, init_beta=cellsweep_init_beta, freeze_ambient_profile=True, max_iter=cellsweep_max_iter, empty_droplet_method="threshold", expected_cells=expected_cells, threads=threads, verbose=1, log_file=cs_utils_log)
    return condition, "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    labels = make_labels(overwrite=args.overwrite)
    print(labels.nunique().to_string())

    conditions = list(labels.columns)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:  # numba/OpenMP is not fork-safe
        for cond, status in ex.map(run_one, conditions, [args.threads] * len(conditions), [args.overwrite] * len(conditions)):
            print(cond, status, flush=True)
