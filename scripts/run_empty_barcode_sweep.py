#!/usr/bin/env python
"""
cellsweep sensitivity to the number of empty barcodes used to estimate the ambient profile
(pbmc8k). Companion to scripts/run_celltype_granularity_sensitivity.py: same cells, same
CellTypist Immune_All_High labels, same settings; only the number of empty droplets changes.

This mirrors the notebook cell "## How many empty droplets do we need?" but writes its runs
next to the label-granularity runs so both sweeps are scored by the same code and the same
cellsweep version.

Usage: python scripts/run_empty_barcode_sweep.py [--overwrite] [--jobs N]
"""
import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from run_celltype_granularity_sensitivity import load_raw, labels_csv, data_dir, expected_cells, cellsweep_max_iter, cellsweep_init_alpha, cellsweep_init_beta

run_dir = os.path.join(data_dir, "empty_barcode_sweep")
os.makedirs(run_dir, exist_ok=True)

EMPTY_BARCODE_NUMBERS = [10, 100, 1_000, 10_000, 50_000, 100_000, 500_000, None]  # None = all of them
SEED = 42


def run_one(n_empty, threads, overwrite=False):
    tag = "all" if n_empty is None else str(n_empty)
    out = os.path.join(run_dir, f"adata_cellsweep_empty_{tag}.h5ad")
    if os.path.exists(out) and not overwrite:
        return tag, "exists"
    from cellsweep import denoise_count_matrix
    labels = pd.read_csv(labels_csv, index_col=0, dtype=str)
    adata = load_raw()
    adata.obs["celltype"] = pd.Categorical(labels["ct_high"].reindex(adata.obs_names).fillna("Empty Droplet"))
    if n_empty is not None:
        empty = adata.obs_names[adata.obs["is_empty"].values]
        rng = np.random.default_rng(SEED)
        keep = set(rng.choice(empty, size=n_empty, replace=False))
        adata = adata[~adata.obs["is_empty"].values | adata.obs_names.isin(keep)].copy()
    denoise_count_matrix(adata, adata_out=out, init_alpha=cellsweep_init_alpha, init_beta=cellsweep_init_beta, freeze_ambient_profile=True, max_iter=cellsweep_max_iter, empty_droplet_method="threshold", expected_cells=expected_cells, threads=threads, verbose=1, log_file=os.path.join(run_dir, f"cellsweep_empty_{tag}.log"))
    return tag, "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:  # numba/OpenMP is not fork-safe
        for tag, status in ex.map(run_one, EMPTY_BARCODE_NUMBERS, [args.threads] * len(EMPTY_BARCODE_NUMBERS), [args.overwrite] * len(EMPTY_BARCODE_NUMBERS)):
            print(tag, status, flush=True)
