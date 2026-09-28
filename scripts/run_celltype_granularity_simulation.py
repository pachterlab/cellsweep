#!/usr/bin/env python
"""
Ground-truth companion to scripts/run_celltype_granularity_sensitivity.py.

Uses simulation1_small_noise (12 true cell types, known real/noise layers) and runs cellsweep
with labels that are coarser or finer than the truth:

    merge_K      K in {1, 2, 3, 6}: true types pooled into K groups (heterogeneous labels)
    truth        K=12
    split_m      each true type randomly split into m in {2, 4, 8} labels (over-clustering,
                 without biological signal)
    leiden_<r>   Leiden on the raw cells (same scanpy pipeline as the notebook), r in {1, 5, 20}
    shuffled     truth labels permuted across cells

Because the simulation records the true ambient/bulk noise per entry, we can score removal
exactly: noise recall, removal precision, real-count retention, off-target marker removal, and
per-cell contamination-fraction accuracy.

Usage: python scripts/run_celltype_granularity_simulation.py [--overwrite] [--jobs N]
Output: notebooks/output/simulation1_small_noise/celltype_granularity_simulation_metrics.csv
"""
import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from scipy.stats import spearmanr

import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
dataset_name = "simulation1_small_noise"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", dataset_name)
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", dataset_name)
run_dir = os.path.join(data_dir, "celltype_granularity")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

raw_path = os.path.join(data_dir, "adata_raw.h5ad")
labels_csv = os.path.join(run_dir, "labels.csv")
metrics_csv = os.path.join(out_dir, "celltype_granularity_simulation_metrics.csv")

expected_cells = 10000
cellsweep_max_iter = 2000


def make_labels(overwrite=False):
    if os.path.exists(labels_csv) and not overwrite:
        return pd.read_csv(labels_csv, index_col=0, dtype=str)
    adata = ad.read_h5ad(raw_path)
    cells = adata[~adata.obs["is_empty"]].copy()
    truth = cells.obs["celltype"].astype(str)
    type_idx = truth.str.replace("Type_", "").astype(int).values
    rng = np.random.default_rng(0)

    labels = pd.DataFrame(index=cells.obs_names)
    # pool true types into K groups with a fixed random assignment
    perm = rng.permutation(12)
    for K in [1, 2, 3, 6]:
        group_of_type = {t: int(np.where(perm == t)[0][0]) % K for t in range(12)}
        labels[f"merge_{K}"] = [f"G{group_of_type[t]}" for t in type_idx]
    labels["truth"] = truth.values
    for m in [2, 4, 8]:
        labels[f"split_{m}"] = [f"{t}_{s}" for t, s in zip(truth.values, rng.integers(0, m, size=len(truth)))]
    for r in [1.0, 5.0, 20.0]:
        tmp = cs_utils.run_scanpy_preprocessing_and_clustering(cells, min_genes=0, min_cells=0, max_mt_percentage=None, n_top_genes=2000, n_pcs=25, n_neighbors=20, leiden_resolution=r, seed=42, verbose=0)
        labels[f"leiden_{r:g}"] = tmp.obs["leiden"].astype(str).reindex(labels.index)
    labels["shuffled"] = rng.permutation(truth.values)
    assert not labels.isna().any().any()
    labels.to_csv(labels_csv)
    return labels


def run_one(condition, threads, overwrite=False):
    out = os.path.join(run_dir, f"adata_cellsweep_{condition}.h5ad")
    if os.path.exists(out) and not overwrite:
        return condition, "exists"
    from cellsweep import denoise_count_matrix
    labels = pd.read_csv(labels_csv, index_col=0, dtype=str)
    adata = ad.read_h5ad(raw_path)
    adata.obs["celltype"] = pd.Categorical(labels[condition].reindex(adata.obs_names).fillna("Empty Droplet"))
    denoise_count_matrix(adata, adata_out=out, init_alpha=0.9, init_beta=0.1, freeze_ambient_profile=True, max_iter=cellsweep_max_iter, empty_droplet_method="threshold", expected_cells=expected_cells, threads=threads, verbose=1, log_file=os.path.join(run_dir, f"cellsweep_{condition}.log"))
    return condition, "done"


def score(condition, labels):
    a = ad.read_h5ad(os.path.join(run_dir, f"adata_cellsweep_{condition}.h5ad"))
    a = a[~a.obs["is_empty"]].copy()
    raw = sp.csr_matrix(a.layers["raw"]).astype(np.float64)
    den = sp.csr_matrix(a.X).astype(np.float64)
    real = sp.csr_matrix(a.layers["real"]).astype(np.float64)
    noise = sp.csr_matrix(a.layers["noise"]).astype(np.float64)
    removed = raw - den
    removed.data = np.clip(removed.data, 0, None)

    correctly_removed = removed.minimum(noise).sum()
    real_retained = den.minimum(real).sum()

    # per-entry error on entries that are nonzero in raw
    err = (den - real)
    # per-cell contamination fraction
    raw_tot = np.asarray(raw.sum(1)).ravel()
    est_frac = np.asarray(removed.sum(1)).ravel() / raw_tot
    true_frac = np.asarray(noise.sum(1)).ravel() / raw_tot

    # off-target markers: markers of type t counted in cells whose TRUE type is not t
    marker_sets = a.uns["marker_sets"]
    truth = labels["truth"].reindex(a.obs_names).values
    off_raw = off_rem = on_raw = on_ret = 0.0
    raw_c, den_c = raw.tocsc(), den.tocsc()
    for t, genes in enumerate(marker_sets):
        genes = np.asarray(genes).astype(int)
        on_mask = truth == f"Type_{t}"
        r = raw_c[:, genes]
        d = den_c[:, genes]
        off_raw += r[~on_mask].sum()
        off_rem += (r[~on_mask] - d[~on_mask]).sum()
        on_raw += r[on_mask].sum()
        on_ret += d[on_mask].sum()

    return dict(
        condition=condition,
        n_labels=labels[condition].nunique(),
        noise_recall=correctly_removed / noise.sum(),
        removal_precision=correctly_removed / removed.sum(),
        real_retained=real_retained / real.sum(),
        fraction_removed=removed.sum() / raw.sum(),
        true_fraction_noise=noise.sum() / raw.sum(),
        entry_rmse=np.sqrt(err.multiply(err).sum() / raw.nnz),
        cell_frac_spearman=spearmanr(est_frac, true_frac).correlation,
        cell_frac_mae=np.mean(np.abs(est_frac - true_frac)),
        offtarget_marker_removed=off_rem / off_raw,
        ontarget_marker_retained=on_ret / on_raw,
        mean_cell_size_per_label=len(labels) / labels[condition].nunique(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    labels = make_labels(overwrite=args.overwrite)
    print(labels.nunique().to_string(), flush=True)
    conditions = list(labels.columns)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=mp.get_context("spawn")) as ex:  # numba/OpenMP is not fork-safe
        for cond, status in ex.map(run_one, conditions, [args.threads] * len(conditions), [args.overwrite] * len(conditions)):
            print(cond, status, flush=True)

    rows = [score(c, labels) for c in conditions]
    df = pd.DataFrame(rows)
    df.to_csv(metrics_csv, index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
