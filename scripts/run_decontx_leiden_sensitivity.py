#!/usr/bin/env python
"""
DecontX sensitivity to the clustering/cell-typing method.

This mirrors the cellsweep Leiden-resolution sensitivity analysis in
notebooks/benchmarking.ipynb (the cell after
"## How sensitive are we to celltype method? Let's compare to Leiden clustering"),
but for DecontX instead of cellsweep.

For each Leiden resolution we:
  1. Cluster the 10X *filtered* cells with the exact same scanpy pipeline the notebook
     uses (cs_utils.run_scanpy_preprocessing_and_clustering).
  2. Run DecontX with those clusters forced in as z (via scripts/run_decontx.R
     --clusters_csv), instead of DecontX's native internal clustering.
  3. Compare the corrected matrix against the *default* DecontX run
     (pbmc8k_output_decontx.*, which used DecontX's own clustering) and against the
     other resolutions.

It then computes the matching metrics for the cellsweep Leiden runs that already exist
on disk (adata_cellsweep_leiden_*.h5ad vs pbmc8k_output_cellsweep.h5ad) so the two
tools' sensitivity can be compared head-to-head. Results are written to
output/pbmc8k/decontx_vs_cellsweep_leiden_sensitivity.csv and scatterplots are saved
alongside.
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

import cellsweep.utils as cs_utils

# ----------------------------------------------------------------------------------
# Config (mirrors notebooks/benchmarking.ipynb for pbmc8k)
# ----------------------------------------------------------------------------------
cellsweep_dir = "/home/jrich/Desktop/cellsweep"
rver_docker_workspace = "/home/ruser/work/cellsweep"
docker = "podman"
dataset_name = "pbmc8k"

data_dir = os.path.join(cellsweep_dir, "notebooks", "data", dataset_name)
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", dataset_name)
os.makedirs(out_dir, exist_ok=True)

# scanpy clustering params from notebooks/config/pbmc8k.yaml
min_genes = 0
min_cells = 0
max_mt_percentage = None
n_top_genes = 2000
n_pcs = 25
n_neighbors = 20
seed = 42
sequencing_technology = "10XV2"
verbose = 1

leiden_resolutions = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]

# DecontX inputs: the genome-prefixed matrix dirs used for the default run (output barcodes
# carry the "GRCh38_" prefix -> importCellRangerV2Sample read the dir containing the GRCh38
# genome subfolder).
matrix_tar_files_dir = os.path.join(data_dir, "matrix_tar_files")
# importCellRangerV2Sample expects the genome-level dir that directly contains
# barcodes.tsv/genes.tsv/matrix.mtx (it prepends the "GRCh38" dir name to barcodes, which is
# why the default decontx output carries the "GRCh38_" prefix).
raw_tar_file_dir = os.path.join(matrix_tar_files_dir, "raw_gene_bc_matrices", "GRCh38")
filtered_tar_file_dir = os.path.join(matrix_tar_files_dir, "filtered_gene_bc_matrices", "GRCh38")
filtered_genome_dir = filtered_tar_file_dir  # clean barcodes for clustering

decontx_default_prefix = os.path.join(data_dir, "pbmc8k_output_decontx")
cellsweep_default_path = os.path.join(data_dir, "pbmc8k_output_cellsweep.h5ad")

overwrite = "--overwrite" in sys.argv


def in_container(path):
    return path.replace(cellsweep_dir, rver_docker_workspace)


# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def strip_genome_prefix(names, prefix="GRCh38_"):
    return [n.replace(prefix, "", 1) for n in names]


def load_decontx(prefix):
    adata = cs_utils.load_adata(prefix, verbose=0)
    adata.obs_names = strip_genome_prefix(list(adata.obs_names))
    adata.var_names_make_unique()
    return adata


def align(adata_a, adata_b):
    """Return X matrices restricted to the shared cells & genes, in the same order."""
    cells = adata_a.obs_names.intersection(adata_b.obs_names)
    genes = adata_a.var_names.intersection(adata_b.var_names)
    A = adata_a[cells, genes]
    B = adata_b[cells, genes]
    Xa = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    Xb = B.X.tocsr() if sp.issparse(B.X) else sp.csr_matrix(B.X)
    return Xa, Xb, len(cells), len(genes)


def safe_corr(a, b):
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(Xbase, Xrun):
    """Metrics describing how far a run is from the baseline corrected matrix."""
    # entry-level over the baseline's nonzero support (mirrors plot_matrix_scatterplot)
    base = Xbase.tocsr()
    rows = np.repeat(np.arange(base.shape[0]), np.diff(base.indptr))
    cols = base.indices
    xv = base.data
    yv = np.asarray(Xrun.tocsr()[rows, cols]).ravel()
    entry_corr = safe_corr(xv, yv)

    # per-cell and per-gene total counts
    cell_base = np.asarray(Xbase.sum(axis=1)).ravel()
    cell_run = np.asarray(Xrun.sum(axis=1)).ravel()
    gene_base = np.asarray(Xbase.sum(axis=0)).ravel()
    gene_run = np.asarray(Xrun.sum(axis=0)).ravel()

    return {
        "entry_pearson_vs_default": entry_corr,
        "cell_total_pearson_vs_default": safe_corr(cell_base, cell_run),
        "gene_total_pearson_vs_default": safe_corr(gene_base, gene_run),
        "total_counts": float(Xrun.sum()),
        "total_counts_default": float(Xbase.sum()),
        "total_counts_ratio_vs_default": float(Xrun.sum() / Xbase.sum()),
        "mean_abs_cell_total_diff": float(np.mean(np.abs(cell_run - cell_base))),
    }


# ----------------------------------------------------------------------------------
# Step 1: cluster filtered cells once per resolution, run DecontX with those clusters
# ----------------------------------------------------------------------------------
print("Loading filtered cells for clustering...", flush=True)
adata_filt_base = cs_utils.load_adata(filtered_genome_dir, verbose=0)
adata_filt_base.var_names_make_unique()
print(f"  filtered matrix: {adata_filt_base.shape}", flush=True)

decontx_runs = {}
for res in leiden_resolutions:
    res_tag = str(res).replace(".", "_")
    out_prefix = os.path.join(data_dir, f"pbmc8k_output_decontx_leiden_{res_tag}")
    clusters_csv = os.path.join(data_dir, f"pbmc8k_decontx_leiden_{res_tag}_clusters.csv")

    if (not os.path.exists(f"{out_prefix}.mtx")) or overwrite:
        print(f"\n=== DecontX with Leiden resolution {res} ===", flush=True)
        adata_tmp = adata_filt_base.copy()
        adata_tmp = cs_utils.run_scanpy_preprocessing_and_clustering(
            adata_tmp, min_genes=min_genes, min_cells=min_cells,
            max_mt_percentage=max_mt_percentage, n_top_genes=n_top_genes,
            n_pcs=n_pcs, n_neighbors=n_neighbors, leiden_resolution=res,
            seed=seed, verbose=verbose,
        )
        n_clusters = adata_tmp.obs["leiden"].nunique()
        print(f"  {n_clusters} Leiden clusters", flush=True)
        adata_tmp.obs[["leiden"]].to_csv(clusters_csv)

        cmd = [
            docker, "run", "--rm", "--security-opt", "label=disable",
            "-w", "/home/ruser/work",
            "-v", f"{cellsweep_dir}:{rver_docker_workspace}",
            "josephrich98/cellsweep_tutorials:decontx.0.1.0",
            "Rscript", f"{rver_docker_workspace}/scripts/run_decontx.R",
            in_container(raw_tar_file_dir),
            in_container(filtered_tar_file_dir),
            sequencing_technology,
            in_container(out_prefix),
            "--dont_prepend_sample_to_barcodes",
            f"--clusters_csv={in_container(clusters_csv)}",
            "--cluster_col=leiden",
        ]
        print("  running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    else:
        print(f"\n=== DecontX Leiden resolution {res}: output exists, skipping run ===", flush=True)

    adata_run = load_decontx(out_prefix)
    # recover cluster count (from saved csv if present)
    if os.path.exists(clusters_csv):
        n_clusters = pd.read_csv(clusters_csv, index_col=0)["leiden"].nunique()
    else:
        n_clusters = np.nan
    decontx_runs[res] = (adata_run, n_clusters)

# ----------------------------------------------------------------------------------
# Step 2: metrics for DecontX vs default DecontX
# ----------------------------------------------------------------------------------
print("\nLoading default DecontX output...", flush=True)
adata_decontx_default = load_decontx(decontx_default_prefix)

rows = []
for res in leiden_resolutions:
    adata_run, n_clusters = decontx_runs[res]
    Xbase, Xrun, ncells, ngenes = align(adata_decontx_default, adata_run)
    m = compare(Xbase, Xrun)
    m.update({"tool": "decontx", "leiden_resolution": res,
              "n_clusters": n_clusters, "n_cells": ncells, "n_genes": ngenes})
    rows.append(m)
    print(f"  decontx res={res}: clusters={n_clusters} "
          f"entry_r={m['entry_pearson_vs_default']:.4f} "
          f"cell_r={m['cell_total_pearson_vs_default']:.4f} "
          f"counts_ratio={m['total_counts_ratio_vs_default']:.4f}", flush=True)

    # Per-resolution scatterplots, mirroring the cellsweep Leiden cell in benchmarking.ipynb
    # (default decontx [internal clustering] on x, the Leiden-z run on y).
    x_axis = "decontx default"
    y_axis = f"leiden ({n_clusters} clusters)"
    cs_utils.plot_matrix_scatterplot(
        adata_decontx_default, adata_run, point_type="matrix",
        density_type="scatter_with_density", scale="log", x_axis=x_axis, y_axis=y_axis,
        out_path=os.path.join(out_dir, f"decontx_leiden_{res}_matrix_scatterplot.png"), show=False)
    cs_utils.plot_matrix_scatterplot(
        adata_decontx_default, adata_run, point_type="cell",
        density_type="scatter_with_kde", scale="log", x_axis=x_axis, y_axis=y_axis,
        out_path=os.path.join(out_dir, f"decontx_leiden_{res}_cell_scatterplot.png"), show=False)
    cs_utils.plot_matrix_scatterplot(
        adata_decontx_default, adata_run, point_type="gene",
        density_type="scatter_with_kde", scale="log", x_axis=x_axis, y_axis=y_axis,
        out_path=os.path.join(out_dir, f"decontx_leiden_{res}_gene_scatterplot.png"), show=False)

# ----------------------------------------------------------------------------------
# Step 3: matching metrics for cellsweep Leiden runs (already on disk)
# ----------------------------------------------------------------------------------
print("\nLoading default cellsweep output...", flush=True)
adata_cs_default = cs_utils.load_adata(cellsweep_default_path, verbose=0)
adata_cs_default.var_names_make_unique()
if "is_empty" in adata_cs_default.obs.columns:
    adata_cs_default = adata_cs_default[~adata_cs_default.obs["is_empty"]].copy()

for res in leiden_resolutions:
    res_tag = str(res).replace(".", "_")
    cs_path = os.path.join(data_dir, f"adata_cellsweep_leiden_{res_tag}.h5ad")
    if not os.path.exists(cs_path):
        print(f"  cellsweep res={res}: file missing ({cs_path}), skipping", flush=True)
        continue
    adata_cs = cs_utils.load_adata(cs_path, verbose=0)
    if "is_empty" in adata_cs.obs.columns:
        adata_cs = adata_cs[~adata_cs.obs["is_empty"]].copy()
    adata_cs.var_names_make_unique()
    n_clusters = adata_cs.obs["celltype"].nunique() if "celltype" in adata_cs.obs.columns else np.nan

    Xbase, Xrun, ncells, ngenes = align(adata_cs_default, adata_cs)
    m = compare(Xbase, Xrun)
    m.update({"tool": "cellsweep", "leiden_resolution": res,
              "n_clusters": n_clusters, "n_cells": ncells, "n_genes": ngenes})
    rows.append(m)
    print(f"  cellsweep res={res}: clusters={n_clusters} "
          f"entry_r={m['entry_pearson_vs_default']:.4f} "
          f"cell_r={m['cell_total_pearson_vs_default']:.4f} "
          f"counts_ratio={m['total_counts_ratio_vs_default']:.4f}", flush=True)

# ----------------------------------------------------------------------------------
# Step 4: save + summarize
# ----------------------------------------------------------------------------------
df = pd.DataFrame(rows)
cols = ["tool", "leiden_resolution", "n_clusters", "n_cells", "n_genes",
        "entry_pearson_vs_default", "cell_total_pearson_vs_default",
        "gene_total_pearson_vs_default", "total_counts", "total_counts_default",
        "total_counts_ratio_vs_default", "mean_abs_cell_total_diff"]
df = df[cols]
csv_path = os.path.join(out_dir, "decontx_vs_cellsweep_leiden_sensitivity.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved metrics to {csv_path}\n", flush=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
print(df.to_string(index=False), flush=True)

# Sensitivity summary: spread of entry/cell correlation and counts ratio across resolutions.
print("\n=== Sensitivity summary (lower spread / higher min-corr = more robust) ===", flush=True)
for tool in ["decontx", "cellsweep"]:
    sub = df[df["tool"] == tool]
    if sub.empty:
        continue
    ec = sub["entry_pearson_vs_default"]
    cc = sub["cell_total_pearson_vs_default"]
    cr = sub["total_counts_ratio_vs_default"]
    print(f"{tool}:", flush=True)
    print(f"  entry r vs default:  min={ec.min():.4f} mean={ec.mean():.4f} "
          f"max={ec.max():.4f} range={ec.max()-ec.min():.4f}", flush=True)
    print(f"  cell-total r vs def: min={cc.min():.4f} mean={cc.mean():.4f} "
          f"max={cc.max():.4f} range={cc.max()-cc.min():.4f}", flush=True)
    print(f"  counts ratio vs def: min={cr.min():.4f} mean={cr.mean():.4f} "
          f"max={cr.max():.4f} range={cr.max()-cr.min():.4f}", flush=True)
