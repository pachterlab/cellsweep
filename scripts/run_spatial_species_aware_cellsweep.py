#!/usr/bin/env python
"""
Reviewer point 7 (spatial xenograft): re-run cellsweep on the Visium HD human-mouse xenograft
(8 um bins) with species-aware cell-type labels.

The manuscript run (notebooks/spatial.ipynb) clusters all non-empty bins together with Leiden.
Because mouse host tissue is only a thin capsule (~4% of tissue area) and every mouse bin carries
human counts, no Leiden cluster is mouse-dominated (the most mouse-rich cluster profile is 26% human),
so human counts in mouse bins are partly absorbed into the cell-type profile rather than the ambient term.
These labelings test how the correction of each species depends on that choice:

    original       Leiden on all non-empty bins (manuscript run; read from adata_cellsweep.h5ad)
    species        Leiden run separately within human-majority and mouse-majority bins
    species_mixed  as `species`, but bins whose content-normalized species fraction is in
                   (MIXED_LO, MIXED_HI) get their own "mixed" label, so that genuine two-species
                   content (the spatial analog of a cross-species doublet) is modeled as signal

Species majority uses the content-normalized human fraction
    fh_norm = (h / s_h) / (h / s_h + m / s_m),
where s_h, s_m are the median library sizes of >=90%-pure human / mouse bins. Per-bin CPM does not
change a within-bin species fraction, so this per-species RNA-content scaling is the relevant normalization.

All runs reuse the raw matrix, empty-bin calls, and cellsweep settings of the manuscript run.

Usage: python scripts/run_spatial_species_aware_cellsweep.py [--overwrite]
Analysis/figures: scripts/analyze_spatial_xenograft_mixing.py
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
run_dir = os.path.join(data_dir, "species_aware")
os.makedirs(run_dir, exist_ok=True)

MIXED_LO, MIXED_HI = 0.25, 0.75
leiden_resolution = 1.0
threads = 32
verbose = 1


def species_counts(adata):
    is_human = (adata.var["genome"] == "GRCh38").values
    X = adata.X.tocsc()
    h = np.asarray(X[:, is_human].sum(axis=1)).ravel()
    m = np.asarray(X[:, ~is_human].sum(axis=1)).ravel()
    return h, m


def leiden_labels(adata, prefix):
    tmp = cs_utils.run_scanpy_preprocessing_and_clustering(adata, min_genes=None, min_cells=None, max_mt_percentage=None, n_top_genes=2000, n_pcs=50, n_neighbors=15, leiden_resolution=leiden_resolution, seed=42, verbose=0)
    return (prefix + "_" + tmp.obs["leiden"].astype(str)).reindex(adata.obs_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labels_path = os.path.join(run_dir, "labels.csv")
    original = ad.read_h5ad(os.path.join(data_dir, "adata_cellsweep.h5ad"))
    adata_raw = ad.AnnData(X=original.layers["raw"].tocsr(), obs=original.obs[["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres", "is_empty", "celltype"]].copy(), var=original.var[["gene_ids", "feature_types", "genome"]].copy())
    adata_raw.obsm["spatial"] = original.obsm["spatial"]
    del original

    if not os.path.exists(labels_path) or args.overwrite:
        is_empty = adata_raw.obs["is_empty"].astype(bool).values
        h, m = species_counts(adata_raw)
        fh = h / np.maximum(h + m, 1)
        s_h = np.median((h + m)[~is_empty & (fh >= 0.9)])
        s_m = np.median((h + m)[~is_empty & (fh <= 0.1)])
        fh_norm = (h / s_h) / np.maximum(h / s_h + m / s_m, 1e-12)
        print(f"median library size: human-pure {s_h:.0f}, mouse-pure {s_m:.0f}")

        labels = pd.DataFrame(index=adata_raw.obs_names)
        labels["fh"], labels["fh_norm"] = fh, fh_norm
        labels["original"] = adata_raw.obs["celltype"].astype(str)

        nonempty = adata_raw[~is_empty]
        majority_human = (fh_norm[~is_empty] >= 0.5)
        species = pd.concat([leiden_labels(nonempty[majority_human], "human"), leiden_labels(nonempty[~majority_human], "mouse")])
        labels["species"] = "empty"
        labels.loc[species.index, "species"] = species.values

        labels["species_mixed"] = labels["species"]
        mixed = (~is_empty) & (fh_norm > MIXED_LO) & (fh_norm < MIXED_HI)
        pure = nonempty[~mixed[~is_empty]]
        pure_h = fh_norm[~is_empty][~mixed[~is_empty]] >= 0.5
        species_pure = pd.concat([leiden_labels(pure[pure_h], "human"), leiden_labels(pure[~pure_h], "mouse")])
        labels.loc[species_pure.index, "species_mixed"] = species_pure.values
        labels.loc[mixed, "species_mixed"] = "mixed"
        labels.to_csv(labels_path)
    labels = pd.read_csv(labels_path, index_col=0)
    for scheme in ["species", "species_mixed"]:
        print(scheme, labels.loc[labels[scheme] != "empty", scheme].str.split("_").str[0].value_counts().to_dict(), "K =", labels[scheme].nunique() - 1)

    for scheme in ["species", "species_mixed"]:
        out_path = os.path.join(run_dir, f"adata_cellsweep_{scheme}.h5ad")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{out_path} exists, skipping")
            continue
        adata = adata_raw.copy()
        adata.obs["celltype"] = labels.loc[adata.obs_names, scheme].astype("category")
        denoise_count_matrix(adata, adata_out=out_path, freeze_ambient_profile=True, empty_droplet_method="threshold", threads=threads, verbose=verbose, log_file=os.path.join(run_dir, f"cellsweep_{scheme}.log"))


if __name__ == "__main__":
    main()
