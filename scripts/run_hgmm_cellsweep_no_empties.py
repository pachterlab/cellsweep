#!/usr/bin/env python
"""CellSweep on the 10x human-mouse mixture with the empty droplets removed.

Mirrors the main hgmm_12k run (notebooks/benchmarking.ipynb) but hands CellSweep only the
called cells, so it falls back to its alternative model: the ambient profile is initialised
from the cell-type mixture and updated during training instead of being frozen to the
empty-droplet profile. The cell-type labels are taken from the main run so that the empty
droplets are the only thing that differs.

Writes hgmm_12k_output_cellsweep_no_empties.h5ad next to the main output.

Usage: python scripts/run_hgmm_cellsweep_no_empties.py [--threads 16] [--overwrite]
"""
import argparse
import os

import anndata as ad

from cellsweep import denoise_count_matrix

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "notebooks", "data", "hgmm_12k")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "notebooks", "output", "hgmm_12k")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_path = os.path.join(DATA_DIR, "hgmm_12k_output_cellsweep_no_empties.h5ad")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"{out_path} exists, skipping")
        return

    adata = ad.read_h5ad(os.path.join(DATA_DIR, "hgmm_12k_output_cellsweep.h5ad"))
    adata = adata[~adata.obs["is_empty"].values].copy()
    adata.X = adata.layers["raw"].copy()
    del adata.layers["raw"]
    for k in ["alpha_hat", "z_hat", "init_alpha", "n_counts"]:
        if k in adata.obs:
            del adata.obs[k]
    adata.uns = {}
    print(f"{adata.n_obs:,} cells, no empty droplets", flush=True)

    denoise_count_matrix(
        adata,
        adata_out=out_path,
        freeze_ambient_profile=True,  # switched off by CellSweep itself when there are no empties
        empty_droplet_method="threshold",
        init_alpha=0.9,
        init_beta=0.1,
        max_iter=2000,
        threads=args.threads,
        verbose=0,
        log_file=os.path.join(OUT_DIR, "cellsweep_no_empties.log"),
    )


if __name__ == "__main__":
    main()
