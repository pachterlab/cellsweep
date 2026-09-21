"""CellSweep on the Janssen et al. (2023) snRNA-seq data with the empty droplets removed.

Mirrors the main Janssen run (notebooks/janssen_snrna.ipynb) but hands CellSweep only the
called nuclei, so it falls back to its alternative model: the ambient profile is initialised
from the cell-type mixture and updated during training instead of being frozen to the
empty-droplet profile. Writes <rep>_output_cellsweep_no_empties.h5ad next to the main output.

Usage: python scripts/run_janssen_cellsweep_no_empties.py [--threads 16] [--overwrite]
"""

import argparse
import os
import sys

import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import janssen_loaders as L
from cellsweep import denoise_count_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for rep in L.REPLICATES:
        out_path = os.path.join(L.DATA_DIR, rep, f"{rep}_output_cellsweep_no_empties.h5ad")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"{rep}: {out_path} exists, skipping")
            continue
        adata = ad.read_h5ad(os.path.join(L.DATA_DIR, rep, "adata_raw_empty100.h5ad"))
        adata = adata[~adata.obs["is_empty"].values].copy()
        print(f"{rep}: {adata.n_obs:,} nuclei, no empty droplets")
        denoise_count_matrix(
            adata,
            adata_out=out_path,
            freeze_ambient_profile=True,  # switched off by CellSweep itself when there are no empties
            empty_droplet_method="threshold",
            threads=args.threads,
            verbose=0,
            log_file=os.path.join(L.OUT_DIR, f"{rep}_cellsweep_no_empties.log"),
        )


if __name__ == "__main__":
    main()
