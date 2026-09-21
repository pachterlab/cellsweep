#!/usr/bin/env python
"""Run scAR on a Janssen et al. (2023) snRNA-seq replicate.

Mirrors scripts/run_scar.py (the settings used for every other dataset in the paper:
prob = 0.995, 200 epochs, batch size 64, micro adjustment) but reads the CellRanger raw and
filtered matrices of one replicate directly and restricts the cells to the nuclei Janssen
et al. keep in their Seurat object, so the output lands on the same grid as the other
methods in scripts/janssen_loaders.py.

Usage: python scripts/run_janssen_scar.py --rep nuc2
"""
import argparse
import os
import sys
import warnings

import numpy as np
import scanpy as sc

warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import janssen_loaders as L


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rep", default="nuc2")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--prob", type=float, default=0.995)
    p.add_argument("--sparsity", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    from scar import model, setup_anndata

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rep_dir = os.path.join(L.DATA_DIR, args.rep)
    genes, cells = L.axes(args.rep)
    out = args.out or os.path.join(L.TOOL_DIR, f"{args.rep}_scar.h5ad")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    print("Loading raw matrix...", flush=True)
    adata_raw = sc.read_10x_h5(os.path.join(rep_dir, "raw_feature_bc_matrix.h5"))
    adata_raw.var_names = L._make_unique(adata_raw.var_names.values.astype(str))
    adata_raw.var["feature_types"] = "Gene Expression"
    print("  raw:", adata_raw.shape, flush=True)

    print("Loading filtered matrix...", flush=True)
    adata = sc.read_10x_h5(os.path.join(rep_dir, "filtered_feature_bc_matrix.h5"))
    adata.var_names = L._make_unique(adata.var_names.values.astype(str))
    adata.var["feature_types"] = "Gene Expression"
    # Janssen et al. keep only the nuclei in their Seurat object; score scAR on the same set.
    adata = adata[cells].copy()
    print("  cells:", adata.shape, flush=True)

    print("Setting up AnnData for scAR...", flush=True)
    setup_anndata(adata=adata, raw_adata=adata_raw, feature_type="Gene Expression",
                  prob=args.prob, kneeplot=False)
    del adata_raw

    print("Training scAR...", flush=True)
    m = model(raw_count=adata,
              ambient_profile=adata.uns["ambient_profile_Gene Expression"],
              feature_type="mRNA", sparsity=args.sparsity,
              device="cuda" if args.cuda else "cpu")
    m.train(epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    print("Inference...", flush=True)
    m.inference(adjust="micro")
    assert m.native_counts.shape == adata.X.shape

    adata.layers["raw"] = adata.X.copy()
    adata.X = m.native_counts
    adata = adata[:, genes].copy()          # the 28,679 genes shared with every other method
    adata.write_h5ad(out)
    print("wrote", out, adata.shape, flush=True)


if __name__ == "__main__":
    main()
