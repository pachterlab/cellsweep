"""scAR diagnostics on simulation1_small_noise (reviewer: is scAR's simulation score a bug?).

Checks, in order:
  1. Input: the ambient profile scAR estimates from the raw matrix vs the true simulated profile.
  2. Reproducibility and settings: retrains scAR (v0.7.0) under
       original  - the settings used in the manuscript (200 epochs, sparsity 1.0, prob 0.975)
       defaults  - scAR's own defaults (400 epochs, sparsity 0.9, prob 0.995)
       oracle    - scAR defaults, but given the TRUE ambient profile
     and runs inference twice per model: scAR's default clip_to_obs=False, and clip_to_obs=True.
  3. Metric ceiling: an "oracle smoother" that outputs the expected native counts
     total_i * (1 - true ambient fraction_i) * true type profile, stochastically rounded as scAR does.
     This is what a perfect scAR decoder would return; it shows how the element-wise simulation
     metric scores a method whose output is a model expectation rather than observed minus noise.

All outputs are scored with the same element-wise confusion components as Figure 2
(scripts/make_simulation_sensitivity_figure.py), on the cells retained by all tools.

Step 2 must run in the scAR environment and step 1/3 + scoring in the cellsweep environment:
  conda run -p ~/miniconda3/envs/scar python scripts/scar_simulation_diagnostics.py train --variant original
  conda run -p ~/miniconda3/envs/scar python scripts/scar_simulation_diagnostics.py train --variant defaults
  conda run -p ~/miniconda3/envs/scar python scripts/scar_simulation_diagnostics.py train --variant oracle
  python scripts/scar_simulation_diagnostics.py score
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

CELLSWEEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CELLSWEEP_DIR, "notebooks", "data", "simulation1_small_noise")
MATRIX_DIR = os.path.join(DATA_DIR, "matrix_tar_files")
OUT_DIR = os.path.join(CELLSWEEP_DIR, "notebooks", "output", "simulation1_small_noise", "scar_diagnostics")

VARIANTS = {
    "original": dict(epochs=200, sparsity=1.0, prob=0.975, oracle_ambient=False),
    "defaults": dict(epochs=400, sparsity=0.9, prob=0.995, oracle_ambient=False),
    "oracle": dict(epochs=400, sparsity=0.9, prob=0.995, oracle_ambient=True),
}


def train(variant, threads, seed=0):
    import anndata as ad
    import scanpy as sc
    import torch
    from scar import model, setup_anndata

    cfg = VARIANTS[variant]
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw = sc.read_10x_mtx(os.path.join(MATRIX_DIR, "raw_gene_bc_matrices", "genome"))
    raw.var["feature_types"] = "Gene Expression"
    adata = sc.read_10x_mtx(os.path.join(MATRIX_DIR, "filtered_gene_bc_matrices", "genome"))
    adata.var["feature_types"] = "Gene Expression"
    setup_anndata(adata=adata, raw_adata=raw, feature_type="Gene Expression", prob=cfg["prob"], kneeplot=False)
    ambient = adata.uns["ambient_profile_Gene Expression"]

    truth = ad.read_h5ad(os.path.join(DATA_DIR, "adata_raw.h5ad"), backed="r")
    true_ambient = truth.var["ambient_profile"].reindex(adata.var_names).to_numpy()
    true_ambient = true_ambient / true_ambient.sum()
    est = np.asarray(ambient).ravel()
    est = est / est.sum()
    print(f"[{variant}] estimated ambient profile vs truth: pearson {np.corrcoef(est, true_ambient)[0, 1]:.4f}, "
          f"L1 {np.abs(est - true_ambient).sum():.4f}", flush=True)
    if cfg["oracle_ambient"]:
        # scAR ignores the ambient_profile argument for AnnData input without batch_key and reads
        # adata.uns["ambient_profile_all"] instead, so the oracle profile must be placed there
        ambient = pd.DataFrame(true_ambient.reshape(-1, 1), index=adata.var_names, columns=["ambient_profile"])
        adata.uns["ambient_profile_all"] = ambient

    t0 = time.time()
    m = model(raw_count=adata, ambient_profile=ambient, feature_type="mRNA", sparsity=cfg["sparsity"], device="cpu")
    m.train(epochs=cfg["epochs"], batch_size=64, verbose=False)
    print(f"[{variant}] trained in {(time.time() - t0) / 60:.1f} min", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for clip in (False, True):
        np.random.seed(seed)
        m.inference(clip_to_obs=clip)
        out = adata.copy()
        out.layers["raw"] = out.X.copy()
        out.X = m.native_counts
        out.obs["noise_ratio"] = np.asarray(m.noise_ratio.todense()).ravel()
        out.uns.pop("ambient_profile_all", None)
        out.write_h5ad(os.path.join(OUT_DIR, f"scar_{variant}_clip{int(clip)}.h5ad"))


def oracle_smoother(raw, seed=0):
    """Expected native counts under the true simulation parameters, stochastically rounded like scAR."""
    P = np.asarray(raw.uns["type_profiles"], dtype=float)
    P = P / P.sum(axis=1, keepdims=True)
    type_idx = raw.obs["celltype"].astype(str).str.replace("Type_", "").astype(int).to_numpy()
    total = np.asarray(raw.X.sum(axis=1)).ravel()
    E = (total * (1 - raw.obs["ambient_fraction"].to_numpy()))[:, None] * P[type_idx]
    rng = np.random.default_rng(seed)
    return np.floor(E) + (rng.random(E.shape) < E - np.floor(E))


def score():
    import anndata as ad
    from scipy import sparse

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_simulation_sensitivity_figure as fig

    raw, tools, common = fig.load_tools()
    raw = raw[common].copy()
    Yr, Yt = fig._rint(raw.X), fig._rint(raw.layers["real"])
    true_noise_frac = fig._rint(raw.layers["noise"]).sum() / Yr.sum()
    is_marker = raw.var["is_marker"].to_numpy().astype(bool)

    outputs = {"CellSweep": tools["CellSweep"], "scAR (manuscript)": tools["scAR"]}
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "scar_*_clip*.h5ad"))):
        a = ad.read_h5ad(f)
        a.var_names_make_unique()
        outputs[os.path.basename(f)[:-5]] = a
    smoother = oracle_smoother(raw)

    matrices = {name: fig._rint(a[common, raw.var_names].X) for name, a in outputs.items()}
    matrices["oracle_smoother_clip0"] = sparse.csr_matrix(smoother)
    matrices["oracle_smoother_clip1"] = sparse.csr_matrix(np.minimum(smoother, Yr.toarray()))

    rows = []
    for name, Yp in matrices.items():
        TP, FP, FN, TN = fig.confusion(Yp, Yt, Yr)
        for scope, cols in (("marker genes", is_marker), ("all genes", slice(None))):
            tp, fp, fn, tn = (M[:, cols].sum() for M in (TP, FP, FN, TN))
            rows.append(dict(output=name, genes=scope, sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                             ppv=tp / (tp + fp), counts_above_raw=(Yp - Yr).maximum(0).sum(),
                             frac_counts_removed=1 - Yp.sum() / Yr.sum(), true_noise_frac=true_noise_frac))
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "scar_diagnostics_metrics.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--variant", choices=list(VARIANTS), required=True)
    t.add_argument("--threads", type=int, default=20)
    sub.add_parser("score")
    args = p.parse_args()
    train(args.variant, args.threads) if args.cmd == "train" else score()
