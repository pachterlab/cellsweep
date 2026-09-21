#!/usr/bin/env python
"""
Why do CellBender, scAR, DecontX and SoupX leave a small subset of cells almost
untouched in the 10x human-mouse mixture (hgmm_12k), while CellSweep does not?

Reviewer 1, comment 3.

For every barcode called as a cell we compute, from the cross-species ground truth,
the fraction of its off-target counts that each tool removed. We then ask what
distinguishes the barcodes that a tool fails on, using

  * standard QC (library size, genes detected, mitochondrial fraction),
  * the per-cell contamination fraction, and
  * a two-component multinomial mixture fit to each barcode's cross-species counts,
    which splits them into "ambient/global contamination" and "a genuine cell of the
    other species" using reference profiles estimated from the data itself.

Outputs a per-cell table, the summary statistics quoted in the manuscript, and a figure
with one panel per method: every barcode by its raw human and mouse counts, with the
residual doublets marked and the barcodes that method failed on circled.

Usage:
    python scripts/hgmm_failure_mode_analysis.py \
        --data-dir notebooks/data/hgmm_12k \
        --out-dir notebooks/output/hgmm_12k
"""
import argparse
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import cellsweep.utils as cs_utils

warnings.filterwarnings("ignore")

TOOLS = ["cellsweep", "soupx", "cellbender", "decontx", "scar", "cellsweep_no_empties"]
TOOL_LABEL = {"cellsweep": "CellSweep", "cellbender": "CellBender", "scar": "scAR",
              "soupx": "SoupX", "decontx": "DecontX",
              "cellsweep_no_empties": "CellSweep (no empties)"}
TOOL_COLOR = {"cellsweep": "#1f77b4", "cellbender": "#d62728", "scar": "#2ca02c",
              "soupx": "#9467bd", "decontx": "#ff7f0e", "cellsweep_no_empties": "#8c564b"}


def species_totals(adata):
    """Per-barcode human and mouse totals plus number of genes detected."""
    adata.var_names_make_unique()
    is_h = np.asarray(adata.var_names.str.startswith("hg19_"))
    is_m = np.asarray(adata.var_names.str.startswith("mm10_"))
    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    return (pd.Series(np.asarray(X[:, is_h].sum(axis=1)).ravel(), index=adata.obs_names),
            pd.Series(np.asarray(X[:, is_m].sum(axis=1)).ravel(), index=adata.obs_names),
            pd.Series(np.diff(X.indptr), index=adata.obs_names))


def build_table(data_dir):
    """Per-cell table: raw counts, per-tool counts, QC, CellSweep alpha_hat."""
    acs = ad.read_h5ad(os.path.join(data_dir, "hgmm_12k_output_cellsweep.h5ad"))
    acs.var_names_make_unique()
    acs = acs[~acs.obs["is_empty"].values].copy()

    is_h = np.asarray(acs.var_names.str.startswith("hg19_"))
    is_m = np.asarray(acs.var_names.str.startswith("mm10_"))
    X_raw = acs.layers["raw"].tocsr()

    df = pd.DataFrame(index=acs.obs_names.copy())
    df["raw_h"] = np.asarray(X_raw[:, is_h].sum(axis=1)).ravel()
    df["raw_m"] = np.asarray(X_raw[:, is_m].sum(axis=1)).ravel()
    df["raw_total"] = df.raw_h + df.raw_m
    df["genome"] = np.where(df.raw_h >= df.raw_m, "hg19", "mm10")
    df["is_doublet"] = acs.obs["is_doublet"].values
    df["alpha_hat"] = acs.obs["alpha_hat"].values
    df["raw_n_genes"] = np.diff(X_raw.indptr)

    mt_h = np.asarray(acs.var_names.str.startswith("hg19_MT-"))
    mt_m = np.asarray(acs.var_names.str.startswith("mm10_mt-"))
    df["mt_frac_raw"] = np.where(
        df.genome == "hg19",
        np.asarray(X_raw[:, mt_h].sum(axis=1)).ravel() / np.maximum(df.raw_h, 1),
        np.asarray(X_raw[:, mt_m].sum(axis=1)).ravel() / np.maximum(df.raw_m, 1))

    df["cellsweep_h"] = np.asarray(acs.X.tocsr()[:, is_h].sum(axis=1)).ravel()
    df["cellsweep_m"] = np.asarray(acs.X.tocsr()[:, is_m].sum(axis=1)).ravel()

    # CellSweep's alternative model, run on the called cells alone
    # (scripts/run_hgmm_cellsweep_no_empties.py)
    ane = ad.read_h5ad(os.path.join(data_dir, "hgmm_12k_output_cellsweep_no_empties.h5ad"))
    ane.var_names_make_unique()
    h, m, _ = species_totals(ane)
    df["cellsweep_no_empties_h"] = h.reindex(df.index).values
    df["cellsweep_no_empties_m"] = m.reindex(df.index).values
    del ane

    loaders = {
        "cellbender": lambda: sc.read_10x_h5(
            os.path.join(data_dir, "hgmm_12k_output_cellbender_filtered.h5"), gex_only=False),
        "soupx": lambda: cs_utils.load_adata(os.path.join(data_dir, "hgmm_12k_output_soupx")),
        "scar": lambda: ad.read_h5ad(os.path.join(data_dir, "hgmm_12k_output_scar.h5ad")),
    }
    for tool, load in loaders.items():
        h, m, _ = species_totals(load())
        df[f"{tool}_h"] = h.reindex(df.index).values
        df[f"{tool}_m"] = m.reindex(df.index).values

    a_dx = cs_utils.load_adata(os.path.join(data_dir, "hgmm_12k_output_decontx"))
    a_dx.obs_names = [n.replace("GRCh38_", "", 1) for n in a_dx.obs_names]
    h, m, _ = species_totals(a_dx)
    df["decontx_h"] = h.reindex(df.index).values
    df["decontx_m"] = m.reindex(df.index).values

    is_hum = (df.genome == "hg19").values
    df["noise_raw"] = np.where(is_hum, df.raw_m, df.raw_h)
    df["signal_raw"] = np.where(is_hum, df.raw_h, df.raw_m)
    df["frac_contam_raw"] = df.noise_raw / df.raw_total
    for t in TOOLS:
        noise = np.where(is_hum, df[f"{t}_m"], df[f"{t}_h"])
        signal = np.where(is_hum, df[f"{t}_h"], df[f"{t}_m"])
        df[f"{t}_noise"] = noise
        df[f"{t}_rm"] = 1 - noise / df.noise_raw.replace(0, np.nan)
        df[f"{t}_sig_ret"] = signal / df.signal_raw
    return df, acs, is_h, is_m


def reference_profiles(df, acs, is_h, is_m):
    """Contamination and other-species-cell reference profiles, per host species."""
    X_raw = acs.layers["raw"].tocsr()
    X_den = acs.X.tocsr()
    pos = pd.Series(np.arange(acs.n_obs), index=acs.obs_names)
    con, cell = {}, {}
    for host, cross in (("hg19", is_m), ("mm10", is_h)):
        other = "mm10" if host == "hg19" else "hg19"
        # contamination profile: cross-species counts pooled over the least
        # contaminated half of the host-species cells
        q = df[df.genome == host].frac_contam_raw.quantile(0.5)
        typ = df[(df.genome == host) & (df.frac_contam_raw < q)].index
        c = np.asarray(X_raw[pos[typ].values][:, cross].sum(axis=0)).ravel()
        con[host] = c / c.sum()
        # cell profile: denoised expression of the least contaminated other-species cells
        q2 = df[df.genome == other].frac_contam_raw.quantile(0.5)
        oth = df[(df.genome == other) & (df.frac_contam_raw < q2)].index
        p = np.asarray(X_den[pos[oth].values][:, cross].sum(axis=0)).ravel()
        cell[host] = p / p.sum()
    return con, cell, pos


def fit_w(counts, p_con, p_cell, iters=500, tol=1e-9):
    """EM for w in  y ~ Multinomial(w * p_cell + (1 - w) * p_con)."""
    nz = counts > 0
    y, A, B = counts[nz], p_cell[nz] + 1e-300, p_con[nz] + 1e-300
    if y.sum() == 0:
        return np.nan
    w = 0.5
    for _ in range(iters):
        num = w * A
        w_new = float((y * (num / (num + (1 - w) * B))).sum() / y.sum())
        if abs(w_new - w) < tol:
            return w_new
        w = w_new
    return w


def fit_all(df, acs, is_h, is_m, con, cell, pos):
    X_raw = acs.layers["raw"].tocsr()
    w = pd.Series(index=df.index, dtype=float)
    for host, cross in (("hg19", is_m), ("mm10", is_h)):
        cells = df.index[df.genome == host]
        sub = X_raw[pos[cells].values][:, cross].tocsr()
        w.loc[cells] = [fit_w(np.asarray(sub[i].todense()).ravel(), con[host], cell[host])
                        for i in range(sub.shape[0])]
    df["w_secondcell"] = w
    df["n_cross"] = df.noise_raw
    df["cross_from_cell"] = df.w_secondcell * df.n_cross
    df["residual_doublet"] = (df.w_secondcell > 0.5) & (df.cross_from_cell > 500)
    return df


def synthetic_controls(df, acs, is_h, is_m, con, cell, pos, n_add=2000, n_cells=300, seed=0):
    """Positive/negative controls for the mixture estimator."""
    rng = np.random.default_rng(seed)
    X_raw, X_den = acs.layers["raw"].tocsr(), acs.X.tocsr()

    def norm(v):
        v = np.maximum(np.asarray(v, float), 0)
        v = v / v.sum()
        v[-1] = max(0.0, 1.0 - v[:-1].sum())
        return v

    out = {"ambient": [], "doublet": []}
    for host, cross in (("hg19", is_m), ("mm10", is_h)):
        other = "mm10" if host == "hg19" else "hg19"
        base = df[(df.genome == host) & (~df.residual_doublet)].index[:n_cells]
        donors = df[(df.genome == other) & (~df.residual_doublet)].index[:n_cells]
        B = X_raw[pos[base].values][:, cross].toarray()
        Dn = X_den[pos[donors].values][:, cross].toarray()
        for i in range(len(base)):
            amb = B[i] + rng.multinomial(n_add, norm(con[host]))
            out["ambient"].append(fit_w(amb, con[host], cell[host]))
            j = rng.integers(Dn.shape[0])
            dbl = B[i] + rng.multinomial(n_add, norm(Dn[j]))
            out["doublet"].append(fit_w(dbl, con[host], cell[host]))
    return {k: np.asarray(v) for k, v in out.items()}


def make_figure(df, out_path):
    """One panel per method: the raw human/mouse counts of every barcode, with the barcodes
    that method leaves more than half of their cross-species counts in circled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5))
    rd = df[df.residual_doublet]
    for letter, tool, ax in zip("ABCDEF", TOOLS, axes.ravel()):
        fail = df[df[f"{tool}_rm"] < 0.5]
        ax.scatter(df.raw_h + 1, df.raw_m + 1, s=4, c="0.8", lw=0, rasterized=True,
                   label=f"singlet-called (n={len(df) - len(rd):,})")
        ax.scatter(rd.raw_h + 1, rd.raw_m + 1, s=18, c="#d62728", lw=0,
                   label=f"residual doublet (n={len(rd):,})")
        ax.scatter(fail.raw_h + 1, fail.raw_m + 1, s=44, facecolors="none", edgecolors="k", lw=0.9,
                   label=f"{TOOL_LABEL[tool]} fails (n={len(fail):,})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("human counts + 1"); ax.set_ylabel("mouse counts + 1")
        ax.set_title(f"{letter}  Barcodes {TOOL_LABEL[tool]} fails on")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)

    # one set of axis limits, so the panels can be read against one another
    lims = [(min(a.get_xlim()[0] for a in axes.ravel()), max(a.get_xlim()[1] for a in axes.ravel())),
            (min(a.get_ylim()[0] for a in axes.ravel()), max(a.get_ylim()[1] for a in axes.ravel()))]
    for a in axes.ravel():
        a.set_xlim(*lims[0]); a.set_ylim(*lims[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="notebooks/data/hgmm_12k")
    ap.add_argument("--out-dir", default="notebooks/output/hgmm_12k")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached per-cell table and controls instead of refitting")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv = os.path.join(args.out_dir, "hgmm_12k_per_cell_failure_analysis.csv")
    npz = os.path.join(args.out_dir, "hgmm_12k_failure_analysis_controls.npz")
    if args.reuse and os.path.exists(csv) and os.path.exists(npz):
        df = pd.read_csv(csv, index_col=0)
        synth = dict(np.load(npz))
        missing = [t for t in TOOLS if f"{t}_rm" not in df.columns]
        if missing:
            # the per-barcode mixture fit does not depend on the tool, so only the removal
            # fractions of the newly added tools have to be recomputed
            print(f"cached table predates {', '.join(missing)}; scoring them", flush=True)
            full, _, _, _ = build_table(args.data_dir)
            cols = [c for t in missing for c in
                    (f"{t}_h", f"{t}_m", f"{t}_noise", f"{t}_rm", f"{t}_sig_ret")]
            df = df.join(full[cols])
            df.to_csv(csv)
    else:
        df, acs, is_h, is_m = build_table(args.data_dir)
        df = df[~df.is_doublet].copy()      # doublets are removed before benchmarking
        con, cell, pos = reference_profiles(df, acs, is_h, is_m)
        df = fit_all(df, acs, is_h, is_m, con, cell, pos)
        synth = synthetic_controls(df, acs, is_h, is_m, con, cell, pos)
        df.to_csv(csv)
        np.savez(npz, **synth)
        print(f"wrote {csv}")
    print("simulated ambient-only: median w = %.3f, %.1f%% called as second cell"
          % (np.nanmedian(synth["ambient"]), 100 * np.nanmean(synth["ambient"] > 0.5)))
    print("simulated doublet:      median w = %.3f, %.1f%% called as second cell"
          % (np.nanmedian(synth["doublet"]), 100 * np.nanmean(synth["doublet"] > 0.5)))
    print("residual doublets: %d / %d (%.2f%%)"
          % (df.residual_doublet.sum(), len(df), 100 * df.residual_doublet.mean()))
    for t in TOOLS:
        f = df[f"{t}_rm"] < 0.5
        print("%-11s failures %4d | clean singlets %.2f%% | residual doublets %.1f%%"
              % (TOOL_LABEL[t], f.sum(), 100 * f[~df.residual_doublet].mean(),
                 100 * f[df.residual_doublet].mean()))

    fig_path = args.figure or os.path.join(args.out_dir, "hgmm_12k_failure_mode_analysis.png")
    make_figure(df, fig_path)


if __name__ == "__main__":
    main()
