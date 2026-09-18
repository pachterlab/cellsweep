"""Marker-level evaluation of background removal on the Janssen et al. (2023) snRNA-seq data.

Writes, for each replicate:
  dotplot_<rep>.csv   per cell type x gene detection fraction and mean expression, per method
  removal_<rep>.csv   in non-PT nuclei, the fraction of PT-marker ("noise") counts removed
                      and the fraction of own-cell-type marker ("signal") counts removed
  accuracy_<rep>.csv  Kendall's tau and RMSLE of each method's per-nucleus estimate against
                      the genotype ground truth
  retention_<rep>.csv how much each method removes from counts that are mostly genuine signal:
                      PT markers inside PT nuclei, and broadly expressed genes everywhere

Usage: python scripts/analyze_janssen_markers.py [--methods m1,m2,...]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import janssen_loaders as L

MIN_CELLS = 30


def cp10k(m):
    tot = np.asarray(m.sum(axis=1)).ravel()
    return sp.diags(1e4 / np.maximum(tot, 1e-9)) @ m


def dotplot_table(m, genes, celltype, panel, method, raw_total):
    """Detection fraction and mean expression per cell type.

    Detection uses Janssen's rounded-count convention. Expression is normalised by each
    nucleus's *uncorrected* total so that a gene's value is comparable across methods:
    re-normalising by the corrected total would inflate whatever a method leaves behind.
    """
    gi = pd.Index(genes).get_indexer(panel)
    sub = m[:, gi]
    norm = sp.diags(1e4 / np.maximum(raw_total, 1e-9)) @ sub
    rows = []
    for t in pd.unique(celltype):
        mask = celltype == t
        if mask.sum() < MIN_CELLS:
            continue
        counts = sub[mask].toarray()
        rows.append(pd.DataFrame({
            "method": method, "celltype": t, "n_cells": int(mask.sum()), "gene": panel,
            "detected": (np.round(counts) > 0).mean(axis=0),
            "mean_cp10k": np.asarray(norm[mask].mean(axis=0)).ravel(),
        }))
    return pd.concat(rows, ignore_index=True)


def removal_table(raw, cor, genes, celltype, pt_markers, markers_by_type, method):
    """Fraction of counts removed from non-PT nuclei, split into noise and signal."""
    idx = pd.Index(genes)
    pt_i = idx.get_indexer(pt_markers)
    rows = []
    for t, own in markers_by_type.items():
        mask = celltype == t
        if mask.sum() < MIN_CELLS:
            continue
        own_i = idx.get_indexer([g for g in own if g in idx])
        for label, cols in (("PT noise", pt_i), ("own-type signal", own_i)):
            r = np.asarray(raw[mask][:, cols].sum(axis=1)).ravel()
            k = np.asarray(cor[mask][:, cols].sum(axis=1)).ravel()
            keep = r > 0
            rows.append({
                "method": method, "celltype": t, "gene_set": label,
                "n_cells": int(keep.sum()),
                # Mean over nuclei of the per-nucleus fraction removed, and the pooled value.
                "frac_removed_mean": float(np.mean(1 - k[keep] / r[keep])),
                "frac_removed_pooled": float(1 - k[keep].sum() / r[keep].sum()),
                "raw_counts": float(r.sum()),
            })
    return pd.DataFrame(rows)


def retention_table(raw, cor, genes, celltype, method):
    """What each method takes out of counts that are mostly genuine signal.

    A nucleus in this data is ~33% (nuc2) or ~18% (nuc3) background, and the background is
    PT RNA, so a calibrated method should remove about that fraction of a PT nucleus's own
    PT-marker counts, and about that fraction of the broadly expressed genes everywhere.
    Removing much more is over-correction that the non-PT marker comparison cannot see.
    """
    idx = pd.Index(genes)
    is_pt = celltype == "PT"
    rows = []
    for label, mask, panel in (("PT markers in PT nuclei", is_pt, L.PT_MARKERS),
                               ("broadly expressed, all nuclei",
                                np.ones(len(celltype), bool), L.CONSTITUTIVE)):
        cols = idx.get_indexer(panel)
        r = np.asarray(raw[mask][:, cols].sum(axis=1)).ravel()
        k = np.asarray(cor[mask][:, cols].sum(axis=1)).ravel()
        keep = r > 0
        rows.append({"method": method, "gene_set": label, "n_cells": int(keep.sum()),
                     "frac_removed_mean": float(np.mean(1 - k[keep] / r[keep]))})
    return pd.DataFrame(rows)


def accuracy_row(raw, cor, bRNA, method):
    """Kendall's tau and RMSLE against the genotype ground truth, Janssen et al.'s definitions.

    Their per-nucleus estimate is the fraction of a nucleus's counts a method removes
    (`1 - colSums(corrected) / colSums(uncorrected)`, mitochondrial genes already excluded),
    scored against the binomial cross-genotype estimate on the CAST nuclei.
    """
    keep = np.isfinite(bRNA)
    r = np.asarray(raw[keep].sum(axis=1)).ravel()
    k = np.asarray(cor[keep].sum(axis=1)).ravel()
    est = 1 - k / np.maximum(r, 1e-9)
    gt = bRNA[keep]
    return {"method": method, "n": int(keep.sum()),
            "tau": float(kendalltau(gt, est).statistic),
            "rmsle": float(np.sqrt(np.mean((np.log1p(gt) - np.log1p(est)) ** 2))),
            "median_estimate": float(np.median(est)),
            "median_ground_truth": float(np.median(gt))}


def summarise(df):
    """Mean over all non-PT nuclei (cell-type rows weighted by how many nuclei they hold)."""
    w = df.assign(num=lambda d: d["frac_removed_mean"] * d["n_cells"])
    g = w.groupby(["method", "gene_set"])[["num", "n_cells"]].sum()
    return (g["num"] / g["n_cells"]).unstack()


def run(methods=None, out_dir=None):
    """Write dotplot_<rep>.csv and removal_<rep>.csv for every method that is available."""
    methods = methods or list(L.METHOD_ORDER)
    out_dir = out_dir or L.OUT_DIR

    # Marker panels are defined once, on the deeper replicate, and reused for both so the
    # two panels of the figure are directly comparable.
    genes2, cells2 = L.axes("nuc2")
    md2 = L.metadata("nuc2")
    raw2 = L.load_method("nuc2", "raw", genes2, cells2)
    ct2 = md2["celltype"].values
    norm2 = cp10k(raw2)
    mean_expr = pd.DataFrame(
        {t: np.asarray(norm2[ct2 == t].mean(axis=0)).ravel()
         for t in pd.unique(ct2) if (ct2 == t).sum() >= MIN_CELLS}, index=genes2)
    markers_by_type = L.celltype_markers(mean_expr)
    pd.Series({t: ",".join(v) for t, v in markers_by_type.items()},
              name="markers").to_csv(os.path.join(out_dir, "celltype_markers.csv"))
    print("cell-type marker panels:")
    for t, v in markers_by_type.items():
        print(f"  {t:8s} {', '.join(v)}")

    panel = L.PT_MARKERS + L.CONSTITUTIVE
    for rep in L.REPLICATES:
        genes, cells = L.axes(rep)
        md = L.metadata(rep)
        ct = md["celltype"].values
        raw = raw2 if rep == "nuc2" else L.load_method(rep, "raw", genes, cells)
        raw_total = np.asarray(raw.sum(axis=1)).ravel()
        bRNA = md["bRNA"].values.astype(float)
        dots, rems, accs, rets = [], [], [], []
        for method in methods:
            try:
                m = raw if method == "raw" else L.load_method(rep, method, genes, cells)
            except (FileNotFoundError, OSError) as exc:
                print(f"  [{rep}] skipping {method}: {exc}")
                continue
            dots.append(dotplot_table(m, genes, ct, panel, method, raw_total))
            if method != "raw":
                rems.append(removal_table(raw, m, genes, ct, L.PT_MARKERS,
                                          markers_by_type, method))
                accs.append(accuracy_row(raw, m, bRNA, method))
                rets.append(retention_table(raw, m, genes, ct, method))
            print(f"  [{rep}] {method}: kept {m.sum() / raw.sum():.3f} of counts")
        pd.concat(dots, ignore_index=True).to_csv(
            os.path.join(out_dir, f"dotplot_{rep}.csv"), index=False)
        if rems:
            pd.concat(rems, ignore_index=True).to_csv(
                os.path.join(out_dir, f"removal_{rep}.csv"), index=False)
        if rets:
            pd.concat(rets, ignore_index=True).to_csv(
                os.path.join(out_dir, f"retention_{rep}.csv"), index=False)
        if accs:
            acc = pd.DataFrame(accs)
            acc.to_csv(os.path.join(out_dir, f"accuracy_{rep}.csv"), index=False)
            print(acc.round(3).to_string(index=False))
    print("done")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default=",".join(L.METHOD_ORDER))
    p.add_argument("--out-dir", default=L.OUT_DIR)
    args = p.parse_args()
    run(args.methods.split(","), args.out_dir)


if __name__ == "__main__":
    main()
