#!/usr/bin/env python
"""
Two single-panel figures for the reviewer point on cell-type label granularity:

  celltype_granularity_signal_vs_noise.{png,pdf}   x = number of cell-type labels
  empty_barcode_signal_vs_noise.{png,pdf}          x = number of empty barcodes

Both plot the same two quantities, measured on lineage marker genes with a fixed reference
annotation (CellTypist Immune_All_Low collapsed to major populations):

  signal retained = fraction of marker counts kept in the cells that express the marker
                    (solid line, filled circles)
  noise retained  = fraction of marker counts kept in cells of other lineages, where those
                    counts are ambient (dashed line, open circles)

A good run keeps signal near 1 and pushes noise toward 0. Each figure's legend is written to a
separate <name>_legend.{png,pdf}. The CSV also carries per-programme series (CD16 monocyte, NK)
that are not drawn in these figures.

Run scripts/run_celltype_granularity_sensitivity.py and scripts/run_empty_barcode_sweep.py first.
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_celltype_granularity_sensitivity import (REF_GROUP, LINEAGE_PANELS, T, B, MONO, MYELOID, ALL, labels, run_dir, out_dir)

empty_run_dir = os.path.join(os.path.dirname(run_dir), "empty_barcode_sweep")
EMPTY_TAGS = ["10", "100", "1000", "10000", "50000", "100000", "500000", "all"]
N_EMPTY_ALL = 728899

# coloured examples: markers of a subpopulation that sits inside a coarser label
EXAMPLES = {
    "CD16 monocyte program (FCGR3A, CDKN1C, MS4A7)": (["FCGR3A", "CDKN1C", "MS4A7"], {"CD16 Mono"}, T | B, "#4C72B0"),
    "NK program (NKG7, GNLY, KLRF1)": (["NKG7", "GNLY", "KLRF1"], {"NK"}, B | MYELOID, "#C44E52"),
}


def series(path, ref):
    """Fraction of marker counts retained, in expressing cells (signal) and in other cells (noise)."""
    a = ad.read_h5ad(path)
    a = a[~a.obs["is_empty"].values].copy()
    a.var_names_make_unique()
    a = a[labels.index]
    raw = sp.csr_matrix(a.layers["raw"], dtype=np.float64).tocsc()
    den = sp.csr_matrix(a.X, dtype=np.float64).tocsc()
    gidx = {g: i for i, g in enumerate(a.var_names)}

    out = {}
    sig_raw = sig_den = noi_raw = noi_den = 0.0
    for panel, (genes, on, off) in LINEAGE_PANELS.items():
        cols = [gidx[g] for g in genes]
        r, d = raw[:, cols], den[:, cols]
        on_mask, off_mask = np.isin(ref, list(on)), np.isin(ref, list(off))
        noi_raw += r[off_mask].sum(); noi_den += d[off_mask].sum()
        if on_mask.sum():
            sig_raw += r[on_mask].sum(); sig_den += d[on_mask].sum()
    out["signal_retained"] = sig_den / sig_raw
    out["noise_retained"] = noi_den / noi_raw

    for name, (genes, on, off, _color) in EXAMPLES.items():
        cols = [gidx[g] for g in genes]
        r, d = raw[:, cols], den[:, cols]
        on_mask, off_mask = np.isin(ref, list(on)), np.isin(ref, list(off))
        out[f"{name}|signal_retained"] = d[on_mask].sum() / r[on_mask].sum()
        out[f"{name}|noise_retained"] = d[off_mask].sum() / r[off_mask].sum()
    return out


def collect():
    ref = labels["ct_low"].map(REF_GROUP).values
    rows = []
    for cond in labels.columns:
        print(f"scoring labels: {cond}", flush=True)
        row = dict(sweep="labels", condition=cond, x=labels[cond].nunique())
        row.update(series(os.path.join(run_dir, f"adata_cellsweep_{cond}.h5ad"), ref))
        rows.append(row)
    for tag in EMPTY_TAGS:
        print(f"scoring empty barcodes: {tag}", flush=True)
        row = dict(sweep="empty", condition=tag, x=N_EMPTY_ALL if tag == "all" else int(tag))
        row.update(series(os.path.join(empty_run_dir, f"adata_cellsweep_empty_{tag}.h5ad"), ref))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "signal_vs_noise_retained.csv"), index=False)
    return df


def draw(ax, df, color="k", label_prefix="", sig_col="signal_retained", noi_col="noise_retained"):
    df = df.sort_values("x")
    ax.plot(df["x"], df[sig_col], "-", color=color, lw=1.2, marker="o", ms=5, mfc=color, mec=color, label=f"{label_prefix}signal retained")
    ax.plot(df["x"], df[noi_col], "--", color=color, lw=1.2, marker="o", ms=5, mfc="none", mec=color, label=f"{label_prefix}noise retained")


def finish(ax, xlabel):
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction of counts retained")
    ax.axhline(1, color="0.85", lw=0.8, zorder=0)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name, tight_bbox=False):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight" if tight_bbox else None)
    plt.close(fig)


def save_legend(handles, labels, name):
    """Legend as its own file, one entry per line."""
    fig = plt.figure(figsize=(3.4, 0.32 * len(labels) + 0.2))
    fig.legend(handles, labels, loc="center", frameon=False, fontsize=8, ncol=1)
    save(fig, name, tight_bbox=True)


def figure_labels(df):
    d = df[df.sweep == "labels"].set_index("condition")
    leiden = d.loc[[c for c in d.index if c.startswith("leiden")]]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    draw(ax, leiden)

    # CellTypist annotations, plotted as squares (High) and diamonds (Low)
    for cond, marker, name in [("ct_high", "s", "CellTypist High"), ("ct_low", "D", "CellTypist Low")]:
        row = d.loc[cond]
        ax.scatter(row["x"], row["signal_retained"], marker=marker, s=55, color="k", zorder=5, label=f"{name}: signal retained")
        ax.scatter(row["x"], row["noise_retained"], marker=marker, s=55, facecolor="none", edgecolor="k", lw=1.2, zorder=5, label=f"{name}: noise retained")
        ax.annotate(name, (row["x"], row["noise_retained"]), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=6.5)

    finish(ax, "Number of celltypes")
    handles, labels_ = ax.get_legend_handles_labels()
    fig.tight_layout()
    save(fig, "celltype_granularity_signal_vs_noise")
    save_legend(handles, labels_, "celltype_granularity_signal_vs_noise_legend")


def figure_empty(df):
    d = df[df.sweep == "empty"]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    draw(ax, d)
    ax.axvline(N_EMPTY_ALL, color="0.6", ls=":", lw=1)
    ax.annotate(f"all {N_EMPTY_ALL:,}", (N_EMPTY_ALL, 0.5), rotation=90, va="center", ha="right", fontsize=6.5, color="0.35")
    finish(ax, "Number of empty barcodes")
    handles, labels_ = ax.get_legend_handles_labels()
    fig.tight_layout()
    save(fig, "empty_barcode_signal_vs_noise")
    save_legend(handles, labels_, "empty_barcode_signal_vs_noise_legend")


if __name__ == "__main__":
    csv = os.path.join(out_dir, "signal_vs_noise_retained.csv")
    df = pd.read_csv(csv) if (os.path.exists(csv) and "--reuse" in sys.argv) else collect()
    figure_labels(df)
    figure_empty(df)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
