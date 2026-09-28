"""Simulation sensitivity panels (reviewer 1, minor point 8).

Figure 2 reports simulation *specificity* only. This script scores simulation
*sensitivity* (fraction of true signal retained) for every tool on
simulation1_small_noise, using the element-wise confusion components of
cellsweep.utils.evaluate_simulation_denoising on rounded counts:

    TP = min(Yp, Yt)   FP = max(0, Yp - Yt)   FN = max(0, Yt - Yp)   TN = max(0, Yr - TP - FP - FN)
    sensitivity = TP / (TP + FN)        specificity = TN / (TN + FP)

where Yr = raw, Yt = true signal (layers["real"]), Yp = tool output, over the cells
retained by all tools. Global marker-gene specificity reproduces the Figure 2 row.

Writes to --out-dir:
  simulation_sensitivity_metrics.csv    global metrics per tool (marker genes and all genes)
  simulation_sensitivity_celltype.csv   marker-gene sensitivity per tool x cell type
  simulation_sensitivity.png/.pdf       panels A-E

Figure 2 (with the simulation sensitivity row) is drawn by scripts/make_fig2_heatmap.py from the metrics CSV.

Usage: python scripts/make_simulation_sensitivity_figure.py [--out-dir DIR]
"""

import argparse
import os

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import sparse

import cellsweep.utils as cs_utils

CELLSWEEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CELLSWEEP_DIR, "notebooks", "data", "simulation1_small_noise")
OUT_DIR = os.path.join(CELLSWEEP_DIR, "notebooks", "output", "simulation1_small_noise", "sensitivity")

TOOLS = ["CellSweep", "CellBender", "DecontX", "scAR", "SoupX"]  # Figure 2 column order
# Categorical slots of the validated default palette (same assignment as make_janssen_figure.py).
METHOD_COLORS = {"CellSweep": "#2a78d6", "CellBender": "#eb6834", "DecontX": "#1baf7a",
                 "scAR": "#eda100", "SoupX": "#4a3aa7"}
METHOD_MARKERS = {"CellSweep": "o", "CellBender": "s", "DecontX": "D", "scAR": "v", "SoupX": "^"}
INK, MUTED, GRID = "#26251f", "#6f6e69", "#e4e3de"
BLUES = LinearSegmentedColormap.from_list(
    "palette_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                     "#256abf", "#184f95", "#0d366b"])
# Panel A label offsets (points); CellSweep and DecontX sit nearly on top of each other.
LABEL_OFFSETS = {"CellSweep": (-14, 10), "CellBender": (-14, 12), "DecontX": (12, -2), "scAR": (10, -4),
                 "SoupX": (-14, 2)}
FLOOR = 1e-5  # log-axis floor for cells/genes that lose no signal


def load_tools():
    raw = cs_utils.load_adata(os.path.join(DATA_DIR, "adata_raw.h5ad"))
    raw.var_names_make_unique()
    tools = {}
    cs = ad.read_h5ad(os.path.join(DATA_DIR, "sim1_output_cellsweep.h5ad"))
    tools["CellSweep"] = cs[~cs.obs["is_empty"]].copy()
    tools["CellBender"] = sc.read_10x_h5(os.path.join(DATA_DIR, "sim1_output_cellbender_filtered.h5"), gex_only=False)
    tools["SoupX"] = cs_utils.load_adata(os.path.join(DATA_DIR, "sim1_output_soupx"))
    dx = cs_utils.load_adata(os.path.join(DATA_DIR, "sim1_output_decontx"))
    dx.obs_names = [n.replace("GRCh38_", "", 1) for n in dx.obs_names]
    tools["DecontX"] = dx
    tools["scAR"] = ad.read_h5ad(os.path.join(DATA_DIR, "sim1_output_scar.h5ad"))
    for a in tools.values():
        a.var_names_make_unique()
    common = sorted(set.intersection(*[set(a.obs_names) for a in tools.values()]))
    return raw, tools, common


def _rint(X):
    X = sparse.csr_matrix(X, dtype=np.float64)
    X.data = np.rint(X.data)
    X.eliminate_zeros()
    return X


def confusion(Yp, Yt, Yr):
    """Element-wise TP/FP/FN/TN matrices, as in evaluate_simulation_denoising."""
    TP = Yp.minimum(Yt)
    FP = (Yp - Yt).maximum(0)
    FN = (Yt - Yp).maximum(0)
    TN = (Yr - TP - FP - FN).maximum(0)
    return TP, FP, FN, TN


def _sum(M, axis):
    return np.asarray(M.sum(axis=axis)).ravel()


def compute(raw, tools, common):
    raw = raw[common].copy()
    genes = raw.var_names
    Yr, Yt = _rint(raw.X), _rint(raw.layers["real"])
    is_marker = raw.var["is_marker"].to_numpy().astype(bool)
    celltype = raw.obs["celltype"].astype(str).to_numpy()
    ambient = raw.obs["ambient_fraction"].to_numpy()
    true_mean = _sum(Yt, 0) / Yt.shape[0]

    global_rows, cell_rows, gene_rows, ct_rows = [], [], [], []
    for tool in TOOLS:
        a = tools[tool][common, genes]
        Yp = _rint(a.X)
        TP, FP, FN, TN = confusion(Yp, Yt, Yr)
        for scope, cols in (("marker genes", is_marker), ("all genes", slice(None))):
            tp, fp, fn, tn = (M[:, cols].sum() for M in (TP, FP, FN, TN))
            global_rows.append(dict(tool=tool, genes=scope, sensitivity=tp / (tp + fn),
                                    specificity=tn / (tn + fp), ppv=tp / (tp + fp),
                                    TP=tp, FP=fp, FN=fn, TN=tn))
        # per cell, marker genes
        tp_c, fn_c = _sum(TP[:, is_marker], 1), _sum(FN[:, is_marker], 1)
        cell_rows.append(pd.DataFrame(dict(tool=tool, celltype=celltype, ambient_fraction=ambient,
                                           TP=tp_c, FN=fn_c, signal=tp_c + fn_c)))
        # per gene, all genes
        tp_g, fn_g = _sum(TP, 0), _sum(FN, 0)
        gene_rows.append(pd.DataFrame(dict(tool=tool, gene=genes, is_marker=is_marker,
                                           true_mean=true_mean, TP=tp_g, FN=fn_g)))
    cells = pd.concat(cell_rows, ignore_index=True)
    cells["signal_removed"] = cells["FN"] / cells["signal"].clip(lower=1)
    genes_df = pd.concat(gene_rows, ignore_index=True)
    genes_df["signal_removed"] = genes_df["FN"] / (genes_df["TP"] + genes_df["FN"]).clip(lower=1)
    ct = cells.groupby(["tool", "celltype"])[["TP", "FN"]].sum().reset_index()
    ct["sensitivity"] = ct["TP"] / (ct["TP"] + ct["FN"])
    return pd.DataFrame(global_rows), cells, genes_df, ct


# ---------------------------------------------------------------- plotting helpers
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _label(ax, letter):
    ax.text(-0.16, 1.08, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", color=INK)


def panel_global(ax, g):
    for tool in TOOLS:
        for scope, fill in (("marker genes", True), ("all genes", False)):
            r = g[(g.tool == tool) & (g.genes == scope)].iloc[0]
            c = METHOD_COLORS[tool]
            ax.scatter(r.specificity, max(1 - r.sensitivity, FLOOR), s=70, marker=METHOD_MARKERS[tool],
                       facecolor=c if fill else "white", edgecolor=c, linewidth=1.8, zorder=3)
        r = g[(g.tool == tool) & (g.genes == "marker genes")].iloc[0]
        ax.annotate(tool, (r.specificity, max(1 - r.sensitivity, FLOOR)), xytext=LABEL_OFFSETS[tool],
                    textcoords="offset points", fontsize=8, color=INK, ha="right" if LABEL_OFFSETS[tool][0] < 0 else "left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Specificity (fraction of noise removed)", fontsize=9, color=INK)
    ax.set_ylabel("1 − sensitivity\n(fraction of true signal removed)", fontsize=9, color=INK)
    ax.scatter([], [], marker="o", facecolor=MUTED, edgecolor=MUTED, label="marker genes")
    ax.scatter([], [], marker="o", facecolor="white", edgecolor=MUTED, label="all genes")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left")
    ax.text(0.03, 0.03, "better: right and down", transform=ax.transAxes, fontsize=8, color=MUTED)
    _style(ax)


def panel_ecdf(ax, cells):
    for tool in TOOLS:
        v = np.sort(np.maximum(cells.loc[cells.tool == tool, "signal_removed"].to_numpy(), FLOOR))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=METHOD_COLORS[tool], lw=2)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.7, 1)
    ax.set_xlabel(f"Fraction of true marker signal removed per cell\n(cells removing none plotted at {FLOOR:g})",
                  fontsize=9, color=INK)
    ax.set_ylabel("Cumulative fraction of cells", fontsize=9, color=INK)
    _style(ax)


def panel_ambient(ax, cells):
    x0 = cells.loc[cells.tool == TOOLS[0], "ambient_fraction"].to_numpy()
    edges = np.quantile(x0, np.linspace(0, 1, 11))
    for tool in TOOLS:
        d = cells[cells.tool == tool]
        # signal removed per cell = total marker FN / total marker signal within the bin
        idx = np.clip(np.digitize(d.ambient_fraction, edges[1:-1]), 0, len(edges) - 2)
        agg = d.assign(bin=idx).groupby("bin").agg(x=("ambient_fraction", "median"), FN=("FN", "sum"),
                                                   S=("signal", "sum"))
        ax.plot(agg.x, np.maximum(agg.FN / agg.S, FLOOR), color=METHOD_COLORS[tool], lw=2,
                marker=METHOD_MARKERS[tool], ms=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("True ambient fraction of cell (decile median)", fontsize=9, color=INK)
    ax.set_ylabel("Fraction of true marker signal removed", fontsize=9, color=INK)
    _style(ax)


def panel_gene_expression(ax, genes_df):
    g0 = genes_df[genes_df.tool == TOOLS[0]]
    expressed = g0.true_mean > 0
    edges = np.quantile(g0.true_mean[expressed], np.linspace(0, 1, 11))
    for tool in TOOLS:
        d = genes_df[(genes_df.tool == tool) & (genes_df.true_mean > 0)]
        idx = np.clip(np.digitize(d.true_mean, edges[1:-1]), 0, len(edges) - 2)
        agg = d.assign(bin=idx).groupby("bin").agg(x=("true_mean", "median"), FN=("FN", "sum"), TP=("TP", "sum"))
        ax.plot(agg.x, np.maximum(agg.FN / (agg.FN + agg.TP), FLOOR), color=METHOD_COLORS[tool], lw=2,
                marker=METHOD_MARKERS[tool], ms=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True mean expression of gene (decile median, all genes)", fontsize=9, color=INK)
    ax.set_ylabel("Fraction of true signal removed", fontsize=9, color=INK)
    _style(ax)


def panel_celltype(ax, fig, ct):
    order = sorted(ct.celltype.unique(), key=lambda s: int(s.split("_")[-1]))
    lost = ct.assign(lost=1 - ct.sensitivity).pivot(index="tool", columns="celltype", values="lost").loc[TOOLS, order]
    sens = ct.pivot(index="tool", columns="celltype", values="sensitivity").loc[TOOLS, order]
    norm = LogNorm(vmin=1e-4, vmax=0.5)
    im = ax.imshow(np.clip(lost.to_numpy(), 1e-4, 0.5), cmap=BLUES, norm=norm, aspect="auto")
    for i in range(len(TOOLS)):
        for j in range(len(order)):
            v = sens.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if lost.iat[i, j] > 0.01 else INK)
    ax.set_yticks(range(len(TOOLS)), TOOLS, fontsize=8, color=INK)
    ax.set_xticks(range(len(order)), [o.replace("_", " ") for o in order], rotation=45, ha="right",
                  fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Marker-gene sensitivity by cell type (text); color = 1 − sensitivity",
                 fontsize=9, color=INK, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("1 − sensitivity", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    cb.outline.set_visible(False)


def make_sensitivity_figure(g, cells, genes_df, ct, out_dir):
    mpl.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.5, wspace=0.3)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])]
    panel_global(axes[0], g)
    panel_ecdf(axes[1], cells)
    panel_ambient(axes[2], cells)
    panel_gene_expression(axes[3], genes_df)
    panel_celltype(axes[4], fig, ct)
    handles = [plt.Line2D([], [], color=METHOD_COLORS[t], marker=METHOD_MARKERS[t], lw=2, ms=6, label=t)
               for t in TOOLS]
    fig.legend(handles=handles, loc="upper center", ncol=len(TOOLS), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.94))
    for ax, letter in zip(axes, "ABCDE"):
        _label(ax, letter)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"simulation_sensitivity.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--replot", action="store_true", help="reuse cached CSVs instead of recomputing")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {k: os.path.join(args.out_dir, f"simulation_sensitivity_{k}.csv") for k in ("metrics", "cells", "genes", "celltype")}

    if args.replot and all(os.path.exists(v) for v in paths.values()):
        g, cells, genes_df, ct = (pd.read_csv(paths[k]) for k in ("metrics", "cells", "genes", "celltype"))
    else:
        raw, tools, common = load_tools()
        print(f"{len(common)} cells retained by all tools")
        g, cells, genes_df, ct = compute(raw, tools, common)
        for k, df in zip(("metrics", "cells", "genes", "celltype"), (g, cells, genes_df, ct)):
            df.to_csv(paths[k], index=False)

    print(g[["tool", "genes", "sensitivity", "specificity", "ppv"]].round(4).to_string(index=False))
    make_sensitivity_figure(g, cells, genes_df, ct, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
