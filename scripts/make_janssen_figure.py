"""Figures for the Janssen et al. (2023) droplet-based snRNA-seq benchmark.

Reads the CSVs written by scripts/analyze_janssen_markers.py and writes:

  janssen_dotplots.png    six dot plots -- uncorrected plus five correction methods --
                          over proximal-tubule markers and broadly expressed genes,
                          with nucleus types grouped into PT and non-PT
  janssen_removal.png     in non-PT nuclei, PT-marker ("noise") counts removed against
                          own-cell-type marker ("signal") counts removed

Usage: python scripts/make_janssen_figure.py [--out-dir DIR] [--rep nuc2]
"""

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_janssen_markers as jm
import janssen_loaders as L

# Categorical slots 1-5 of the validated default palette, plus a neutral for uncorrected data.
METHOD_COLORS = {
    "CellSweep": "#2a78d6",
    "SoupX": "#4a3aa7",
    "CellBender": "#eb6834",
    "DecontX": "#1baf7a",
    "scAR": "#eda100",
    "CellSweep (no empties)": "#b07aa1",
    "DecontX (empty)": "#e87ba4",
    "raw": "#9a9a95",
}
# Sequential single-hue blue ramp, steps 100 -> 700 of the same palette.
BLUES = LinearSegmentedColormap.from_list(
    "palette_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                     "#256abf", "#184f95", "#0d366b"])
INK, MUTED = "#26251f", "#6f6e69"
PANEL_TITLE = "uncorrected"
# Methods shown in the figures, in the tool order of the PBMC supplementary figure. DecontX (empty)
# and CellSweep (no empties) are still scored by analyze_janssen_markers.py but left out of the
# plots.
FIGURE_METHODS = ["raw", "CellSweep", "SoupX", "CellBender", "DecontX", "scAR"]


def _panel_label(ax, letter):
    ax.text(-0.17, 1.07, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", color=INK)


def _despine(ax):
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def celltype_order(df):
    """PT first, then the non-PT types ordered by abundance."""
    n = df.groupby("celltype")["n_cells"].first()
    non_pt = [t for t in n.sort_values(ascending=False).index if t != "PT"]
    return ["PT"] + non_pt


def scanpy_dotplot(ax, sub, types, panel_genes, title, show_y, show_x):
    """One method's dot plot, drawn by scanpy's DotPlot from the precomputed per-type values:
    size = fraction of nuclei detected, colour = mean expression relative to the peak cell type."""
    import anndata as ad
    import scanpy as sc

    size_df = sub.pivot(index="celltype", columns="gene", values="detected").reindex(index=types, columns=panel_genes)
    color_df = sub.pivot(index="celltype", columns="gene", values="rel_expr").reindex(index=types, columns=panel_genes)
    # DotPlot needs an AnnData to lay out groups and genes; the values themselves come from the two frames
    dummy = ad.AnnData(np.zeros((len(types), len(panel_genes)), dtype=np.float32),
                       obs=pd.DataFrame({"celltype": pd.Categorical(types, categories=types)}, index=types),
                       var=pd.DataFrame(index=panel_genes))
    n_pt = len(L.PT_MARKERS)
    dp = sc.pl.DotPlot(dummy, panel_genes, groupby="celltype", categories_order=types, ax=ax,
                       dot_size_df=size_df, dot_color_df=color_df, vmin=0, vmax=1,
                       var_group_positions=[(0, n_pt - 1), (n_pt, len(panel_genes) - 1)],
                       var_group_labels=["PT markers", "broadly expressed"], var_group_rotation=0)
    # size_exponent=1 keeps dot area proportional to the detected fraction; dot_min/max fixed so all panels share one scale
    dp.style(cmap="Reds", dot_min=0, dot_max=1, size_exponent=1, smallest_dot=0, largest_dot=DOT_MAX_SIZE)
    dp.legend(show=False)
    dp.make_figure()
    main_ax = dp.ax_dict["mainplot_ax"]
    main_ax.set_xticklabels(panel_genes if show_x else [], rotation=90, fontsize=7, style="italic")
    main_ax.tick_params(axis="x", length=3 if show_x else 0)
    main_ax.set_yticklabels(types if show_y else [], fontsize=7.5)
    main_ax.tick_params(axis="y", length=3 if show_y else 0)
    dp.ax_dict["gene_group_ax"].set_title(title, fontsize=10.5, pad=12,
                                          fontweight="bold" if title == "CellSweep" else "normal")
    for t in dp.ax_dict["gene_group_ax"].texts:
        t.set_fontsize(8)
    return dp


DOT_MAX_SIZE = 110


def build_dotplots(df, rep, out_path):
    types = celltype_order(df)
    panel_genes = L.PT_MARKERS + L.CONSTITUTIVE
    methods = [m for m in FIGURE_METHODS if m in set(df["method"])]
    # Colour is expression relative to the highest level that gene reaches in any nucleus
    # type under any method -- almost always PT in the uncorrected panel. Absolute CP10K
    # would be swamped by Malat1 and leave every marker unreadable.
    df = df.copy()
    df["rel_expr"] = df["mean_cp10k"] / df.groupby("gene")["mean_cp10k"].transform("max")

    ncol = 3
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.3 * nrow + 0.6), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.885, top=0.93, bottom=0.12, wspace=0.12, hspace=0.08)
    dp = None
    for i, method in enumerate(methods):
        ax = axes[i // ncol][i % ncol]
        dp = scanpy_dotplot(ax, df[df["method"] == method], types, panel_genes,
                            PANEL_TITLE if method == "raw" else method,
                            show_y=(i % ncol == 0), show_x=(i + ncol >= len(methods)))
        if i % ncol == 0:
            # PT / non-PT row groups, bracketed to the left of the cell-type names
            main_ax = dp.ax_dict["mainplot_ax"]
            trans = main_ax.get_yaxis_transform()
            for lo, hi, label in [(0, 0, "PT"), (1, len(types) - 1, "non-PT")]:
                main_ax.plot([-0.2, -0.22, -0.22, -0.2], [lo - 0.3, lo - 0.3, hi + 0.3, hi + 0.3], transform=trans,
                             color="black", lw=0.8, clip_on=False)
                main_ax.text(-0.25, (lo + hi) / 2, label, transform=trans, rotation=90, ha="center", va="center",
                             fontsize=8.5)
    for j in range(len(methods), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    # one shared legend, drawn with scanpy's own size-legend and colorbar code
    dp.size_title = "Fraction of nuclei\nin group (%)"
    dp.color_legend_title = "Mean expression,\nrelative to peak cell type"
    dp._plot_size_legend(fig.add_axes([0.9, 0.36, 0.09, 0.1]))
    dp._plot_colorbar(fig.add_axes([0.9, 0.58, 0.075, 0.018]), Normalize(vmin=0, vmax=1))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    print("wrote", out_path)


def build_removal(removals, out_path):
    """Noise removed against signal removed, one panel per replicate."""
    reps = list(removals)
    fig, axes = plt.subplots(1, len(reps), figsize=(4.6 * len(reps), 4.3), squeeze=False)
    for letter, (ax, rep) in zip("ABCD", zip(axes[0], reps)):
        df = removals[rep]
        agg = jm.summarise(df)
        ax.axhline(1.0, color="#d8d7d2", lw=1, ls="--")
        ax.axvline(0.0, color="#d8d7d2", lw=1, ls="--")
        ax.plot(0, 1, marker="*", ms=16, color="#b8b7b2", zorder=1)
        ax.annotate("ideal", (0, 1), textcoords="offset points", xytext=(9, 11),
                    fontsize=8, color=MUTED, va="center")
        pts = [(method, agg.loc[method, "own-type signal"], agg.loc[method, "PT noise"])
               for method in FIGURE_METHODS if method in agg.index]
        for method, x, y in pts:
            ax.plot(x, y, "o", ms=10, color=METHOD_COLORS[method], zorder=3,
                    markeredgecolor="white", markeredgewidth=1.2)
        # Methods can land on top of one another, so labels are pushed apart vertically and
        # tied back to their point with a leader line.
        lo, hi = ax.get_ylim()
        gap = 0.062 * (hi - lo)
        placed = []
        for method, x, y in sorted(pts, key=lambda t: -t[2]):
            ly = y if not placed else min(y, placed[-1] - gap)
            placed.append(ly)
            if abs(ly - y) > 1e-9:
                ax.plot([x, x + 0.012 * (ax.get_xlim()[1] - ax.get_xlim()[0])], [y, ly],
                        color="#c9c8c3", lw=0.8, zorder=2)
            ax.annotate(method, (x, ly), textcoords="offset points", xytext=(11, 0),
                        fontsize=8.5, color=INK, va="center",
                        fontweight="bold" if method == "CellSweep" else "normal")
        ax.set_xlabel("own-cell-type marker counts removed\n(signal: lower is better)",
                      fontsize=9)
        ax.set_ylabel("PT marker counts removed\n(background: higher is better)", fontsize=9)
        ax.set_title(rep, fontsize=10.5, color=INK)
        _panel_label(ax, letter)
        ax.set_xlim(-0.045, max(0.28, agg["own-type signal"].max() * 1.5))
        ax.set_ylim(-0.05, 1.08)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    print("wrote", out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=L.OUT_DIR)
    p.add_argument("--rep", default="nuc2", help="replicate shown in the dot-plot figure")
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.out_dir, f"dotplot_{args.rep}.csv"))
    build_dotplots(df, args.rep, os.path.join(args.out_dir, "janssen_dotplots.png"))

    removals = {}
    for rep in L.REPLICATES:
        path = os.path.join(args.out_dir, f"removal_{rep}.csv")
        if os.path.exists(path):
            removals[rep] = pd.read_csv(path)
    if removals:
        build_removal(removals, os.path.join(args.out_dir, "janssen_removal.png"))
        for rep, d in removals.items():
            print(f"\n=== {rep}: mean fraction removed in non-PT nuclei ===")
            print(jm.summarise(d).round(3).to_string())


if __name__ == "__main__":
    main()
