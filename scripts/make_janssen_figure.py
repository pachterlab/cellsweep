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
    "CellBender": "#eb6834",
    "DecontX": "#1baf7a",
    "DecontX (empty)": "#eda100",
    "SoupX": "#4a3aa7",
    "raw": "#9a9a95",
}
# Sequential single-hue blue ramp, steps 100 -> 700 of the same palette.
BLUES = LinearSegmentedColormap.from_list(
    "palette_blue", ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                     "#256abf", "#184f95", "#0d366b"])
INK, MUTED = "#26251f", "#6f6e69"
PANEL_TITLE = "uncorrected"


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


def dot_panel(ax, sub, types, panel_genes, norm, size_scale, show_y, show_x):
    """One method's dot plot: size = fraction of nuclei detected, colour = mean expression."""
    piv_d = sub.pivot(index="celltype", columns="gene", values="detected").reindex(
        index=types, columns=panel_genes)
    piv_e = sub.pivot(index="celltype", columns="gene", values="rel_expr").reindex(
        index=types, columns=panel_genes)
    xs, ys = np.meshgrid(np.arange(len(panel_genes)), np.arange(len(types)))
    ax.scatter(xs.ravel(), ys.ravel(), s=piv_d.values.ravel() * size_scale,
               c=piv_e.values.ravel(), cmap=BLUES, norm=norm,
               linewidths=0.4, edgecolors="#ffffff")
    ax.set_xlim(-0.8, len(panel_genes) - 0.2)
    ax.set_ylim(len(types) - 0.5, -0.8)
    ax.set_xticks(np.arange(len(panel_genes)))
    ax.set_yticks(np.arange(len(types)))
    ax.set_xticklabels(panel_genes if show_x else [], rotation=90, fontsize=7, style="italic")
    ax.set_yticklabels(types if show_y else [], fontsize=7.5)
    ax.tick_params(length=0, colors=INK)
    # Divider between the PT-marker block and the constitutive block, and between PT and non-PT.
    ax.axvline(len(L.PT_MARKERS) - 0.5, color="#d8d7d2", lw=1.0)
    ax.axhline(0.5, color="#d8d7d2", lw=1.0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#f0efec", lw=0.8)
    _despine(ax)


def build_dotplots(df, rep, out_path):
    types = celltype_order(df)
    panel_genes = L.PT_MARKERS + L.CONSTITUTIVE
    methods = [m for m in L.METHOD_ORDER if m in set(df["method"])]
    # Colour is expression relative to the highest level that gene reaches in any nucleus
    # type under any method -- almost always PT in the uncorrected panel. Absolute CP10K
    # would be swamped by Malat1 and leave every marker unreadable.
    df = df.copy()
    df["rel_expr"] = df["mean_cp10k"] / df.groupby("gene")["mean_cp10k"].transform("max")
    norm = Normalize(vmin=0, vmax=1)
    size_scale = 105.0

    ncol = 3
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.15 * nrow + 0.9),
                             squeeze=False)
    for i, method in enumerate(methods):
        ax = axes[i // ncol][i % ncol]
        dot_panel(ax, df[df["method"] == method], types, panel_genes, norm, size_scale,
                  show_y=(i % ncol == 0), show_x=(i + ncol >= len(methods)))
        label = PANEL_TITLE if method == "raw" else method
        ax.set_title(label, fontsize=10.5, color=INK, pad=26,
                     fontweight="bold" if method == "CellSweep" else "normal")
    for j in range(len(methods), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    # Gene-block and nucleus-group labels, drawn once on the top-left panel.
    for ax in axes[0]:
        if not ax.get_visible():
            continue
        ax.text((len(L.PT_MARKERS) - 1) / 2, -1.45, "PT markers", ha="center", fontsize=8,
                color=MUTED)
        ax.text(len(L.PT_MARKERS) + (len(L.CONSTITUTIVE) - 1) / 2, -1.45,
                "broadly expressed", ha="center", fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        if not ax.get_visible():
            continue
        ax.text(-0.29, 0, "PT", transform=ax.get_yaxis_transform(), rotation=90,
                ha="center", va="center", fontsize=8.5, color=MUTED, fontweight="bold")
        ax.text(-0.29, (len(types) + 1) / 2, "non-PT", transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=8.5, color=MUTED,
                fontweight="bold")

    fig.subplots_adjust(left=0.11, right=0.895, top=0.88, bottom=0.15, wspace=0.16,
                        hspace=0.45)
    cax = fig.add_axes([0.915, 0.55, 0.011, 0.28])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=BLUES), cax=cax)
    cb.set_label("mean expression,\nrelative to peak cell type", fontsize=7.5, color=INK)
    cb.ax.tick_params(labelsize=7, length=2, colors=INK)
    cb.outline.set_visible(False)

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="#6da7ec",
                      markeredgecolor="white", markersize=np.sqrt(f * size_scale),
                      label=f"{f:.0%}") for f in (0.25, 0.5, 0.75, 1.0)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.9, 0.24), frameon=False,
               fontsize=7.5, labelspacing=1.05, handletextpad=0.9,
               title="nuclei detecting", title_fontsize=7.5)
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
               for method in L.METHOD_ORDER if method in agg.index]
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
