"""Assemble the Janssen et al. (2023) snRNA-seq benchmark figure.

Reads the CSVs written by notebooks/janssen_snrna.ipynb plus the published benchmark values
shipped with the Janssen code archive, and writes a six-panel figure.

Usage:
    python scripts/make_janssen_figure.py [--data-dir DIR] [--out-dir DIR] [--out PATH]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPLICATES = ["nuc2", "nuc3"]

# Categorical slots 1-5 of the validated default palette, plus a neutral for the uncorrected data.
METHOD_COLORS = {
    "CellSweep": "#2a78d6",
    "CellBender": "#eb6834",
    "DecontX": "#1baf7a",
    "DecontX (empty)": "#eda100",
    "SoupX": "#4a3aa7",
    "raw": "#9a9a95",
}
METHOD_ORDER = ["CellSweep", "CellBender", "DecontX", "DecontX (empty)", "SoupX"]
REP_COLORS = {"nuc2": "#2a78d6", "nuc3": "#eb6834"}


def _despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _panel_label(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def panel_ground_truth(ax, percell):
    """(A) How much background these nuclei actually carry."""
    for rep in REPLICATES:
        gt = percell[rep]["bRNA"].dropna()
        ax.hist(gt, bins=32, range=(0, 0.8), histtype="step", linewidth=2,
                color=REP_COLORS[rep], label=f"{rep} (median {gt.median():.0%})")
    ax.set_xlabel("genotype-estimated background fraction")
    ax.set_ylabel("nuclei")
    ax.set_title("Background RNA per nucleus", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)


def panel_knee(ax, knee):
    """(B) The shallow knee typical of high-background droplet data."""
    for rep in REPLICATES:
        counts, n_cells = knee[rep]["counts"], int(knee[rep]["n_cells"])
        ax.plot(np.arange(1, len(counts) + 1), counts, linewidth=1.5,
                color=REP_COLORS[rep], label=rep)
        ax.plot(n_cells, counts[n_cells - 1], "o", color=REP_COLORS[rep], ms=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.axhline(100, color="0.35", ls=":", linewidth=1.2)
    ax.text(1.3, 110, "noncellular cutoff (100 UMI)", fontsize=7, color="#52514e", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("barcode rank")
    ax.set_ylabel("UMI counts")
    ax.set_title("Knee plots (dot = last called cell)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _despine(ax)


def panel_scatter(ax, df, rep, accuracy, letter_axis=True):
    """(C, D) CellSweep's per-nucleus estimate against the genotype ground truth."""
    g = df.dropna(subset=["bRNA"])
    ax.scatter(g["bRNA"], g["est"], s=7, alpha=0.3, color=METHOD_COLORS["CellSweep"],
               edgecolors="none", rasterized=True)
    lim = 0.9
    ax.plot([0, lim], [0, lim], color="0.35", ls="--", linewidth=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("genotype ground truth")
    ax.set_ylabel("CellSweep estimate")
    ax.set_title(f"{rep}  ($\\tau$ = {accuracy['tau']:.2f}, RMSLE = {accuracy['rmsle']:.3f}, "
                 f"n = {int(accuracy['n']):,})", fontsize=9)
    _despine(ax)


def _grouped_bars(ax, table, metric, ylabel, title, replicates=REPLICATES,
                  include_raw=False, fmt="{:.2f}"):
    """(E, F) One bar per method, grouped by replicate; every bar is directly labelled."""
    methods = (["raw"] if include_raw else []) + METHOD_ORDER
    methods = [m for m in methods if any((rep, m) in table.index for rep in replicates)]
    x = np.arange(len(replicates))
    width = 0.82 / len(methods)
    top = max(table.loc[(rep, m), metric] for rep in replicates for m in methods
              if (rep, m) in table.index)
    for i, method in enumerate(methods):
        vals = [table.loc[(rep, method), metric] if (rep, method) in table.index else np.nan
                for rep in replicates]
        pos = x - 0.41 + width * (i + 0.5)
        ax.bar(pos, vals, width * 0.84, color=METHOD_COLORS[method], label=method)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(p, v + 0.02 * top, fmt.format(v), ha="center", va="bottom",
                        fontsize=7, color="#52514e", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(replicates)
    ax.set_xlim(-0.6, len(replicates) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.margins(y=0.26)
    _despine(ax)
    return methods


def build_figure(percell, knee, accuracy, estimation_table, marker_table, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)

    panel_ground_truth(axes[0, 0], percell)
    panel_knee(axes[0, 1], knee)
    panel_scatter(axes[0, 2], percell["nuc2"], "nuc2", accuracy["nuc2"])
    panel_scatter(axes[1, 0], percell["nuc3"], "nuc3", accuracy["nuc3"])

    methods = _grouped_bars(axes[1, 1], estimation_table, "tau",
                            "Kendall's $\\tau$ vs. ground truth",
                            "Per-nucleus estimation accuracy")
    # Janssen et al. report marker metrics for nuc2 only, so that panel is nuc2 alone.
    _grouped_bars(axes[1, 2], marker_table, "expression_fraction",
                  "fraction of non-PT nuclei expressing",
                  "PT-marker leakage, nuc2 (lower is better)",
                  replicates=["nuc2"], include_raw=True, fmt="{:.3f}")

    handles = [Patch(facecolor=METHOD_COLORS[m], label=m) for m in ["raw"] + methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.035))

    for ax, letter in zip(axes.ravel(), "ABCDEF"):
        _panel_label(ax, letter)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    print("wrote", out_path)
    return fig


def build_sensitivity_figure(sweep, out_path):
    """Companion figure: how the noncellular-barcode cutoff moves the result."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    for rep in REPLICATES:
        s = sweep[sweep["replicate"] == rep].sort_values("umi")
        knee = s.iloc[-1]
        for ax, col in zip(axes, ["tau", "frac_alpha_above_0p9"]):
            ax.plot(s["umi"], s[col], "o-", color=REP_COLORS[rep], label=rep, ms=6)
            ax.plot(knee["umi"], knee[col], "o", color=REP_COLORS[rep], ms=11,
                    markerfacecolor="none", markeredgewidth=1.6)
    axes[0].set_ylabel("Kendall's $\\tau$ vs. ground truth")
    axes[0].axhline(0, color="0.75", linewidth=0.8, zorder=0)
    axes[1].set_ylabel("fraction of nuclei with $\\alpha_i > 0.9$")
    for ax, letter in zip(axes, "AB"):
        ax.set_xscale("log")
        ax.set_xlabel("noncellular barcode UMI cutoff")
        ax.legend(frameon=False, fontsize=8)
        _despine(ax)
        _panel_label(ax, letter)
    axes[1].text(0.97, 0.06, "open circle = cutoff at the called-cell count",
                 transform=axes[1].transAxes, ha="right", fontsize=7, color="#52514e")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")
    print("wrote", out_path)
    return fig


def load_inputs(data_dir, out_dir):
    percell = {rep: pd.read_csv(os.path.join(out_dir, f"{rep}_percell.csv"), index_col=0)
               for rep in REPLICATES}
    knee = {rep: np.load(os.path.join(out_dir, f"{rep}_knee.npz")) for rep in REPLICATES}
    accuracy = (pd.read_csv(os.path.join(out_dir, "cellsweep_estimation_accuracy.csv"))
                  .set_index("replicate").to_dict("index"))

    bench = pd.read_csv(os.path.join(data_dir, "benchmark_metrics.csv"))
    bench = bench[bench["default"] & bench["replicate"].isin(REPLICATES)].copy()
    bench["method"] = np.where(bench["param"].str.contains("emptyTrue", na=False),
                               bench["method"] + " (empty)", bench["method"])

    estimation_table = (bench[bench["evaluation_category"] == "estimation_accuracy"]
                        .pivot_table(index=["replicate", "method"], columns="metric", values="value"))
    marker_table = (bench[bench["evaluation_category"] == "marker_evaluation"]
                    .pivot_table(index=["replicate", "method"], columns="metric", values="value"))

    cs_markers = pd.read_csv(os.path.join(out_dir, "marker_metrics.csv"))
    cs_markers = cs_markers[cs_markers["method"] == "CellSweep"]
    for _, row in cs_markers.iterrows():
        for metric in ("expression_fraction", "log_ratio_expression", "lfc"):
            marker_table.loc[(row["replicate"], "CellSweep"), metric] = row[metric]
    for rep in REPLICATES:
        estimation_table.loc[(rep, "CellSweep"), "tau"] = accuracy[rep]["tau"]
        estimation_table.loc[(rep, "CellSweep"), "rmsle"] = accuracy[rep]["rmsle"]

    return percell, knee, accuracy, estimation_table.sort_index(), marker_table.sort_index()


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.path.join(here, "notebooks", "data", "janssen2023"))
    p.add_argument("--out-dir", default=os.path.join(here, "notebooks", "output", "janssen2023"))
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_path = args.out or os.path.join(args.out_dir, "janssen_snrna_figure.png")

    percell, knee, accuracy, estimation_table, marker_table = load_inputs(args.data_dir, args.out_dir)
    print(estimation_table.round(3).to_string())
    print(marker_table.round(3).to_string())
    build_figure(percell, knee, accuracy, estimation_table, marker_table, out_path)

    sweep_path = os.path.join(args.out_dir, "empty_cutoff_sweep.csv")
    if os.path.exists(sweep_path):
        build_sensitivity_figure(pd.read_csv(sweep_path),
                                 os.path.join(args.out_dir, "janssen_cutoff_sensitivity.png"))


if __name__ == "__main__":
    main()
