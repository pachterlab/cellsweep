"""Figure 2: performance heatmap across ambient RNA correction methods.

Rows are normalized scores (sensitivity = fraction of signal retained, specificity =
fraction of noise removed) per benchmark dataset, followed by idempotency and CPU runtime.
Cells are colored on RdYlGn clipped to [0.5, 1].

The simulation rows are read from the marker-gene rows of
simulation_sensitivity_metrics.csv, written by scripts/make_simulation_sensitivity_figure.py.
The remaining rows are the values reported in the manuscript for the human-mouse
mixture, PBMC 8k, and 8-cubed benchmarks (see notebooks/benchmarking.ipynb and
scripts/visualize_8cube.py).

Usage: python scripts/make_fig2_heatmap.py [--metrics CSV] [--out PATH_WITHOUT_EXT]
"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CELLSWEEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_METRICS = os.path.join(CELLSWEEP_DIR, "notebooks", "output", "simulation1_small_noise", "sensitivity",
                           "simulation_sensitivity_metrics.csv")
OUT = os.path.join(CELLSWEEP_DIR, "notebooks", "output", "Fig2")

TOOLS = ["CellSweep", "CellBender", "DecontX", "scAR", "SoupX"]
NA = np.nan
OTHER_ROWS = [
    ("Human-Mouse sensitivity", [0.981, 0.991, 0.981, 0.975, 0.983]),
    ("Human-Mouse specificity", [0.987, 0.956, 0.712, 0.987, 0.734]),
    ("PBMC sensitivity", [0.949, 0.997, 0.940, 0.854, 0.977]),
    ("PBMC specificity", [0.913, 0.704, 0.913, 0.768, 0.839]),
    ("8 cubed sensitivity", [0.975, 0.986, 0.970, NA, 0.987]),
    ("8 cubed specificity", [0.914, 0.918, 0.898, NA, 0.404]),
]
IDEMPOTENT = ["Yes", "No", "Yes", "No", "Yes"]
RUNTIME_MIN = [1, 180, 3, 200, 2]
FAST_MIN = 3  # runtimes at or below this are colored good, otherwise bad


def simulation_rows(metrics_csv):
    m = pd.read_csv(metrics_csv)
    m = m[m["genes"] == "marker genes"].set_index("tool")
    return [
        ("Simulation sensitivity", [m.loc[t, "sensitivity"] for t in TOOLS]),
        ("Simulation specificity", [m.loc[t, "specificity"] for t in TOOLS]),
    ]


def make_heatmap(rows, out_base):
    cmap = plt.get_cmap("RdYlGn")
    norm = mpl.colors.Normalize(vmin=0.5, vmax=1.0)
    n_rows = len(rows) + 2

    fig, ax = plt.subplots(figsize=(7, 0.62 * n_rows + 0.8))

    def cell(i, j, color, text, text_color):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, ec="white", lw=2))
        ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=10, color=text_color)

    for i, (_, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            if np.isnan(v):
                cell(i, j, "lightgray", "N/A", "black")
            else:
                cell(i, j, cmap(norm(max(v, 0.5))), f"{v:.3f}", "white" if (v >= 0.9 or v < 0.6) else "black")
    i = len(rows)
    for j, v in enumerate(IDEMPOTENT):
        cell(i, j, cmap(norm(1.0 if v == "Yes" else 0.5)), v, "white")
    i += 1
    for j, v in enumerate(RUNTIME_MIN):
        cell(i, j, cmap(norm(0.98 if v <= FAST_MIN else 0.5)), str(v), "white")

    ax.set_xlim(0, len(TOOLS))
    ax.set_ylim(n_rows, 0)
    ax.set_yticks(np.arange(n_rows) + 0.5, [r[0] for r in rows] + ["Idempotent", "Runtime on CPU (min)"])
    ax.set_xticks(np.arange(len(TOOLS)) + 0.5, TOOLS, rotation=30, ha="left")
    ax.xaxis.tick_top()

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, shrink=0.8)
    cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=["≤0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
    cb.set_label("Normalized score")

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_base}.png and {out_base}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", default=SIM_METRICS, help="simulation_sensitivity_metrics.csv")
    p.add_argument("--out", default=OUT, help="output path without extension")
    args = p.parse_args()
    make_heatmap(simulation_rows(args.metrics) + OTHER_ROWS, args.out)


if __name__ == "__main__":
    main()
