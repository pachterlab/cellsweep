#!/usr/bin/env python
"""
Reviewer point 7: supplementary figure for the standard (non-xenograft) Visium HD 3' Mouse Brain dataset,
from the run in scripts/run_spatial_mouse_brain.py and the analysis in scripts/analyze_spatial_mouse_brain_markers.py.

Panels, composed into notebooks/output/visium_mouse_brain/mouse_brain_supplement.png:
    A  knee plot of barcodes ranked by UMI count
    B  histogram of alpha_hat over non-empty bins
    C  spatial map of binned alpha_hat
    D  mean counts per bin against distance from the source region, raw vs cellsweep, for three
       spatially restricted genes (Ttr, Pmch, Hcrt)

Panel D recomputes the source regions exactly as analyze_spatial_mouse_brain_markers.py does: the raw counts of a
gene are smoothed (Gaussian, SIGMA_BINS), the "source" is the set of bins holding the top 50% of the smoothed mass
dilated by SOURCE_DILATE_UM, and distance is the Euclidean distance transform away from that core. Counts far from
the source cannot have come from the source cells, so preferential removal there is the signature we are after.

Usage: python scripts/make_spatial_mouse_brain_figure.py
"""
import os
import json

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import matplotlib.patheffects as pe
import seaborn as sns

import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "visium_mouse_brain")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "visium_mouse_brain")
panel_dir = os.path.join(out_dir, "panels")
os.makedirs(panel_dir, exist_ok=True)

EXPECTED_CELLS = 350_000
BIN_UM = 8
SIGMA_BINS = 3
SOURCE_DILATE_UM = 40
FAR_UM = 400
DIST_EDGES = [0, 40, 100, 200, 400, 800, 1600, np.inf]
GENES = {"Ttr": "#d95f02", "Pmch": "#7570b3", "Hcrt": "#1b9e77"}
ALPHA_BINS = [0, 0.25, 0.5, 0.75, 1.0]
ALPHA_LABELS = [f"{ALPHA_BINS[i]}-{ALPHA_BINS[i + 1]}" for i in range(len(ALPHA_BINS) - 1)]
PANEL_H = 1000   # px; every panel is rendered at this height so the rows line up

# Anatomical regions the Leiden clusters identify unambiguously, with the label offset in bins from the
# region centroid. Marker support: cortex Nrgn/Neurod6/Slc17a7; striatum Drd1/Adora2a/Ppp1r1b/Penk;
# thalamus Prkcd/Tcf7l2; hypothalamus Hcrt/Pmch/Agrp; white matter Mobp/Plp1/Mal/Mag; choroid plexus Ttr/Folr1.
# At this clustering resolution cortex and hippocampus are one cluster (22 carries Spink8, Fibcd1 and Itpka
# alongside the cortical markers), so they share an outline.
REGIONS = [
    ("Cortex / hippocampus", ["22", "15", "12"]),
    ("Striatum", ["13"]),
    ("Thalamus", ["18"]),
    ("Hypothalamus", ["4"]),
    ("White matter", ["10", "19", "20", "9"]),
    ("Choroid plexus", ["11"]),
]
REGION_SMOOTH_BINS = 6
REGION_MIN_BINS = 400
REGION_MAX_PARTS = 3

plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans"})


def panel_path(letter):
    return os.path.join(panel_dir, f"panel_{letter}.png")


def panel_knee(adata):
    raw_only = ad.AnnData(X=adata.layers["raw"], obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    cutoff = cs_utils.knee_plot(raw_only, transpose=True, expected_cells=EXPECTED_CELLS, title="", out_path=panel_path("A"), show=False)
    print(f"[A] UMI cutoff {cutoff:.0f}")
    return cutoff


def panel_alpha_histogram(alpha):
    fig, ax = plt.subplots(figsize=(5.4, 5), constrained_layout=True)
    ax.hist(alpha, bins=100, color="blue")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\hat{\alpha}_i$", fontsize=20)
    ax.set_ylabel("Number of bins", fontsize=18)
    ax.grid(axis="y", color="lightgray")
    ax.set_axisbelow(True)
    fig.savefig(panel_path("B"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[B] median alpha_hat {np.median(alpha):.3f}; >0.1 {(alpha > 0.1).mean():.3f}; >0.5 {(alpha > 0.5).mean():.3f}")


def grid_affine(obs):
    """Least-squares affine from (array_row, array_col, 1) to full-resolution pixel col (coef_x) and row (coef_y)."""
    A = np.column_stack([obs["array_row"].values, obs["array_col"].values, np.ones(len(obs))])
    coef_x, *_ = np.linalg.lstsq(A, obs["pxl_col_in_fullres"].values, rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, obs["pxl_row_in_fullres"].values, rcond=None)
    resid = np.abs(A @ coef_x - obs["pxl_col_in_fullres"].values).max()
    print(f"[C] affine grid->pixel max residual {resid:.1f} px")
    return coef_x, coef_y


def grid_to_pixel(obs):
    coef_x, coef_y = grid_affine(obs)
    return lambda rr, cc: (np.column_stack([rr, cc, np.ones(len(rr))]) @ coef_x,
                           np.column_stack([rr, cc, np.ones(len(rr))]) @ coef_y)


def region_outlines(ax, obs, to_pixel, scalef):
    """Outline and label the anatomical regions that the Leiden clusters identify unambiguously."""
    from skimage import measure

    labels = obs["celltype"].astype(str).values
    rr, cc = obs["array_row"].values, obs["array_col"].values
    shape = (rr.max() + 2, cc.max() + 2)
    for name, clusters in REGIONS:
        mask = np.zeros(shape, dtype=float)
        sel = np.isin(labels, clusters)
        mask[rr[sel], cc[sel]] = 1.0
        # smooth and threshold so the outline follows the region, not individual bins
        smooth = ndi.gaussian_filter(mask, REGION_SMOOTH_BINS)
        keep = smooth > 0.45
        keep = ndi.binary_closing(keep, np.ones((5, 5)))
        comps, n = ndi.label(keep)
        if not n:
            continue
        sizes = ndi.sum(keep, comps, range(1, n + 1))
        for comp in np.argsort(sizes)[::-1][:REGION_MAX_PARTS]:
            if sizes[comp] < REGION_MIN_BINS:
                continue
            for contour in measure.find_contours((comps == comp + 1).astype(float), 0.5):
                x, y = to_pixel(contour[:, 0], contour[:, 1])
                ax.plot(x * scalef, y * scalef, color="white", lw=2.2, path_effects=[pe.Stroke(linewidth=4.0, foreground="0.15"), pe.Normal()])
        # put the label at the most interior point of the largest part, so it lands inside a
        # C-shaped region such as the cortical ribbon rather than in the hole it wraps around
        biggest = (comps == np.argmax(sizes) + 1)
        inner = ndi.distance_transform_edt(biggest)
        cy, cx = np.unravel_index(np.argmax(inner), inner.shape)
        lx, ly = to_pixel(np.array([float(cy)]), np.array([float(cx)]))
        ax.text(lx[0] * scalef, ly[0] * scalef, name, color="white", fontsize=14, ha="center", va="center",
                path_effects=[pe.Stroke(linewidth=3.5, foreground="0.15"), pe.Normal()])
        print(f"[C]   {name}: {int(sel.sum()):,} bins, mean alpha {obs.loc[sel, 'alpha_hat'].mean():.3f}")


def panel_alpha_map(obs, below_cutoff):
    """obs: non-empty bins, coloured by alpha_hat. below_cutoff: in-tissue bins under the UMI cutoff, which
    cellsweep treats as empty (they define the ambient profile and get no alpha_hat), drawn in grey.

    The bins are drawn as one raster on the bin grid, mapped onto the H&E by the grid->pixel affine, so each
    8 um bin fills its own square. Scatter markers leave sub-pixel gaps through which the H&E shows as a pink
    moire."""
    spatial_dir = os.path.join(data_dir, "binned_outputs", "square_008um", "spatial")
    with open(os.path.join(spatial_dir, "scalefactors_json.json")) as f:
        scalefactors = json.load(f)
    scalef = scalefactors["tissue_hires_scalef"]
    image = np.array(Image.open(os.path.join(spatial_dir, "tissue_hires_image.png")))

    alpha_bin = pd.cut(obs["alpha_hat"], bins=ALPHA_BINS, labels=ALPHA_LABELS, include_lowest=True)
    colors = sns.color_palette("viridis", n_colors=len(ALPHA_LABELS)).as_hex()
    below_label, below_color = "in tissue, below\nUMI cutoff (empty)", "lightgray"

    both = pd.concat([obs, below_cutoff])
    rows, cols = both["array_row"].values, both["array_col"].values
    rgba = np.zeros((rows.max() + 1, cols.max() + 1, 4))
    rgba[below_cutoff["array_row"].values, below_cutoff["array_col"].values] = to_rgba(below_color)
    codes = alpha_bin.cat.codes.values
    rgba[obs["array_row"].values, obs["array_col"].values] = np.array([to_rgba(c) for c in colors])[codes]

    # imshow puts grid cell (row r, col c) at data (u=c, v=r); map it to hires pixels
    coef_x, coef_y = grid_affine(obs)
    grid_to_hires = Affine2D(np.array([[coef_x[1], coef_x[0], coef_x[2]],
                                       [coef_y[1], coef_y[0], coef_y[2]],
                                       [0, 0, 1]])).scale(scalef)

    x, y = obs["pxl_col_in_fullres"].values * scalef, obs["pxl_row_in_fullres"].values * scalef
    fig, ax = plt.subplots(figsize=(8.2, 7.4), constrained_layout=True)
    ax.imshow(image)
    ax.imshow(rgba, interpolation="nearest", transform=grid_to_hires + ax.transData)
    region_outlines(ax, obs, grid_to_pixel(obs), scalef)
    ax.set_xlim(x.min() - 25, x.max() + 25)
    ax.set_ylim(y.max() + 25, y.min() - 25)
    ax.axis("off")
    handles = [Line2D([], [], ls="", marker="s", ms=13, color=c, label=l) for l, c in zip(ALPHA_LABELS + [below_label], colors + [below_color])]
    legend = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=15, title=r"$\hat{\alpha}_i$")
    legend.get_title().set_fontsize(16)
    fig.savefig(panel_path("C"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[C] map: {alpha_bin.value_counts().reindex(ALPHA_LABELS).to_dict()}; {len(below_cutoff):,} in-tissue bins below cutoff")


def gene_decay(adata, raw, cs, gene):
    """Mean counts per bin in each distance stratum away from the gene's source region, raw and cellsweep."""
    rows, cols = adata.obs["array_row"].values, adata.obs["array_col"].values
    j = adata.var_names.get_loc(gene)
    r = np.asarray(raw[:, j].todense()).ravel()
    c = np.asarray(cs[:, j].todense()).ravel()
    grid = np.zeros((rows.max() + 1, cols.max() + 1))
    np.add.at(grid, (rows, cols), r)
    vals = ndi.gaussian_filter(grid, SIGMA_BINS)[rows, cols]
    order = np.argsort(vals)[::-1]
    n_top = np.searchsorted(np.cumsum(vals[order]), 0.5 * vals.sum()) + 1
    core = np.zeros(grid.shape, dtype=bool)
    core[rows[order[:n_top]], cols[order[:n_top]]] = True
    d = (ndi.distance_transform_edt(~core) * BIN_UM)[rows, cols]
    # the first stratum uses >= so that the core itself (d == 0) is included; it is then exactly the
    # on-source set (d <= SOURCE_DILATE_UM) that the removal percentages are quoted over
    strata = [((d >= lo) if i == 0 else (d > lo)) & (d <= hi) for i, (lo, hi) in enumerate(zip(DIST_EDGES[:-1], DIST_EDGES[1:]))]
    on, far = d <= SOURCE_DILATE_UM, d > FAR_UM
    print(f"[D] {gene}: {100 * (1 - c[on].sum() / r[on].sum()):.1f}% removed at source, "
          f"{100 * (1 - c[far].sum() / r[far].sum()):.1f}% removed >{FAR_UM} um away")
    return (np.array([r[m].mean() if m.any() else np.nan for m in strata]),
            np.array([c[m].mean() if m.any() else np.nan for m in strata]))


def panel_gene_decay(curves):
    labels = [f"{int(lo)}-{int(hi)}" if np.isfinite(hi) else f">{int(lo)}" for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
    fig, ax = plt.subplots(figsize=(6.6, 5), constrained_layout=True)
    for gene, color in GENES.items():
        r, c = curves[gene]
        ax.plot(labels, r, ":", marker="o", ms=6, mfc="white", color=color, lw=1.8)
        ax.plot(labels, c, "-", marker="o", ms=6, color=color, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("Distance from source region (um)", fontsize=17)
    ax.set_ylabel("Mean counts per bin", fontsize=17)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="lightgray")
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=c, lw=2.5, label=g) for g, c in GENES.items()]
    handles += [Line2D([], [], color="0.35", ls=":", marker="o", ms=6, mfc="white", lw=1.8, label="raw"),
                Line2D([], [], color="0.35", ls="-", marker="o", ms=6, lw=2.2, label="cellsweep")]
    ax.legend(handles=handles, frameon=False, fontsize=13, ncol=2)
    fig.savefig(panel_path("D"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compose():
    """Two rows (A B / C D), every panel scaled to PANEL_H, with a capital letter at the top left of each."""
    letters = ["A", "B", "C", "D"]
    imgs = {}
    for letter in letters:
        im = Image.open(panel_path(letter)).convert("RGB")
        imgs[letter] = im.resize((int(im.width * PANEL_H / im.height), PANEL_H), Image.LANCZOS)
    font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 90)
    pad, label_h = 40, 110
    rows = [["A", "B"], ["C", "D"]]
    row_w = [sum(imgs[l].width for l in r) + pad * (len(r) - 1) for r in rows]
    W = max(row_w)
    H = len(rows) * (label_h + PANEL_H) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for r in rows:
        x = (W - (sum(imgs[l].width for l in r) + pad * (len(r) - 1))) // 2
        for letter in r:
            draw.text((x + 10, y + 5), letter, fill="black", font=font)
            canvas.paste(imgs[letter], (x, y + label_h))
            x += imgs[letter].width + pad
        y += label_h + PANEL_H + pad
    path = os.path.join(out_dir, "mouse_brain_supplement.png")
    canvas.save(path)
    print(f"wrote {path} {canvas.size}")


def main():
    adata = ad.read_h5ad(os.path.join(data_dir, "adata_cellsweep.h5ad"))
    panel_knee(adata)
    non_empty = ~adata.obs["is_empty"].astype(bool).values
    alpha = adata.obs.loc[non_empty, "alpha_hat"].values
    panel_alpha_histogram(alpha)

    below_cutoff = adata.obs.loc[~non_empty & (adata.obs["in_tissue"].values == 1)].copy()
    adata = adata[non_empty].copy()
    raw, cs = adata.layers["raw"].tocsc(), adata.X.tocsc()
    panel_alpha_map(adata.obs, below_cutoff)
    panel_gene_decay({g: gene_decay(adata, raw, cs, g) for g in GENES})
    compose()


if __name__ == "__main__":
    main()
