#!/usr/bin/env python
"""
Reviewer point 7: are cross-species counts in the Visium HD human-mouse xenograft ambient contamination
or genuine two-species content, and why did Fig. 4E show correction of human but not mouse bins?

Sections
    1. Species counts, ambient-pool composition, tumor/host compartments, and distances of every bin to the
       tissue edge and to the tumor-host interface (compartment boundary inside the tissue).
    2. Cells per 8 um bin from the 10x Space Ranger 4.0.1 segmentation (nucleus-expanded cells rasterized onto
       the 2 um grid; nucleus centroids), and species mixing in 8 um bins vs segmented single cells.
    3. Species purity under a raw-count threshold and a content-normalized threshold.
    4. Cross-species (minority) and same-species (majority) counts removed by cellsweep, by majority species,
       purity, and distance to the interface, for each labeling in scripts/run_spatial_species_aware_cellsweep.py.
    5. Revised Fig. 4E restricted to species-pure bins, for both species.
    6. Tissue-border conclusion re-tested on species-pure bins, separating distance to the tissue edge from
       distance to the tumor-host interface, with an alpha estimate that uses only species labels.

Usage: python scripts/analyze_spatial_xenograft_mixing.py
Outputs: notebooks/output/visium_human_mouse/reviewer7/
"""
import os
import json

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.draw import polygon as draw_polygon

import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "visium_human_mouse")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "visium_human_mouse", "reviewer7")
cache_dir = os.path.join(data_dir, "reviewer7_cache")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

BIN_UM = 8
SMOOTH_SIGMA_BINS = 3        # Gaussian smoothing of species counts (24 um) to define compartments
PURITY = 0.9                 # species-pure: >= 90% of counts from one species
RUNS = {
    "original": os.path.join(data_dir, "adata_cellsweep.h5ad"),
    "species": os.path.join(data_dir, "species_aware", "adata_cellsweep_species.h5ad"),
    "species_mixed": os.path.join(data_dir, "species_aware", "adata_cellsweep_species_mixed.h5ad"),
}
DIST_BINS = [-1, 16, 48, 96, 200, 400, np.inf]
DIST_LABELS = ["0-16", "16-48", "48-96", "96-200", "200-400", ">400"]


def species_split(adata, X=None):
    is_human = (adata.var["genome"] == "GRCh38").values
    X = (adata.X if X is None else X).tocsc()
    return np.asarray(X[:, is_human].sum(axis=1)).ravel(), np.asarray(X[:, ~is_human].sum(axis=1)).ravel()


# =====================================================================================
# 1. Species counts, compartments, distances
# =====================================================================================
def build_bin_table():
    path = os.path.join(cache_dir, "bins.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    original = ad.read_h5ad(RUNS["original"], backed="r")
    obs = original.obs[["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres", "is_empty", "celltype", "alpha_hat"]].copy()
    raw = sc.read_10x_h5(os.path.join(data_dir, "binned_outputs", "square_008um", "raw_feature_bc_matrix.h5"), gex_only=False)
    raw = raw[obs.index]
    obs["h"], obs["m"] = species_split(raw)
    obs["tot"] = obs["h"] + obs["m"]
    obs["fh"] = obs["h"] / obs["tot"].clip(lower=1)
    obs["is_empty"] = obs["is_empty"].astype(bool)
    obs["celltype"] = obs["celltype"].astype(str)

    R, C = obs["array_row"].max() + 1, obs["array_col"].max() + 1
    rr, cc = obs["array_row"].values, obs["array_col"].values
    def grid(values):
        g = np.zeros((R, C))
        g[rr, cc] = values
        return g
    tissue = grid(obs["in_tissue"].values) > 0
    hs, ms = ndi.gaussian_filter(grid(obs["h"].values), SMOOTH_SIGMA_BINS), ndi.gaussian_filter(grid(obs["m"].values), SMOOTH_SIGMA_BINS)
    fh_smooth = hs / np.maximum(hs + ms, 1e-12)
    human_region, mouse_region = tissue & (fh_smooth >= 0.5), tissue & (fh_smooth < 0.5)
    d_edge = ndi.distance_transform_edt(tissue) * BIN_UM
    d_to_mouse = ndi.distance_transform_edt(~mouse_region) * BIN_UM
    d_to_human = ndi.distance_transform_edt(~human_region) * BIN_UM
    obs["compartment"] = np.where(human_region[rr, cc], "tumor (human)", np.where(mouse_region[rr, cc], "host (mouse)", "outside"))
    obs["fh_smooth"] = fh_smooth[rr, cc]
    obs["d_edge"] = d_edge[rr, cc]
    obs["d_iface"] = np.where(human_region, d_to_mouse, d_to_human)[rr, cc]
    obs.to_parquet(path)
    return obs


def ambient_composition(bins):
    original = ad.read_h5ad(RUNS["original"], backed="r")
    is_human = (original.var["genome"] == "GRCh38").values
    ambient = original.var["ambient"].values
    empty = bins[bins["is_empty"]]
    rows = [
        ("ambient profile used by cellsweep", ambient[is_human].sum() / ambient.sum()),
        ("empty bins in tissue", empty.loc[empty["in_tissue"] == 1, "h"].sum() / empty.loc[empty["in_tissue"] == 1, "tot"].sum()),
        ("empty bins outside tissue", empty.loc[empty["in_tissue"] == 0, "h"].sum() / empty.loc[empty["in_tissue"] == 0, "tot"].sum()),
        ("non-empty bins (pooled)", bins.loc[~bins["is_empty"], "h"].sum() / bins.loc[~bins["is_empty"], "tot"].sum()),
    ]
    return pd.DataFrame(rows, columns=["pool", "human_fraction"])


# =====================================================================================
# 2. Cells per bin (segmentation) and mixing in bins vs segmented cells
# =====================================================================================
def affine_to_array(positions):
    A = np.c_[positions["pxl_col_in_fullres"], positions["pxl_row_in_fullres"], np.ones(len(positions))]
    coef_row = np.linalg.lstsq(A, positions["array_row"].values, rcond=None)[0]
    coef_col = np.linalg.lstsq(A, positions["array_col"].values, rcond=None)[0]
    max_resid = max(np.abs(A @ coef_row - positions["array_row"]).max(), np.abs(A @ coef_col - positions["array_col"]).max())
    return coef_row, coef_col, max_resid


def build_cells_per_bin(bins):
    path = os.path.join(cache_dir, "cells_per_bin.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    seg_dir = os.path.join(data_dir, "segmented_outputs")
    p2 = pd.read_parquet(os.path.join(data_dir, "binned_outputs", "square_002um", "spatial", "tissue_positions.parquet"))
    coef_row, coef_col, resid = affine_to_array(p2.sample(200_000, random_state=0))
    print(f"2 um pixel->array affine max residual: {resid:.3f} bins")
    R2, C2 = p2["array_row"].max() + 1, p2["array_col"].max() + 1

    # rasterize nucleus-expanded cells onto the 2 um grid
    labels = np.zeros((R2, C2), dtype=np.int32)
    with open(os.path.join(seg_dir, "cell_segmentations.geojson")) as f:
        features = json.load(f)["features"]
    for i, feature in enumerate(features, start=1):
        ring = np.asarray(feature["geometry"]["coordinates"][0])
        A = np.c_[ring, np.ones(len(ring))]
        r, c = draw_polygon(A @ coef_row + 0.5, A @ coef_col + 0.5, shape=(R2, C2))
        labels[r, c] = i

    # each 8 um bin covers a 4x4 block of 2 um bins; block origin from the 8 um bin center
    A8 = np.c_[bins["pxl_col_in_fullres"], bins["pxl_row_in_fullres"], np.ones(len(bins))]
    r0, c0 = np.round(A8 @ coef_row - 1.5).astype(int), np.round(A8 @ coef_col - 1.5).astype(int)
    block = np.stack([labels[np.clip(r0 + dr, 0, R2 - 1), np.clip(c0 + dc, 0, C2 - 1)] for dr in range(4) for dc in range(4)], axis=1)
    s = np.sort(block, axis=1)
    n_cells = ((np.diff(s, axis=1) != 0) & (s[:, 1:] > 0)).sum(axis=1) + (s[:, 0] > 0)

    with open(os.path.join(seg_dir, "nucleus_segmentations.geojson")) as f:
        nuclei = json.load(f)["features"]
    centroids = np.array([np.asarray(n["geometry"]["coordinates"][0]).mean(axis=0) for n in nuclei])
    _, nearest = cKDTree(bins[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values).query(centroids)
    n_nuclei = np.bincount(nearest, minlength=len(bins))

    out = pd.DataFrame({"n_cells_overlap": n_cells, "n_2um_in_cells": (block > 0).sum(axis=1), "n_nuclei": n_nuclei}, index=bins.index)
    out.to_parquet(path)
    return out


def build_segmented_cells(bins):
    path = os.path.join(cache_dir, "segmented_cells.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    seg_dir = os.path.join(data_dir, "segmented_outputs")
    cells_adata = sc.read_10x_h5(os.path.join(seg_dir, "filtered_feature_cell_matrix.h5"), gex_only=False)
    cells = pd.DataFrame(index=cells_adata.obs_names)
    cells["h"], cells["m"] = species_split(cells_adata)
    cells["tot"] = cells["h"] + cells["m"]
    cells["fh"] = cells["h"] / cells["tot"].clip(lower=1)
    with open(os.path.join(seg_dir, "cell_segmentations.geojson")) as f:
        features = json.load(f)["features"]
    centroids = pd.DataFrame({f"cellid_{ft['properties']['cell_id']:09d}-1": np.asarray(ft["geometry"]["coordinates"][0]).mean(axis=0) for ft in features}, index=["px", "py"]).T
    cells = cells.join(centroids)
    _, nearest = cKDTree(bins[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values).query(cells[["px", "py"]].values)
    for col in ["compartment", "d_iface", "d_edge"]:
        cells[col] = bins[col].values[nearest]
    cells.to_parquet(path)
    return cells


def mixing_by_distance(df):
    df = df[df["compartment"] != "outside"].copy()
    df["majority_species_fraction"] = np.maximum(df["fh"], 1 - df["fh"])
    df["dist"] = pd.cut(df["d_iface"], DIST_BINS, labels=DIST_LABELS)
    df["compartment_species_fraction"] = np.where(df["compartment"] == "tumor (human)", df["fh"], 1 - df["fh"])
    return df.groupby(["compartment", "dist"], observed=True).agg(
        n=("fh", "size"),
        median_library_size=("tot", "median"),
        median_other_species_fraction=("compartment_species_fraction", lambda x: 1 - x.median()),
        frac_mixed_10_90=("fh", lambda x: ((x > 0.1) & (x < 0.9)).mean()),
        frac_pure_90=("majority_species_fraction", lambda x: (x >= PURITY).mean()),
    ).reset_index()


# =====================================================================================
# 3-4. Purity and per-run correction
# =====================================================================================
def add_purity(bins):
    ne = bins[~bins["is_empty"]].copy()
    s_h = ne.loc[ne["fh"] >= PURITY, "tot"].median()
    s_m = ne.loc[ne["fh"] <= 1 - PURITY, "tot"].median()
    ne["fh_norm"] = (ne["h"] / s_h) / (ne["h"] / s_h + ne["m"] / s_m).clip(lower=1e-12)
    ne["majority"] = np.where(ne["fh"] >= 0.5, "human", "mouse")
    ne["purity"] = np.maximum(ne["fh"], 1 - ne["fh"])
    ne["purity_norm"] = np.maximum(ne["fh_norm"], 1 - ne["fh_norm"])
    return ne, s_h, s_m


def load_run_species_counts(run, path, index):
    cache = os.path.join(cache_dir, f"species_counts_{run}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    adata = ad.read_h5ad(path)
    adata = adata[index]
    h, m = species_split(adata)
    df = pd.DataFrame({"h_cs": h, "m_cs": m, "alpha_hat": adata.obs["alpha_hat"].values, "celltype": adata.obs["celltype"].astype(str).values}, index=index)
    df.to_parquet(cache)
    return df


def correction_table(ne, runs_counts, group_cols):
    rows = []
    for run, cs_counts in runs_counts.items():
        df = ne.join(cs_counts[["h_cs", "m_cs", "alpha_hat"]], rsuffix="_run")
        is_h = df["majority"] == "human"
        df["minor_raw"], df["minor_cs"] = np.where(is_h, df["m"], df["h"]), np.where(is_h, df["m_cs"], df["h_cs"])
        df["major_raw"], df["major_cs"] = np.where(is_h, df["h"], df["m"]), np.where(is_h, df["h_cs"], df["m_cs"])
        g = df.groupby(group_cols, observed=True).agg(n=("tot", "size"), minor_raw=("minor_raw", "sum"), minor_cs=("minor_cs", "sum"), major_raw=("major_raw", "sum"), major_cs=("major_cs", "sum"), median_alpha_hat=("alpha_hat_run", "median")).reset_index()
        g["pct_cross_species_removed"] = 100 * (1 - g["minor_cs"] / g["minor_raw"])
        g["pct_same_species_removed"] = 100 * (1 - g["major_cs"] / g["major_raw"])
        g.insert(0, "run", run)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


# =====================================================================================
# Figures
# =====================================================================================
def plot_species_map(bins, cells_per_bin, alpha_col="alpha_hat"):
    R, C = bins["array_row"].max() + 1, bins["array_col"].max() + 1
    def grid(values, mask):
        g = np.full((R, C), np.nan)
        g[bins.loc[mask, "array_row"], bins.loc[mask, "array_col"]] = values[mask]
        return g
    in_tissue = (bins["in_tissue"] == 1).values
    nonempty = (~bins["is_empty"]).values
    compartment = grid((bins["compartment"] == "tumor (human)").astype(float).values, in_tissue)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    im = axes[0].imshow(grid(bins["fh"].values, nonempty), cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Human fraction of counts (non-empty bins)")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    im = axes[1].imshow(grid(np.minimum(cells_per_bin["n_nuclei"].values, 3), nonempty), cmap="viridis", vmin=0, vmax=3, interpolation="nearest")
    axes[1].set_title("Nuclei per 8 um bin (10x segmentation; 3 = 3+)")
    fig.colorbar(im, ax=axes[1], fraction=0.04)
    im = axes[2].imshow(grid(bins[alpha_col].values, nonempty), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("cellsweep alpha_hat (manuscript run)")
    fig.colorbar(im, ax=axes[2], fraction=0.04)
    for ax in axes:
        ax.contour(np.nan_to_num(compartment, nan=1.0), levels=[0.5], colors="k", linewidths=0.5)
        ax.axis("off")
    fig.suptitle("Black contour: tumor (human) / host (mouse) compartment boundary; white inside tissue: empty (below UMI cutoff)", y=0.86)
    fig.savefig(os.path.join(out_dir, "species_map.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cells_per_bin(cells_per_bin, ne):
    cpb = cells_per_bin.loc[ne.index]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in [(axes[0], "n_nuclei", "Nucleus centroids per 8 um bin"), (axes[1], "n_cells_overlap", "Segmented cells overlapping an 8 um bin")]:
        counts = cpb[col].value_counts().sort_index()
        ax.bar(counts.index, 100 * counts.values / counts.sum(), color="#4c72b0")
        ax.set_xlabel(title)
        ax.set_ylabel("% of non-empty bins")
        ax.set_title(f"median = {cpb[col].median():.0f}, mean = {cpb[col].mean():.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cells_per_bin.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mixing_vs_distance(mix_bins, mix_cells):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, comp, color in [(axes[0], "tumor (human)", "#0047b3"), (axes[1], "host (mouse)", "#cc5500")]:
        for df, style, label in [(mix_bins, "-o", "8 um bins"), (mix_cells, "--s", "segmented cells")]:
            d = df[df["compartment"] == comp]
            ax.plot(d["dist"].astype(str), 100 * d["median_other_species_fraction"], style, color=color, label=label)
        ax.set_title(f"{comp} compartment")
        ax.set_xlabel("Distance to tumor-host interface (um)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Median % counts from the other species")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mixing_vs_interface_distance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_correction_by_purity(tab):
    runs = tab["run"].unique()
    fig, axes = plt.subplots(2, len(runs), figsize=(5 * len(runs), 8), sharey="row", squeeze=False)
    for j, run in enumerate(runs):
        for i, (species, color) in enumerate([("human", "#0047b3"), ("mouse", "#cc5500")]):
            ax = axes[i, j]
            d = tab[(tab["run"] == run) & (tab["majority"] == species)]
            x = np.arange(len(d))
            ax.bar(x - 0.2, d["pct_cross_species_removed"], 0.4, color=color, label="cross-species counts removed")
            ax.bar(x + 0.2, d["pct_same_species_removed"], 0.4, color=color, alpha=0.35, label="same-species counts removed")
            ax.set_xticks(x, [f"{p}\n(n={n:,})" for p, n in zip(d["purity_bin"].astype(str), d["n"])], fontsize=8)
            ax.set_title(f"{run}: {species}-majority bins")
            ax.set_ylim(0, 100)
            if j == 0:
                ax.set_ylabel("% of counts removed")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8)
        axes[1, j].set_xlabel("Majority-species fraction of raw counts")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "correction_by_purity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_joint_scatter(ne, cs_counts, run, pure_col="purity"):
    pure = ne[ne[pure_col] >= PURITY]
    obs_raw = pd.DataFrame({"human_counts_total_": pure["h"], "mouse_counts_total_": pure["m"], "genome": pure["majority"]})
    obs_cs = pd.DataFrame({"human_counts_total_": cs_counts.loc[pure.index, "h_cs"], "mouse_counts_total_": cs_counts.loc[pure.index, "m_cs"], "genome": pure["majority"]})
    # keep every mouse-majority pure bin and subsample human-majority bins so both species are visible
    rng = np.random.default_rng(0)
    human_idx = pure.index[pure["majority"] == "human"]
    keep = pure.index[pure["majority"] == "mouse"].append(pd.Index(rng.choice(human_idx, size=min(20_000, len(human_idx)), replace=False)))
    dummy = lambda obs: ad.AnnData(obs=obs.loc[keep], var=pd.DataFrame(index=["g"]), X=np.zeros((len(keep), 1), dtype=np.float32))
    cs_utils.plot_cross_species_joint_scatterplot(dummy(obs_raw), dummy(obs_cs), processed_name="cellsweep", x_name="human", y_name="mouse", x_axis="human_counts_total_", y_axis="mouse_counts_total_", genome_column="genome", marginal_type="histogram", fill_histogram=False, show_marginal_ticks=True, show_point_movement=True, out_path=os.path.join(out_dir, f"joint_scatter_pure{int(PURITY * 100)}_{pure_col}_{run}.png"), show=False)


def plot_border(border):
    fig, axes = plt.subplots(1, len(border["run"].unique()), figsize=(5.5 * border["run"].nunique(), 4.2), sharey=True, squeeze=False)
    for ax, run in zip(axes[0], border["run"].unique()):
        d = border[border["run"] == run]
        for iface, color in zip(d["iface_stratum"].cat.categories, ["#d62728", "#ff9896", "#7f7f7f"]):
            dd = d[d["iface_stratum"] == iface]
            if len(dd):
                ax.plot(dd["edge_dist"].astype(str), dd["mean_alpha_hat"], "-o", color=color, label=f"interface {iface} um")
        ax.set_title(f"{run}: species-pure human bins")
        ax.set_xlabel("Distance to tissue edge (um)")
        ax.tick_params(axis="x", rotation=45)
    axes[0, 0].set_ylabel("Mean alpha_hat")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "border_alpha_edge_vs_interface.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    bins = build_bin_table()
    amb = ambient_composition(bins)
    amb.to_csv(os.path.join(out_dir, "ambient_species_composition.csv"), index=False)
    print(amb.to_string(index=False))
    tissue = bins[bins["in_tissue"] == 1]
    print(f"tumor (human) compartment: {100 * (tissue['compartment'] == 'tumor (human)').mean():.1f}% of in-tissue bins")

    # ---- 2. cells per bin, mixing in bins vs cells
    cells_per_bin = build_cells_per_bin(bins)
    ne, s_h, s_m = add_purity(bins)
    ne = ne.join(cells_per_bin)
    print(f"median library size of >=90%-pure bins: human {s_h:.0f}, mouse {s_m:.0f} (ratio {s_h / s_m:.2f})")
    cpb_summary = ne[["n_nuclei", "n_cells_overlap"]].describe().T
    cpb_summary.to_csv(os.path.join(out_dir, "cells_per_bin_summary.csv"))
    print(cpb_summary.to_string())
    plot_cells_per_bin(cells_per_bin, ne)
    plot_species_map(bins, cells_per_bin)

    cells = build_segmented_cells(bins)
    cells = cells[cells["tot"] >= ne["tot"].min()]
    mix_bins, mix_cells = mixing_by_distance(ne), mixing_by_distance(cells)
    mix = pd.concat([mix_bins.assign(unit="8um_bin"), mix_cells.assign(unit="segmented_cell")])
    mix.to_csv(os.path.join(out_dir, "mixing_vs_interface_distance.csv"), index=False)
    print(mix.round(3).to_string(index=False))
    plot_mixing_vs_distance(mix_bins, mix_cells)

    # ---- 3. purity thresholds
    purity_counts = pd.DataFrame({
        "raw_counts": ne.assign(pure=ne["purity"] >= PURITY).groupby("majority")["pure"].agg(["size", "sum"]).rename(columns={"size": "n_majority", "sum": "n_pure"}).stack(),
        "content_normalized": ne.assign(majority=np.where(ne["fh_norm"] >= 0.5, "human", "mouse"), pure=ne["purity_norm"] >= PURITY).groupby("majority")["pure"].agg(["size", "sum"]).rename(columns={"size": "n_majority", "sum": "n_pure"}).stack(),
    })
    purity_counts.to_csv(os.path.join(out_dir, "purity_counts.csv"))
    print(purity_counts.to_string())

    # ---- 4. correction per run
    runs_counts = {run: load_run_species_counts(run, path, ne.index) for run, path in RUNS.items() if os.path.exists(path)}
    ne["purity_bin"] = pd.cut(ne["purity"], [0.5, 0.8, 0.9, 0.95, 0.99, 1.0001], right=False, labels=["50-80%", "80-90%", "90-95%", "95-99%", ">=99%"])
    ne["dist"] = pd.cut(ne["d_iface"], DIST_BINS, labels=DIST_LABELS)
    tab_purity = correction_table(ne, runs_counts, ["majority", "purity_bin"])
    tab_purity.to_csv(os.path.join(out_dir, "correction_by_purity.csv"), index=False)
    tab_pure = correction_table(ne[ne["purity"] >= PURITY], runs_counts, ["majority"])
    tab_pure.to_csv(os.path.join(out_dir, "correction_pure_bins.csv"), index=False)
    tab_dist = correction_table(ne, runs_counts, ["majority", "dist"])
    tab_dist.to_csv(os.path.join(out_dir, "correction_by_interface_distance.csv"), index=False)
    cols = ["run", "majority", "n", "median_alpha_hat", "pct_cross_species_removed", "pct_same_species_removed"]
    print(tab_pure[cols].round(2).to_string(index=False))
    print(tab_purity[cols[:2] + ["purity_bin"] + cols[2:]].round(2).to_string(index=False))
    print(tab_dist[cols[:2] + ["dist"] + cols[2:]].round(2).to_string(index=False))
    plot_correction_by_purity(tab_purity)

    # ---- 5. revised Fig. 4E
    for run, cs_counts in runs_counts.items():
        plot_joint_scatter(ne, cs_counts, run)

    # ---- 6. border
    ambient_mouse_share = 1 - amb.loc[0, "human_fraction"]
    border_rows = []
    pure_human = ne[ne["fh"] >= PURITY].copy()
    pure_human["iface_stratum"] = pd.cut(pure_human["d_iface"], [-1, 100, 300, np.inf], labels=["<=100", "100-300", ">300"])
    pure_human["edge_dist"] = pd.cut(pure_human["d_edge"], DIST_BINS, labels=DIST_LABELS)
    pure_human["alpha_species"] = (1 - pure_human["fh"]) / ambient_mouse_share
    for run, cs_counts in runs_counts.items():
        d = pure_human.join(cs_counts[["alpha_hat"]], rsuffix="_run")
        g = d.groupby(["iface_stratum", "edge_dist"], observed=True).agg(n=("tot", "size"), mean_alpha_hat=("alpha_hat_run", "mean"), median_alpha_hat=("alpha_hat_run", "median"), mean_alpha_species=("alpha_species", "mean"), median_library_size=("tot", "median")).reset_index()
        g.insert(0, "run", run)
        border_rows.append(g)
        far = d[d["iface_stratum"] == ">300"]
        print(f"[{run}] pure human bins >300 um from interface (n={len(far)}): Spearman(alpha_hat, alpha_species) = {far[['alpha_hat_run', 'alpha_species']].corr('spearman').iloc[0, 1]:.3f}; Spearman(alpha_hat, d_edge) = {far[['alpha_hat_run', 'd_edge']].corr('spearman').iloc[0, 1]:.3f}")
    border = pd.concat(border_rows, ignore_index=True)
    border.to_csv(os.path.join(out_dir, "border_alpha_edge_vs_interface.csv"), index=False)
    print(border.round(3).to_string(index=False))
    plot_border(border)


if __name__ == "__main__":
    main()
