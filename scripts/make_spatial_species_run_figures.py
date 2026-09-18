#!/usr/bin/env python
"""
Visium HD xenograft figures in notebooks/output/visium_human_mouse/, from the runs in
scripts/run_spatial_purity_filtered_cellsweep.py, which exclude bins that may contain two species before
correcting the rest:

    purity     bins < 90% pure are excluded          -> adata_cellsweep_pure90.h5ad
    interface  bins within 16 um of the tumor-host    -> adata_cellsweep_interface16.h5ad
               interface are excluded

Figures:
    visium_human_mouse_cellsweep_joint_scatterplot.png   human vs mouse counts, raw vs cellsweep (interface run),
                                                         species coloured at the ambient-implied boundary
    visium_human_mouse_cellsweep_joint_scatterplot_majority_rule.png  same points, coloured by the 50% majority rule
    visium_human_mouse_cellsweep_joint_scatterplot_purity_rule.png    purity run, 50% majority (sensitivity)

Species colouring. A human cell with ambient fraction alpha has mouse fraction alpha * (1 - a_h) and so can never
fall below a_h human, where a_h is the human share of the ambient profile (0.876 here); a mouse cell with ambient
fraction alpha sits at alpha * a_h human. The boundary between "explainable as a contaminated human cell" and
"contains mouse material" is therefore fh = a_h, not fh = 0.5. Bins between 0.5 and a_h are contaminated mouse
cells that the majority rule paints as human, and they are exactly the bins that lose human counts while keeping
their mouse counts. a_h is read from the ambient profile of the run, not hard-coded.
    visium_hd_alpha_hat.png                              alpha_hat map, purity run, excluded bins in grey
    visium_hd_alpha_hat_interface_excluded.png           alpha_hat map, interface run
    visium_hd_alpha_hat_exclusion_comparison.png         the two maps side by side

Usage: python scripts/make_spatial_species_run_figures.py
"""
import os
import json

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import squidpy as sq
from PIL import Image

import cellsweep.utils as cs_utils

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "visium_human_mouse")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "visium_human_mouse")
os.makedirs(out_dir, exist_ok=True)

PURITY = 0.9
INTERFACE_UM = 16
RUNS = {
    "purity": (os.path.join(data_dir, "purity_filtered", f"adata_cellsweep_pure{int(PURITY * 100)}.h5ad"), f"excluded: <{int(PURITY * 100)}% species-pure"),
    "interface": (os.path.join(data_dir, "purity_filtered", f"adata_cellsweep_interface{INTERFACE_UM}.h5ad"), f"excluded: within {INTERFACE_UM} um of interface"),
}
N_HUMAN_SUBSAMPLE = 20_000
ALPHA_BINS = [0, 0.25, 0.5, 0.75, 1.0]
ALPHA_LABELS = [f"{ALPHA_BINS[i]}-{ALPHA_BINS[i + 1]}" for i in range(len(ALPHA_BINS) - 1)]
DIST_BINS = [-1, 16, 48, 96, 200, 400, np.inf]
DIST_LABELS = ["0-16", "16-48", "48-96", "96-200", "200-400", ">400"]


def species_counts(adata, X):
    is_human = (adata.var["genome"] == "GRCh38").values
    X = X.tocsc()
    return np.asarray(X[:, is_human].sum(axis=1)).ravel(), np.asarray(X[:, ~is_human].sum(axis=1)).ravel()


def ambient_human_fraction(adata):
    """Human share of the ambient profile; the run's column name depends on the cellsweep version."""
    column = next(c for c in ["ambient_hat", "ambient_profile", "ambient"] if c in adata.var.columns)
    is_human = (adata.var["genome"] == "GRCh38").values
    ambient = adata.var[column].values
    return ambient[is_human].sum() / ambient.sum()


def load_run(path):
    adata = ad.read_h5ad(path)
    a_h = ambient_human_fraction(adata)
    adata = adata[~adata.obs["is_empty"].astype(bool)].copy()
    df = adata.obs[["pxl_row_in_fullres", "pxl_col_in_fullres", "genome", "purity", "alpha_hat"]].copy()
    df["h"], df["m"] = species_counts(adata, adata.layers["raw"])
    df["h_cs"], df["m_cs"] = species_counts(adata, adata.X)
    df["fh"] = df["h"] / (df["h"] + df["m"]).clip(lower=1)
    df["genome_majority"] = np.where(df["fh"] >= 0.5, "human", "mouse")
    df["genome_boundary"] = np.where(df["fh"] > a_h, "human", "mouse")
    print(f"   ambient human fraction {a_h:.3f}; species calls: majority rule {df['genome_majority'].value_counts().to_dict()}, "
          f"boundary rule {df['genome_boundary'].value_counts().to_dict()}, relabelled {int((df['genome_majority'] != df['genome_boundary']).sum()):,}")
    bins = pd.read_parquet(os.path.join(data_dir, "reviewer7_cache", "bins.parquet"))
    df["d_iface"] = bins["d_iface"].reindex(df.index).values
    excluded = bins[~bins["is_empty"] & ~bins.index.isin(df.index)][["pxl_row_in_fullres", "pxl_col_in_fullres"]]
    return df, excluded


def report(name, df):
    print(f"[{name}] {len(df):,} corrected bins; median alpha_hat {df['alpha_hat'].median():.3f}")
    for genome in ["human", "mouse"]:
        d = df[df["genome_majority"] == genome]
        if not len(d):
            continue
        minor_raw, minor_cs = (d["m"], d["m_cs"]) if genome == "human" else (d["h"], d["h_cs"])
        major_raw, major_cs = (d["h"], d["h_cs"]) if genome == "human" else (d["m"], d["m_cs"])
        print(f"   {genome}-majority (n={len(d):,}): {100 * (1 - minor_cs.sum() / minor_raw.sum()):.1f}% cross-species counts removed, "
              f"{100 * (1 - major_cs.sum() / major_raw.sum()):.1f}% same-species removed, median alpha_hat {d['alpha_hat'].median():.3f}")
    human = df[df["genome_majority"] == "human"].copy()
    human["dist"] = pd.cut(human["d_iface"], DIST_BINS, labels=DIST_LABELS)
    g = human.groupby("dist", observed=True)["alpha_hat"].agg(["size", "mean", "median"]).round(4)
    print(f"   alpha_hat by distance to interface (human-majority bins):\n{g.to_string()}")
    d = df.copy()
    d["purity_bin"] = pd.cut(d["purity"], [0.5, 0.8, 0.9, 0.95, 0.99, 1.0001], right=False, labels=["50-80%", "80-90%", "90-95%", "95-99%", ">=99%"])
    is_h = d["genome_majority"] == "human"
    d["minor_raw"], d["minor_cs"] = np.where(is_h, d["m"], d["h"]), np.where(is_h, d["m_cs"], d["h_cs"])
    t = d.groupby(["genome_majority", "purity_bin"], observed=True).apply(lambda x: pd.Series({"n": len(x), "pct_cross_species_removed": 100 * (1 - x["minor_cs"].sum() / max(x["minor_raw"].sum(), 1e-9))}))
    print(f"   cross-species counts removed by purity:\n{t.round(1).to_string()}")


def joint_scatter(df, fname, label_col="genome_boundary", keep=None):
    """keep: the bins to plot. Passing the same set for both colourings makes the two figures directly comparable."""
    if keep is None:
        rng = np.random.default_rng(0)
        # every bin either rule calls mouse, plus a subsample of the unambiguous human bins
        mouse_either = df.index[(df["genome_majority"] == "mouse") | (df["genome_boundary"] == "mouse")]
        rest = df.index.difference(mouse_either)
        keep = mouse_either.append(pd.Index(rng.choice(rest, size=min(N_HUMAN_SUBSAMPLE, len(rest)), replace=False)))

    def dummy(h, m):
        obs = pd.DataFrame({"human_counts_total_": h.loc[keep].values, "mouse_counts_total_": m.loc[keep].values, "genome": df.loc[keep, label_col].values}, index=keep)
        return ad.AnnData(obs=obs, var=pd.DataFrame(index=["g"]), X=np.zeros((len(keep), 1), dtype=np.float32))

    cs_utils.plot_cross_species_joint_scatterplot(
        dummy(df["h"], df["m"]), dummy(df["h_cs"], df["m_cs"]),
        processed_name="cellsweep", x_name="human", y_name="mouse",
        x_axis="human_counts_total_", y_axis="mouse_counts_total_", genome_column="genome",
        marginal_type="histogram", fill_histogram=False, show_marginal_ticks=True, show_point_movement=True,
        out_path=os.path.join(out_dir, fname), show=False)
    return keep


def build_map_adata(df, excluded, excluded_label):
    # include_lowest: alpha_hat == 0 would otherwise fall outside the first interval
    alpha_bin = pd.cut(df["alpha_hat"], bins=ALPHA_BINS, labels=ALPHA_LABELS, include_lowest=True).astype(str)
    values = pd.concat([alpha_bin, pd.Series(excluded_label, index=excluded.index)])
    coords = np.vstack([df[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(), excluded[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()])
    adata = ad.AnnData(X=np.zeros((len(values), 1), dtype=np.float32), obs=pd.DataFrame(index=values.index), var=pd.DataFrame(index=["g"]))
    adata.obs["alpha_hat_bin"] = pd.Categorical(values.values, categories=ALPHA_LABELS + [excluded_label])
    adata.obsm["spatial"] = coords
    spatial_dir = os.path.join(data_dir, "binned_outputs", "square_008um", "spatial")
    with open(os.path.join(spatial_dir, "scalefactors_json.json")) as f:
        scalefactors = json.load(f)
    adata.uns["spatial"] = {"square_008um": {"images": {"hires": np.array(Image.open(os.path.join(spatial_dir, "tissue_hires_image.png"))), "lowres": np.array(Image.open(os.path.join(spatial_dir, "tissue_lowres_image.png")))}, "scalefactors": scalefactors}}
    print(f"   map: {adata.obs['alpha_hat_bin'].value_counts().reindex(ALPHA_LABELS + [excluded_label]).to_dict()}")
    return adata


def main():
    palette = ListedColormap(sns.color_palette("viridis", n_colors=len(ALPHA_LABELS)).as_hex() + ["lightgrey"])
    maps = {}
    for name, (path, excluded_label) in RUNS.items():
        if not os.path.exists(path):
            print(f"[{name}] {path} missing, skipping")
            continue
        df, excluded = load_run(path)
        report(name, df)
        maps[name] = (build_map_adata(df, excluded, excluded_label), excluded_label)
        if name == "interface":
            keep = joint_scatter(df, "visium_human_mouse_cellsweep_joint_scatterplot.png", label_col="genome_boundary")
            joint_scatter(df, "visium_human_mouse_cellsweep_joint_scatterplot_majority_rule.png", label_col="genome_majority", keep=keep)
        else:
            joint_scatter(df, "visium_human_mouse_cellsweep_joint_scatterplot_purity_rule.png", label_col="genome_majority")

    for name, (adata, excluded_label) in maps.items():
        sq.pl.spatial_scatter(adata, color="alpha_hat_bin", size=4, palette=palette)
        fname = "visium_hd_alpha_hat.png" if name == "purity" else "visium_hd_alpha_hat_interface_excluded.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close()

    if len(maps) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(22, 11))
        for ax, name in zip(axes, ["purity", "interface"]):
            adata, excluded_label = maps[name]
            sq.pl.spatial_scatter(adata, color="alpha_hat_bin", size=4, palette=palette, ax=ax, fig=fig, title=excluded_label, legend_loc="right margin" if name == "interface" else None)
        fig.savefig(os.path.join(out_dir, "visium_hd_alpha_hat_exclusion_comparison.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
