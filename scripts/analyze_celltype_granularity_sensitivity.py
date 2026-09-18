#!/usr/bin/env python
"""
Metrics and figure for cellsweep's sensitivity to cell-type label granularity (pbmc8k).
Run scripts/run_celltype_granularity_sensitivity.py first.

Total counts removed per cell is insensitive to *which* counts are removed, so every metric
here looks at specific genes, specific cells, or downstream structure. Evaluation cells are
grouped with a fixed reference annotation (CellTypist Immune_All_Low collapsed to major
populations) that does not change across runs.

1. Lineage-marker specificity. For marker panels with a well-defined expressing lineage
   (myeloid, B, T, NK, platelet, erythrocyte), the fraction of raw counts removed from
   off-target cells (should be high: ambient) vs. from on-target cells (should be low: real).
2. Subtype markers inside a coarse label. Genes that separate two populations that share a
   CellTypist-High label (CD8A in CD8 vs CD4 T; FCGR3A in CD16 vs CD14 monocytes; TCL1A in
   naive vs memory B; SLC4A10 in MAIT vs other T). Retention in the expressing population and
   log2 fold change between the pair, before vs after denoising.
3. Per-cell and per-gene removal *fractions* (not totals) compared with the default run.
4. Downstream structure. kNN purity of the reference populations in PCA of the denoised data,
   and kNN purity of each run's *own* labels (does denoising manufacture separation between the
   labels it was given? the shuffled-label run is the negative control).

Outputs (notebooks/output/pbmc8k/celltype_granularity/):
    summary_metrics.csv, lineage_marker_metrics.csv, subtype_marker_metrics.csv,
    celltype_granularity_sensitivity.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "pbmc8k")
run_dir = os.path.join(data_dir, "celltype_granularity")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "pbmc8k", "celltype_granularity")
os.makedirs(out_dir, exist_ok=True)

sim_metrics_csv = os.path.join(cellsweep_dir, "notebooks", "output", "simulation1_small_noise", "celltype_granularity_simulation_metrics.csv")

labels = pd.read_csv(os.path.join(run_dir, "labels.csv"), index_col=0, dtype=str)
DEFAULT = "ct_high"

# reference populations from CellTypist Immune_All_Low
REF_GROUP = {
    "Tcm/Naive helper T cells": "CD4 T", "Tem/Effector helper T cells": "CD4 T", "Regulatory T cells": "CD4 T", "Treg(diff)": "CD4 T",
    "Tcm/Naive cytotoxic T cells": "CD8 T", "Tem/Temra cytotoxic T cells": "CD8 T", "Tem/Trm cytotoxic T cells": "CD8 T",
    "MAIT cells": "MAIT",
    "CD16+ NK cells": "NK", "NK cells": "NK",
    "Naive B cells": "Naive B", "Memory B cells": "Memory B",
    "Classical monocytes": "CD14 Mono", "Non-classical monocytes": "CD16 Mono",
    "DC1": "cDC", "DC2": "cDC", "pDC": "pDC",
    "Megakaryocytes/platelets": "Platelet", "HSC/MPP": "HSC",
}
T = {"CD4 T", "CD8 T", "MAIT"}
B = {"Naive B", "Memory B"}
MONO = {"CD14 Mono", "CD16 Mono"}
MYELOID = MONO | {"cDC", "pDC"}
ALL = set(REF_GROUP.values())

# panel -> genes, on-target populations, off-target populations (others are ignored)
LINEAGE_PANELS = {
    "Myeloid (LYZ, S100A8/9, CST3, FCN1)": (["LYZ", "S100A8", "S100A9", "CST3", "FCN1"], MYELOID, T | B | {"NK"}),
    "B (MS4A1, CD79A/B)": (["MS4A1", "CD79A", "CD79B"], B, T | MONO | {"NK"}),
    "T (CD3D/E, TRAC)": (["CD3D", "CD3E", "TRAC"], T, B | MYELOID),
    "NK (NKG7, GNLY, KLRF1)": (["NKG7", "GNLY", "KLRF1"], {"NK"}, B | MYELOID),
    "Platelet (PPBP, PF4)": (["PPBP", "PF4"], {"Platelet"}, ALL - {"Platelet"}),
    "Erythrocyte (HBB, HBA1/2)": (["HBB", "HBA1", "HBA2"], set(), ALL),
}

# gene -> (expressing population, sibling population sharing its CellTypist-High label)
SUBTYPE_MARKERS = {
    "CD8A": ("CD8 T", "CD4 T"), "CD8B": ("CD8 T", "CD4 T"),
    "SLC4A10": ("MAIT", "CD4 T"), "KLRB1": ("MAIT", "CD4 T"),
    "FCGR3A": ("CD16 Mono", "CD14 Mono"), "CDKN1C": ("CD16 Mono", "CD14 Mono"), "MS4A7": ("CD16 Mono", "CD14 Mono"),
    "CD14": ("CD14 Mono", "CD16 Mono"), "VCAN": ("CD14 Mono", "CD16 Mono"), "S100A12": ("CD14 Mono", "CD16 Mono"),
    "TCL1A": ("Naive B", "Memory B"), "IGHD": ("Naive B", "Memory B"),
}

N_NEIGHBORS = 15
N_PCS = 25


def condition_order():
    k = labels.nunique()
    order = sorted([c for c in labels.columns if c != "shuffled"], key=lambda c: (k[c], c))
    return order + ["shuffled"]


def pretty(c):
    names = {"all_one": "Single label", "lineage": "Lymphoid/myeloid", "ct_high": "CellTypist High (default)", "ct_low": "CellTypist Low", "shuffled": "Shuffled High (control)"}
    if c in names:
        return f"{names[c]} (K={labels[c].nunique()})"
    return f"Leiden r={c.split('_')[1]} (K={labels[c].nunique()})"


def load(condition, genes):
    a = ad.read_h5ad(os.path.join(run_dir, f"adata_cellsweep_{condition}.h5ad"))
    a = a[~a.obs["is_empty"].values].copy()
    a.var_names_make_unique()
    a = a[labels.index]
    raw = sp.csr_matrix(a.layers["raw"], dtype=np.float64)
    den = sp.csr_matrix(a.X, dtype=np.float64)
    return a, raw, den


def knn_purity(emb, groups):
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1).fit(emb)
    idx = nn.kneighbors(emb, return_distance=False)[:, 1:]
    g = np.asarray(groups)
    return float(np.mean(g[idx] == g[:, None]))


def embed(X, hvg_idx):
    ad_tmp = ad.AnnData(X=X[:, hvg_idx].copy())
    sc.pp.normalize_total(ad_tmp, target_sum=1e4)
    sc.pp.log1p(ad_tmp)
    sc.pp.scale(ad_tmp, max_value=10)
    sc.tl.pca(ad_tmp, n_comps=N_PCS, random_state=0)
    return ad_tmp.obsm["X_pca"]


def main():
    ref = labels["ct_low"].map(REF_GROUP)
    assert ref.notna().all(), labels.loc[ref.isna(), "ct_low"].unique()
    ref = ref.values
    order = condition_order()

    summary, lineage_rows, subtype_rows = [], [], []
    per_cell_frac, per_gene_frac = {}, {}
    hvg_idx = None
    raw_embed = None

    for cond in order:
        print(f"scoring {cond}", flush=True)
        a, raw, den = load(cond, None)
        var_names = a.var_names
        removed = raw - den
        removed.data = np.clip(removed.data, 0, None)

        if hvg_idx is None:
            # fixed HVG set from the raw cells, shared by every run
            tmp = ad.AnnData(X=raw.copy(), var=pd.DataFrame(index=var_names))
            sc.pp.highly_variable_genes(tmp, n_top_genes=2000, flavor="seurat_v3")
            hvg_idx = np.where(tmp.var["highly_variable"].values)[0]
            raw_embed = embed(raw, hvg_idx)
            raw_ref_purity = knn_purity(raw_embed, ref)

        raw_cell = np.asarray(raw.sum(1)).ravel()
        per_cell_frac[cond] = np.asarray(removed.sum(1)).ravel() / raw_cell
        raw_gene = np.asarray(raw.sum(0)).ravel()
        keep_gene = raw_gene >= 100
        per_gene_frac[cond] = np.asarray(removed.sum(0)).ravel()[keep_gene] / raw_gene[keep_gene]

        rawc, denc = raw.tocsc(), den.tocsc()
        gidx = {g: i for i, g in enumerate(var_names)}

        # 1. lineage-marker specificity
        off_raw_all = off_rem_all = on_raw_all = on_rem_all = 0.0
        for panel, (genes, on, off) in LINEAGE_PANELS.items():
            cols = [gidx[g] for g in genes]
            r, d = rawc[:, cols], denc[:, cols]
            on_mask, off_mask = np.isin(ref, list(on)), np.isin(ref, list(off))
            off_raw, off_rem = r[off_mask].sum(), (r[off_mask] - d[off_mask]).sum()
            on_raw, on_rem = r[on_mask].sum(), (r[on_mask] - d[on_mask]).sum()
            lineage_rows.append(dict(condition=cond, panel=panel, offtarget_removed=off_rem / off_raw,
                                     ontarget_removed=(on_rem / on_raw) if on_raw > 0 else np.nan,
                                     offtarget_raw_counts=off_raw, ontarget_raw_counts=on_raw))
            if on_raw > 0:
                off_raw_all += off_raw; off_rem_all += off_rem; on_raw_all += on_raw; on_rem_all += on_rem

        # 2. subtype markers
        for g, (pos, neg) in SUBTYPE_MARKERS.items():
            c = gidx[g]
            r, d = np.asarray(rawc[:, c].todense()).ravel(), np.asarray(denc[:, c].todense()).ravel()
            pm, nm = ref == pos, ref == neg
            # pseudobulk CPM using each matrix's own library sizes
            den_cell = np.asarray(den.sum(1)).ravel()
            cpm = lambda x, lib, m: 1e6 * x[m].sum() / lib[m].sum()
            lfc_raw = np.log2((cpm(r, raw_cell, pm) + 1) / (cpm(r, raw_cell, nm) + 1))
            lfc_den = np.log2((cpm(d, den_cell, pm) + 1) / (cpm(d, den_cell, nm) + 1))
            subtype_rows.append(dict(condition=cond, gene=g, expressing=pos, sibling=neg,
                                     retained_in_expressing=d[pm].sum() / r[pm].sum(),
                                     removed_in_sibling=1 - d[nm].sum() / r[nm].sum(),
                                     log2fc_raw=lfc_raw, log2fc_denoised=lfc_den, log2fc_gain=lfc_den - lfc_raw))

        # 4. downstream structure
        emb = embed(den, hvg_idx)
        own = labels[cond].values
        summary.append(dict(
            condition=cond, label=pretty(cond), n_labels=labels[cond].nunique(),
            median_cells_per_label=float(labels[cond].value_counts().median()),
            min_cells_per_label=int(labels[cond].value_counts().min()),
            fraction_counts_removed=removed.sum() / raw.sum(),
            median_alpha_hat=float(np.median(a.obs["alpha_hat"])),
            n_cells_reassigned=int((np.asarray(a.uns["celltype_names"]).astype(str)[a.obs["z_hat"].values.astype(int) - 1] != own).sum()),  # z_hat is a 1-based index into celltype_names
            beta_hat=float(a.uns["beta_hat"]),
            loglike=float(a.uns["loglike"]),
            pooled_offtarget_marker_removed=off_rem_all / off_raw_all,
            pooled_ontarget_marker_removed=on_rem_all / on_raw_all,
            ref_knn_purity=knn_purity(emb, ref), ref_knn_purity_raw=raw_ref_purity,
            own_label_knn_purity=knn_purity(emb, own), own_label_knn_purity_raw=knn_purity(raw_embed, own),
        ))

    summary = pd.DataFrame(summary)
    for other in [DEFAULT, "ct_low"]:
        summary[f"cell_removed_frac_spearman_vs_{other}"] = [spearmanr(per_cell_frac[c], per_cell_frac[other]).correlation for c in summary.condition]
        summary[f"gene_removed_frac_spearman_vs_{other}"] = [spearmanr(per_gene_frac[c], per_gene_frac[other]).correlation for c in summary.condition]
    summary["marker_specificity"] = summary["pooled_offtarget_marker_removed"] - summary["pooled_ontarget_marker_removed"]
    summary["own_label_purity_gain"] = summary["own_label_knn_purity"] - summary["own_label_knn_purity_raw"]
    summary["ref_purity_gain"] = summary["ref_knn_purity"] - summary["ref_knn_purity_raw"]
    lineage = pd.DataFrame(lineage_rows)
    subtype = pd.DataFrame(subtype_rows)

    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    lineage.to_csv(os.path.join(out_dir, "lineage_marker_metrics.csv"), index=False)
    subtype.to_csv(os.path.join(out_dir, "subtype_marker_metrics.csv"), index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 50)
    print(summary.drop(columns=["label"]).round(3).to_string(index=False))
    print(lineage.pivot(index="panel", columns="condition", values="offtarget_removed")[order].round(3).to_string())
    print(lineage.pivot(index="panel", columns="condition", values="ontarget_removed")[order].round(3).to_string())
    print(subtype.pivot(index="gene", columns="condition", values="retained_in_expressing")[order].round(3).to_string())
    print(subtype.pivot(index="gene", columns="condition", values="log2fc_gain")[order].round(3).to_string())

    make_figure(summary, lineage, subtype, order)


def make_figure(summary, lineage, subtype, order):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    s = summary.set_index("condition").loc[order]
    main_conds = [c for c in order if c != "shuffled"]
    ms = s.loc[main_conds]
    x = ms["n_labels"].values
    named = {"all_one": "1 label", "lineage": "lineage", "ct_high": "CT High", "ct_low": "CT Low"}
    ct_color, leiden_color, coarse_color, shuf_color = "#DD8452", "#4C72B0", "#8C8C8C", "#C44E52"

    def ccolor(c):
        return leiden_color if c.startswith("leiden") else ct_color if c.startswith("ct_") else coarse_color

    def points(ax, series, label_points=True, shuffled_value=None):
        leiden = [c for c in main_conds if c.startswith("leiden")]
        ax.plot(s.loc[leiden, "n_labels"], series.loc[leiden], "-", color=leiden_color, lw=1, zorder=2)
        for c in main_conds:
            ax.scatter(s.loc[c, "n_labels"], series.loc[c], color=ccolor(c), s=20, zorder=3)
            if label_points and c in named:
                ax.annotate(named[c], (s.loc[c, "n_labels"], series.loc[c]), textcoords="offset points", xytext=(4, 3), fontsize=6)
        if shuffled_value is not None:
            ax.scatter(s.loc["shuffled", "n_labels"], shuffled_value, marker="x", color=shuf_color, s=28, zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("Number of labels K (log)")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    # a: pooled lineage-marker removal, off vs on target
    ax = axes[0, 0]
    points(ax, s["pooled_offtarget_marker_removed"], shuffled_value=s.loc["shuffled", "pooled_offtarget_marker_removed"])
    points(ax, s["pooled_ontarget_marker_removed"], label_points=False, shuffled_value=s.loc["shuffled", "pooled_ontarget_marker_removed"])
    ax.text(0.98, 0.62, "off-target counts\n(should be removed)", transform=ax.transAxes, ha="right", fontsize=6.5)
    ax.text(0.98, 0.12, "on-target counts\n(should be kept)", transform=ax.transAxes, ha="right", fontsize=6.5)
    ax.set_ylabel("Fraction of lineage-marker counts removed")
    ax.set_title("a  Lineage markers, pooled", loc="left", fontsize=8.5)

    # b / c: per panel on-target (over-correction) and off-target (under-correction)
    # line = merged-High labels then Leiden; squares = CellTypist High/Low; x = shuffled
    line_conds = ["all_one", "lineage"] + [c for c in main_conds if c.startswith("leiden")]
    panel_colors = dict(zip(LINEAGE_PANELS.keys(), plt.cm.tab10.colors))
    for ax, col, title, ylabel in [
        (axes[0, 1], "ontarget_removed", "b  Over-correction (on-target removed)", "Fraction removed from expressing cells"),
        (axes[0, 2], "offtarget_removed", "c  Under-correction (off-target removed)", "Fraction removed from non-expressing cells"),
    ]:
        for panel, sub in lineage.groupby("panel", sort=False):
            sub = sub.set_index("condition")
            if sub[col].isna().all():
                continue
            color = panel_colors[panel]
            ax.plot(s.loc[line_conds, "n_labels"], sub.loc[line_conds, col], "-o", ms=2.5, lw=1, color=color, label=panel)
            ax.scatter(s.loc[["ct_high", "ct_low"], "n_labels"], sub.loc[["ct_high", "ct_low"], col], marker="s", s=22, color=color, edgecolor="k", lw=0.5, zorder=4)
            ax.scatter(s.loc["shuffled", "n_labels"], sub.loc["shuffled", col], marker="x", color=color, s=22)
        ax.set_xscale("log")
        ax.set_xlabel("Number of labels K (log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[0, 2].legend(fontsize=5.5, frameon=False, loc="lower left")
    axes[0, 1].text(0.97, 0.97, "line: merged High, then Leiden\nsquares: CellTypist High / Low\nx: shuffled High", transform=axes[0, 1].transAxes, ha="right", va="top", fontsize=6)

    # d: subtype-marker retention heatmap
    ax = axes[1, 0]
    piv = subtype.pivot(index="gene", columns="condition", values="retained_in_expressing")[order].loc[list(SUBTYPE_MARKERS.keys())]
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0.8, vmax=1.0)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="w" if v < 0.9 else "k")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{g} in {SUBTYPE_MARKERS[g][0]}" for g in piv.index], fontsize=6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{named.get(c, c.replace('leiden_', 'r=')) if c != 'shuffled' else 'shuffled'} ({labels[c].nunique()})" for c in order], rotation=90, fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Fraction retained (clipped at 0.8)", fontsize=6)
    ax.set_title("d  Subtype markers in the expressing subtype", loc="left", fontsize=8.5)

    # e: model parameters absorb contamination as K grows
    ax = axes[1, 1]
    points(ax, s["fraction_counts_removed"], shuffled_value=s.loc["shuffled", "fraction_counts_removed"])
    ax2 = ax.twinx()
    leiden = [c for c in main_conds if c.startswith("leiden")] 
    ax2.plot(ms["n_labels"], ms["beta_hat"], ":", color="k", lw=1)
    ax2.scatter(ms["n_labels"], ms["beta_hat"], marker="d", s=10, color="k")
    ax2.set_ylabel("beta_hat (dotted)", fontsize=7)
    ax2.spines["right"].set_visible(True)
    ax.set_ylabel("Fraction of all counts removed")
    ax.set_title("e  Total removal and beta_hat", loc="left", fontsize=8.5)

    # f: circularity control
    ax = axes[1, 2]
    points(ax, s["own_label_purity_gain"], shuffled_value=s.loc["shuffled", "own_label_purity_gain"])
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_ylim(-0.05, 0.05)
    ax.set_ylabel(f"{N_NEIGHBORS}-NN purity of input labels,\ndenoised minus raw")
    ax.set_title("f  Label self-reinforcement", loc="left", fontsize=8.5)

    handles = [plt.Line2D([], [], marker="o", ls="", color=ct_color, label="CellTypist"), plt.Line2D([], [], marker="o", ls="-", color=leiden_color, label="Leiden"),
               plt.Line2D([], [], marker="o", ls="", color=coarse_color, label="Merged CellTypist High"), plt.Line2D([], [], marker="x", ls="", color=shuf_color, label="Shuffled High (control)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(out_dir, f"celltype_granularity_sensitivity.{ext}"), dpi=300)
    plt.close(fig)

    if os.path.exists(sim_metrics_csv):
        sim = pd.read_csv(sim_metrics_csv)
        sim_main = sim[sim.condition != "shuffled"].sort_values("n_labels")
        shuf = sim[sim.condition == "shuffled"].iloc[0]
        styles = {"leiden": "o", "merge": "s", "split": "^", "truth": "*"}
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
        for ax, cols, ylab in [
            (axes[0], ["fraction_removed", "true_fraction_noise"], "Fraction of counts"),
            (axes[1], ["offtarget_marker_removed", "ontarget_marker_retained", "real_retained"], "Fraction"),
            (axes[2], ["cell_frac_spearman"], "Spearman, estimated vs true\nper-cell contamination"),
        ]:
            for col in cols:
                line, = ax.plot(sim_main["n_labels"], sim_main[col], "-", lw=1, label=col.replace("_", " "))
                for _, row in sim_main.iterrows():
                    ax.scatter(row["n_labels"], row[col], marker=styles[row["condition"].split("_")[0]], s=18, color=line.get_color(), zorder=3)
                ax.scatter(shuf["n_labels"], shuf[col], marker="x", color="#C44E52", s=30, zorder=4)
            ax.axvline(12, color="#2CA02C", lw=1, ls=":")
            ax.set_xscale("log")
            ax.set_xlabel("Number of labels (true K = 12)")
            ax.set_ylabel(ylab)
            ax.legend(fontsize=6, frameon=False)
        fig.suptitle("Simulation with ground truth: merged types (squares), truth (star), random splits (triangles), Leiden (circles), shuffled (red x)", fontsize=7.5)
        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(out_dir, f"celltype_granularity_simulation.{ext}"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
