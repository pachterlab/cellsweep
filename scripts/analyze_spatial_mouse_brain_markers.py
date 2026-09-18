#!/usr/bin/env python
"""
Reviewer point 7: empirical evaluation of cellsweep on a standard (non-xenograft) Visium HD tissue, where no
ground truth exists, via marker specificity before and after correction.

Dataset: Visium HD 3' Mouse Brain (fresh frozen), 8 um bins; cellsweep run by scripts/run_spatial_mouse_brain.py.

A. Spatially restricted genes (genome-wide, data-driven).
   For every gene with >= MIN_GENE_COUNTS raw counts, the raw counts are smoothed (Gaussian, SIGMA_BINS) and the
   gene's "source" is the set of bins holding the top 50% of smoothed mass, dilated by SOURCE_DILATE_UM. Genes whose
   source covers <= MAX_SOURCE_AREA of the tissue are spatially restricted (e.g. Ttr: choroid plexus; Pmch: lateral
   hypothalamus). Counts of these genes far from their source cannot come from the source cells, so a decontamination
   method should remove off-source counts while retaining on-source counts. Broadly expressed genes are the control.
B. Cluster marker specificity with labels independent of cellsweep.
   Space Ranger graph-based clusters and differential expression (binned_outputs/square_008um/analysis). For the top
   markers of each cluster: fraction of the marker's counts that fall in its own cluster, before vs after, and
   the fraction of on-target counts retained. Controls: ubiquitous genes (lowest across-cluster variation).

Because contamination differs between regions and clusters, every retention is also compared with the retention
*expected* if the gene's counts were removed at the same rate as all counts in the same bins
(expected = sum_bins raw_g * r_bin / sum_bins raw_g, with r_bin = cellsweep total / raw total of the bin).
observed / expected < 1 means the gene's counts there were removed preferentially; > 1 means preferentially kept.

Usage: python scripts/analyze_spatial_mouse_brain_markers.py
Outputs: notebooks/output/visium_mouse_brain/reviewer7/
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

cellsweep_dir = "/home/jrich/Desktop/cellsweep"
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "visium_mouse_brain")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "visium_mouse_brain", "reviewer7")
os.makedirs(out_dir, exist_ok=True)

BIN_UM = 8
SIGMA_BINS = 3
SOURCE_DILATE_UM = 40
MAX_SOURCE_AREA = 0.05
BROAD_MIN_AREA = 0.25
MIN_GENE_COUNTS = 2000
FAR_UM = 400
N_MARKERS = 10
EXAMPLE_GENES = ["Ttr", "Pmch"]
DIST_EDGES = [0, 40, 100, 200, 400, 800, 1600, np.inf]

_G = {}  # per-process globals for the gene loop


def _init(rows, cols, shape, tissue, raw_cols, cs_cols, bin_retention):
    _G.update(rows=rows, cols=cols, shape=shape, tissue=tissue, raw=raw_cols, cs=cs_cols, r=bin_retention)


def gene_source_stats(j):
    rows, cols, shape, tissue = _G["rows"], _G["cols"], _G["shape"], _G["tissue"]
    raw = np.asarray(_G["raw"][:, j].todense()).ravel()
    cs = np.asarray(_G["cs"][:, j].todense()).ravel()
    grid = np.zeros(shape)
    np.add.at(grid, (rows, cols), raw)
    smooth = ndi.gaussian_filter(grid, SIGMA_BINS)
    vals = smooth[rows, cols]
    order = np.argsort(vals)[::-1]
    n_top = np.searchsorted(np.cumsum(vals[order]), 0.5 * vals.sum()) + 1
    core = np.zeros(shape, dtype=bool)
    core[rows[order[:n_top]], cols[order[:n_top]]] = True
    dist = ndi.distance_transform_edt(~core) * BIN_UM
    d = dist[rows, cols]
    on = d <= SOURCE_DILATE_UM
    far = d > FAR_UM
    return {
        "j": j,
        "source_area_frac": on.sum() / tissue,
        "raw_total": raw.sum(), "cs_total": cs.sum(),
        "raw_on": raw[on].sum(), "cs_on": cs[on].sum(),
        "raw_far": raw[far].sum(), "cs_far": cs[far].sum(),
        "expected_on": (raw[on] * _G["r"][on]).sum(), "expected_far": (raw[far] * _G["r"][far]).sum(),
        "raw_decay": np.array([raw[(d > lo) & (d <= hi)].mean() if ((d > lo) & (d <= hi)).any() else np.nan for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]),
        "cs_decay": np.array([cs[(d > lo) & (d <= hi)].mean() if ((d > lo) & (d <= hi)).any() else np.nan for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]),
    }


def load():
    adata = ad.read_h5ad(os.path.join(data_dir, "adata_cellsweep.h5ad"))
    adata = adata[~adata.obs["is_empty"].astype(bool)].copy()
    raw = adata.layers["raw"].tocsc()
    cs = adata.X.tocsc()
    return adata, raw, cs


def analysis_restricted_genes(adata, raw, cs):
    rows, cols = adata.obs["array_row"].values, adata.obs["array_col"].values
    shape = (rows.max() + 1, cols.max() + 1)
    totals = np.asarray(raw.sum(axis=0)).ravel()
    genes = np.where(totals >= MIN_GENE_COUNTS)[0]
    print(f"genes with >= {MIN_GENE_COUNTS} counts: {len(genes)}")
    bin_retention = np.asarray(cs.sum(axis=1)).ravel() / np.maximum(np.asarray(raw.sum(axis=1)).ravel(), 1e-12)
    with ProcessPoolExecutor(max_workers=32, initializer=_init, initargs=(rows, cols, shape, len(rows), raw, cs, bin_retention)) as ex:
        results = list(ex.map(gene_source_stats, genes, chunksize=8))
    decay = {adata.var_names[r["j"]]: (r.pop("raw_decay"), r.pop("cs_decay")) for r in results}
    df = pd.DataFrame(results)
    df.index = adata.var_names[df.pop("j")]
    df["on_source_frac_raw"] = df["raw_on"] / df["raw_total"]
    df["on_source_frac_cs"] = df["cs_on"] / df["cs_total"]
    df["pct_on_source_removed"] = 100 * (1 - df["cs_on"] / df["raw_on"])
    df["pct_far_removed"] = 100 * (1 - df["cs_far"] / df["raw_far"].replace(0, np.nan))
    df["pct_total_removed"] = 100 * (1 - df["cs_total"] / df["raw_total"])
    df["on_source_obs_over_expected"] = df["cs_on"] / df["expected_on"]
    df["far_obs_over_expected"] = df["cs_far"] / df["expected_far"].replace(0, np.nan)
    df["class"] = np.where(df["source_area_frac"] <= MAX_SOURCE_AREA, "restricted", np.where(df["source_area_frac"] >= BROAD_MIN_AREA, "broad", "intermediate"))
    df.to_csv(os.path.join(out_dir, "restricted_genes_source_stats.csv"))

    summary = df.groupby("class").agg(n_genes=("raw_total", "size"), median_on_source_frac_raw=("on_source_frac_raw", "median"), median_on_source_frac_cs=("on_source_frac_cs", "median"), median_pct_on_source_removed=("pct_on_source_removed", "median"), median_pct_far_removed=("pct_far_removed", "median"), median_pct_total_removed=("pct_total_removed", "median"), median_on_source_obs_over_expected=("on_source_obs_over_expected", "median"), median_far_obs_over_expected=("far_obs_over_expected", "median"))
    summary.to_csv(os.path.join(out_dir, "restricted_genes_summary.csv"))
    print(summary.round(3).to_string())
    restricted = df[df["class"] == "restricted"]
    print(f"restricted genes with increased on-source fraction: {(restricted['on_source_frac_cs'] > restricted['on_source_frac_raw']).mean() * 100:.1f}%")
    print(restricted.sort_values("raw_total", ascending=False).head(25)[["raw_total", "source_area_frac", "on_source_frac_raw", "on_source_frac_cs", "pct_on_source_removed", "pct_far_removed", "on_source_obs_over_expected", "far_obs_over_expected"]].round(3).to_string())

    # figures: example decay curves + genome-wide scatter
    fig, axes = plt.subplots(1, len(EXAMPLE_GENES) + 1, figsize=(5 * (len(EXAMPLE_GENES) + 1), 4.2))
    labels = [f"{int(lo)}-{int(hi)}" if np.isfinite(hi) else f">{int(lo)}" for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
    for ax, g in zip(axes, EXAMPLE_GENES):
        if g not in decay:
            continue
        r, c = decay[g]
        ax.plot(labels, r, "-o", color="#a6c8ff", label="raw")
        ax.plot(labels, c, "-o", color="#0047b3", label="cellsweep")
        ax.set_yscale("log")
        ax.set_title(f"{g}: {df.loc[g, 'pct_on_source_removed']:.0f}% removed at source, {df.loc[g, 'pct_far_removed']:.0f}% >{FAR_UM} um away")
        ax.set_xlabel("Distance from source region (um)")
        ax.set_ylabel("Mean counts per bin")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(frameon=False)
    ax = axes[-1]
    for cls, color in [("broad", "#bbbbbb"), ("intermediate", "#8da0cb"), ("restricted", "#d62728")]:
        d = df[df["class"] == cls]
        ax.scatter(d["on_source_frac_raw"], d["on_source_frac_cs"], s=6, color=color, alpha=0.6, label=f"{cls} (n={len(d)})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Fraction of gene counts in source region (raw)")
    ax.set_ylabel("... (cellsweep)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "restricted_genes.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # spatial maps of the example genes
    fig, axes = plt.subplots(2, len(EXAMPLE_GENES), figsize=(6 * len(EXAMPLE_GENES), 11), squeeze=False)
    for j, g in enumerate(EXAMPLE_GENES):
        gi = adata.var_names.get_loc(g)
        for i, (mat, name) in enumerate([(raw, "raw"), (cs, "cellsweep")]):
            v = np.asarray(mat[:, gi].todense()).ravel()
            ax = axes[i, j]
            order = np.argsort(v)
            ax.scatter(cols[order], -rows[order], c=np.log1p(v[order]), s=0.05, cmap="magma", vmin=0, vmax=np.log1p(np.asarray(raw[:, gi].todense()).max()))
            ax.set_title(f"{g} ({name}), log1p counts")
            ax.set_aspect("equal")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "restricted_gene_maps.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


def analysis_cluster_markers(adata, raw, cs):
    ana = os.path.join(data_dir, "binned_outputs", "square_008um", "analysis")
    clusters = pd.read_csv(os.path.join(ana, "clustering", "gene_expression_graphclust", "clusters.csv"), index_col=0)["Cluster"]
    de = pd.read_csv(os.path.join(ana, "diffexp", "gene_expression_graphclust", "differential_expression.csv"))
    labels = clusters.reindex(adata.obs_names)
    keep = labels.notna().values
    labels = labels[keep].astype(int).values
    raw, cs = raw.tocsr()[keep], cs.tocsr()[keep]
    print(f"bins with Space Ranger graphclust label: {keep.sum()} / {len(keep)}; clusters: {len(np.unique(labels))}")
    gene_index = pd.Index(adata.var_names)

    K = np.unique(labels)
    onehot = np.zeros((len(labels), len(K)))
    onehot[np.arange(len(labels)), np.searchsorted(K, labels)] = 1
    import scipy.sparse as sp
    L = sp.csr_matrix(onehot)
    raw_by_cluster = np.asarray((L.T @ raw).todense())   # K x G sums
    cs_by_cluster = np.asarray((L.T @ cs).todense())
    bin_retention = np.asarray(cs.sum(axis=1)).ravel() / np.maximum(np.asarray(raw.sum(axis=1)).ravel(), 1e-12)
    expected_by_cluster = np.asarray((L.T @ sp.diags(bin_retention) @ raw).todense())  # K x G, raw counts weighted by bin retention
    sizes = onehot.sum(axis=0)

    rows = []
    for k in K:
        col_fc, col_p, col_mean = f"Cluster {k} Log2 fold change", f"Cluster {k} Adjusted p value", f"Cluster {k} Mean Counts"
        cand = de[(de[col_p] < 0.05) & (de[col_mean] >= 0.1) & de["Feature Name"].isin(gene_index)].sort_values(col_fc, ascending=False).head(N_MARKERS)
        for g in cand["Feature Name"]:
            rows.append((k, g, "marker"))
    # controls: highly expressed genes with the lowest across-cluster CV of mean expression
    means = raw_by_cluster / sizes[:, None]
    expr = np.asarray(raw.sum(axis=0)).ravel()
    top = np.argsort(expr)[::-1][:500]
    cv = means[:, top].std(axis=0) / means[:, top].mean(axis=0)
    for gi in top[np.argsort(cv)[:30]]:
        rows.append((K[np.argmax(means[:, gi])], adata.var_names[gi], "ubiquitous control"))

    out = []
    for k, g, kind in rows:
        gi = gene_index.get_loc(g)
        ki = np.searchsorted(K, k)
        r_tot, c_tot = raw_by_cluster[:, gi].sum(), cs_by_cluster[:, gi].sum()
        r_in, c_in = raw_by_cluster[ki, gi], cs_by_cluster[ki, gi]
        r_other_mean = (r_tot - r_in) / (sizes.sum() - sizes[ki])
        c_other_mean = (c_tot - c_in) / (sizes.sum() - sizes[ki])
        out.append({"cluster": k, "gene": g, "kind": kind,
                    "in_cluster_frac_raw": r_in / r_tot, "in_cluster_frac_cs": c_in / max(c_tot, 1e-12),
                    "log2_enrichment_raw": np.log2((r_in / sizes[ki] + 1e-6) / (r_other_mean + 1e-6)),
                    "log2_enrichment_cs": np.log2((c_in / sizes[ki] + 1e-6) / (c_other_mean + 1e-6)),
                    "pct_in_cluster_retained": 100 * c_in / r_in, "pct_off_cluster_retained": 100 * (c_tot - c_in) / max(r_tot - r_in, 1e-12),
                    "in_cluster_obs_over_expected": c_in / expected_by_cluster[ki, gi],
                    "off_cluster_obs_over_expected": (c_tot - c_in) / max(expected_by_cluster[:, gi].sum() - expected_by_cluster[ki, gi], 1e-12)})
    out = pd.DataFrame(out).drop_duplicates(["cluster", "gene", "kind"])
    out["delta_in_cluster_frac"] = out["in_cluster_frac_cs"] - out["in_cluster_frac_raw"]
    out["delta_log2_enrichment"] = out["log2_enrichment_cs"] - out["log2_enrichment_raw"]
    # >1: in-cluster counts kept preferentially relative to off-cluster counts, beyond bin-level removal rates
    out["specificity_gain"] = out["in_cluster_obs_over_expected"] / out["off_cluster_obs_over_expected"]
    out.to_csv(os.path.join(out_dir, "cluster_marker_specificity.csv"), index=False)
    summary = out.groupby("kind").agg(n=("gene", "size"), median_in_cluster_frac_raw=("in_cluster_frac_raw", "median"), median_in_cluster_frac_cs=("in_cluster_frac_cs", "median"), frac_increased=("delta_in_cluster_frac", lambda x: (x > 0).mean()), median_delta_log2_enrichment=("delta_log2_enrichment", "median"), median_pct_in_cluster_retained=("pct_in_cluster_retained", "median"), median_pct_off_cluster_retained=("pct_off_cluster_retained", "median"), median_in_cluster_obs_over_expected=("in_cluster_obs_over_expected", "median"), median_off_cluster_obs_over_expected=("off_cluster_obs_over_expected", "median"), frac_off_removed_preferentially=("off_cluster_obs_over_expected", lambda x: (x < 1).mean()), median_specificity_gain=("specificity_gain", "median"), frac_specificity_gain_gt1=("specificity_gain", lambda x: (x > 1).mean()))
    summary.to_csv(os.path.join(out_dir, "cluster_marker_specificity_summary.csv"))
    print(summary.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for kind, color in [("ubiquitous control", "#bbbbbb"), ("marker", "#0047b3")]:
        d = out[out["kind"] == kind]
        axes[0].scatter(d["in_cluster_frac_raw"], d["in_cluster_frac_cs"], s=12, color=color, alpha=0.7, label=f"{kind} (n={len(d)})")
        axes[1].scatter(d["off_cluster_obs_over_expected"], d["in_cluster_obs_over_expected"], s=12, color=color, alpha=0.7, label=kind)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("Fraction of gene counts in its cluster (raw)")
    axes[0].set_ylabel("... (cellsweep)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1, color="k", ls="--", lw=0.8)
    axes[1].axvline(1, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Off-cluster counts retained, observed / expected")
    axes[1].set_ylabel("In-cluster counts retained, observed / expected")
    fig.suptitle("Space Ranger graph-based cluster markers (labels independent of cellsweep)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cluster_marker_specificity.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    adata, raw, cs = load()
    print(f"non-empty bins: {adata.n_obs}; median alpha_hat {adata.obs['alpha_hat'].median():.3f}; counts removed {100 * (1 - cs.sum() / raw.sum()):.1f}%")
    analysis_restricted_genes(adata, raw, cs)
    analysis_cluster_markers(adata, raw, cs)


if __name__ == "__main__":
    main()
