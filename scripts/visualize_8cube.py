import os
import anndata as ad
import pandas as pd
import numpy as np
import itertools
import argparse
import resource
import sys
import cellsweep.utils as cs_utils

debug = False
plates = ["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"]  # ["igvf_003"]  #? debug
print_custom_markers = True
overwrite = False  # overridden by --overwrite

parser = argparse.ArgumentParser(description="Run plates processing pipeline.")
parser.add_argument("--plates", nargs="+", default=["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"], help="List of plate names (default: all plates)",)
parser.add_argument("--tools", nargs="+", default=[], choices=["cellbender", "soupx", "decontx"], help="Alternate tools to include alongside cellsweep (default: none).",)
parser.add_argument("--celltypes", nargs="+", default=None, help="Restrict per-celltype gene-count plots to these celltypes (case-insensitive). Default: all celltypes.",)
parser.add_argument("--plot-types", dest="plot_types", nargs="+", default=None, choices=["joint", "gene_counts", "gene_counts_tissue"], help="Plot families to produce: 'joint' (per-plate cross-tissue scatterplot), 'gene_counts' (per-celltype gene-count scatterplots), 'gene_counts_tissue' (tissue-aggregate gene-count scatterplot). Default: all.",)
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plot files (default: skip plots that already exist).",)
args = parser.parse_args()
plates = args.plates
alternate_tools = args.tools
celltypes = args.celltypes
plot_types = args.plot_types
overwrite = args.overwrite
include_cellbender = "cellbender" in alternate_tools


# Set max RAM usage in bytes
max_ram_gb = 500  # GB
MAX_RAM = max_ram_gb * 1024**3

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, MAX_RAM))

cellsweep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "8cubed")
eight_cubed_markers_path = os.path.join(data_dir, "8_cube_marker_genes.csv")
gene_id_name_map_path = os.path.join(data_dir, "gene_id_name_map.csv")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "8cubed")
os.makedirs(out_dir, exist_ok=True)
custom_markers = {
    'CortexHippocampus': ["Snap25", "Nrxn3", "Nrxn1"],  # Snap25: found in plate 3, tissue heart, cluster 36; Nrxn3, Nrxn1: found in plate 11, tissue gastroc, cluster 38
    'Heart': ["Tnnt2", "Myh6"],
    'Liver': ["Alb"],  # found in plate 5, tissue heart, cluster 30
    'HypothalamusPituitary': [],
    'Gonads': [],
    'Adrenal': ["Star"],  # found in plate 9, tissue kidney, cluster 14
    'Kidney': ["Slc34a1"],  # found in plate 10, tissue gastroc, cluster 32
    'Gastrocnemius': ["Myh4"]  # found in plate 10, tissue kidney, clusters 0,28,29,30
}

# custom_markers = {
#     'CortexHippocampus': ["Nrxn3", "Nrxn1", "Meis2", "Slc17a7", "Mir124a-1hg", "Snap25"],
#     'Heart': [],
#     'Liver': ["Cyp1a2", "Ttr", "Alb"],
#     'HypothalamusPituitary': [],
#     'Gonads': [],
#     'Adrenal': ["Chga", "Star"],
#     'Kidney': ["Slc5a2", "Slc34a1", "Akr1c21"],
#     'Gastrocnemius': ["Myh4", "Myh2", "Myh1"]
# }

all_custom_markers_start_with_ensmug = all(gene.startswith("ENSMUG") for genes in custom_markers.values() for gene in genes)
gene_name_to_id = None


def log_sensitivity_specificity(stats_df, out_dir, celltypes=None):
    """Print and save a sensitivity/specificity stat log derived from the signal/noise stats.

    Sensitivity (TPR) = signal_retained (fraction of true signal counts kept);
    specificity (TNR) = noise_removed (fraction of contaminating counts removed). This mirrors
    benchmarking.ipynb cell 56, which count-weights signal_retained/noise_removed per tool by the
    raw signal/noise counts. A per-tool count-weighted summary is printed (and written to CSV),
    along with a per-(tool, plate, tissue, celltype) breakdown (restricted to `celltypes` if given).
    """
    if stats_df is None or len(stats_df) == 0:
        print("No signal/noise stats available; skipping sensitivity/specificity log.")
        return

    def _wavg(values, weights):
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
        return float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan

    # per-tool count-weighted summary (weights = raw signal / raw noise counts, like the notebook)
    summary_rows = []
    for tool, d in stats_df.groupby("tool"):
        summary_rows.append({
            "tool": tool,
            "sensitivity_weighted": _wavg(d["sensitivity"], d["signal_raw_counts"]),
            "specificity_weighted": _wavg(d["specificity"], d["noise_raw_counts"]),
            "sensitivity_mean": float(np.nanmean(d["sensitivity"])),
            "specificity_mean": float(np.nanmean(d["specificity"])),
            "n_groups": int(len(d)),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("tool")
    summary_path = os.path.join(out_dir, "sensitivity_specificity_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n================ SENSITIVITY / SPECIFICITY STAT LOG ================")
    print("Sensitivity = signal retained (TPR); Specificity = noise removed (TNR)")
    print("\nPer-tool summary (count-weighted across plate/tissue/celltype groups):")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # focused per-(tool, plate, tissue, celltype) breakdown
    detail = stats_df
    if celltypes is not None:
        wanted = {c.lower() for c in celltypes}
        detail = detail[detail["celltype"].astype(str).str.lower().isin(wanted)]
    if len(detail) > 0:
        cols = ["tool", "plate", "tissue", "celltype", "n_cells", "sensitivity", "specificity"]
        detail = detail[cols].sort_values(["plate", "tissue", "celltype", "tool"])
        label = "requested celltypes" if celltypes is not None else "all celltypes"
        print(f"\nPer-(tool, plate, tissue, celltype) breakdown ({label}):")
        print(detail.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote per-tool sensitivity/specificity summary to {summary_path}")
    print("===================================================================\n")
    return summary_df


def compute_signal_noise_stats(dict_of_adata_dicts, custom_markers, gene_name_to_id, out_dir):
    """For each (tool, plate, tissue, celltype), report fraction of signal retained and fraction
    of noise removed.

    Signal = this tissue's own marker genes (custom_markers[this_tissue]); noise = the OTHER
    tissue's marker genes in the same plate (cross-tissue contamination that should be removed).
    e.g. plate igvf_003, tissue Heart -> the other tissue is CortexHippocampus, so its markers
    (Snap25, Nrxn3, Nrxn1) are the noise tracked in Heart cells. Subsetting mirrors the
    plate-tissue-celltype gene scatterplots in make_8cubed_plots (common cells, Tissue, celltype).

    By the time this runs, custom_markers holds ENSMUSG gene IDs and gene_name_to_id maps
    name -> id, so it is inverted here to display readable gene symbols.
    """
    id_to_name = {v: k for k, v in gene_name_to_id.items()} if gene_name_to_id else {}

    def _sum(adata, genes):
        # sum counts over the given genes and all cells in adata (genes already filtered to var_names)
        if len(genes) == 0 or adata.n_obs == 0:
            return 0.0
        return float(adata[:, genes].X.sum())

    adata_raw_dict = dict_of_adata_dicts.get("raw", {})
    rows = []
    for tool, adata_dict in dict_of_adata_dicts.items():
        if tool == "raw":
            continue
        for plate, adata_processed in adata_dict.items():
            adata_raw = adata_raw_dict.get(plate)
            if adata_raw is None:
                print(f"  Warning: no raw adata for plate {plate}; skipping stats.")
                continue

            # restrict both to the cells they share (same as make_8cubed_plots)
            common_cells = adata_raw.obs_names.intersection(adata_processed.obs_names)
            adata_raw_p = adata_raw[common_cells]
            adata_proc_p = adata_processed[common_cells]

            tissues = adata_proc_p.obs["Tissue"].unique().tolist()
            if len(tissues) != 2:
                print(f"  Warning: plate {plate} does not have exactly 2 tissues ({tissues}); skipping stats.")
                continue

            for this_tissue in tissues:
                other_tissue = tissues[1] if this_tissue == tissues[0] else tissues[0]

                raw_tissue = adata_raw_p[adata_raw_p.obs["Tissue"] == this_tissue]
                proc_tissue = adata_proc_p[adata_proc_p.obs["Tissue"] == this_tissue]

                # keep only marker genes present in both matrices so raw/processed sums are comparable
                def _present(genes):
                    return [g for g in (genes or []) if g in raw_tissue.var_names and g in proc_tissue.var_names]
                signal_markers = _present(custom_markers.get(this_tissue, []))
                noise_markers = _present(custom_markers.get(other_tissue, []))

                for celltype in raw_tissue.obs["celltype"].dropna().drop_duplicates().values:
                    raw_sub = raw_tissue[raw_tissue.obs["celltype"] == celltype]
                    proc_sub = proc_tissue[proc_tissue.obs["celltype"] == celltype]
                    if raw_sub.n_obs == 0 or proc_sub.n_obs == 0:
                        continue

                    sig_raw, sig_proc = _sum(raw_sub, signal_markers), _sum(proc_sub, signal_markers)
                    noise_raw, noise_proc = _sum(raw_sub, noise_markers), _sum(proc_sub, noise_markers)

                    # clamp to [0, 1] like the simulation's min/max(0,.) confusion components:
                    # a tool can retain at most all signal (min) and remove at most all noise (max(0,.))
                    signal_retained = min(1.0, sig_proc / sig_raw) if sig_raw > 0 else np.nan
                    noise_removed = max(0.0, 1 - noise_proc / noise_raw) if noise_raw > 0 else np.nan

                    # Sensitivity (TPR) = fraction of true signal counts retained = signal_retained.
                    # Specificity (TNR) = fraction of contaminating noise counts removed = noise_removed.
                    # (Same definition as benchmarking.ipynb's signal_retained / noise_removed.)
                    rows.append({
                        "tool": tool,
                        "plate": plate,
                        "tissue": this_tissue,
                        "other_tissue": other_tissue,
                        "celltype": celltype,
                        "n_cells": int(proc_sub.n_obs),
                        "signal_markers": ",".join(id_to_name.get(g, g) for g in signal_markers),
                        "noise_markers": ",".join(id_to_name.get(g, g) for g in noise_markers),
                        "signal_retained": signal_retained,
                        "noise_removed": noise_removed,
                        "sensitivity": signal_retained,
                        "specificity": noise_removed,
                        "signal_raw_counts": sig_raw,
                        "signal_proc_counts": sig_proc,
                        "noise_raw_counts": noise_raw,
                        "noise_proc_counts": noise_proc,
                    })

    stats_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "signal_noise_stats_by_plate_tissue_celltype.csv")
    stats_df.to_csv(out_path, index=False)
    print(f"Wrote signal/noise stats ({len(stats_df)} rows) to {out_path}")
    return stats_df


# Convert custom marker gene symbols to ENSMUSG gene IDs (the custom_markers above are symbols).
# Done up front (independent of which tools are loaded) so SoupX/DecontX-only runs still work.
if custom_markers is not None and len(custom_markers) > 0 and not all_custom_markers_start_with_ensmug:
    gene_name_to_id = pd.read_csv(gene_id_name_map_path).set_index("gene_name")["gene_id"].to_dict()
    for tissue in custom_markers:
        gene_ids = []
        for gene_name in custom_markers[tissue]:
            gene_id = gene_name_to_id.get(gene_name)
            if gene_id is not None:
                gene_ids.append(gene_id)
        custom_markers[tissue] = gene_ids
    all_custom_markers_start_with_ensmug = True


def _attach_obs_metadata(adata_tool, adata_raw_ref, cols=("Tissue", "celltype")):
    """Transfer obs metadata (Tissue, celltype) from the raw adata onto a processed adata.

    SoupX/DecontX outputs are reconstructed from bare matrix files and carry no obs columns,
    so the metadata that make_8cubed_plots/compute_signal_noise_stats need is copied over
    here by aligning on barcode. Restricting to shared barcodes also mirrors the intersection
    those downstream functions perform anyway.
    """
    common = adata_tool.obs_names.intersection(adata_raw_ref.obs_names)
    adata_tool = adata_tool[common].copy()
    ref_obs = adata_raw_ref.obs.loc[common]
    for col in cols:
        if col in ref_obs.columns:
            adata_tool.obs[col] = ref_obs[col].values
    return adata_tool


def _load_alternate_tool(tool, plate, adata_raw_ref):
    tool_path = os.path.join(data_dir, plate, f"{tool}.h5ad")
    if not os.path.exists(tool_path):
        print(f"  File {tool_path} does not exist, skipping {tool} for plate {plate}...")
        return None
    print(f"  Loading {tool} for plate {plate}...")
    adata_tool = ad.read_h5ad(tool_path)

    # The viz only needs X, barcodes and gene ids. CellBender h5ads carry ~30GB of unused layers
    # (plus obsm/obsp/etc); drop them so the multi-tool, multi-plate working set stays in memory.
    for _slot in (adata_tool.layers, adata_tool.obsm, adata_tool.obsp, adata_tool.varm, adata_tool.varp):
        for _k in list(_slot.keys()):
            del _slot[_k]
    adata_tool.uns = {}

    if tool == "cellbender":
        if "Subpool" in adata_tool.obs_names[0]:
            adata_tool.obs_names = adata_tool.obs_names.str.replace("Subpool", "Sublibrary", regex=False)
        if adata_tool.obs_names[0].endswith("-0") or adata_tool.obs_names[0].endswith("-1"):  # strip "-0/-1" suffix from barcodes added by CellBender
            adata_tool.obs_names = [bc[:-2] for bc in adata_tool.obs_names]
        if not adata_tool.var_names[0].startswith("ENSMUSG") and "gene_id" in adata_tool.var.columns:
            adata_tool.var_names = adata_tool.var["gene_id"].astype(str)  # Assign gene_id as the new index

    adata_tool.var_names_make_unique()
    adata_tool = _attach_obs_metadata(adata_tool, adata_raw_ref)
    if adata_tool.n_obs == 0:
        print(f"  WARNING: {tool} for plate {plate} has 0 barcodes overlapping raw after the index transform — check the file (e.g. missing/renamed barcodes). Skipping {tool} for {plate}.")
        return None
    if debug:  # filter to the same 5000 cells as above for debugging
        adata_tool = adata_tool[adata_tool.obs_names.isin(barcodes), :].copy()
    return adata_tool


try:
    adata_raw_filtered_dict, adata_cellsweep_dict = {}, {}
    adata_tool_dicts = {tool: {} for tool in alternate_tools}
    for plate in plates:
        print(f"Loading data for plate {plate}...")
        adata_raw_filtered_path = os.path.join(data_dir, plate, "raw_counts_removed_empty_barcodes.h5ad")
        if not os.path.exists(adata_raw_filtered_path):
            print(f"  File {adata_raw_filtered_path} does not exist, skipping...")
            continue
        adata_raw_filtered = ad.read_h5ad(adata_raw_filtered_path)
        adata_raw_filtered.var_names_make_unique()
        if debug:  # filter to 5000 cells for debugging
            np.random.seed(42)
            adata_raw_filtered = adata_raw_filtered[np.random.choice(adata_raw_filtered.n_obs, size=5000, replace=False), :].copy()
            barcodes = adata_raw_filtered.obs_names
        adata_raw_filtered_dict[plate] = adata_raw_filtered

        adata_cellsweep_path = os.path.join(data_dir, plate, "cellsweep.h5ad")
        if not os.path.exists(adata_cellsweep_path):
            print(f"  File {adata_cellsweep_path} does not exist, skipping...")
            continue
        adata_cellsweep = ad.read_h5ad(adata_cellsweep_path)
        adata_cellsweep = adata_cellsweep[~adata_cellsweep.obs["is_empty"]].copy()
        adata_cellsweep.var_names_make_unique()
        if debug:  # filter to the same 5000 cells as above for debugging
            adata_cellsweep = adata_cellsweep[adata_cellsweep.obs_names.isin(barcodes), :].copy()
        adata_cellsweep_dict[plate] = adata_cellsweep
        adata_cellsweep = None  # free memory

        for tool in alternate_tools:
            adata_tool = _load_alternate_tool(tool, plate, adata_raw_filtered_dict[plate])
            if adata_tool is not None:
                adata_tool_dicts[tool][plate] = adata_tool

        adata_raw_filtered = None  # free memory

    if custom_markers is not None and len(custom_markers) > 0 and not all_custom_markers_start_with_ensmug:
        raise ValueError("Custom markers contain gene names that were not found in the gene_id/name map; cannot proceed.")

    dict_of_adata_dicts = {
        "raw": adata_raw_filtered_dict,
        "cellsweep": adata_cellsweep_dict,
    }
    for tool in alternate_tools:
        dict_of_adata_dicts[tool] = adata_tool_dicts[tool]
    print("Computing signal/noise stats per plate-tissue-celltype...")
    stats_df = compute_signal_noise_stats(dict_of_adata_dicts, custom_markers, gene_name_to_id, out_dir)
    log_sensitivity_specificity(stats_df, out_dir, celltypes=celltypes)
    print("Generating 8cubed plots...")
    cs_utils.make_8cubed_plots(dict_of_adata_dicts, eight_cubed_markers_path, custom_markers=custom_markers, gene_name_to_id=gene_name_to_id, print_custom_markers=print_custom_markers, out_dir=out_dir, overwrite=overwrite, celltypes=celltypes, plot_types=plot_types)
except MemoryError:
    print("❌ Memory limit exceeded — exiting")  # might just print 'Segmentation fault (core dumped)' rather than this
    sys.exit(1)
