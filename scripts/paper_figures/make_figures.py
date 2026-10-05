"""Assemble the paper's figures from the panels written by the jobs in scripts/paper_figures/jobs.txt.

usage: python scripts/paper_figures/make_figures.py [NAME ...] [--copy-to PAPER_FIGURES_DIR] [--list]

NAME is a file name in the paper's Figures/ directory without .pdf (e.g. Fig3 Supp8); default: every figure listed
below. Each figure is written to notebooks/output/paper_figures/figures/NAME.pdf (+ .png preview); with --copy-to it
is also copied into the paper's Figures/ directory. Underscore-prefixed files there are intermediate
panels.

File names follow the figure numbers in the text: Fig<n>.pdf is Fig. n and Supp<n>.pdf is Fig. S<n>.
"""
import argparse
import os
import shutil
import sys

from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from assemble_figure import OUT_DIR as F, assemble, combine, pad_width, side_label, titled  # noqa: E402

REPO = os.path.abspath(os.path.join(here, "..", ".."))
O = os.path.join(REPO, "notebooks", "output")
ASSETS = os.path.join(here, "assets")
TOOLS = ["soupx", "cellbender", "decontx", "scar"]
TOOL_LABELS = ["SoupX", "CellBender", "DecontX", "scAR"]


def copy_pdf(src, name):
    """Figures made in one piece by a notebook or script: copy the PDF (and PNG preview, if any)."""
    os.makedirs(F, exist_ok=True)
    shutil.copy(src, os.path.join(F, f"{name}.pdf"))
    png = os.path.splitext(src)[0] + ".png"
    if os.path.exists(png):
        shutil.copy(png, os.path.join(F, f"{name}.png"))
    print(f"wrote {F}/{name}.pdf (from {src})")


# ---------- main figures ----------
def fig2():
    copy_pdf(f"{O}/Fig2.pdf", "Fig2")   # summary_heatmap.ipynb


def fig3():
    h = f"{O}/hgmm_12k"
    assemble([[("A", f"{h}/cellsweep_human_mouse_contamination_histograms_by_cells.png"),
               ("B", f"{h}/cellsweep_joint_scatterplot.png")]], "Fig3", width_in=7.0)


def fig4():
    rows = [[("A", f"{O}/smartseq_10x/smartseq_cellsweep_human_mouse_contamination_histograms_by_cells.png"),
             ("B", f"{O}/smartseq_10x/smartseq_human_mouse_cellsweep_joint_scatterplot.png")],
            [("C", f"{O}/ATAC_hgmm_10k/cellsweep_human_mouse_contamination_histograms_by_cells.png"),
             ("D", f"{O}/ATAC_hgmm_10k/cellsweep_joint_scatterplot.png")],
            [("E", f"{O}/visium_human_mouse/visium_human_mouse_cellsweep_joint_scatterplot.png"),
             ("F", f"{O}/visium_human_mouse/visium_hd_alpha_hat_interface_excluded.png")]]
    assemble(rows, "Fig4", width_in=7.0, row_labels=["Smart-seq2", "ATAC-seq", "Visium HD"], row_label_width=0.06)


def fig5():
    o = f"{O}/pbmc8k"
    dots = combine([f"{o}/dotplot_raw_with_raw_clusters_cellbender_fig2.png", f"{o}/dotplot_cellsweep_with_raw_clusters_cellbender_fig2.png"],
                   ["Raw", "Processed"], f"{F}/_Fig5C.png", title_x=0.33)
    assemble([[("A", f"{o}/cellsweep_vs_raw_matrix_expression_scatterplot.png"),
               ("B", f"{o}/cellsweep_vs_raw_cell_expression_scatterplot.png")],
              [("C", dots), ("D", f"{o}/cluster_tightness_bars.png")]],   # D: benchmarking.ipynb, "Cluster tightness"
             "Fig5", width_in=7.0)


# 8 cubed panels (8cube.ipynb, cell 35): plate -> [(tissue and cell type as in the file names,
# tissue and cell type as shown)], in the order they appear in Fig6 and in Supp13/Supp14
CUBE_CELLTYPES = {"igvf_003": [("Heart", "atrial cardiac myocyte", "Heart", "Atrial Cardiac Myocyte"),
                               ("CortexHippocampus", "glutamatergic neuron", "Cortex/Hippocampus", "Glutamatergic Neuron")],
                  "igvf_009": [("Kidney", "proximal tubule epithelial cell", "Kidney", "proximal tubule epithelial cell"),
                               ("Adrenal", "zona fasciculata", "Adrenal", "zona fasciculata")]}


def cube_scatter(plate, tissue, ct, tool):
    return (f"{O}/8cubed/plate_{plate}/tissue_celltype_gene_scatterplots/"
            f"plate_{plate}_tissue_{tissue}_celltype_{ct}_{tool}_gene_counts_scatterplot.png")


def fig6():
    spec = [(L, plate, *cts) for L, (plate, cts) in zip("BCDE", [(pl, c) for pl in CUBE_CELLTYPES for c in CUBE_CELLTYPES[pl]])]
    p = {L: titled(cube_scatter(pl, t, c, "cellsweep"), [f"Plate: {pl}", f"Tissue: {tt}", f"Celltype: {ct}"], f"{F}/_Fig6{L}.png", title_size=14)
         for L, pl, t, c, tt, ct in spec}
    a = pad_width(f"{ASSETS}/Fig6A_8cube_schematic.png", 0.62, f"{F}/_Fig6A.png")   # static schematic
    assemble([[("A", a)], [("B", p["B"]), ("C", p["C"])], [("D", p["D"]), ("E", p["E"])]], "Fig6", width_in=7.0)


def fig7():
    i = f"{O}/pbmc8k/idempotency"
    assemble([[("A", f"{i}/diff_counts.png"), ("B", f"{i}/diff_cells.png")]], "Fig7", width_in=7.0)


def fig8():
    r = f"{O}/pbmc8k/runtime"
    main = Image.open(f"{r}/runtime_ascending_comparison.png").convert("RGBA")
    leg = Image.open(f"{r}/runtime_legend.png").convert("RGBA")
    scale = 0.2 * main.width / leg.width   # legend about 20% of the plot width
    leg = leg.resize((int(leg.width * scale), int(leg.height * scale)), Image.LANCZOS)
    main.alpha_composite(leg, (int(0.58 * main.width), int(0.03 * main.height)))   # right of the inset, near the top
    os.makedirs(F, exist_ok=True)
    main.save(f"{F}/_Fig8.png")
    assemble([[(None, f"{F}/_Fig8.png")]], "Fig8", width_in=5.0)


def fig9():
    s = f"{O}/simulation1_small_noise"
    assemble([[("A", f"{ASSETS}/Fig9A_simulation_schematic.png"), ("B", f"{s}/dotplot_raw.png")],   # A: static schematic
              [("C", f"{s}/cross_species_joint_scatterplot_total_signal_vs_total_noise_cellsweep.png"), ("D", f"{s}/dotplot_cellsweep.png")]],
             "Fig9", width_in=7.0)


def fig10():   # panels saved by melanoma_discovery.ipynb, Section 7
    m = f"{O}/melanoma_discovery"
    assemble([[("A", f"{m}/fig10a_sensitivity_specificity.png"), ("B", f"{m}/fig10b_antigen_counts_t_cells.png")],
              [("C", f"{m}/fig10c_lineage_markers_removed_in_lineage.png"), ("D", f"{m}/fig10d_lineage_markers_removed_elsewhere.png")]],
             "Fig10", width_in=7.0)


def supp18():   # Fig. S18
    g = f"{O}/pbmc8k/celltype_granularity"
    assemble([[("A", f"{g}/celltype_granularity_signal_vs_noise.png"), ("B", f"{g}/empty_barcode_signal_vs_noise.png")]], "Supp18", width_in=7.0)


# ---------- supplementary figures ----------
def supp1():
    h, ne = f"{O}/hgmm_12k", f"{O}/hgmm_12k_no_empties"
    rows = [[(a, f"{h}/{t}_human_mouse_contamination_histograms_by_cells.png"), (b, f"{h}/{t}_joint_scatterplot.png")]
            for t, a, b in zip(TOOLS, "ACEG", "BDFH")]
    rows.append([("I", f"{ne}/cellsweep_human_mouse_contamination_histograms_by_cells.png"), ("J", f"{ne}/cellsweep_joint_scatterplot.png")])
    assemble(rows, "Supp1", width_in=7.0, row_labels=TOOL_LABELS + ["CellSweep\nno_empties"], row_label_width=0.07)


def supp2():
    h = f"{O}/hgmm_12k"
    letters = iter("ABCDEFGHIJKLMNO")
    rows = [[(next(letters), f"{h}/{t}_vs_raw_{k}_expression_scatterplot.png") for k in ("matrix", "cell", "gene")]
            for t in ["cellsweep"] + TOOLS]
    assemble(rows, "Supp2", width_in=7.0, row_labels=["CellSweep"] + TOOL_LABELS, row_label_width=0.06)


def supp3():
    assemble([[(None, f"{O}/hgmm_12k/hgmm_12k_per_tool_failures.png")]], "Supp3", width_in=7.0)   # hgmm_failure_modes.ipynb


def supp4():
    k = f"{O}/kidney_nuclei_10k"
    assemble([[(L, f"{k}/cellsweep_vs_raw_{m}_expression_scatterplot.png") for L, m in zip("ABC", ("matrix", "cell", "gene"))]], "Supp4", width_in=7.0)


def supp5():   # Janssen et al. accuracy (Kendall tau vs RMSLE), janssen_snrna.ipynb
    copy_pdf(f"{O}/janssen2023/janssen_accuracy.pdf", "Supp5")


def supp6():   # Fig. S6
    v = f"{O}/visium_human_mouse"
    assemble([[("A", f"{v}/knee_plot.png"), ("B", f"{v}/alpha_hat_histogram_interface_excluded.png")]], "Supp6", width_in=7.0)


def supp7():   # Fig. S7
    v = f"{O}/visium_human_mouse"
    c = pad_width(f"{v}/visium_hd_alpha_hat.png", 0.6, f"{F}/_Supp6C.png")
    assemble([[("A", f"{v}/visium_human_mouse_cellsweep_joint_scatterplot_majority_rule.png"),
               ("B", f"{v}/visium_human_mouse_cellsweep_joint_scatterplot_purity_rule.png")], [("C", c)]], "Supp7", width_in=7.0)


def supp8():   # Fig. S8: composed by spatial_mouse_brain.ipynb
    os.makedirs(F, exist_ok=True)
    im = Image.open(f"{O}/visium_mouse_brain/mouse_brain_supplement.png").convert("RGB")
    im.save(f"{F}/Supp8.pdf", resolution=300)
    im.save(f"{F}/Supp8.png")
    print(f"wrote {F}/Supp8.pdf")


def supp9():   # Fig. S9
    o, ne = f"{O}/pbmc8k", f"{O}/pbmc8k_no_empties"
    letters = iter("ABCDEFGHIJKLMNO")
    rows = []
    for i, (t, d) in enumerate([(t, o) for t in TOOLS] + [("cellsweep", ne)]):
        dots = combine([f"{d}/dotplot_raw_with_raw_clusters_cellbender_fig2.png", f"{d}/dotplot_{t}_with_raw_clusters_cellbender_fig2.png"],
                       ["Raw", "Processed"] if i == 0 else ["", ""], f"{F}/_Supp9_{t}_{i}.png", title_x=0.33)
        rows.append([(next(letters), f"{d}/{t}_vs_raw_matrix_expression_scatterplot.png"),
                     (next(letters), f"{d}/{t}_vs_raw_cell_expression_scatterplot.png"), (next(letters), dots)])
    assemble(rows, "Supp9", width_in=7.0, row_labels=TOOL_LABELS + ["CellSweep\nno_empties"], row_label_width=0.07)


def supp10():   # Fig. S10
    o = f"{O}/pbmc8k"
    letters = iter("ABCDEFGHIJ")
    rows = [[(next(letters), f"{o}/{t}_knee_plot.png"), (next(letters), f"{o}/{t}_vs_raw_gene_expression_scatterplot.png")]
            for t in ["cellsweep"] + TOOLS]
    assemble(rows, "Supp10", width_in=6.0, row_labels=["CellSweep"] + TOOL_LABELS, row_label_width=0.08)


def supp11():   # Fig. S11
    o = f"{O}/pbmc8k"
    letters = iter("ABCDEFGHIJKL")
    rows = [[(next(letters), f"{o}/{t}_vs_cellsweep_{k}_expression_scatterplot.png") for k in ("matrix", "cell", "gene")] for t in TOOLS]
    assemble(rows, "Supp11", width_in=7.0, row_labels=TOOL_LABELS, row_label_width=0.07)


def supp12():   # Fig. S12
    o = f"{O}/pbmc8k"
    letters = iter("ABCDEFGHIJKLMNO")
    kinds = ("nonmonocyte_monocyte_marker_scatterplot", "monocyte_monocyte_marker_scatterplot", "pbmc_correlation_scatterplot")
    rows = [[(next(letters), f"{o}/{t}_{k}.png") for k in kinds] for t in ["cellsweep"] + TOOLS]
    assemble(rows, "Supp12", width_in=7.0, row_labels=["CellSweep"] + TOOL_LABELS, row_label_width=0.07)


def cube_tools(plate, name):   # Figs. S14 (igvf_003) and S15 (igvf_009): the other tools on the Fig6 cell types
    tools = [("soupx", "SoupX"), ("cellbender", "CellBender"), ("decontx", "DecontX")]
    letters = iter("ABCDEF")
    rows = []
    for r, (tool, _) in enumerate(tools):
        row = []
        for t, c, tt, ct in CUBE_CELLTYPES[plate]:
            path = cube_scatter(plate, t, c, tool)
            if r == 0:   # tissue and cell type above the top row only
                path = titled(path, [f"Tissue: {tt}", f"Celltype: {ct}"], f"{F}/_{name}_{t}.png", title_size=14)
            row.append((next(letters), path))
        rows.append(row)
    assemble(rows, name, width_in=7.0, row_labels=[label for _, label in tools], row_label_width=0.07)


def supp15():   # Fig. S15: 8cube.ipynb, Trem2 section
    t = f"{O}/8cubed/trem2"
    assemble([[(L, f"{t}/{g}_counts_per_tissue_trem2.png") for L, g in zip("ABC", ("Alb", "Myh4", "Ttn"))]], "Supp15", width_in=8.4)


def supp16():   # Fig. S16
    i = f"{O}/pbmc8k/idempotency"
    tools = ["cellsweep"] + TOOLS + ["cellsweep_no_empties"]
    labels = ["CellSweep"] + TOOL_LABELS + ["cellsweep\nno_empties"]
    p = [side_label(f"{i}/{t}_per_cell_absolute_difference_overlay.png", lab, f"{F}/_Supp16_{t}.png", label_frac=0.17)
         for t, lab in zip(tools, labels)]
    letters = iter("ABCDEF")
    assemble([[(next(letters), p[r * 2 + c]) for c in range(2)] for r in range(3)], "Supp16", width_in=7.0)


def supp17():   # Fig. S17
    s = f"{O}/simulation1_small_noise"
    letters = iter("ABCDEFGHIJ")
    rows = [[(next(letters), f"{d}/cross_species_joint_scatterplot_total_signal_vs_total_noise_{t}.png"), (next(letters), f"{d}/dotplot_{t}.png")]
            for t, d in [(t, s) for t in TOOLS] + [("cellsweep", s + "_no_empties")]]
    assemble(rows, "Supp17", width_in=7.0, row_labels=TOOL_LABELS + ["cellsweep\nno_empties"], row_label_width=0.07)


def supp19():   # Fig. S19
    p = f"{O}/pbmc8k"
    res = ["0.1", "0.5", "1.0", "1.5", "2.0", "5.0"]
    letters = iter("ABCDEF")
    assemble([[(next(letters), f"{p}/cellsweep_leiden_{r}_cell_scatterplot.png") for r in res[i:i + 3]] for i in (0, 3)], "Supp19", width_in=7.0)


def supp20():   # Fig. S20
    p = f"{O}/pbmc8k"
    assemble([[("A", f"{p}/knee_plot.png"), ("B", f"{p}/cellular_barcodes_upset.png")],
              [(L, f"{p}/cellsweep_emptydrops_vs_thresholding_{k}_scatterplot.png") for L, k in zip("CDE", ("matrix", "cell", "gene"))]],
             "Supp20", width_in=7.0)


def supp21():   # Fig. S21
    p = f"{O}/pbmc8k"
    paths = [f"{p}/cellsweep_empty_droplets_{k}_cell_scatterplot.png" for k in ["1000", "10000", "50000", "100000", "500000"]]
    blank = f"{F}/_blank_supp22.png"   # white panel so the second row's two panels keep the first row's size
    os.makedirs(F, exist_ok=True)
    Image.new("RGBA", Image.open(paths[0]).size, (255, 255, 255, 255)).save(blank)
    letters = iter("ABCDE")
    assemble([[(next(letters), q) for q in paths[:3]], [(next(letters), q) for q in paths[3:]] + [(" ", blank)]], "Supp21", width_in=7.0)


def supp22():   # Fig. S22
    p = f"{O}/pbmc33k"
    dots = combine([f"{p}/dotplot_raw_with_raw_clusters_cellbender_fig2.png", f"{p}/dotplot_cellsweep_with_raw_clusters_cellbender_fig2.png"],
                   ["", ""], f"{F}/_Supp22_dots.png")
    assemble([[(L, f"{p}/cellsweep_vs_raw_{k}_expression_scatterplot.png") for L, k in zip("ABC", ("matrix", "cell", "gene"))],
              [("D", dots), ("E", f"{p}/cellsweep_nonmonocyte_monocyte_marker_scatterplot.png"),
               ("F", f"{p}/cellsweep_monocyte_monocyte_marker_scatterplot.png"), ("G", f"{p}/cellsweep_pbmc_correlation_scatterplot.png")]],
             "Supp22", width_in=7.0)


def supp23():   # Fig. S23
    m = f"{O}/pbmc_mouse_5k"
    assemble([[(L, f"{m}/cellsweep_vs_raw_{k}_expression_scatterplot.png") for L, k in zip("ABC", ("matrix", "cell", "gene"))]], "Supp23", width_in=7.0)


def supp24():   # Fig. S24
    m = f"{O}/melanoma"
    assemble([[(L, f"{m}/cellsweep_vs_raw_{k}_expression_scatterplot.png") for L, k in zip("ABC", ("matrix", "cell", "gene"))]], "Supp24", width_in=7.0)


def sweep(src, name):   # Figs. S26-S29: notebooks/parameter_sweep.ipynb
    return lambda: copy_pdf(f"{O}/parameter_sweep/{src}.pdf", name)


# name -> (figure number, builder). Not listed: Fig1 (schematic, not generated from data).
FIGURES = {
    "Fig2": ("Fig. 2", fig2), "Fig3": ("Fig. 3", fig3), "Fig4": ("Fig. 4", fig4), "Fig5": ("Fig. 5", fig5),
    "Fig6": ("Fig. 6", fig6), "Fig7": ("Fig. 7", fig7), "Fig8": ("Fig. 8", fig8), "Fig9": ("Fig. 9", fig9), "Fig10": ("Fig. 10", fig10),
    "Supp1": ("Fig. S1", supp1), "Supp2": ("Fig. S2", supp2), "Supp3": ("Fig. S3", supp3), "Supp4": ("Fig. S4", supp4),
    "Supp5": ("Fig. S5", supp5), "Supp6": ("Fig. S6", supp6),
    "Supp7": ("Fig. S7", supp7), "Supp8": ("Fig. S8", supp8), "Supp9": ("Fig. S9", supp9), "Supp10": ("Fig. S10", supp10),
    "Supp11": ("Fig. S11", supp11), "Supp12": ("Fig. S12", supp12), "Supp13": ("Fig. S13", lambda: cube_tools("igvf_003", "Supp13")),
    "Supp14": ("Fig. S14", lambda: cube_tools("igvf_009", "Supp14")), "Supp15": ("Fig. S15", supp15), "Supp16": ("Fig. S16", supp16),
    "Supp17": ("Fig. S17", supp17), "Supp18": ("Fig. S18", supp18), "Supp19": ("Fig. S19", supp19), "Supp20": ("Fig. S20", supp20),
    "Supp21": ("Fig. S21", supp21), "Supp22": ("Fig. S22", supp22), "Supp23": ("Fig. S23", supp23), "Supp24": ("Fig. S24", supp24),
    "Supp25": ("Fig. S25", sweep("heatmaps_default", "Supp25")),
    "Supp26": ("Fig. S26", sweep("convergence_check", "Supp26")),
    "Supp27": ("Fig. S27", sweep("kappa_sweep", "Supp27")),
    "Supp28": ("Fig. S28", sweep("init_alpha_sweep", "Supp28")),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help="figures to build (default: all)")
    ap.add_argument("--copy-to", default=None, help="the paper's Figures/ directory; built PDFs are copied there")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    args = ap.parse_args()
    if args.list:
        for name, (label, _) in FIGURES.items():
            print(f"{name:22s} {label}")
        return
    unknown = [n for n in args.names if n not in FIGURES]
    if unknown:
        sys.exit(f"unknown figure(s): {', '.join(unknown)}; see --list")
    failed = []
    for name in args.names or list(FIGURES):
        label, build = FIGURES[name]
        print(f"--- {name} ({label})")
        try:
            build()
        except FileNotFoundError as e:   # a panel is missing: its job in jobs.txt has not run
            print(f"  skipped, missing input: {e.filename or e}")
            failed.append(name)
            continue
        if args.copy_to:
            shutil.copy(os.path.join(F, f"{name}.pdf"), os.path.join(args.copy_to, f"{name}.pdf"))
            print(f"  copied to {args.copy_to}/{name}.pdf")
    if failed:
        sys.exit(f"not built (missing inputs): {', '.join(failed)}")


if __name__ == "__main__":
    main()
