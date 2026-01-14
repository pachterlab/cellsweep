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
include_cellbender = False
print_custom_markers = True
overwrite = False

#!!! erase
parser = argparse.ArgumentParser(description="Run plates processing pipeline.")
parser.add_argument("--plates", nargs="+", default=["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"], help="List of plate names (default: igvf_004)",)
args = parser.parse_args()
plates = args.plates
#!!! erase


# Set max RAM usage in bytes
max_ram_gb = 500  # GB
MAX_RAM = max_ram_gb * 1024**3

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, MAX_RAM))

cellsweep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "8cubed")
eight_cubed_markers_path = os.path.join(data_dir, "8_cube_marker_genes.csv")
gene_id_name_map_path = os.path.join(data_dir, "gene_id_name_map.csv")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "8cubed_tmp")
os.makedirs(out_dir, exist_ok=True)
custom_markers = {
    'CortexHippocampus': ["Snap25", "Nrxn3", "Nrxn1"],  # Snap25: found in plate 3, tissue heart, cluster 36; Nrxn3, Nrxn1: found in plate 11, tissue gastroc, cluster 38
    'Heart': [],
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

try:
    adata_raw_filtered_dict, adata_cellsweep_dict, adata_cellbender_dict = {}, {}, {}
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
        adata_raw_filtered = None  # free memory

        adata_cellsweep_path = os.path.join(data_dir, plate, "cellsweep.h5ad")
        if not os.path.exists(adata_cellsweep_path):
            print(f"  File {adata_cellsweep_path} does not exist, skipping...")
            continue
        adata_cellsweep = ad.read_h5ad(adata_cellsweep_path)
        adata_cellsweep = adata_cellsweep[~adata_cellsweep.obs["is_empty"]].copy()
        # if "leiden" not in adata_cellsweep.obs.columns:  #!!! erase
        #     adata_cellsweep.obs["leiden"] = adata_raw_filtered_dict[plate].obs["leiden"].reindex(adata_cellsweep.obs_names)  #!!! erase
        adata_cellsweep.var_names_make_unique()
        if debug:  # filter to the same 5000 cells as above for debugging
            adata_cellsweep = adata_cellsweep[adata_cellsweep.obs_names.isin(barcodes), :].copy()
        adata_cellsweep_dict[plate] = adata_cellsweep
        adata_cellsweep = None  # free memory

        if not include_cellbender and all_custom_markers_start_with_ensmug:
            continue
        
        adata_cellbender_path = os.path.join(data_dir, plate, "cellbender.h5ad")
        if not os.path.exists(adata_cellbender_path):
            print(f"  File {adata_cellbender_path} does not exist, skipping...")
            continue
        adata_cellbender = ad.read_h5ad(adata_cellbender_path)
        
        if custom_markers is not None and len(custom_markers) > 0 and gene_name_to_id is None:
            if "gene_name" not in adata_cellbender.var.columns or "gene_id" not in adata_cellbender.var.columns or adata_cellbender.var["gene_name"].str.startswith("ENSMUSG").all() or not adata_cellbender.var["gene_id"].str.startswith("ENSMUSG").all():
                gene_name_to_id = pd.read_csv(gene_id_name_map_path).set_index("gene_name")["gene_id"].to_dict()
            else:    
                gene_name_to_id = adata_cellbender.var.set_index("gene_name")["gene_id"].to_dict()
        
        if not all_custom_markers_start_with_ensmug:
            for tissue in custom_markers:
                gene_ids = []
                for gene_name in custom_markers[tissue]:
                    gene_id = gene_name_to_id.get(gene_name)
                    if gene_id is not None:
                        gene_ids.append(gene_id)
                custom_markers[tissue] = gene_ids
            all_custom_markers_start_with_ensmug = True
        if not include_cellbender:
            adata_cellbender = None  # free memory
            continue

        if adata_cellbender.obs_names[0].endswith("-0") or adata_cellbender.obs_names[0].endswith("-1"):  # strip "-0/-1" suffix from barcodes added by CellBender
            adata_cellbender.obs_names = [bc[:-2] for bc in adata_cellbender.obs_names]
        if not adata_cellbender.var_names[0].startswith("ENSMUSG"):
            adata_cellbender.var_names = adata_cellbender.var["gene_id"].astype(str)  # Assign gene_id as the new index
        adata_cellbender.var_names_make_unique()
        if debug:  # filter to the same 5000 cells as above for debugging
            adata_cellbender = adata_cellbender[adata_cellbender.obs_names.isin(barcodes), :].copy()
        adata_cellbender_dict[plate] = adata_cellbender
        adata_cellbender = None  # free memory
    
    if custom_markers is not None and len(custom_markers) > 0 and not all_custom_markers_start_with_ensmug:
        raise ValueError("Custom markers contain gene names that were not found in CellBender data; cannot proceed.")

    # np.random.seed(42)
    # adata_cellsweep_dict["igvf_003"] = adata_cellsweep_dict["igvf_003"][np.random.choice(adata_cellsweep_dict["igvf_003"].n_obs, size=5000, replace=False), :].copy()  #? debug

    dict_of_adata_dicts = {
        "raw": adata_raw_filtered_dict,
        "cellsweep": adata_cellsweep_dict,
        "cellbender": adata_cellbender_dict,
    }
    if not include_cellbender:
        dict_of_adata_dicts.pop("cellbender")
    print("Generating 8cubed plots...")
    breakpoint()
    cs_utils.make_8cubed_plots(dict_of_adata_dicts, eight_cubed_markers_path, custom_markers=custom_markers, gene_name_to_id=gene_name_to_id, print_custom_markers=print_custom_markers, out_dir=out_dir, overwrite=overwrite)
except MemoryError:
    print("❌ Memory limit exceeded — exiting")  # might just print 'Segmentation fault (core dumped)' rather than this
    sys.exit(1)
