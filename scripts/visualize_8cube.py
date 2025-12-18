import os
import anndata as ad
import pandas as pd
import numpy as np
import cellsweep.utils as cs_utils
cellsweep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", "8cubed")
plates = ["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"]  # ["igvf_003"]  #? debug
adata_raw_filtered_dict, adata_cellsweep_dict = {}, {}
for plate in plates:
    print(f"Loading data for plate {plate}...")
    adata_raw_filtered_path = os.path.join(data_dir, plate, "raw_counts_removed_empty_barcodes.h5ad")
    if not os.path.exists(adata_raw_filtered_path):
        print(f"  File {adata_raw_filtered_path} does not exist, skipping...")
        continue
    adata_raw_filtered = ad.read_h5ad(adata_raw_filtered_path)
    adata_raw_filtered.var_names_make_unique()
    adata_raw_filtered_dict[plate] = adata_raw_filtered

    adata_cellsweep_path = os.path.join(data_dir, plate, "cellsweep.h5ad")
    if not os.path.exists(adata_cellsweep_path):
        print(f"  File {adata_cellsweep_path} does not exist, skipping...")
        continue
    adata_cellsweep = ad.read_h5ad(adata_cellsweep_path)
    adata_cellsweep = adata_cellsweep[~adata_cellsweep.obs["is_empty"]].copy()
    adata_cellsweep.var_names_make_unique()
    adata_cellsweep_dict[plate] = adata_cellsweep

eight_cubed_markers_path = os.path.join(data_dir, "8_cube_marker_genes.csv")
out_dir = os.path.join(cellsweep_dir, "notebooks", "output", "8cubed")

# np.random.seed(42)
# adata_cellsweep_dict["igvf_003"] = adata_cellsweep_dict["igvf_003"][np.random.choice(adata_cellsweep_dict["igvf_003"].n_obs, size=5000, replace=False), :].copy()  #? debug

dict_of_adata_dicts = {
    "raw": adata_raw_filtered_dict,
    "cellsweep": adata_cellsweep_dict,
}
cs_utils.make_8cubed_plots(dict_of_adata_dicts, eight_cubed_markers_path, out_dir=out_dir)