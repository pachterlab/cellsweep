import os
import sys
import numpy as np
import pandas as pd
import itertools
import yaml
import requests
import matplotlib.pyplot as plt
import anndata as ad
from collections import OrderedDict
import seaborn as sns
import scanpy as sc
from cellsweep import denoise_count_matrix
import cellsweep.utils as cs_utils

cellsweep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

adata_raw_parent_dir = "/mnt/data1"
adata_filtered_dir = "/mnt/data1/8_cube"

debug = False
verbose = 2  # 2 debug, 1 info, 0 warning, -1 error, -2 critical
overwrite = False  # overwrite existing files
threads = 32  # for cellsweep and CellBender (if use_cuda=False)


dataset_name = "8cubed"

# create directories for data, output
data_dir = os.path.join(cellsweep_dir, "notebooks", "data", dataset_name)
os.makedirs(data_dir, exist_ok=True)

out_dir = os.path.join(cellsweep_dir, "notebooks", "output", dataset_name)
os.makedirs(out_dir, exist_ok=True)

eight_cubed_markers_path = os.path.join(data_dir, "8_cube_marker_genes.csv")

cellsweep_max_iter = 500
cellsweep_beta = 0.1
cellsweep_init_alpha = 0.9

if not os.path.exists(adata_raw_parent_dir):
    raise ValueError(f"adata_raw_parent_dir {adata_raw_parent_dir} does not exist.")
if not os.path.exists(adata_filtered_dir):
    raise ValueError(f"adata_filtered_dir {adata_filtered_dir} does not exist.")


plate_to_tissues = {}
plates = ["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"]
for plate in plates:
    plate_dir = os.path.join(adata_raw_parent_dir, plate)
    plate_to_tissues[plate] = [tissue for tissue in os.listdir(plate_dir)]
        
# plates = ["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"]
# adata_raw_dict = {}
# for plate in plates:
#     adata_raw_dict[plate] = ad.read_h5ad(os.path.join(data_dir, plate, "raw_counts.h5ad"))

expected_cells = {
    'igvf_003': 639885,
    'igvf_004': 745458,
    'igvf_005': 694748,
    'igvf_007': 786528,
    'igvf_008b': 504134,
    'igvf_009': 641097,
    'igvf_010': 659512,
    'igvf_011': 721541
}


for plate in plates:   #? memory management
    adata_path_cellsweep = os.path.join(data_dir, plate, "cellsweep.h5ad")
    if os.path.exists(adata_path_cellsweep) and not overwrite:
        print(f"  Cellsweep output for plate {plate} already exists at {adata_path_cellsweep}, skipping...")
        continue
    print(f"Processing Cellsweep for plate {plate}...")
    adata_raw = ad.read_h5ad(os.path.join(data_dir, plate, "raw_counts.h5ad"))   #? memory management
    cellsweep_log_path = os.path.join(data_dir, plate, "cellsweep.log")
    
    adata_cellsweep = denoise_count_matrix(adata_raw, adata_out=adata_path_cellsweep, beta=cellsweep_beta, freeze_ambient_profile=True, init_alpha=cellsweep_init_alpha, max_iter=cellsweep_max_iter, empty_droplet_method="threshold", expected_cells=expected_cells[plate], threads=threads, verbose=verbose, log_file=cellsweep_log_path)

    del adata_raw   #? memory management
    del adata_cellsweep   #? memory management
