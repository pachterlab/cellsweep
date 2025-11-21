import os
import subprocess
import scanpy as sc
import cellmender.utils as cm_utils

cellmender_dir = "/data/cellmender"   #!!!! os.path.dirname(os.path.abspath(""))
rver_docker_workspace = "/home/ruser/work/cellmender"

adata_path_raw = '/data/cellmender/notebooks/data/pbmc8k/idempotency/pbmc8k_raw_gene_bc_matrices_h5.h5'
matrix_tar_files_dir = '/data/cellmender/notebooks/data/pbmc8k/idempotency/matrix_tar_files'
raw_tar_file_dir = os.path.join(matrix_tar_files_dir, "raw_gene_bc_matrices", "GRCh38")
filtered_tar_file_dir = os.path.join(matrix_tar_files_dir, "filtered_gene_bc_matrices", "GRCh38")
expected_cells = 8381
dataset_name = "pbmc8k"  # options: pbmc8k
iterations = 4
verbose = 2  # 2 debug, 1 info, 0 warning, -1 error, -2 critical
overwrite = False  # overwrite existing files
scar_env = "/home/jrich/miniconda3/envs/scar_gpu"
use_cuda = True
threads = 8
data_dir = '/data/cellmender/notebooks/data/pbmc8k/idempotency'


adata_raw = cm_utils.load_adata(adata_path_raw, verbose=verbose)
adata_raw.var_names_make_unique()

adata_raw = cm_utils.infer_empty_droplets(adata_raw, method="threshold", expected_cells=expected_cells, verbose=verbose)  # adds adata.obs["is_empty"]

matrix_tar_files_dir_scar = matrix_tar_files_dir
raw_tar_file_dir_scar = raw_tar_file_dir
filtered_tar_file_dir_scar = filtered_tar_file_dir
adatas_scar = [adata_raw]
adatas_scar_concat = [adata_raw]
for it in range(1, iterations+1):
    print(f"Iteration {it} / {iterations}")
    scar_out_prefix = os.path.join(data_dir, f"scar_iteration{it}")
    adata_path_scar = os.path.join(data_dir, f"adata_scar_iteration{it}.h5ad")
    if not os.path.exists(f"{adata_path_scar}.mtx") or overwrite:
        runtime = "--cuda" if use_cuda else ""
        conda_run_flag = "-p" if "/" in scar_env else "-n"
        subprocess.run(f"conda run {conda_run_flag} {scar_env} python {cellmender_dir}/scripts/run_scar.py -r {raw_tar_file_dir_scar} -f {filtered_tar_file_dir_scar} -o {adata_path_scar} {runtime} --epochs 200", shell=True, check=True)
        
    adata_scar = cm_utils.load_adata(adata_path_scar)
    adata_scar.var_names_make_unique()
    adatas_scar.append(adata_scar)

    matrix_tar_files_dir_scar = f"{scar_out_prefix}_matrix_tar_files"
    raw_tar_file_dir_scar = os.path.join(matrix_tar_files_dir_scar, "raw_gene_bc_matrices", "GRCh38")
    filtered_tar_file_dir_scar = os.path.join(matrix_tar_files_dir_scar, "filtered_gene_bc_matrices", "GRCh38")

    # merge adatas_scar_concat[it-1] into it, and fill NaN with True
    adata_scar.obs["is_empty"] = False
    adata_prev = adatas_scar_concat[it-1]
    cells_prev = set(adata_prev.obs_names)  # 1. find missing cells
    cells_curr = set(adata_scar.obs_names)
    missing_cells = list(cells_prev - cells_curr)
    adata_missing = adata_prev[missing_cells].copy()  # 2. subset missing cells
    adata_missing.obs["is_empty"] = True
    adata_scar = sc.concat([adata_scar, adata_missing], join="outer", merge="unique", label=None, index_unique=None)  # 3. concat
    adatas_scar_concat.append(adata_scar)
    
    if not os.path.exists(matrix_tar_files_dir_scar) or overwrite:
        _ = cm_utils.write_10x_like(adata_scar, matrix_tar_files_dir_scar, gzip_output=False, is_empty_col="is_empty", cluster_col=None, genome="GRCh38", write_raw=True, write_filtered=True)