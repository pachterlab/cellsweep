import os
import sys
import subprocess
import anndata as ad
import numpy as np
import argparse
from cellsweep import denoise_count_matrix
import cellsweep.utils as cs_utils

import resource

# Set max RAM usage in bytes
max_ram_gb = 500  # 300 GB
MAX_RAM = max_ram_gb * 1024**3

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, MAX_RAM))




cellsweep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

adata_raw_parent_dir = "/mnt/data1"
adata_filtered_dir = "/mnt/data1/8_cube_cellbender_raw"

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

cellsweep_max_iter = 1000
cellsweep_beta = 0.1
cellsweep_init_alpha = 0.9

# Docker/Podman settings for the R-based alternate tools (SoupX, DecontX).
# Mirrors the configuration used in notebooks/benchmarking.ipynb.
docker = "podman"  # "docker" or "podman" - if podman and SELinux is enforcing, run `sudo setenforce 0` first
rver_docker_workspace = "/home/ruser/work/cellsweep"
soupx_image = "josephrich98/cellsweep_tutorials:soupx.0.1.0"
decontx_image = "josephrich98/cellsweep_tutorials:decontx.0.1.0"

parser = argparse.ArgumentParser(description="Run plates processing pipeline.")
parser.add_argument("--plates", nargs="+", default=["igvf_003", "igvf_004", "igvf_005", "igvf_007", "igvf_008b", "igvf_009", "igvf_010", "igvf_011"], help="List of plate names (default: all plates)",)
parser.add_argument("--tools", nargs="+", default=["cellsweep"], choices=["cellsweep", "soupx", "decontx"], help="Denoising tools to run (default: cellsweep). Alternate tools: soupx, decontx.",)
args = parser.parse_args()
plates = args.plates
tools = args.tools

if not os.path.exists(adata_raw_parent_dir):
    raise ValueError(f"adata_raw_parent_dir {adata_raw_parent_dir} does not exist.")
if not os.path.exists(adata_filtered_dir):
    raise ValueError(f"adata_filtered_dir {adata_filtered_dir} does not exist.")


plate_to_tissues = {}
for plate in plates:
    plate_dir = os.path.join(adata_raw_parent_dir, plate)
    plate_to_tissues[plate] = [tissue for tissue in os.listdir(plate_dir)]

expected_cells = {
    'igvf_003': 643226,
    'igvf_004': 679838,
    'igvf_005': 722870,
    'igvf_007': 879650,
    'igvf_008b': 606911,
    'igvf_009': 772261,
    'igvf_010': 844946,
    'igvf_011': 806290
}


def _to_container(path):
    """Map a host path under cellsweep_dir to its mounted location inside the container."""
    return path.replace(cellsweep_dir, rver_docker_workspace)


def _write_10x_for_plate(plate):
    """Write 10x-like raw/filtered matrices (and leiden clusters.csv) for a plate.

    Returns the paths dict from cs_utils.write_10x_like. The raw_counts.h5ad for the
    8cube already carries the `is_empty` and `leiden` obs columns the alternate tools need.
    """
    matrix_dir = os.path.join(data_dir, plate, "matrix_tar_files")
    adata_raw_path = os.path.join(data_dir, plate, "raw_counts.h5ad")
    if not os.path.exists(adata_raw_path):
        raise FileNotFoundError(f"Raw counts for plate {plate} not found at {adata_raw_path}")
    print(f"  Writing 10x-like matrices for plate {plate} to {matrix_dir}...")
    adata_raw = ad.read_h5ad(adata_raw_path)
    adata_raw.var_names_make_unique()
    is_empty = adata_raw.obs["is_empty"].astype(bool).values
    # Write the "raw" matrix from empty droplets only. SoupX and DecontX only use the empty
    # droplets from the raw matrix (SoupX estimates the soup from them; DecontX uses them as the
    # ambient background), and the full all-droplet matrix can exceed R's Matrix-package 2^31-1
    # non-zero limit (R's readMM then fails to parse it). Empties-only keeps it readable, gives an
    # identical soup profile for SoupX (cells are excluded from soup estimation regardless), and is
    # the recommended ambient background for DecontX.
    paths = cs_utils.write_10x_like(
        adata_raw,
        matrix_dir,
        gzip_output=False,
        is_empty_col="is_empty",
        cluster_col="leiden",
        write_raw=True,
        write_filtered=True,
        raw_mask=is_empty,
    )
    paths["matrix_dir"] = matrix_dir

    # Upper UMI bound for SoupX's soup/background estimation. The 8cube raw matrices are
    # pre-filtered (empty droplets sit well above SoupX's default (0,100] range), so estimate
    # the soup from the actual empty droplets. estimateSoup selects droplets with UMIs strictly
    # below this bound, so use (max empty-droplet UMI + 1): includes every empty droplet while
    # still excluding all real cells (which have strictly more UMIs than any empty droplet).
    if "n_counts" in adata_raw.obs.columns:
        empty_counts = adata_raw.obs["n_counts"].values[is_empty]
    else:
        empty_counts = np.asarray(adata_raw.X.sum(axis=1)).ravel()[is_empty]
    paths["soup_range_max"] = float(empty_counts.max()) + 1.0 if empty_counts.size else 100.0

    del adata_raw  # memory management
    return paths


def run_soupx(plate):
    out_path = os.path.join(data_dir, plate, "soupx.h5ad")
    if os.path.exists(out_path) and not overwrite:
        print(f"SoupX output for plate {plate} already exists at {out_path}, skipping...")
        return
    print(f"Processing SoupX for plate {plate}...")
    paths = _write_10x_for_plate(plate)
    soupx_out_prefix = os.path.join(data_dir, plate, "soupx_out")
    cmd = [
        docker, "run", "--rm", "--security-opt", "label=disable",
        "-w", "/home/ruser/work",
        "-v", f"{cellsweep_dir}:{rver_docker_workspace}",
        soupx_image,
        "Rscript", _to_container(os.path.join(cellsweep_dir, "scripts", "run_soupx.R")),
        _to_container(paths["matrix_dir"]),
        _to_container(paths["clusters"]),
        _to_container(soupx_out_prefix),
        "leiden",
        str(paths["soup_range_max"]),
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    adata_soupx = _load_soupx_output(soupx_out_prefix)
    adata_soupx.var_names_make_unique()
    print(f"  Writing {out_path}...")
    adata_soupx.write_h5ad(out_path)
    del adata_soupx  # memory management


def _load_soupx_output(soupx_out_prefix):
    """Load SoupX output into an AnnData (cells x genes).

    run_soupx.R writes either a single soupx_out.mtx (small datasets) or, for matrices too large
    for R's Matrix package, per-cluster-batch files (soupx_out_batch{N}.mtx). Batches are combined
    here with scipy, which handles >2^31 non-zeros via 64-bit indices.
    """
    if os.path.exists(soupx_out_prefix + ".mtx"):
        return cs_utils.load_adata(soupx_out_prefix)

    from scipy import io as scio, sparse as sp
    import pandas as pd

    genes = pd.read_csv(soupx_out_prefix + "_genes.csv", header=None)[0].astype(str).values
    n_batches = int(open(soupx_out_prefix + "_nbatches.txt").read().strip())
    print(f"  Combining {n_batches} SoupX cluster-batch outputs...")
    mats, barcodes = [], []
    for b in range(1, n_batches + 1):
        mats.append(scio.mmread(f"{soupx_out_prefix}_batch{b}.mtx").tocsc())  # genes x cells
        barcodes.append(pd.read_csv(f"{soupx_out_prefix}_batch{b}_barcodes.csv", header=None)[0].astype(str).values)
    X = sp.hstack(mats).T.tocsr()  # cells x genes
    obs = pd.DataFrame(index=np.concatenate(barcodes))
    var = pd.DataFrame(index=genes)
    return ad.AnnData(X=X, obs=obs, var=var)


def run_decontx(plate):
    out_path = os.path.join(data_dir, plate, "decontx.h5ad")
    if os.path.exists(out_path) and not overwrite:
        print(f"DecontX output for plate {plate} already exists at {out_path}, skipping...")
        return
    print(f"Processing DecontX for plate {plate}...")
    paths = _write_10x_for_plate(plate)
    decontx_out_prefix = os.path.join(data_dir, plate, "decontx_out")
    cmd = [
        docker, "run", "--rm", "--security-opt", "label=disable",
        "-w", "/home/ruser/work",
        "-v", f"{cellsweep_dir}:{rver_docker_workspace}",
        decontx_image,
        "Rscript", _to_container(os.path.join(cellsweep_dir, "scripts", "run_decontx.R")),
        _to_container(paths["raw"]),
        _to_container(paths["filtered"]),
        paths["technology"],
        _to_container(decontx_out_prefix),
        "--dont_prepend_sample_to_barcodes",
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    adata_decontx = cs_utils.load_adata(decontx_out_prefix)
    adata_decontx.var_names_make_unique()
    adata_decontx.write_h5ad(out_path)
    del adata_decontx  # memory management


def run_cellsweep(plate):
    adata_path_cellsweep = os.path.join(data_dir, plate, "cellsweep.h5ad")
    if os.path.exists(adata_path_cellsweep) and not overwrite:
        print(f"Cellsweep output for plate {plate} already exists at {adata_path_cellsweep}, skipping...")
        return
    print(f"Processing Cellsweep for plate {plate}...")
    adata_raw = ad.read_h5ad(os.path.join(data_dir, plate, "raw_counts.h5ad"))
    cellsweep_log_path = os.path.join(data_dir, plate, "cellsweep.log")

    adata_cellsweep = denoise_count_matrix(adata_raw, adata_out=adata_path_cellsweep, beta=cellsweep_beta, freeze_ambient_profile=True, init_alpha=cellsweep_init_alpha, max_iter=cellsweep_max_iter, empty_droplet_method="threshold", expected_cells=expected_cells[plate], threads=threads, verbose=verbose, log_file=cellsweep_log_path)

    adata_cellsweep = None  # memory management
    del adata_raw   # memory management


tool_runners = {
    "cellsweep": run_cellsweep,
    "soupx": run_soupx,
    "decontx": run_decontx,
}

try:
    for plate in plates:
        for tool in tools:
            tool_runners[tool](plate)
except MemoryError:
    print("❌ Memory limit exceeded — exiting")  # might just print 'Segmentation fault (core dumped)' rather than this
    sys.exit(1)
