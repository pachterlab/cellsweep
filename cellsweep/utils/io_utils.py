"""IO Utils"""

import os
import gzip
import shutil
import subprocess
import numpy as np
import pandas as pd
import logging
from scipy import io, sparse
import anndata as ad
import tarfile
from .logger_utils import setup_logger


def read_r_matrix_into_anndata(file_prefix):
    """
    Read in the matrix output from R SoupX processing into an AnnData object.
    Assumed files: 
    1. {file_prefix}.mtx
    2. {file_prefix}_genes.csv
    3. {file_prefix}_barcodes.csv
    """
    if file_prefix is None:
        print("No file prefix provided for reading R matrix into AnnData. Returning None.")
        return None

    for expected_file in [f"{file_prefix}.mtx", f"{file_prefix}_genes.csv", f"{file_prefix}_barcodes.csv"]:
        if not os.path.exists(expected_file):
            raise FileNotFoundError(f"Expected file not found: {expected_file}")

    X = io.mmread(f"{file_prefix}.mtx").T.tocsr()
    genes = pd.read_csv(f"{file_prefix}_genes.csv", header=None)[0].to_list()
    barcodes = pd.read_csv(f"{file_prefix}_barcodes.csv", header=None)[0].to_list()

    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = barcodes
    return adata

def concat_on_barcodes(adatas):
    # --- 1. union of all barcodes ---
    all_barcodes = sorted(set().union(*[a.obs_names for a in adatas]))

    aligned = []
    for a in adatas:
        # a.obs_names is indexed, so we create a new AnnData with reindexed obs

        # --- reindex the obs ---
        new_obs = a.obs.reindex(all_barcodes)

        # --- reindex X ---
        # Convert X to a DataFrame to reindex rows easily
        X_df = pd.DataFrame.sparse.from_spmatrix(a.X, index=a.obs_names, columns=a.var_names)
        X_df = X_df.reindex(all_barcodes).fillna(0)

        # --- rebuild AnnData ---
        a2 = ad.AnnData(
            X=X_df.sparse.to_coo(),
            obs=new_obs,
            var=a.var.copy()
        )
        aligned.append(a2)

    # --- 3. concatenate genes ---
    combined = ad.concat(aligned, axis=1, join="outer", index_unique=None)

    return combined

def read_kb_mtx_as_adata(kb_count_counts_dir, use_gene_symbols=True):
    import scanpy as sc
    matrix_path = os.path.join(kb_count_counts_dir, "cells_x_genes.mtx")
    barcodes_path = os.path.join(kb_count_counts_dir, "cells_x_genes.barcodes.txt")
    genes_path = os.path.join(kb_count_counts_dir, "cells_x_genes.genes.names.txt" if use_gene_symbols else "cells_x_genes.genes.txt")

    # for kb python's approach: from kb_python.utils import import_matrix_as_anndata, adata = import_matrix_as_anndata(matrix_path, barcodes_path, genes_path)  # avoided to prevent circular import

    adata = sc.read_mtx(matrix_path)
    barcodes = pd.read_csv(barcodes_path, header=None)[0].values
    genes = pd.read_csv(genes_path, header=None)[0].values

    if adata.n_obs == len(genes) and adata.n_vars == len(barcodes):
        adata = adata.T
    
    assert adata.n_vars == len(genes), f"Number of genes in matrix ({adata.n_vars}) does not match number of genes in {genes_path} ({len(genes)})"
    assert adata.n_obs == len(barcodes), f"Number of barcodes in matrix ({adata.n_obs}) does not match number of barcodes in {barcodes_path} ({len(barcodes)})"

    adata.var_names = genes
    adata.obs_names = barcodes
    
    return adata

# anndata object, h5 path, h5ad path, a 10x matrix directory (containing matrix.mtx, genes.tsv, barcodes.tsv), or an R matrix prefix ({prefix}.mtx, {prefix}_genes.csv, {prefix}_barcodes.csv)
def load_adata(adata, multiple_anndatas=False, merge_multiple_adatas=False, gex_only = True, backed=None, logger=None, verbose=0, quiet=False, copy_anndata=True):
    if logger is None:
        logger = setup_logger(verbose=verbose, quiet=quiet)
    if isinstance(adata, str):
        if adata.endswith(".h5ad"):
            if not os.path.exists(adata):
                raise FileNotFoundError(f"File not found: {adata!r}")
            logger.info(f"Loading adata from {adata!r}")
            adata = ad.read_h5ad(adata)
        elif adata.endswith(".h5"):
            if not os.path.exists(adata):
                raise FileNotFoundError(f"File not found: {adata!r}")
            logger.info(f"Loading adata from {adata!r}")
            import scanpy as sc
            import h5py
            with h5py.File(adata, "r") as f:
                genomes = list(f.keys())
            if len(genomes) == 1:
                adata = sc.read_10x_h5(adata, gex_only=gex_only)
            else:
                logger.info(f"Multiple genomes found in {adata!r}: {genomes}. Loading each separately.")
                adatas = []
                for genome in genomes:
                    logger.info(f"Loading genome {genome!r} from {adata!r}")
                    adata_tmp = sc.read_10x_h5(adata, genome=genome, gex_only=gex_only)
                    # adata_tmp.var_names = genome + "_" + adata_tmp.var_names  # at least for the sample hgmm12k dataset, gene names are already prepended with genome
                    adata_tmp.var_names_make_unique()
                    adata_tmp.var['genome'] = genome
                    adatas.append(adata_tmp)
                adata = concat_on_barcodes(adatas)
                assert adata.obs_names.is_unique, f"Non-unique obs names found"
        elif os.path.exists(f"{adata}.mtx"):
            logger.info(f"Loading adata from matrix files with prefix {adata!r}")
            adata = read_r_matrix_into_anndata(adata)
        elif os.path.isdir(adata):
            if multiple_anndatas:
                adata = load_and_merge_anndatas(adata, merge=merge_multiple_adatas, backed=backed)
            else:
                import scanpy as sc
                logger.info(f"Searching recursively for 10x-style dataset under {adata!r}")

                found_dirs = []
                tenx_style, kb_style = False, False
                for root, dirs, files in os.walk(adata):
                    files_lower = [f.lower() for f in files]
                    has_matrix = any(f in files_lower for f in ["matrix.mtx", "matrix.mtx.gz"])
                    has_barcodes = any(f in files_lower for f in ["barcodes.tsv", "barcodes.tsv.gz"])
                    has_genes = any(f in files_lower for f in ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"])
                    if has_matrix and has_barcodes and has_genes:
                        found_dirs.append(root)
                        tenx_style = True
                    
                    has_matrix_kb_python = any(f in files_lower for f in ["cells_x_genes.mtx"])
                    has_barcodes_kb_python = any(f in files_lower for f in ["cells_x_genes.barcodes.txt"])
                    has_genes_kb_python = any(f in files_lower for f in ["cells_x_genes.genes.txt", "cells_x_genes.genes.names.txt"])
                    if has_matrix_kb_python and has_barcodes_kb_python and has_genes_kb_python:
                        found_dirs.append(root)
                        kb_style = True

                if len(found_dirs) == 0:
                    raise FileNotFoundError(f"No valid 10x dataset found under {adata!r}. Expected matrix.mtx, barcodes.tsv, and genes.tsv or features.tsv.")
                elif len(found_dirs) > 1:
                    raise RuntimeError(
                        f"Multiple 10x-style datasets found under {adata!r}:\n" +
                        "\n".join(found_dirs) +
                        "\nPlease specify one directory explicitly."
                    )

                if tenx_style:
                    tenx_dir = found_dirs[0]
                    logger.info(f"Found 10x dataset in {tenx_dir!r}")
                    
                    use_gene_symbols = os.path.exists(os.path.join(tenx_dir, "genes.tsv"))
                    adata = sc.read_10x_mtx(
                        tenx_dir,
                        var_names="gene_symbols" if use_gene_symbols else "gene_ids",
                        make_unique=True
                    )
                elif kb_style:
                    kb_count_counts_unfiltered_dir = found_dirs[0]
                    logger.info(f"Found kb dataset in {kb_count_counts_unfiltered_dir!r}")

                    use_gene_symbols = os.path.exists(os.path.join(kb_count_counts_unfiltered_dir, "cells_x_genes.genes.names.txt"))
                    
                    adata = read_kb_mtx_as_adata(kb_count_counts_unfiltered_dir, use_gene_symbols=use_gene_symbols)
                else:
                    raise ValueError("Unrecognized 10x-style dataset format.")
        else:
            raise ValueError(f"Invalid adata input {adata!r}. Expected a path to an .h5ad file, an .h5 file, a matrix-containing directory, or an AnnData object.")
    elif isinstance(adata, ad.AnnData):
        if copy_anndata:
            adata = adata.copy()
    else:
        raise ValueError(f"Invalid adata input {adata!r}. Expected a path to an .h5ad file, an .h5 file, a matrix-containing directory, or an AnnData object.")
    return adata


def write_10x_like(
    adata,
    parent_dir,
    gzip_output=False,
    genome="genome",
    is_empty_col="is_empty",
    cluster_col="leiden",
    write_raw=True,
    write_filtered=True,
    transpose_matrix=True
):
    """
    Write an AnnData object to a 10x-like directory structure.

    Structure:
      <parent_dir>/
          raw_gene_bc_matrices/
              <genome>/
                barcodes.tsv[.gz]
                genes.tsv[.gz]
                matrix.mtx[.gz]
            filtered_gene_bc_matrices/
              <genome>/
                barcodes.tsv[.gz]
                genes.tsv[.gz]
                matrix.mtx[.gz]
        #   clusters.csv

    Parameters
    ----------
    adata : anndata.AnnData
        Object containing X (cell x gene), obs, var.
        adata.obs must include columns specified by `is_empty_col` and `cluster_col`.

    parent_dir : str
        Output directory path to create/write into.

    gzip_output : bool, default False
        Whether to gzip-compress .tsv and .mtx files.

    is_empty_col : str, default "is_empty"
        Column name in adata.obs marking empty droplets (bool).

    cluster_col : str, default "leiden"
        Column name in adata.obs with cluster labels.

    Returns
    -------
    dict
        Paths of all generated files.
    """
    os.makedirs(parent_dir, exist_ok=True)
    paths = {}

    def _write_10x_subdir(subdir_name, mask):
        subdir = os.path.join(parent_dir, subdir_name, genome)
        os.makedirs(subdir, exist_ok=True)

        suffix = ".gz" if gzip_output else ""
        barcodes_path = os.path.join(subdir, f"barcodes.tsv{suffix}")
        genes_path = os.path.join(subdir, "genes.tsv") if not gzip_output else os.path.join(subdir, "features.tsv.gz")
        matrix_path = os.path.join(subdir, f"matrix.mtx{suffix}")

        if os.path.exists(barcodes_path) and os.path.exists(genes_path) and os.path.exists(matrix_path):
            print(f"Found existing 10x files in {subdir!r}. Skipping write.")
            return subdir  # {"barcodes": barcodes_path, "genes": genes_path, "matrix": matrix_path}

        X = adata.X[mask]
        if sparse.issparse(X):
            X = X.tocoo()
        else:
            X = sparse.coo_matrix(X)
        
        if transpose_matrix:  # needed for scanpy's read 10x function
            X = X.T  # genes x cells

        genes = adata.var_names
        barcodes = adata.obs_names[mask]

        if gzip_output:
            with gzip.open(barcodes_path, "wt") as f:
                for b in barcodes:
                    f.write(f"{b}\n")

            with gzip.open(genes_path, "wt") as f:
                for g in genes:
                    # Write: gene_id, gene_name, feature_type
                    f.write(f"{g}\t{g}\tGene Expression\n")

            with gzip.open(matrix_path, "wb") as f:
                io.mmwrite(f, X)
        else:
            pd.Series(barcodes).to_csv(barcodes_path, index=False, header=False)
            pd.DataFrame({"gene_id": genes, "gene_name": genes}).to_csv(
                genes_path, sep="\t", index=False, header=False
            )
            io.mmwrite(matrix_path, X)

        return subdir  # {"barcodes": barcodes_path, "genes": genes_path, "matrix": matrix_path}

    # Raw matrix (all cells)
    if write_raw:
        paths["raw"] = _write_10x_subdir("raw_gene_bc_matrices", mask=np.ones(adata.n_obs, dtype=bool))

    clusters_path = os.path.join(parent_dir, "clusters.csv")
    paths["clusters"] = clusters_path
    if cluster_col is not None and not os.path.exists(clusters_path) and cluster_col in adata.obs.columns and is_empty_col in adata.obs.columns:
        # Clusters
        adata_filtered = adata[~adata.obs[is_empty_col].astype(bool)].copy()
        adata_filtered.obs[[cluster_col]].to_csv(clusters_path)

    if write_filtered:
        # Filtered matrix (is_empty == False)
        if is_empty_col not in adata.obs.columns:
            raise KeyError(f"Missing column '{is_empty_col}' in adata.obs")

        mask_filtered = ~adata.obs[is_empty_col].astype(bool).values
        paths["filtered"] = _write_10x_subdir("filtered_gene_bc_matrices", mask=mask_filtered)

    # technology
    technology = "10XV3" if gzip_output else "10XV2"
    paths["technology"] = technology

    # soupx inputs: parent_dir, paths["clusters"], soupx_out_prefix
    # decontx inputs: paths["raw"], paths["filtered"], paths["technology"], decontx_out_prefix

    return paths



def load_and_merge_anndatas(directory, join="outer", backed=None, label=None, merge=False):
    """
    Load all .h5ad files in a directory and merge into a single AnnData.
    
    Parameters
    ----------
    directory : str
        Path containing .h5ad files.
    join : str
        How to merge gene/cell sets. Usually "outer" or "inner".
    label : str or None
        Column name in .obs storing the source filename for each batch.
        Use None to disable.
    """
    try:
        from tqdm import tqdm
        tq = tqdm
    except ImportError:
        # no tqdm → identity function
        tq = lambda x, *args, **kwargs: x

    # List .h5ad files
    files = sorted(
        [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".h5ad")]
    )

    if not files:
        raise ValueError(f"No .h5ad files found in: {directory}")

    print(f"Found {len(files)} files:")
    for f in files:
        print("  -", f)

    # Load them
    # adatas = [ad.read_h5ad(f) for f in files]
    adatas = [ad.read_h5ad(f, backed=backed) for f in tq(files, desc="Loading AnnData files")]

    # Add batch labels (optional)
    if label is not None:
        for a, fname in zip(adatas, files):
            a.obs[label] = os.path.basename(fname)

    # Merge
    if merge:
         return ad.concat(adatas, join=join, label=label, fill_value=0)
    else:
        return adatas