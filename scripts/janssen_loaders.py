"""Shared loaders for the Janssen et al. (2023) snRNA-seq benchmark.

Every correction method is returned on the same (cell x gene) grid: the 16,714 / 4,266
nuclei and 28,679 genes of the published Seurat objects (CellRanger's features minus the
13 mitochondrial genes, which Janssen et al. drop).
"""

import os

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
import h5py

CS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CS_DIR, "notebooks", "data", "janssen2023")
OUT_DIR = os.path.join(CS_DIR, "notebooks", "output", "janssen2023")
TOOL_DIR = os.path.join(OUT_DIR, "tools")

REPLICATES = ["nuc2", "nuc3"]
METHOD_ORDER = ["raw", "CellSweep", "CellBender", "DecontX", "DecontX (empty)", "SoupX"]
# The ten proximal-tubule markers Janssen et al. use for their marker-leakage evaluation
# (Snakemake_benchmark/input/top10_PT_markers.RDS).
PT_MARKERS = ["Slc34a1", "Miox", "Pck1", "Slc4a4", "Ttc36", "Lrp2", "Fbp1", "Cyp2e1",
              "Fut9", "Khk"]


def axes(rep):
    """The gene and cell names shared by every matrix."""
    d = os.path.join(DATA_DIR, rep)
    genes = np.array(open(os.path.join(d, "seurat_genes.txt")).read().split("\n")[:-1] or
                     open(os.path.join(d, "seurat_genes.txt")).read().split("\n"))
    genes = np.array([g for g in genes if g])
    cells = np.array([c for c in open(os.path.join(d, "seurat_cells.txt")).read().split("\n") if c])
    return genes, cells


def metadata(rep):
    md = pd.read_csv(os.path.join(DATA_DIR, rep, "seurat_metadata.csv"))
    md = md.set_index("cell")
    _, cells = axes(rep)
    return md.loc[cells]


def _read_10x_h5(path):
    """CellRanger / CellBender HDF5 -> (csc genes x barcodes, gene names, barcodes)."""
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = g["shape"][:]
        m = sp.csc_matrix((g["data"][:].astype(np.float64), g["indices"][:], g["indptr"][:]),
                          shape=tuple(shape))
        names = np.array([x.decode() for x in g["features"]["name"][:]])
        bcs = np.array([x.decode() for x in g["barcodes"][:]])
    return m, _make_unique(names), bcs


def _make_unique(names):
    """R's make.unique, which is how Seurat/CellRanger de-duplicate gene symbols."""
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return np.array(out)


def _align(m, row_names, col_names, genes, cells):
    """Reorder a (gene x barcode) matrix onto the shared axes, returned cells x genes."""
    ri = pd.Index(row_names).get_indexer(genes)
    ci = pd.Index(col_names).get_indexer(cells)
    assert (ri >= 0).all() and (ci >= 0).all(), "missing genes or cells"
    return sp.csr_matrix(m.tocsr()[ri][:, ci].T)


def _read_mtx_triplet(prefix, genes, cells):
    m = scipy.io.mmread(prefix + ".mtx").tocsr()
    g = np.array([x for x in open(prefix + "_genes.csv").read().split("\n") if x])
    b = np.array([x for x in open(prefix + "_barcodes.csv").read().split("\n") if x])
    return _align(m, g, b, genes, cells)


def load_method(rep, method, genes, cells):
    """Corrected counts for one method, as a cells x genes CSR matrix."""
    if method == "raw":
        m, g, b = _read_10x_h5(os.path.join(DATA_DIR, rep, "raw_feature_bc_matrix.h5"))
        return _align(m, g, b, genes, cells)
    if method == "CellSweep":
        # The h5ad also holds the ~1.5M noncellular barcodes; read the CSR blocks straight
        # out of the file rather than materialising the whole AnnData.
        path = os.path.join(DATA_DIR, rep, f"{rep}_output_cellsweep_empty100.h5ad")
        with h5py.File(path, "r") as f:
            shape = tuple(f["X"].attrs["shape"])
            m = sp.csr_matrix((f["X"]["data"][:].astype(np.float64), f["X"]["indices"][:],
                               f["X"]["indptr"][:]), shape=shape)
            obs = np.array([x.decode() for x in f["obs"]["_index"][:]])
            n_var = f["var"]["_index"].shape[0]
        # The h5ad was written by scanpy, whose var_names_make_unique uses "-1" suffixes where
        # Seurat's make.unique uses ".1"; match on feature position in the CellRanger file
        # instead, which both share.
        with h5py.File(os.path.join(DATA_DIR, rep, "raw_feature_bc_matrix.h5"), "r") as f:
            raw_names = _make_unique(np.array([x.decode() for x in f["matrix"]["features"]["name"][:]]))
        assert len(raw_names) == n_var
        ci = pd.Index(obs).get_indexer(cells)
        gi = pd.Index(raw_names).get_indexer(genes)
        assert (ci >= 0).all() and (gi >= 0).all()
        return sp.csr_matrix(m[ci][:, gi])
    if method == "CellBender":
        # The main output keeps every input barcode; the _filtered companion keeps only the
        # barcodes CellBender itself calls as cells, which is not the same set as Janssen's.
        path = os.path.join(TOOL_DIR, f"cb_{rep}", f"{rep}_cellbender.h5")
        m, g, b = _read_10x_h5(path)
        return _align(m, g, b, genes, cells)
    suffix = {"DecontX": "decontx", "DecontX (empty)": "decontx_empty", "SoupX": "soupx"}[method]
    return _read_mtx_triplet(os.path.join(TOOL_DIR, f"{rep}_{suffix}"), genes, cells)


# Broadly expressed genes, used as the "constitutive" half of the dot plot: uniform across
# every annotated nucleus type in the uncorrected data (min/max mean CP10K within ~2x) and
# not proximal-tubule restricted. Removing these is over-correction, not decontamination.
CONSTITUTIVE = ["Malat1", "Fth1", "Ftl1", "Tpt1", "Psap", "App"]


def celltype_markers(mean_expr, min_cp10k=8.0, n_per_type=8):
    """Per-cell-type marker genes, ranked by specificity against proximal tubule.

    `mean_expr` is a gene x cell-type table of mean CP10K in the uncorrected matrix. Ambient
    contamination in this dataset flows almost entirely from PT (85% of the nuclei), so
    scoring a gene by its expression in a type relative to PT is conservative: contamination
    can only pull the score down, never manufacture a spurious marker.
    """
    out = {}
    for t in mean_expr.columns:
        if t == "PT":
            continue
        score = mean_expr[t] / (mean_expr["PT"] + 1.0)
        eligible = score[(mean_expr[t] >= min_cp10k) & ~mean_expr.index.isin(PT_MARKERS)]
        out[t] = list(eligible.sort_values(ascending=False).head(n_per_type).index)
    return out


def write_soup_profile(rep, umi_max=100):
    """Precompute SoupX's soup profile: gene counts pooled over droplets with 1..umi_max-1 UMIs.

    This is exactly what SoupX's `estimateSoup(soupRange = c(0, 100))` computes. Doing it here
    avoids subsetting a 6.8M-column matrix inside R, which is the one step that makes the stock
    SoupX path impractical on a full CellRanger raw matrix.
    """
    path = os.path.join(DATA_DIR, rep, "raw_feature_bc_matrix.h5")
    with h5py.File(path, "r") as f:
        g = f["matrix"]                                  # CSC: one column per barcode
        data = g["data"][:].astype(np.int64)
        indices, indptr = g["indices"][:], g["indptr"][:]
        names = _make_unique(np.array([x.decode() for x in g["features"]["name"][:]]))
        n_gene = int(g["shape"][0])
    lengths = np.diff(indptr)
    totals = np.zeros(len(lengths), dtype=np.int64)
    nonempty = lengths > 0
    totals[nonempty] = np.add.reduceat(data, indptr[:-1][nonempty])
    in_soup = np.repeat((totals > 0) & (totals < umi_max), lengths)
    counts = np.bincount(indices[in_soup], weights=data[in_soup], minlength=n_gene)
    out = os.path.join(DATA_DIR, rep, "soup_profile.csv")
    pd.DataFrame({"gene": names, "counts": counts}).to_csv(out, index=False)
    return out
