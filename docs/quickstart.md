# Quickstart

CellSweep has a single main function, {func}`cellsweep.denoise`. It takes a raw count
matrix in an AnnData object and returns a denoised count matrix in another AnnData object.

The input must contain **all barcodes**, including empty droplets (e.g. Cell Ranger's
`raw_feature_bc_matrix`, not the filtered matrix), and a `celltype` label for each cell.
See {doc}`input_format` for the full specification.

## Python API

```python
import cellsweep

adata_cellsweep = cellsweep.denoise(
    "adata_raw.h5ad",                  # path to an .h5ad file, or an AnnData object
    adata_out="adata_cellsweep.h5ad",  # optional: also write the result to disk
    expected_cells=1000,               # helps when empty droplets must be inferred
    threads=8,
)
```

The returned object holds the denoised counts in `adata_cellsweep.X` and the raw counts in
`adata_cellsweep.layers["raw"]`. The empty droplets are used to fit the model but are removed
from the output by default; pass `keep_empties=True` to keep them. See {ref}`outputs` for the
other fields CellSweep adds.

For the full list of options:

```python
help(cellsweep.denoise)
```

## Command line interface

```bash
cellsweep denoise -o adata_cellsweep.h5ad adata_raw.h5ad

# for help
cellsweep denoise --help
```

See {doc}`cli` for every flag.

## Multiple samples

CellSweep fits a single ambient profile, global contamination profile and global
contamination fraction per run. Ambient contamination is specific to each sample, so run
CellSweep separately on each sequencing sample (e.g. each 10x channel or plate), with that
sample's own non-cellular barcodes, rather than on a matrix that combines several samples,
batches or plates.

## Next steps

- Work through the {doc}`tutorials/intro` tutorial for a complete PBMC example, from
  downloading data to cell-type annotation and plotting.
- Browse the utilities in {mod}`cellsweep.utils` for preprocessing, I/O and plotting.
