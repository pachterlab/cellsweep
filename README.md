# cellsweep

Sweep out noisy counts from single-cell RNA-seq data with CellSweep!

Documentation: https://cellsweep.readthedocs.io

![alt text](https://github.com/pachterlab/cellsweep/blob/main/figures/logo.png?raw=true)

## Install
### Basic use
```
pip install cellsweep
```

### To run notebooks:
```
pip install cellsweep[analysis]
```

### To remake figures from the paper:
```
git clone https://github.com/pachterlab/cellsweep.git
cd cellsweep
conda env create -f environment.yml
pip install cellsweep[analysis]==0.1.1
```

## Quickstart
CellSweep has a single function denoise that takes a raw count matrix in an AnnData object and produces a denoised count matrix in another AnnData object. See a simple, fully worked example in the `notebooks/intro.ipynb` Jupyter Notebook.

### Python API
```python
import cellsweep
adata_cellsweep = cellsweep.denoise(adata_raw_path, adata_out=adata_cellsweep_path)  # see below for expected structure

# for help
help(cellsweep.denoise)
```

### Command line interface
```
cellsweep denoise -o adata_cellsweep.h5ad adata_raw.h5ad  # see below for expected structure

# for help
cellsweep denoise --help
```

There are many utility functions in the `cellsweep.utils` module for data processing, plotting, and analysis. See examples in our Jupyter Notebooks.

## Anndata object
The input Anndata object/h5ad file should have the following structure:
- `adata.X` : cell count matrix (cells x genes)
- `adata.obs`:
    - `adata.obs[celltype_key]`: a column indicating the cell type of each cell.
    - `adata.obs[is_empty_key]` (optional): a boolean column indicating whether each cell is an empty droplet. If not provided, CellSweep will infer empty droplets using the `empty_droplet_method` argument.

If a column is provided in any part of `adata`, then it will take priority over default internal calculations. For example, if `adata.obs[is_empty_key]` is provided, then CellSweep will not infer empty droplets and will use the provided column instead.

Additional column inputs can be provided in advanced use cases. See the documentation for details.

CellSweep returns a denoised Anndata object (and writes it to `adata_out`, if provided) with an updated `adata.X` and the following added fields. By default the empty droplets are removed from the output so it only contains the real cells; pass `keep_empties=True` (CLI: `--keep-empties`) to keep every input barcode.
- `adata.layers["raw"]` : raw count matrix
- `adata.obs`:
    - `adata.obs["alpha_hat"]` : final optimized alpha values
    - `adata.obs["contamination_fraction"]` : total contamination fraction per cell, `(1 - beta_hat) * alpha_hat + beta_hat`
    - `adata.obs["z_hat"]` : final cell-type assignments
- `adata.var`:
    - `adata.var["ambient_hat"]` : final optimized ambient distribution
    - `adata.var["bulk_hat"]` : global noise distribution
- `adata.uns`:
    - `adata.uns["p_hat"]` : final optimized matrix of cell-type profiles (K x G)
    - `adata.uns["beta_hat"]` : final optimized beta
    - `adata.uns["loglike"]` : final log-likelihood (this is the relative log-likelihood, not the complete log-likelihood)


### Multiple samples
CellSweep fits a single ambient profile, global contamination profile and global contamination fraction per run. Ambient contamination is specific to each sample, so run CellSweep separately on each sequencing sample (e.g. each 10x channel or plate), with that sample's own non-cellular barcodes, rather than on a matrix that combines several samples, batches or plates.

## Tutorials
We have several Jupyter Notebooks demonstrating the use of CellSweep for denoising count matrices and analyzing the results. See the `notebooks` folder in the repository.